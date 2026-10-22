# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
import io
from types import SimpleNamespace

import pytest
import torch
from transformer_engine.pytorch.optimizers import FusedAdam

from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """These tests use synthetic tensors and need no downloaded test assets."""


@pytest.fixture(scope="module", autouse=True)
def distributed_gpu():
    Utils.initialize_distributed()


def make_optimizer(params, offload_fraction, overlap=False, **kwargs):
    return HybridDeviceOptimizer(
        params,
        offload_fraction=offload_fraction,
        cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=FusedAdam,
        param_update_in_fp32=True,
        overlap_cpu_optimizer_d2h_h2d=overlap,
        lr=0.01,
        weight_decay=0.1,
        **kwargs,
    )


def make_params():
    return [
        torch.nn.Parameter(torch.linspace(0.25, 1.0, 16, device="cuda", dtype=dtype))
        for dtype in (torch.bfloat16, torch.bfloat16, torch.float32)
    ]


def take_step(optimizer, params, step):
    for i, param in enumerate(params):
        param.grad = torch.full_like(param, (i + 1) * step / 16)
    optimizer.step()
    optimizer.zero_grad()


def test_deferred_optimizer_allocates_masters_only_for_owned_shards():
    param = torch.nn.Parameter(torch.ones(1024 * 1024, device="cuda", dtype=torch.bfloat16))
    allocated = torch.cuda.memory_allocated()
    deferred = make_optimizer([param], 0.0, defer_init=True)
    assert torch.cuda.memory_allocated() == allocated
    assert "defer_init" not in deferred.defaults
    assert deferred.param_groups[0]["params"][0] is param

    # DistributedOptimizer replaces the deferred object after assigning shards.
    shard = param.detach()[: param.numel() // 2]
    initialized = HybridDeviceOptimizer([shard], **deferred.defaults)
    assert initialized.param_to_fp32_param[shard].numel() == param.numel() // 2


@pytest.mark.parametrize("offload_fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("overlap", [False, True])
def test_dummy_step_preserves_loaded_compute_and_stale_master_weights(offload_fraction, overlap):
    params = make_params()
    optimizer = make_optimizer(params, offload_fraction, overlap)
    masters = {p: p.clone() for p in optimizer.param_to_fp32_param.values()}
    # Model loading happens after construction, leaving masters stale until state loading.
    with torch.no_grad():
        for param in params:
            param.fill_(2.0)
    before = [param.clone() for param in params]
    optimizer.dummy_step()
    torch.cuda.synchronize()

    for param, expected in zip(params, before):
        assert torch.equal(param, expected)
        assert param.grad is None
    for master, expected in masters.items():
        assert torch.equal(master, expected)
    assert all(group["lr"] == 0.01 for group in optimizer.param_groups)
    for state in optimizer.state.values():
        assert torch.count_nonzero(state["exp_avg"]) == 0
        assert torch.count_nonzero(state["exp_avg_sq"]) == 0


@pytest.mark.parametrize("offload_fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("overlap", [False, True])
def test_safe_checkpoint_resume_matches_uninterrupted_adam(offload_fraction, overlap):
    params = make_params()
    optimizer = make_optimizer(params, offload_fraction, overlap)
    for step in range(1, 4):
        take_step(optimizer, params, step)

    checkpoint = optimizer.state_dict()
    # DistributedOptimizer saves a common step in outer groups for FusedAdam.
    for group in checkpoint["param_groups"]:
        group["step"] = 3
    serialized = io.BytesIO()
    torch.save(checkpoint, serialized)
    serialized.seek(0)
    checkpoint = torch.load(serialized, map_location="cpu", weights_only=True)

    restored_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
    restored = make_optimizer(restored_params, offload_fraction, overlap)
    restored.load_state_dict(checkpoint)
    for step in range(4, 7):
        take_step(optimizer, params, step)
        take_step(restored, restored_params, step)
        torch.cuda.synchronize()
        if restored.gpu_optimizer:
            assert all(g["step"] == step for g in restored.gpu_optimizer.param_groups)
        for param, restored_param in zip(params, restored_params):
            assert torch.equal(param, restored_param)
            for key in ("exp_avg", "exp_avg_sq", "master_param"):
                assert torch.equal(optimizer.state[param][key], restored.state[restored_param][key])


def test_dp_zero_scatter_restores_state_devices_and_resumes_adam():
    """Exercise the real Gloo scatter loader with mixed CPU/GPU-owned states."""
    params = make_params()[:2]
    optimizer = make_optimizer(params, 0.5)
    for step in range(1, 4):
        take_step(optimizer, params, step)

    restored_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
    restored = make_optimizer(restored_params, 0.5)
    common_state = copy.deepcopy(optimizer.state_dict())
    for group in common_state["param_groups"]:
        group["step"] = 3
    restored.load_state_dict(common_state)
    # Only the ranks' own slices are used after scattering the coalesced buffers.
    world_size = torch.distributed.get_world_size()
    local_numel = sum(p.numel() for p in params)
    world_numel = world_size * local_numel
    dtype = (torch.bfloat16, torch.float32)
    wrapper = DistributedOptimizer.__new__(DistributedOptimizer)
    wrapper.optimizer = restored
    wrapper.config = SimpleNamespace(
        optimizer="adam", use_precision_aware_optimizer_no_fp8_or_ds_fp8=True
    )
    wrapper.data_parallel_group_gloo = torch.distributed.new_group(backend="gloo")
    wrapper.model_param_group_index_map = {p: (0, i) for i, p in enumerate(restored_params)}
    wrapper.gbuf_ranges = [
        {
            dtype: [
                {
                    "param_map": {
                        p: {"gbuf_local": Range(i * 16, (i + 1) * 16)}
                        for i, p in enumerate(restored_params)
                    }
                }
            ]
        }
    ]
    wrapper.buffers = [
        SimpleNamespace(
            params=restored_params,
            numel_unpadded=world_numel,
            buckets=[
                SimpleNamespace(
                    grad_data=torch.empty(world_numel, device="cuda"), numel_unpadded=world_numel
                )
            ],
        )
    ]
    state = {"numel_unpadded": world_numel}
    for key in ("param", "exp_avg", "exp_avg_sq"):
        state_key = "master_param" if key == "param" else key
        local = torch.cat([optimizer.state[p][state_key].detach().cpu() for p in params])
        state[key] = local.repeat(world_size)
    checkpoint = {0: {dtype: state}} if torch.distributed.get_rank() == 0 else None
    wrapper.load_parameter_state_from_dp_zero(checkpoint)

    for inner_optimizer in restored.sub_optimizers:
        for inner_param, inner_state in inner_optimizer.state.items():
            original = restored.inner_param_to_orig_param[inner_param]
            assert inner_state is restored.state[original]
            assert inner_state["exp_avg"].device == inner_param.device
            assert inner_state["exp_avg_sq"].device == inner_param.device
    for step in range(4, 7):
        take_step(optimizer, params, step)
        take_step(restored, restored_params, step)
        torch.cuda.synchronize()
        for param, restored_param in zip(params, restored_params):
            assert torch.equal(param, restored_param)
            for key in ("exp_avg", "exp_avg_sq", "master_param"):
                assert torch.equal(optimizer.state[param][key], restored.state[restored_param][key])
