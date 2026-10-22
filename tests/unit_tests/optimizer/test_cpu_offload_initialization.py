# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Distributed offload must not allocate masters before parameter sharding."""

import gc
import io
import os
import weakref
from types import SimpleNamespace

import pytest
import torch
from transformer_engine.pytorch.optimizers import FusedAdam

import megatron.core.optimizer as optimizer_module
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer


@pytest.mark.parametrize('fraction', [0.0, 0.5, 1.0])
def test_deferred_offload_allocates_only_shard_masters_and_updates(fraction):
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    params = [
        torch.nn.Parameter(
            torch.full((256, 256), 0.25 + index / 16, device='cuda', dtype=torch.bfloat16)
        )
        for index in range(4)
    ]
    options = dict(
        offload_fraction=fraction,
        cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=FusedAdam,
        param_update_in_fp32=True,
        overlap_cpu_optimizer_d2h_h2d=False,
        lr=0.01,
        betas=(0.9, 0.995),
        eps=1e-8,
        weight_decay=0.1,
    )
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        before = torch.cuda.memory_allocated()
        shell = HybridDeviceOptimizer(params, defer_init=True, **options)
        assert torch.cuda.memory_allocated() == before
        shell_ref = weakref.ref(shell)
        # Emulate DistributedOptimizer's ownership: the first half of each tensor.
        shards = [param.detach().view(-1)[: param.numel() // 2] for param in params]
        optimizer = HybridDeviceOptimizer(shards, **shell.defaults)
        del shell
        assert shell_ref() is None  # No unsharded optimizer survives with GC disabled.
        masters = list(optimizer.param_to_inner_param.values())
        assert sum(p.numel() for p in masters) == sum(p.numel() for p in shards)
        assert all(p.dtype == torch.float32 for p in masters)
        expected_cpu = sum(p.numel() for p in shards) * fraction
        assert sum(p.numel() for p in masters if p.device.type == 'cpu') == expected_cpu
        reference = [p.detach().float().clone() for p in shards]
        ref_optimizer = FusedAdam(reference, lr=0.01, betas=(0.9, 0.995), eps=1e-8, weight_decay=0.1)
        for _ in range(2):
            for shard, ref in zip(shards, reference):
                shard.decoupled_grad = torch.full_like(shard, 0.125, dtype=torch.float32)
                ref.grad = torch.full_like(ref, 0.125)
            optimizer.step()
            ref_optimizer.step()
            for master, ref in zip(masters, reference):
                torch.testing.assert_close(master.to('cuda'), ref, rtol=1e-6, atol=1e-7)
            for index, (full, shard, ref) in enumerate(zip(params, shards, reference)):
                torch.testing.assert_close(shard, ref.bfloat16(), rtol=0, atol=0)
                # Updating one owned shard must leave the rest of the model alone.
                tail = full.detach().view(-1)[shard.numel() :]
                assert torch.all(tail == tail[0])
                assert tail[0] == 0.25 + index / 16
        checkpoint = io.BytesIO()
        torch.save(optimizer.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = torch.load(checkpoint, weights_only=True)
        assert restored['param_groups'][0]['cpu_optimizer_cls'] is torch.optim.AdamW
        assert restored['param_groups'][0]['gpu_optimizer_cls'] is FusedAdam
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()


def test_distributed_factory_defers_unsharded_offload(monkeypatch):
    captured = []

    def wrap(base, *args, **kwargs):
        assert isinstance(base, HybridDeviceOptimizer)
        assert not hasattr(base, 'param_to_fp32_param')
        captured.append(base)
        return SimpleNamespace()

    monkeypatch.setattr(optimizer_module, 'DistributedOptimizer', wrap)
    param = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
    optimizer_module._get_megatron_optimizer_based_on_param_groups(
        config=OptimizerConfig(
            bf16=True,
            optimizer='adam',
            lr=0.01,
            optimizer_cpu_offload=True,
            use_precision_aware_optimizer=True,
            use_distributed_optimizer=True,
        ),
        model_chunks=[torch.nn.Module()],
        param_groups=[{'params': [param]}],
        pg_collection=SimpleNamespace(tp=None),
    )
    assert len(captured) == 1


def test_distributed_offload_checkpoint_continuation(tmp_path):
    import copy

    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    if int(os.environ.get('WORLD_SIZE', 1)) != 2:
        pytest.skip('requires exactly two ranks for mixed CPU/GPU owned shards')
    Utils.initialize_model_parallel(context_parallel_size=2)
    try:
        config = TransformerConfig(
            num_layers=1, hidden_size=256, num_attention_heads=4,
            context_parallel_size=2, bf16=True, params_dtype=torch.bfloat16,
        )

        def build(state=None):
            module = torch.nn.Module()
            module.config = config
            module.weights = torch.nn.ParameterList([
                torch.nn.Parameter(torch.zeros(256, 256, dtype=torch.bfloat16, device='cuda'))
                for _ in range(8)
            ])
            if state is not None:
                module.load_state_dict(state)
            model = DistributedDataParallel(
                config,
                DistributedDataParallelConfig(
                    use_distributed_optimizer=True, grad_reduce_in_fp32=True
                ),
                module,
            )
            optimizer = optimizer_module.get_megatron_optimizer(
                OptimizerConfig(
                    bf16=True, optimizer='adam', lr=1e-6, use_distributed_optimizer=True,
                    optimizer_cpu_offload=True, optimizer_offload_fraction=0.5,
                    use_precision_aware_optimizer=True, clip_grad=0.0,
                ),
                [model],
            )
            return model, optimizer

        def step(model, optimizer):
            model.zero_grad_buffer()
            optimizer.zero_grad()
            for param in model.parameters():
                param.main_grad.fill_(0.125)
            assert optimizer.step()[0]
            torch.cuda.synchronize()

        continuous, opt_a = build()
        for _ in range(3):
            step(continuous, opt_a)
        saved_model = copy.deepcopy(continuous.module.state_dict())
        metadata = copy.deepcopy(opt_a.state_dict())
        paths = [str(tmp_path / 'optimizer.pt')]
        torch.distributed.broadcast_object_list(paths, src=0)
        opt_a.save_parameter_state(paths[0])
        # The trainer constructs the optimizer before loading model weights.
        # Its masters are still the initial values when dummy_step allocates state.
        resumed, opt_b = build()
        resumed.module.load_state_dict(saved_model)
        rng = torch.cuda.get_rng_state()
        opt_b.load_state_dict(metadata)
        opt_b.load_parameter_state(paths[0])
        torch.testing.assert_close(torch.cuda.get_rng_state(), rng, rtol=0, atol=0)
        for name, value in resumed.module.state_dict().items():
            torch.testing.assert_close(value, saved_model[name], rtol=0, atol=0)
        for _ in range(2):
            step(continuous, opt_a)
            step(resumed, opt_b)
            for name, value in resumed.module.state_dict().items():
                torch.testing.assert_close(
                    value, continuous.module.state_dict()[name], rtol=0, atol=0
                )
            for child_a, child_b in zip(opt_a.chained_optimizers, opt_b.chained_optimizers):
                a, b = child_a.optimizer, child_b.optimizer
                for group_a, group_b in zip(a.param_groups, b.param_groups):
                    for param_a, param_b in zip(group_a['params'], group_b['params']):
                        master_a = a.param_to_inner_param[param_a]
                        master_b = b.param_to_inner_param[param_b]
                        torch.testing.assert_close(master_a, master_b, rtol=0, atol=0)
                        for key in ('exp_avg', 'exp_avg_sq'):
                            assert b.state[param_b][key].device == master_b.device
                            torch.testing.assert_close(
                                a.state[param_a][key], b.state[param_b][key], rtol=0, atol=0
                            )
    finally:
        Utils.destroy_model_parallel()
