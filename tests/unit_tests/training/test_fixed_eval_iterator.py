# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from megatron.core.rerun_state_machine import RerunMode
from megatron.training.datasets.data_samplers import MegatronPretrainingSampler
from megatron.training.datasets.fixed_eval import FixedEvalDataIterator, reset_fixed_eval_iterator


def test_repeated_eval_and_restored_iterator_start_from_same_samples():
    def build():
        sampler = MegatronPretrainingSampler(16, 0, 2, 0, 1)
        loader = DataLoader(
            list(range(16)), batch_sampler=sampler, generator=torch.Generator().manual_seed(1234)
        )
        return FixedEvalDataIterator(loader)

    iterator = build()
    state = SimpleNamespace(get_mode=lambda: RerunMode.DISABLED)
    with patch('megatron.core.rerun_state_machine.get_rerun_state_machine', return_value=state):
        for current in (iterator, iterator, build()):
            reset_fixed_eval_iterator(current)
            assert [next(current).tolist() for _ in range(2)] == [[0, 1], [2, 3]]


def test_reset_does_not_advance_training_or_test_iterators():
    training, test = iter(range(16)), iter(range(16))
    loader = DataLoader(list(range(16)), batch_size=2)
    validation = FixedEvalDataIterator(loader)
    reset_fixed_eval_iterator([None, [validation]])
    reset_fixed_eval_iterator(validation)
    assert next(training) == next(test) == 0


def test_iterator_creation_preserves_global_rng():
    loader = DataLoader(
        list(range(16)), batch_size=2, generator=torch.Generator().manual_seed(1234)
    )
    before = torch.random.get_rng_state().clone()
    validation = FixedEvalDataIterator(loader)
    validation.reset_for_eval()
    validation.reset_for_eval()
    assert torch.equal(before, torch.random.get_rng_state())
