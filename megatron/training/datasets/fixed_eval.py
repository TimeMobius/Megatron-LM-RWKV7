# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Restartable validation without buffering the validation set in memory."""

from megatron.core.rerun_state_machine import RerunDataIterator


class FixedEvalDataIterator(RerunDataIterator):
    """Rebuild a validation iterator from the same sampler position each time."""

    def __init__(self, dataloader, cyclic_factory=None):
        super().__init__(iter(()))
        self._dataloader = dataloader
        self._cyclic_factory = cyclic_factory
        generator = getattr(dataloader, "generator", None)
        self._generator_state = generator.get_state().clone() if generator is not None else None

    def reset_for_eval(self):
        """Discard prefetched data and restart the original validation sample order."""
        sampler = self._dataloader.batch_sampler
        if hasattr(sampler, "consumed_samples"):
            sampler.consumed_samples = 0
        if self._generator_state is not None:
            self._dataloader.generator.set_state(self._generator_state)
        self.advance()
        self.iterable = (
            iter(self._dataloader)
            if self._cyclic_factory is None
            else iter(self._cyclic_factory(self._dataloader))
        )


def reset_fixed_eval_iterator(data_iterator):
    """Reset validation iterators, including the list used by virtual PP."""
    if data_iterator is None:
        return
    if isinstance(data_iterator, (list, tuple)):
        for iterator in data_iterator:
            reset_fixed_eval_iterator(iterator)
    elif isinstance(data_iterator, FixedEvalDataIterator):
        data_iterator.reset_for_eval()
    else:
        raise ValueError("fixed_eval_samples requires a restartable validation DataLoader")
