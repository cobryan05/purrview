"""Trainers release their DataLoader workers when a job ends.

The training_server process outlives every job it runs, so whatever a job fails
to hand back is leaked for the life of the server rather than reclaimed at exit.

Observed in production on 2026-08-14: training_server sat at 1022 of its 1024
descriptors with 987 open pipes and seven orphaned worker processes, with no job
running. The next training died on ``[Errno 24] Too many open files`` while
opening an unrelated temp file — a symptom pointing nowhere near the cause.

The leak is specifically ultralytics' ``InfiniteDataLoader``, which assigns
``self.iterator = super().__iter__()`` in ``__init__`` and holds that reference
for the object's whole life so workers can be reused across epochs. A plain
torch ``DataLoader`` with ``persistent_workers=False`` — the default, and what
the classifier trainers use — keeps no such reference and is freed by
refcounting when the training loop's local goes away.

So this deliberately drives each loader's own public ``close()`` rather than
torch internals. ``DataLoader``'s public surface is only
``check_worker_number_rationality`` and ``multiprocessing_context``; for plain
loaders the supported shutdown is dropping the last reference, which the helper
forces with a gc pass.
"""

import gc
from unittest.mock import MagicMock

from apps.training_server.trainers.base import Trainer


class _ClosableLoader:
    """Stands in for ultralytics' InfiniteDataLoader, which has a public close()."""

    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class TestClosesWhatCanBeClosed:
    def test_closes_every_loader_that_offers_it(self):
        a, b = _ClosableLoader(), _ClosableLoader()

        Trainer.shutdown_dataloaders(a, b)

        assert a.close_calls == 1
        assert b.close_calls == 1

    def test_a_loader_without_close_is_left_to_refcounting(self):
        """A plain torch DataLoader has no close(). Touching its private
        _iterator would be both unsupported and pointless: with
        persistent_workers=False that attribute is always None."""

        class _PlainLoader:
            pass

        Trainer.shutdown_dataloaders(_PlainLoader())  # must not raise

    def test_none_loaders_are_skipped(self):
        """Callers pass optional loaders straight through — the ultralytics
        trainer reads them off attributes that may not exist yet."""
        Trainer.shutdown_dataloaders(None, None)  # must not raise


class TestNeverBreaksTheJob:
    """Cleanup runs in a `finally`. It must never be what fails a training run
    that otherwise succeeded — that would turn a leak into a lost model."""

    def test_a_failing_close_does_not_propagate(self):
        exploding = MagicMock()
        exploding.close.side_effect = RuntimeError("boom")

        Trainer.shutdown_dataloaders(exploding)  # must not raise

    def test_one_bad_loader_does_not_strand_the_rest(self):
        """Giving up on the first failure would leak every loader after it."""
        exploding = MagicMock()
        exploding.close.side_effect = RuntimeError("boom")
        healthy = _ClosableLoader()

        Trainer.shutdown_dataloaders(exploding, healthy)

        assert healthy.close_calls == 1


class TestGarbageCollectionBackstop:
    def test_runs_a_collection_pass(self):
        """The backstop for loaders with no close(): an iterator pinned by a
        reference cycle or an exception traceback is only freed by a gc pass,
        and freeing it is what triggers torch's own worker shutdown."""
        collected = []
        real_collect = gc.collect

        def _spy(*args, **kwargs):
            collected.append(True)
            return real_collect(*args, **kwargs)

        gc.collect = _spy
        try:
            Trainer.shutdown_dataloaders(_ClosableLoader())
        finally:
            gc.collect = real_collect

        assert collected, "expected a gc pass to free iterators held by cycles/tracebacks"


class TestAgainstRealTorch:
    """The behaviour the helper is designed around, asserted against real torch
    rather than assumed — these are the facts that make close() the right call
    and _iterator the wrong one."""

    def test_a_default_dataloader_holds_no_iterator_reference(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        loader = DataLoader(TensorDataset(torch.zeros(4, 2)), batch_size=2, num_workers=0)
        for _ in loader:
            break

        # getattr rather than attribute access: the premise holds whether the
        # attribute is None *or* gone in a future torch, and a rename should not
        # fail this as an AttributeError from the assert line itself.
        assert getattr(loader, "_iterator", None) is None, (
            "persistent_workers=False keeps no iterator reference, so a helper "
            "reaching for _iterator would silently do nothing"
        )

    def test_ultralytics_infinite_loader_exposes_public_close(self):
        """The public API this helper relies on. If a version bump removes it,
        this fails here rather than as a descriptor leak weeks later."""
        from ultralytics.data.build import InfiniteDataLoader

        assert callable(getattr(InfiniteDataLoader, "close", None))
