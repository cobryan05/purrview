"""Trainers release their DataLoader workers when a job ends.

The training_server process outlives every job it runs, so whatever a job fails
to hand back is leaked for the life of the server rather than reclaimed at exit.

Observed in production on 2026-08-14: training_server sat at 1022 of its 1024
descriptors with 987 open pipes and seven orphaned worker processes, with no job
running. The next training died on ``[Errno 24] Too many open files`` while
opening an unrelated temp file — a symptom pointing nowhere near the cause.

There are three loader shapes here, and they need three different things:

1. Ultralytics' ``InfiniteDataLoader`` assigns ``self.iterator =
   super().__iter__()`` in ``__init__`` and holds that reference for the
   object's whole life so workers can be reused across epochs. It has a public
   ``close()``; drive that.
2. A plain torch ``DataLoader`` with ``persistent_workers=False`` — the default,
   and what the classifier trainers use — keeps no such reference and is freed
   by refcounting when the training loop's local goes away. Nothing to do.
3. A plain torch ``DataLoader`` with ``persistent_workers=True`` — libreyolo's
   default, used by the object detection trainer — parks its iterator on
   ``loader._iterator`` and reuses it across epochs, but ``DataLoader`` has no
   ``close()``. Torch's own shutdown is that iterator's ``__del__``, so dropping
   the reference is the supported mechanism; there is just no public name for
   it. Case 3 was missed originally, which made the helper a silent no-op for
   every object detection run.

Observed again in production on 2026-08-19: a *cancelled* detection job left
eight orphaned workers holding ~1.5 GB, still there ten minutes later. Only
cancelled and errored runs leaked, because libreyolo builds its Trainer as a
local of ``model.train()`` — on success the frame returns and refcounting reaps
everything, but a raise pins that frame in the exception's traceback for as long
as the traceback lives. Hence ``shutdown_dataloaders_in_traceback``.
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
        """A plain torch DataLoader has no close(). With persistent_workers=False
        its _iterator is always None, so there is nothing to release and the
        loader is left entirely to refcounting."""

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
    rather than assumed — these are the facts that decide which of the three
    shutdown paths above each loader shape needs."""

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

    def test_a_persistent_dataloader_does_hold_an_iterator_reference(self):
        """The premise for case 3, and the one that makes it different from the
        default loader above: persistent_workers=True *does* park an iterator,
        and there is no close() to reach it through."""
        from torch.utils.data import DataLoader

        loader = _persistent_loader()
        try:
            for _ in loader:
                break

            assert loader._iterator is not None
            assert not callable(
                getattr(DataLoader, "close", None)
            ), "DataLoader grew a close(); prefer it over clearing _iterator"
        finally:
            loader._iterator = None


def _persistent_loader(num_workers: int = 2):
    """A real DataLoader shaped like libreyolo's: plain torch, persistent workers."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    return DataLoader(
        TensorDataset(torch.zeros(8, 2)),
        batch_size=2,
        num_workers=num_workers,
        persistent_workers=True,
    )


def _assert_workers_exited(workers, timeout: float = 15.0) -> None:
    """Worker shutdown is asynchronous — torch signals, then the children exit."""
    for w in workers:
        w.join(timeout=timeout)
    alive = [w.pid for w in workers if w.is_alive()]
    assert not alive, f"worker processes still running after shutdown: {alive}"


class TestPersistentWorkersAreActuallyReaped:
    """The regression that matters: real worker processes, really gone.

    Asserting on `_iterator is None` alone would pass against a helper that
    merely nulled the attribute without torch ever shutting the children down.
    """

    def test_shutdown_releases_a_persistent_loaders_workers(self):
        loader = _persistent_loader()
        for _ in loader:
            break
        workers = list(loader._iterator._workers)
        assert all(w.is_alive() for w in workers), "fixture never started its workers"

        Trainer.shutdown_dataloaders(loader)

        assert loader._iterator is None
        _assert_workers_exited(workers)

    def test_workers_stranded_by_a_traceback_are_reaped(self):
        """The production failure end to end: the loader is reachable *only*
        through the raising frame, exactly as libreyolo's trainer is."""

        class _LibreyoloishTrainer:
            def __init__(self):
                self.train_loader = _persistent_loader()

            def train(self):
                for _ in self.train_loader:
                    break
                # Stands in for the cancel raised from our epoch callback.
                raise RuntimeError("Training cancelled by user")

        def _model_train():
            # `trainer` is a local and is never returned — the only thing that
            # will still reference it after this raises is the traceback.
            trainer = _LibreyoloishTrainer()
            trainer.train()

        workers = []
        try:
            _model_train()
        except RuntimeError as exc:
            tb = exc.__traceback__
            while tb is not None:
                t = tb.tb_frame.f_locals.get("trainer")
                if t is not None:
                    workers = list(t.train_loader._iterator._workers)
                tb = tb.tb_next
            assert workers, "test could not reach the loader it means to assert on"
            assert all(w.is_alive() for w in workers)

            Trainer.shutdown_dataloaders_in_traceback(exc)

        _assert_workers_exited(workers)

    def test_the_success_path_is_a_no_op(self):
        """Called unconditionally from a finally, so it has to be free when
        nothing went wrong — and must not disturb a loader still in use."""
        loader = _persistent_loader()
        try:
            for _ in loader:
                break

            Trainer.shutdown_dataloaders_in_traceback(None)

            assert loader._iterator is not None
        finally:
            loader._iterator = None
