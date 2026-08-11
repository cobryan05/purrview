"""Regression tests for the implicit keep-warm pin in Manager.recognize().

Before this behaviour existed, a recognize() call from a caller that held no pin
(the /classify endpoint is the live example) took a temporary pin, ran inference,
and unpinned in a finally: — dropping the only pin and therefore destroying the
ort.InferenceSession. The next call rebuilt it. Under CUDA that rebuilds a GPU
context and host-side arenas per call, and the process RSS grows monotonically
(observed in production: 1502 loads of one classifier in 33 minutes).
"""

import asyncio
import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest

from apps.helpers.types import InferenceResult
from apps.inference_server.manager import Manager

MODEL_METADATA = {"yolo": 8, "classes": ["cat", "dog"], "class_map": {}}


class _CountingProvider:
    """Stands in for an ONNX provider, recording how many were constructed.

    ``weights`` is whatever bytes were on disk at load time, so a test can tell
    *which* version of a model file the live session was actually built from.
    """

    def __init__(self, counter: list, weights: bytes = b""):
        self._counter = counter
        self.weights = weights

    async def processImage(self, image, **kwargs):
        return InferenceResult(source_width=1, source_height=1, detections=[], inference_time=0.01)


@pytest.fixture
def load_counter(monkeypatch):
    """Patch load_yolo_onnx to count how many sessions get built."""
    calls: list = []

    def _fake_load(path):
        calls.append(path)
        return _CountingProvider(calls, weights=Path(path).read_bytes())

    monkeypatch.setattr("apps.inference_server.manager.load_yolo_onnx", _fake_load)
    return calls


@pytest.fixture
async def keepwarm_manager(tmp_path, load_counter):
    (tmp_path / "clf.onnx").write_bytes(b"fake onnx")
    (tmp_path / "clf.json").write_text(json.dumps(MODEL_METADATA))
    mgr = Manager(models_path=str(tmp_path))
    await mgr.list_models()
    return mgr


def _img() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


class TestImplicitKeepWarmPin:
    async def test_repeated_pinless_recognize_loads_model_once(self, keepwarm_manager, load_counter):
        """The regression itself: N pinless calls must build exactly one session."""
        for _ in range(5):
            await keepwarm_manager.recognize("clf", _img(), conf_thresh=0.5)

        assert len(load_counter) == 1

    async def test_model_stays_loaded_after_pinless_recognize(self, keepwarm_manager):
        await keepwarm_manager.recognize("clf", _img(), conf_thresh=0.5)

        assert "clf" in keepwarm_manager._pinned_models
        assert Manager.IMPLICIT_PIN_ID in keepwarm_manager._pinned_models["clf"].pins

    async def test_implicit_pin_is_refreshed_not_accumulated(self, keepwarm_manager):
        """Reusing one pin id keeps the pin dict from growing per call."""
        for _ in range(10):
            await keepwarm_manager.recognize("clf", _img(), conf_thresh=0.5)

        assert list(keepwarm_manager._pinned_models["clf"].pins) == [Manager.IMPLICIT_PIN_ID]

    async def test_idle_model_still_unloads_once_implicit_pin_expires(self, keepwarm_manager, load_counter):
        """Keeping the model warm must not defeat the auto-unload path."""
        await keepwarm_manager.recognize("clf", _img(), conf_thresh=0.5)

        # Age the implicit pin past its expiry, as the periodic worker would see it.
        keepwarm_manager._pinned_models["clf"].pins[Manager.IMPLICIT_PIN_ID].expiry = 0.0
        keepwarm_manager._cleanup_expired_pins()

        assert "clf" not in keepwarm_manager._pinned_models

        # And a later call reloads it rather than erroring.
        await keepwarm_manager.recognize("clf", _img(), conf_thresh=0.5)
        assert len(load_counter) == 2

    async def test_caller_pin_is_refreshed_and_no_implicit_pin_added(self, keepwarm_manager, load_counter):
        pin_id = await keepwarm_manager.pin_model("clf")
        keepwarm_manager._pinned_models["clf"].pins[pin_id].expiry = 1.0

        await keepwarm_manager.recognize("clf", _img(), conf_thresh=0.5, pin_id=pin_id)

        pins = keepwarm_manager._pinned_models["clf"].pins
        assert list(pins) == [pin_id], "a caller-held pin must not also take the implicit pin"
        assert pins[pin_id].expiry > 1.0, "caller pin expiry should be refreshed"
        assert len(load_counter) == 1


class TestPinIdReuse:
    async def test_explicit_pin_id_refreshes_existing_pin(self, keepwarm_manager):
        first = await keepwarm_manager.pin_model("clf", pin_id="shared")
        second = await keepwarm_manager.pin_model("clf", pin_id="shared")

        assert first == second == "shared"
        assert list(keepwarm_manager._pinned_models["clf"].pins) == ["shared"]

    async def test_pin_model_still_generates_unique_ids_by_default(self, keepwarm_manager):
        a = await keepwarm_manager.pin_model("clf")
        b = await keepwarm_manager.pin_model("clf")

        assert a != b
        assert len(keepwarm_manager._pinned_models["clf"].pins) == 2


class TestRetrainedModelIsReloaded:
    """Retraining a model writes new weights over `$MODELS_ROOT/<name>.onnx` from a
    *different process* (tasks_server's `_training_job_runner._save_artifact`), so the
    inference server gets no notification at all. If pin_model keeps serving the
    session it loaded from the previous file, every request after a retrain answers
    from the old weights — silently, until the implicit keep-warm pin happens to
    idle out. Observed live: a retrained detector scored 0.0006 on an image its
    new weights scored 0.94 on, so the UI showed zero detections.
    """

    @staticmethod
    def _overwrite(tmp_path, content: bytes) -> None:
        """Land new weights the way an out-of-process retrain does.

        mtime is pushed forward explicitly: the writes in a test happen well
        inside one filesystem timestamp granule, so relying on the natural mtime
        would make this pass or fail on timing rather than on behaviour.
        """
        target = tmp_path / "clf.onnx"
        target.write_bytes(content)
        stat = target.stat()
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    async def test_recognize_after_overwrite_uses_new_weights(self, keepwarm_manager, tmp_path, load_counter):
        await keepwarm_manager.recognize("clf", _img(), conf_thresh=0.5)
        assert keepwarm_manager._pinned_models["clf"].inference.weights == b"fake onnx"

        self._overwrite(tmp_path, b"retrained onnx")
        await keepwarm_manager.recognize("clf", _img(), conf_thresh=0.5)

        assert keepwarm_manager._pinned_models["clf"].inference.weights == b"retrained onnx", (
            "recognize() served the session built from the pre-retrain file; the "
            "model on disk changed but the loaded ort.InferenceSession did not"
        )
        assert len(load_counter) == 2, "new weights on disk must trigger exactly one reload"

    async def test_unchanged_file_is_not_reloaded(self, keepwarm_manager, load_counter):
        """The reload must be keyed on the file actually changing — otherwise this
        undoes the keep-warm pin and rebuilds a CUDA session on every call."""
        for _ in range(5):
            await keepwarm_manager.recognize("clf", _img(), conf_thresh=0.5)

        assert len(load_counter) == 1

    async def test_caller_held_pin_also_sees_new_weights(self, keepwarm_manager, tmp_path):
        pin_id = await keepwarm_manager.pin_model("clf")

        self._overwrite(tmp_path, b"retrained onnx")
        await keepwarm_manager.pin_model("clf", pin_id=pin_id)

        assert keepwarm_manager._pinned_models["clf"].inference.weights == b"retrained onnx"

    async def test_torn_write_keeps_serving_old_session_then_recovers(
        self, keepwarm_manager, tmp_path, load_counter, monkeypatch
    ):
        """_save_artifact lands new weights with a plain write_bytes(), so a request
        can arrive while the file is a truncated ONNX. That must not fail the
        request, and must not strand the model on the old weights either."""
        await keepwarm_manager.recognize("clf", _img(), conf_thresh=0.5)

        self._overwrite(tmp_path, b"half-written on")
        monkeypatch.setattr(
            "apps.inference_server.manager.load_yolo_onnx",
            lambda path: (_ for _ in ()).throw(ValueError("truncated onnx")),
        )

        await keepwarm_manager.recognize("clf", _img(), conf_thresh=0.5)
        assert keepwarm_manager._pinned_models["clf"].inference.weights == b"fake onnx"

        # The rest of the write lands, bumping mtime again — next call picks it up.
        def _real_load(path):
            load_counter.append(path)
            return _CountingProvider(load_counter, weights=Path(path).read_bytes())

        monkeypatch.setattr("apps.inference_server.manager.load_yolo_onnx", _real_load)
        self._overwrite(tmp_path, b"retrained onnx")
        await keepwarm_manager.recognize("clf", _img(), conf_thresh=0.5)

        assert keepwarm_manager._pinned_models["clf"].inference.weights == b"retrained onnx"

    async def test_stat_failure_keeps_serving_existing_session(self, keepwarm_manager, tmp_path, load_counter):
        """A model file that has gone missing (mid-redeploy write, or unlinked) must
        not take the loaded model down with it — keep answering from memory."""
        await keepwarm_manager.recognize("clf", _img(), conf_thresh=0.5)
        (tmp_path / "clf.onnx").unlink()

        await keepwarm_manager.recognize("clf", _img(), conf_thresh=0.5)

        assert keepwarm_manager._pinned_models["clf"].inference.weights == b"fake onnx"
        assert len(load_counter) == 1


class TestDeleteModelWithImplicitPin:
    async def test_delete_succeeds_when_only_implicit_pin_held(self, keepwarm_manager):
        """A keep-warm pin means 'used recently', not 'reserved by a caller'."""
        await keepwarm_manager.recognize("clf", _img(), conf_thresh=0.5)

        await keepwarm_manager.delete_model("clf")

        assert "clf" not in keepwarm_manager._avail_models
        assert "clf" not in keepwarm_manager._pinned_models

    async def test_delete_still_blocked_by_a_real_caller_pin(self, keepwarm_manager):
        await keepwarm_manager.pin_model("clf")

        with pytest.raises(ValueError, match="currently pinned"):
            await keepwarm_manager.delete_model("clf")


class TestDeleteModelRacesLoad:
    """A pinless recognize() and a concurrent delete_model() can interleave around
    the model's initial (slow) load. Without serializing the two, delete_model can
    see the model as "not yet pinned" mid-load, delete its files, and then have
    pin_model finish the load anyway and re-insert a "ghost" entry into
    _pinned_models — one that is loaded in memory but absent from _avail_models,
    so a later delete_model 404s before ever being able to clean it up.
    """

    async def test_delete_during_slow_load_leaves_no_ghost_pinned_model(self, tmp_path, monkeypatch):
        (tmp_path / "clf.onnx").write_bytes(b"fake onnx")
        (tmp_path / "clf.json").write_text(json.dumps(MODEL_METADATA))
        mgr = Manager(models_path=str(tmp_path))
        await mgr.list_models()

        load_started = threading.Event()
        release_load = threading.Event()

        def _slow_load(path):
            load_started.set()
            release_load.wait(timeout=5)
            return _CountingProvider([])

        monkeypatch.setattr("apps.inference_server.manager.load_yolo_onnx", _slow_load)

        recognize_task = asyncio.create_task(mgr.recognize("clf", _img(), conf_thresh=0.5))
        while not load_started.is_set():
            await asyncio.sleep(0.01)

        # Give delete_model a chance to observe the model as not-yet-pinned before
        # the slow load (and therefore pin_model) completes.
        delete_task = asyncio.create_task(mgr.delete_model("clf"))
        await asyncio.sleep(0.05)
        release_load.set()

        await asyncio.gather(recognize_task, delete_task, return_exceptions=True)

        if "clf" not in mgr._avail_models:
            assert "clf" not in mgr._pinned_models, (
                "model was deleted from _avail_models but a load that raced the "
                "deletion re-inserted it into _pinned_models — a ghost entry that "
                "a future delete_model('clf') can never reach again"
            )
