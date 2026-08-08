"""Regression tests for the implicit keep-warm pin in Manager.recognize().

Before this behaviour existed, a recognize() call from a caller that held no pin
(the /classify endpoint is the live example) took a temporary pin, ran inference,
and unpinned in a finally: — dropping the only pin and therefore destroying the
ort.InferenceSession. The next call rebuilt it. Under CUDA that rebuilds a GPU
context and host-side arenas per call, and the process RSS grows monotonically
(observed in production: 1502 loads of one classifier in 33 minutes).
"""

import json

import numpy as np
import pytest

from apps.helpers.types import InferenceResult
from apps.inference_server.manager import Manager

MODEL_METADATA = {"yolo": 8, "classes": ["cat", "dog"], "class_map": {}}


class _CountingProvider:
    """Stands in for an ONNX provider, recording how many were constructed."""

    def __init__(self, counter: list):
        self._counter = counter

    async def processImage(self, image, **kwargs):
        return InferenceResult(source_width=1, source_height=1, detections=[], inference_time=0.01)


@pytest.fixture
def load_counter(monkeypatch):
    """Patch load_yolo_onnx to count how many sessions get built."""
    calls: list = []

    def _fake_load(path):
        calls.append(path)
        return _CountingProvider(calls)

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
