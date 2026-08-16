"""The StreamWatcher's inference call — the seam between the task and the models.

This exists because the watcher shipped unable to detect anything at all: the payload
it built omitted a required field, and the resulting ValidationError was swallowed by
the same `except Exception` that guards genuine inference failures, so every frame
silently produced zero detections. The pure engine tests all passed throughout, because
they feed synthetic `Detection`s and never exercise this call.

The lesson those tests encode: assert on what comes *back* from `_run_inference`, not
just on what the engine does with it.
"""

from types import SimpleNamespace
from typing import Any, List

import pytest

from apps.helpers.tracking.tracker import Detection
from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask

CAT = "label-cat"


def _bbox(x1: float, y1: float, x2: float, y2: float, label_uuid: str = CAT) -> SimpleNamespace:
    """A detection bbox as the inference client returns it.

    Carries both corner and x/y/width/height forms: the task reads the corners, while
    the hierarchy merger's `iou_xywh` reads the latter.
    """
    return SimpleNamespace(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        x=x1,
        y=y1,
        width=x2 - x1,
        height=y2 - y1,
        label_uuid=label_uuid,
        label_name="Cat",
    )


class _FakeInferenceApi:
    """Records the payloads it is handed, so the conf floor can be asserted."""

    def __init__(self, detections: List[Any]) -> None:
        self._detections = detections
        self.payloads: List[Any] = []

    def detect(self, model_name: str, payload: Any) -> SimpleNamespace:
        self.payloads.append(payload)
        return SimpleNamespace(result=SimpleNamespace(detections=list(self._detections)))


def _task() -> StreamWatcherTask:
    """A task with only the attributes `_run_inference` touches.

    Built without `_init` so the test needs no manager, DB or watch profile — the call
    under test is deliberately narrow.
    """
    task = StreamWatcherTask.__new__(StreamWatcherTask)
    task._pin_ids = {}
    task._label_ancestry = {}
    task._inference_ms = []
    return task


class TestRunInference:
    async def test_returns_detections(self):
        """The regression: this returned [] for every frame, for every model."""
        api = _FakeInferenceApi([SimpleNamespace(bbox=_bbox(0, 0, 10, 10), confidence=0.9)])

        detections = await _task()._run_inference(api, "base64-image", ["cat-model"])

        assert len(detections) == 1, "watcher produced no detections — the payload is invalid again"
        assert isinstance(detections[0], Detection)
        assert detections[0].label_uuid == CAT
        assert detections[0].confidence == pytest.approx(0.9)

    async def test_requests_a_low_confidence_floor(self):
        """Thresholding belongs to the tracker, not to this call.

        ByteTrack's second association pass exists to recover boxes a conventional
        threshold discards, so filtering here would silently reduce it to plain SORT.
        """
        api = _FakeInferenceApi([])

        await _task()._run_inference(api, "base64-image", ["cat-model"])

        assert len(api.payloads) == 1
        assert api.payloads[0].conf_thresh == pytest.approx(StreamWatcherTask._INFERENCE_CONF_FLOOR)
        assert StreamWatcherTask._INFERENCE_CONF_FLOOR <= 0.2, "floor too high for two-stage association"

    async def test_no_models_is_not_a_call(self):
        api = _FakeInferenceApi([SimpleNamespace(bbox=_bbox(0, 0, 10, 10), confidence=0.9)])

        assert await _task()._run_inference(api, "base64-image", []) == []
        assert api.payloads == []

    async def test_inference_failure_is_contained(self):
        """A failing model must yield no detections, not kill the frame loop."""

        class _Boom:
            def detect(self, model_name: str, payload: Any):
                raise RuntimeError("model exploded")

        assert await _task()._run_inference(_Boom(), "base64-image", ["cat-model"]) == []
