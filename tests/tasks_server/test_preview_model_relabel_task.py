"""Tests for PreviewModelRelabelTask.

Exercises the IoU-matching/candidate-building logic against a fake
InferenceBackend (see `_filter_chain_runner.InferenceBackend`) injected
directly into `_process_image` — no real inference server or db_client HTTP
round-trip involved (mirrors ApplyFilterChainTask's test pattern).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from apps.tasks_server.tasks._filter_chain_runner import Detection
from apps.tasks_server.tasks.PreviewModelRelabelTask import _MAX_BBOX_UUIDS, PreviewModelRelabelTask

_MOCK_MANAGER = MagicMock()
_MOCK_MANAGER.update_task_heartbeat = AsyncMock()


def _make_task(params: dict) -> PreviewModelRelabelTask:
    task = PreviewModelRelabelTask(task_config_uuid="cfg-uuid", params=params, manager=_MOCK_MANAGER)
    task.set_instance_uuid("11111111-1111-1111-1111-111111111111")
    return task


def _base_params(**overrides):
    params = {
        "model_name": "my_detector",
        "conf_thresh": 0.25,
        "iou_thresh": 0.5,
        "image_uuids": ["img-1"],
        "bbox_uuids": ["bbox-1"],
    }
    params.update(overrides)
    return params


def _bbox(uuid="bbox-1", label_uuid="cat", x=0.1, y=0.1, width=0.2, height=0.2):
    # Mirrors the camelCase dict shape returned by batch_images_by_uuids
    # (ImageRecordRead.model_dump(by_alias=True), alias_generator=to_camel).
    return {"uuid": uuid, "labelUuid": label_uuid, "x": x, "y": y, "width": width, "height": height}


def _image(uuid="img-1", filename="/data/x.jpg", bboxes=None):
    return {"uuid": uuid, "filename": filename, "bboxes": bboxes or []}


class _FakeBackend:
    """Fake InferenceBackend — returns a fixed detection list regardless of input."""

    def __init__(self, detections):
        self._detections = detections

    async def detect(self, model_name, image_bytes, conf_thresh):
        return self._detections


class TestGuards:
    async def test_missing_model_name_raises(self):
        task = _make_task(_base_params(model_name=""))
        with pytest.raises(ValueError, match="model_name"):
            await task._init(task._params, None)

    async def test_missing_image_uuids_raises(self):
        task = _make_task(_base_params(image_uuids=[]))
        with pytest.raises(ValueError, match="image_uuids"):
            await task._init(task._params, None)

    async def test_missing_bbox_uuids_raises(self):
        task = _make_task(_base_params(bbox_uuids=[]))
        with pytest.raises(ValueError, match="bbox_uuids"):
            await task._init(task._params, None)

    async def test_over_cap_bbox_uuids_raises(self):
        params = _base_params(bbox_uuids=[f"bbox-{i}" for i in range(_MAX_BBOX_UUIDS + 1)])
        task = _make_task(params)
        with pytest.raises(ValueError, match="exceeding the hard cap"):
            await task._init(task._params, None)

    async def test_exactly_at_cap_is_allowed(self):
        params = _base_params(bbox_uuids=[f"bbox-{i}" for i in range(_MAX_BBOX_UUIDS)])
        task = _make_task(params)
        await task._init(task._params, None)
        assert len(task._bbox_uuids) == _MAX_BBOX_UUIDS

    async def test_invalid_model_name_rejected(self):
        task = _make_task(_base_params(model_name="../escape"))
        with pytest.raises(ValueError, match="Invalid model_name"):
            await task._init(task._params, None)


class TestMatching:
    """Exercises `_process_image` directly with a fake backend, bypassing the
    HTTP/db_client plumbing (pin_model, batch fetch)."""

    async def _run_process_image(self, task, detections, bbox_uuid="bbox-1", current_label="cat"):
        await task._init(task._params, None)
        task._load_image_bytes = AsyncMock(return_value=b"fake-image-bytes")
        backend = _FakeBackend(detections)
        img = _image(bboxes=[_bbox(uuid=bbox_uuid, label_uuid=current_label)])
        await task._process_image(
            images_api=MagicMock(),
            backend=backend,
            img=img,
            label_names={"cat": "Cat", "dog": "Dog"},
        )
        return task._candidates

    async def test_matching_detection_produces_candidate(self):
        task = _make_task(_base_params())
        detections = [Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9, label_uuid="dog")]
        candidates = await self._run_process_image(task, detections)
        assert len(candidates) == 1
        c = candidates[0]
        assert c["bboxUuid"] == "bbox-1"
        assert c["imageUuid"] == "img-1"
        assert c["imagePath"] == "/data/x.jpg"
        assert c["currentLabelUuid"] == "cat"
        assert c["currentLabelName"] == "Cat"
        assert c["proposedLabelUuid"] == "dog"
        assert c["proposedLabelName"] == "Dog"
        assert c["confidence"] == 0.9

    async def test_no_op_relabel_excluded(self):
        task = _make_task(_base_params())
        # Detection overlaps and matches, but proposes the *same* label already on the bbox.
        detections = [Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9, label_uuid="cat")]
        candidates = await self._run_process_image(task, detections)
        assert candidates == []

    async def test_unmapped_detection_excluded(self):
        task = _make_task(_base_params())
        # Detection overlaps but has no label_uuid (unmapped class) — excluded even
        # though it would otherwise be the only/best match.
        detections = [Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9, label_uuid=None)]
        candidates = await self._run_process_image(task, detections)
        assert candidates == []

    async def test_non_overlapping_detection_excluded(self):
        task = _make_task(_base_params())
        detections = [Detection(x=0.8, y=0.8, width=0.1, height=0.1, confidence=0.9, label_uuid="dog")]
        candidates = await self._run_process_image(task, detections)
        assert candidates == []

    async def test_highest_confidence_detection_wins(self):
        task = _make_task(_base_params())
        detections = [
            Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.4, label_uuid="dog"),
            Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.95, label_uuid="bird"),
            Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.6, label_uuid="fish"),
        ]
        candidates = await self._run_process_image(task, detections)
        assert len(candidates) == 1
        assert candidates[0]["proposedLabelUuid"] == "bird"
        assert candidates[0]["confidence"] == 0.95

    async def test_non_candidate_bbox_on_same_image_ignored(self):
        task = _make_task(_base_params(bbox_uuids=["bbox-1"]))
        await task._init(task._params, None)
        task._load_image_bytes = AsyncMock(return_value=b"fake-image-bytes")
        backend = _FakeBackend([Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9, label_uuid="dog")])
        # img has two bboxes; only bbox-1 is in the candidate set.
        img = _image(
            bboxes=[
                _bbox(uuid="bbox-1", label_uuid="cat"),
                _bbox(uuid="bbox-2", label_uuid="cat", x=0.1, y=0.1, width=0.2, height=0.2),
            ]
        )
        await task._process_image(
            images_api=MagicMock(),
            backend=backend,
            img=img,
            label_names={"cat": "Cat", "dog": "Dog"},
        )
        assert len(task._candidates) == 1
        assert task._candidates[0]["bboxUuid"] == "bbox-1"

    async def test_no_detections_produces_no_candidates(self):
        task = _make_task(_base_params())
        candidates = await self._run_process_image(task, detections=[])
        assert candidates == []
