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
        assert task._no_op_count == 1
        assert task._no_detection_count == 0

    async def test_unmapped_detection_excluded(self):
        task = _make_task(_base_params())
        # Detection overlaps but has no label_uuid (unmapped class) — excluded even
        # though it would otherwise be the only/best match.
        detections = [Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9, label_uuid=None)]
        candidates = await self._run_process_image(task, detections)
        assert candidates == []
        assert task._no_detection_count == 1

    async def test_non_overlapping_detection_excluded(self):
        task = _make_task(_base_params())
        detections = [Detection(x=0.8, y=0.8, width=0.1, height=0.1, confidence=0.9, label_uuid="dog")]
        candidates = await self._run_process_image(task, detections)
        assert candidates == []
        assert task._no_detection_count == 1

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
        assert task._no_detection_count == 1

    async def test_image_load_failure_counted_and_reported(self):
        task = _make_task(_base_params())
        await task._init(task._params, None)
        task._load_image_bytes = AsyncMock(side_effect=FileNotFoundError("/data/x.jpg"))
        backend = _FakeBackend([])
        img = _image(bboxes=[_bbox(uuid="bbox-1", label_uuid="cat")])
        with pytest.raises(FileNotFoundError):
            await task._process_image(
                images_api=MagicMock(),
                backend=backend,
                img=img,
                label_names={"cat": "Cat", "dog": "Dog"},
            )
        # _process_image itself propagates — it's the _run() batch loop (not exercised
        # here) that catches per-image failures and increments _images_failed.


class TestUnmappedClassFallback:
    """A detection whose model class isn't mapped to a label_uuid should still
    produce a candidate if its raw class_str matches a project label's name by
    name — mirrors the manual single-image "Run Model" UI's tolerance for
    unmapped classes."""

    async def _run_process_image(self, task, detections, name_to_label_uuid):
        await task._init(task._params, None)
        task._name_to_label_uuid = name_to_label_uuid
        task._load_image_bytes = AsyncMock(return_value=b"fake-image-bytes")
        backend = _FakeBackend(detections)
        img = _image(bboxes=[_bbox(uuid="bbox-1", label_uuid="cat")])
        await task._process_image(
            images_api=MagicMock(),
            backend=backend,
            img=img,
            label_names={"cat": "Cat", "dog": "Dog"},
        )
        return task._candidates

    async def test_unmapped_class_resolved_by_name(self):
        task = _make_task(_base_params())
        detections = [Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.86, label_uuid=None, class_str="Dog")]
        candidates = await self._run_process_image(task, detections, {"Dog": "dog", "Cat": "cat"})
        assert len(candidates) == 1
        assert candidates[0]["proposedLabelUuid"] == "dog"
        assert candidates[0]["confidence"] == 0.86

    async def test_unmapped_class_with_no_name_match_excluded(self):
        task = _make_task(_base_params())
        detections = [Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9, label_uuid=None, class_str="Bird")]
        candidates = await self._run_process_image(task, detections, {"Dog": "dog", "Cat": "cat"})
        assert candidates == []

    async def test_unmapped_class_with_no_class_str_excluded(self):
        task = _make_task(_base_params())
        detections = [Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9, label_uuid=None, class_str=None)]
        candidates = await self._run_process_image(task, detections, {"Dog": "dog", "Cat": "cat"})
        assert candidates == []


class TestBboxCropMode:
    """Exercises `_process_image_bbox_crop`: crops each candidate bbox out of a
    real in-memory image and matches the crop's best detection directly, with
    no IoU check."""

    @staticmethod
    def _make_image_bytes(width=200, height=200):
        from io import BytesIO

        from PIL import Image as PILImage

        img = PILImage.new("RGB", (width, height), color=(10, 20, 30))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    async def _run_process_image(self, task, detections, name_to_label_uuid=None):
        await task._init(task._params, None)
        task._name_to_label_uuid = name_to_label_uuid or {}
        task._load_image_bytes = AsyncMock(return_value=self._make_image_bytes())
        backend = _FakeBackend(detections)
        img = _image(bboxes=[_bbox(uuid="bbox-1", label_uuid="cat")])
        await task._process_image(
            images_api=MagicMock(),
            backend=backend,
            img=img,
            label_names={"cat": "Cat", "dog": "Dog"},
        )
        return task._candidates

    async def test_bbox_crop_ignores_geometry_and_uses_best_detection(self):
        task = _make_task(_base_params(crop_mode="bbox"))
        # Detection coordinates are relative to the *crop*, not the source image,
        # and don't overlap the original bbox geometry at all — crop mode must not
        # apply an IoU check, so this still produces a candidate.
        detections = [Detection(x=0.0, y=0.0, width=1.0, height=1.0, confidence=0.86, label_uuid="dog")]
        candidates = await self._run_process_image(task, detections)
        assert len(candidates) == 1
        assert candidates[0]["proposedLabelUuid"] == "dog"
        assert candidates[0]["confidence"] == 0.86

    async def test_bbox_crop_unmapped_class_resolved_by_name(self):
        task = _make_task(_base_params(crop_mode="bbox"))
        detections = [Detection(x=0.0, y=0.0, width=1.0, height=1.0, confidence=0.7, label_uuid=None, class_str="Dog")]
        candidates = await self._run_process_image(task, detections, {"Dog": "dog"})
        assert len(candidates) == 1
        assert candidates[0]["proposedLabelUuid"] == "dog"

    async def test_bbox_crop_no_detections_produces_no_candidates(self):
        task = _make_task(_base_params(crop_mode="bbox"))
        candidates = await self._run_process_image(task, detections=[])
        assert candidates == []


class TestCropModeGuard:
    async def test_invalid_crop_mode_raises(self):
        task = _make_task(_base_params(crop_mode="whole-image"))
        with pytest.raises(ValueError, match="crop_mode"):
            await task._init(task._params, None)

    async def test_default_crop_mode_is_image(self):
        task = _make_task(_base_params())
        await task._init(task._params, None)
        assert task._crop_mode == "image"


class TestLoadImageBytesResolvesFilesRoot:
    """`filename` (as returned by batch_images_by_uuids) is FILES_ROOT-relative,
    not a path valid against the tasks_server process's cwd — `_load_image_bytes`
    must resolve it against FILES_ROOT before opening (regression test for a bug
    where every filesystem read failed with FileNotFoundError, silently producing
    zero candidates)."""

    async def test_relative_filename_resolved_against_files_root(self, tmp_path, monkeypatch):
        import apps.tasks_server.tasks.PreviewModelRelabelTask as mod

        monkeypatch.setattr(mod, "FILES_ROOT", tmp_path)
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "img.jpg").write_bytes(b"fake-jpeg-bytes")

        task = _make_task(_base_params())
        await task._init(task._params, None)
        result = await task._load_image_bytes("subdir/img.jpg", images_api=MagicMock())
        assert result == b"fake-jpeg-bytes"

    async def test_path_escaping_files_root_raises(self, tmp_path, monkeypatch):
        import apps.tasks_server.tasks.PreviewModelRelabelTask as mod

        monkeypatch.setattr(mod, "FILES_ROOT", tmp_path)
        task = _make_task(_base_params())
        await task._init(task._params, None)
        with pytest.raises(FileNotFoundError):
            await task._load_image_bytes("../../etc/passwd", images_api=MagicMock())
