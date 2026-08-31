"""Tests for PreviewModelRelabelTask.

Exercises the IoU-matching/candidate-building logic against a fake
InferenceBackend (see `_filter_chain_runner.InferenceBackend`) injected
directly into `_process_image` — no real inference server or db_client HTTP
round-trip involved (mirrors ApplyFilterChainTask's test pattern).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.tasks_server.tasks._filter_chain_runner import Detection
from apps.tasks_server.tasks.PreviewModelRelabelTask import PreviewModelRelabelTask

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

    async def test_missing_bbox_uuids_raises_in_bbox_crop_mode_even_with_add_new(self):
        # crop_mode="bbox" has nothing to crop without candidate bboxes, so the
        # image+add_new exemption must not apply here.
        task = _make_task(_base_params(bbox_uuids=[], crop_mode="bbox", add_new=True))
        with pytest.raises(ValueError, match="bbox_uuids"):
            await task._init(task._params, None)

    async def test_empty_bbox_uuids_allowed_in_image_mode_with_add_new(self):
        # Image-scoped "run on filtered images" flow: no existing candidate
        # bboxes at all, relying entirely on add_new to produce candidates.
        task = _make_task(_base_params(bbox_uuids=[], crop_mode="image", add_new=True))
        await task._init(task._params, None)
        assert task._bbox_uuids == set()

    async def test_large_bbox_uuids_list_is_allowed(self):
        # No hard ceiling — the client warns/caps large runs; the task accepts them.
        n = 10_000
        params = _base_params(bbox_uuids=[f"bbox-{i}" for i in range(n)])
        task = _make_task(params)
        await task._init(task._params, None)
        assert len(task._bbox_uuids) == n

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


class TestAddNewBoxes:
    """add_new (whole-image mode): detections overlapping no existing bbox on the
    image become brand-new box candidates (bboxUuid=None, isNew=True)."""

    async def _run(self, task, detections, image_bboxes):
        await task._init(task._params, None)
        task._load_image_bytes = AsyncMock(return_value=b"fake-image-bytes")
        backend = _FakeBackend(detections)
        img = _image(bboxes=image_bboxes)
        await task._process_image(
            images_api=MagicMock(),
            backend=backend,
            img=img,
            label_names={"cat": "Cat", "dog": "Dog"},
        )
        return task._candidates

    async def test_add_new_off_ignores_non_overlapping_detection(self):
        task = _make_task(_base_params())  # add_new defaults False
        detections = [Detection(x=0.7, y=0.7, width=0.1, height=0.1, confidence=0.9, label_uuid="dog")]
        candidates = await self._run(task, detections, [_bbox(uuid="bbox-1", label_uuid="cat")])
        assert candidates == []
        assert task._new_boxes_found == 0

    async def test_add_new_emits_new_box_for_non_overlapping_detection(self):
        task = _make_task(_base_params(add_new=True))
        detections = [Detection(x=0.7, y=0.7, width=0.1, height=0.1, confidence=0.88, label_uuid="dog")]
        candidates = await self._run(task, detections, [_bbox(uuid="bbox-1", label_uuid="cat")])
        assert len(candidates) == 1
        c = candidates[0]
        assert c["isNew"] is True
        assert c["bboxUuid"] is None
        assert c["currentLabelUuid"] is None
        assert c["proposedLabelUuid"] == "dog"
        assert c["confidence"] == 0.88
        assert c["geometry"] == {"x": 0.7, "y": 0.7, "width": 0.1, "height": 0.1}
        assert task._new_boxes_found == 1

    async def test_add_new_skips_detection_overlapping_a_non_candidate_box(self):
        # A detection landing on an existing, already-correct, non-candidate box
        # must not be re-added as new.
        task = _make_task(_base_params(add_new=True))
        detections = [Detection(x=0.6, y=0.6, width=0.2, height=0.2, confidence=0.9, label_uuid="dog")]
        image_bboxes = [
            _bbox(uuid="bbox-1", label_uuid="cat", x=0.1, y=0.1),  # candidate, elsewhere
            _bbox(uuid="bbox-2", label_uuid="dog", x=0.6, y=0.6, width=0.2, height=0.2),  # non-candidate overlap
        ]
        candidates = await self._run(task, detections, image_bboxes)
        assert candidates == []
        assert task._new_boxes_found == 0

    async def test_add_new_and_relabel_not_double_emitted(self):
        # One detection overlapping the candidate → relabel; a second detection
        # overlapping nothing → new box. The relabel detection is not also "new".
        task = _make_task(_base_params(add_new=True))
        detections = [
            Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9, label_uuid="dog"),  # overlaps candidate
            Detection(x=0.8, y=0.8, width=0.1, height=0.1, confidence=0.7, label_uuid="dog"),  # new
        ]
        candidates = await self._run(task, detections, [_bbox(uuid="bbox-1", label_uuid="cat")])
        assert len(candidates) == 2
        relabel = [c for c in candidates if not c.get("isNew")]
        new = [c for c in candidates if c.get("isNew")]
        assert len(relabel) == 1 and relabel[0]["bboxUuid"] == "bbox-1"
        assert len(new) == 1 and new[0]["geometry"]["x"] == 0.8

    async def test_boxless_image_with_add_new_still_produces_new_box_candidate(self):
        # Image-scoped "run on filtered images" flow: the image carries no bboxes
        # at all (so bbox_uuids has nothing to match on this image), but a
        # detection should still surface as a new-box candidate.
        task = _make_task(_base_params(bbox_uuids=[], crop_mode="image", add_new=True))
        detections = [Detection(x=0.2, y=0.2, width=0.1, height=0.1, confidence=0.75, label_uuid="dog")]
        candidates = await self._run(task, detections, image_bboxes=[])
        assert len(candidates) == 1
        assert candidates[0]["isNew"] is True
        assert candidates[0]["bboxUuid"] is None
        assert candidates[0]["proposedLabelUuid"] == "dog"
        assert task._new_boxes_found == 1

    async def test_boxless_image_without_add_new_is_skipped_entirely(self):
        # Without add_new there is nothing this image could ever produce, so
        # _process_image should skip inference rather than call the backend.
        # bbox_uuids has a candidate, just not one that lives on *this* image.
        task = _make_task(_base_params(bbox_uuids=["other-bbox"], crop_mode="image", add_new=False))
        await task._init(task._params, None)
        task._load_image_bytes = AsyncMock(side_effect=AssertionError("should not load image bytes"))
        backend = _FakeBackend([Detection(x=0.2, y=0.2, width=0.1, height=0.1, confidence=0.75, label_uuid="dog")])
        img = _image(bboxes=[])
        await task._process_image(
            images_api=MagicMock(),
            backend=backend,
            img=img,
            label_names={"cat": "Cat", "dog": "Dog"},
        )
        assert task._candidates == []


class TestOnlyAddNewBoxes:
    """relabel_existing=False ("Only add new boxes"): existing bboxes are left
    alone entirely, so a run proposes new boxes and nothing else."""

    async def _run(self, task, detections, image_bboxes):
        await task._init(task._params, None)
        task._load_image_bytes = AsyncMock(return_value=b"fake-image-bytes")
        await task._process_image(
            images_api=MagicMock(),
            backend=_FakeBackend(detections),
            img=_image(bboxes=image_bboxes),
            label_names={"cat": "Cat", "dog": "Dog"},
        )
        return task._candidates

    async def test_relabel_candidate_is_not_proposed(self):
        # The same detection/bbox pair that TestAddNewBoxes'
        # test_add_new_and_relabel_not_double_emitted turns into a relabel plus a
        # new box must yield only the new box here.
        task = _make_task(_base_params(add_new=True, relabel_existing=False))
        detections = [
            Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9, label_uuid="dog"),  # overlaps candidate
            Detection(x=0.8, y=0.8, width=0.1, height=0.1, confidence=0.7, label_uuid="dog"),  # new
        ]
        candidates = await self._run(task, detections, [_bbox(uuid="bbox-1", label_uuid="cat")])
        assert [c.get("isNew") for c in candidates] == [True]
        assert candidates[0]["geometry"]["x"] == 0.8

    async def test_existing_boxes_still_block_new_box_overlap(self):
        # Not relabelling an existing box is not the same as ignoring it: a
        # detection sitting on one is still not "new".
        task = _make_task(_base_params(add_new=True, relabel_existing=False))
        detections = [Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9, label_uuid="dog")]
        candidates = await self._run(task, detections, [_bbox(uuid="bbox-1", label_uuid="cat")])
        assert candidates == []
        assert task._new_boxes_found == 0

    async def test_without_add_new_raises(self):
        # Neither half enabled: the run could only ever complete empty.
        task = _make_task(_base_params(crop_mode="image", add_new=False, relabel_existing=False))
        with pytest.raises(ValueError, match="relabel_existing"):
            await task._init(task._params, None)

    async def test_bbox_crop_mode_raises(self):
        # Bbox-crop mode has no whole-image detections to find new boxes with.
        task = _make_task(_base_params(crop_mode="bbox", add_new=True, relabel_existing=False))
        with pytest.raises(ValueError, match="relabel_existing"):
            await task._init(task._params, None)


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

    async def test_bbox_crop_remaps_geometry_using_clamped_crop_region(self):
        # A candidate bbox extends past the image's right/bottom edge, so the crop
        # is clamped to the visible region. The detection covers the whole crop, so
        # the proposed geometry must equal the *clamped* image-relative region —
        # remapping with the unclamped bbox width/height overshoots past the image.
        task = _make_task(_base_params(crop_mode="bbox"))
        await task._init(task._params, None)
        task._name_to_label_uuid = {}
        task._load_image_bytes = AsyncMock(return_value=self._make_image_bytes(width=100, height=100))
        # bbox spans image-relative [0.8, 0.8]..[1.2, 1.2]; clamped to [0.8,0.8]..[1.0,1.0].
        img = _image(bboxes=[_bbox(uuid="bbox-1", label_uuid="cat", x=0.8, y=0.8, width=0.4, height=0.4)])
        detections = [Detection(x=0.0, y=0.0, width=1.0, height=1.0, confidence=0.9, label_uuid="dog")]
        await task._process_image(
            images_api=MagicMock(),
            backend=_FakeBackend(detections),
            img=img,
            label_names={"cat": "Cat", "dog": "Dog"},
        )
        assert len(task._candidates) == 1
        geom = task._candidates[0]["geometry"]
        # Visible crop is [0.8,0.8]..[1.0,1.0] => width/height 0.2, not the unclamped 0.4.
        assert geom["x"] == pytest.approx(0.8, abs=0.01)
        assert geom["y"] == pytest.approx(0.8, abs=0.01)
        assert geom["width"] == pytest.approx(0.2, abs=0.01)
        assert geom["height"] == pytest.approx(0.2, abs=0.01)


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


class TestPreviewRunPersistence:
    """`preview_run_uuid` streams candidates into a persisted preview run instead
    of shipping thousands of them back through the one-shot task result.

    ⚠️ The task's "never writes to the DB" invariant is about *annotation*
    tables. preview_candidates is scratch state; bboxes/labels/tags stay
    untouched here.
    """

    @staticmethod
    def _fake_api(monkeypatch):
        """Captures AddPreviewCandidatesPayload calls without an HTTP round-trip."""
        calls = []

        class _FakeApi:
            def __init__(self, _client):
                pass

            def add_preview_candidates(self, run_uuid, payload):
                calls.append((run_uuid, payload))

        monkeypatch.setattr(
            "apps.tasks_server.tasks.PreviewModelRelabelTask.PreviewRunsApi",
            _FakeApi,
        )
        return calls

    async def test_without_a_run_uuid_nothing_is_written(self, monkeypatch):
        calls = self._fake_api(monkeypatch)
        task = _make_task(_base_params())
        await task._init(task._params, None)
        task._candidates = [{"imageUuid": "img-1", "bboxUuid": "bbox-1", "proposedLabelUuid": "dog", "confidence": 0.9}]
        await task._flush_candidates(force=True)
        assert calls == []
        # Untouched: the in-memory result path is still the only one.
        assert len(task._candidates) == 1
        assert task._build_result()["candidates"] == task._candidates

    async def test_forced_flush_writes_and_clears_the_buffer(self, monkeypatch):
        calls = self._fake_api(monkeypatch)
        task = _make_task(_base_params(preview_run_uuid="run-1"))
        await task._init(task._params, None)
        task._candidates = [
            {
                "imageUuid": "img-1",
                "bboxUuid": "bbox-1",
                "currentLabelUuid": "cat",
                "proposedLabelUuid": "dog",
                "confidence": 0.9,
                "geometry": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
            }
        ]
        await task._flush_candidates(force=True)

        assert len(calls) == 1
        run_uuid, payload = calls[0]
        assert run_uuid == "run-1"
        assert payload.conflict_policy == "last"
        assert len(payload.candidates) == 1
        written = payload.candidates[0]
        assert written.bbox_uuid == "bbox-1"
        assert written.current_label_uuid == "cat"
        assert written.proposed_label_uuid == "dog"
        assert written.confidence == 0.9
        assert written.geometry == {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}
        # source_model records which model in a multi-model run produced this.
        assert written.source_model == "my_detector"

        # Buffer cleared so a resume cannot re-insert it, but the count survives.
        assert task._candidates == []
        assert task._candidates_written == 1

    async def test_flush_is_batched_until_the_threshold(self, monkeypatch):
        from apps.tasks_server.tasks.PreviewModelRelabelTask import _CANDIDATE_FLUSH_SIZE

        calls = self._fake_api(monkeypatch)
        task = _make_task(_base_params(preview_run_uuid="run-1"))
        await task._init(task._params, None)

        task._candidates = [
            {"imageUuid": f"img-{i}", "bboxUuid": f"bbox-{i}", "proposedLabelUuid": "dog", "confidence": 0.5}
            for i in range(_CANDIDATE_FLUSH_SIZE - 1)
        ]
        await task._flush_candidates()
        assert calls == [], "flushed before reaching the batch size"

        task._candidates.append(
            {"imageUuid": "img-x", "bboxUuid": "bbox-x", "proposedLabelUuid": "dog", "confidence": 0.5}
        )
        await task._flush_candidates()
        assert len(calls) == 1
        assert len(calls[0][1].candidates) == _CANDIDATE_FLUSH_SIZE

    async def test_flushed_candidates_leave_the_checkpoint(self, monkeypatch):
        """A resumed run must not re-insert what it already wrote."""
        self._fake_api(monkeypatch)
        task = _make_task(_base_params(preview_run_uuid="run-1"))
        await task._init(task._params, None)
        task._candidates = [{"imageUuid": "img-1", "bboxUuid": "bbox-1", "proposedLabelUuid": "dog", "confidence": 0.9}]
        await task._flush_candidates(force=True)

        resume = task._build_resume_data()
        assert resume["candidates"] == []
        assert resume["candidates_written"] == 1

        resumed = _make_task(_base_params(preview_run_uuid="run-1"))
        await resumed._init(resumed._params, resume)
        assert resumed._candidates == []
        assert resumed._candidates_written == 1

    async def test_a_failed_flush_keeps_the_buffer_for_retry(self, monkeypatch):
        class _BoomApi:
            def __init__(self, _client):
                pass

            def add_preview_candidates(self, run_uuid, payload):
                raise RuntimeError("db down")

        monkeypatch.setattr("apps.tasks_server.tasks.PreviewModelRelabelTask.PreviewRunsApi", _BoomApi)
        task = _make_task(_base_params(preview_run_uuid="run-1"))
        await task._init(task._params, None)
        task._candidates = [{"imageUuid": "img-1", "bboxUuid": "bbox-1", "proposedLabelUuid": "dog", "confidence": 0.9}]
        with pytest.raises(RuntimeError):
            await task._flush_candidates(force=True)
        assert len(task._candidates) == 1, "a failed flush dropped the candidates it could not write"
        assert task._candidates_written == 0

    async def test_result_carries_the_run_uuid_instead_of_the_candidates(self, monkeypatch):
        self._fake_api(monkeypatch)
        task = _make_task(_base_params(preview_run_uuid="run-1"))
        await task._init(task._params, None)
        task._candidates = [{"imageUuid": "img-1", "bboxUuid": "bbox-1", "proposedLabelUuid": "dog", "confidence": 0.9}]
        await task._flush_candidates(force=True)

        result = task._build_result()
        assert result["preview_run_uuid"] == "run-1"
        assert result["candidates"] == []
        # The count still reflects everything written, not the empty buffer.
        assert result["candidate_count"] == 1

    async def test_conflict_policy_is_threaded_through(self, monkeypatch):
        calls = self._fake_api(monkeypatch)
        task = _make_task(_base_params(preview_run_uuid="run-1", conflict_policy="first"))
        await task._init(task._params, None)
        task._candidates = [{"imageUuid": "img-1", "bboxUuid": "bbox-1", "proposedLabelUuid": "dog", "confidence": 0.9}]
        await task._flush_candidates(force=True)
        assert calls[0][1].conflict_policy == "first"

    async def test_invalid_conflict_policy_raises(self):
        task = _make_task(_base_params(conflict_policy="middle"))
        with pytest.raises(ValueError, match="conflict_policy"):
            await task._init(task._params, None)


class TestBatchConcurrency:
    """The batch loop runs images concurrently (Semaphore + gather), mirroring
    ApplyFilterChainTask. Inference is ~97% of the per-image cost and only ~2/3
    of that round-trip is actual model work, so a serial loop leaves the shared
    inference server idle between requests.
    """

    @staticmethod
    def _patched_run(task, images, detect, monkeypatch, batch_size=None):
        """Drive `_run()` with the DB/inference edges faked out.

        Returns a context manager; `images` is the list of camelCase image dicts
        `batch_images_by_uuids` should hand back, `detect` the coroutine standing
        in for `_HttpInferenceBackend.detect`.
        """
        if batch_size is not None:
            monkeypatch.setattr("apps.tasks_server.tasks.PreviewModelRelabelTask._IMAGE_BATCH_SIZE", batch_size)

        by_uuid = {img["uuid"]: img for img in images}

        class _FakeImagesApi:
            def __init__(self, _client):
                pass

            def batch_images_by_uuids(self, payload):
                return MagicMock(images=[by_uuid[u] for u in payload.image_uuids if u in by_uuid])

        class _FakeLabelsApi:
            def __init__(self, _client):
                pass

            def list_labels(self):
                return MagicMock(labels=[MagicMock(uuid="cat", name="Cat"), MagicMock(uuid="dog", name="Dog")])

        monkeypatch.setattr("apps.tasks_server.tasks.PreviewModelRelabelTask.ImagesApi", _FakeImagesApi)
        monkeypatch.setattr("apps.tasks_server.tasks.PreviewModelRelabelTask.LabelsApi", _FakeLabelsApi)
        # Reads the model's sidecar off disk; nothing on disk in a unit test.
        monkeypatch.setattr(
            "apps.tasks_server.tasks.PreviewModelRelabelTask._read_model_kind", lambda _n: "object_detector"
        )
        task._pin_model = AsyncMock()
        task._load_image_bytes = AsyncMock(return_value=b"fake-jpeg")
        return patch("apps.tasks_server.tasks.PreviewModelRelabelTask._HttpInferenceBackend.detect", new=detect)

    @staticmethod
    def _images(count):
        return [
            _image(uuid=f"img-{i}", filename=f"img_{i}.jpg", bboxes=[_bbox(uuid=f"bbox-{i}", label_uuid="cat")])
            for i in range(count)
        ]

    async def test_counters_match_serial_over_many_images(self, monkeypatch):
        """Out-of-order completion must not disturb the totals, including across
        a batch boundary that isn't a divisor of the image count."""
        count = 12
        images = self._images(count)

        async def _detect(self, model_name, image_bytes, conf_thresh):
            return [Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9, label_uuid="dog", class_str=None)]

        task = _make_task(
            _base_params(
                image_uuids=[f"img-{i}" for i in range(count)],
                bbox_uuids=[f"bbox-{i}" for i in range(count)],
                concurrency=4,
            )
        )
        await task._init(task._params, None)
        with self._patched_run(task, images, _detect, monkeypatch, batch_size=5):
            result = await task._run()

        assert result["images_processed"] == count
        assert result["images_failed"] == 0
        # Every bbox is labelled "cat" and every detection proposes "dog", so
        # each image yields exactly one candidate relabel.
        assert result["candidate_count"] == count
        assert len(task._processed_image_uuids) == count

    async def test_images_actually_overlap(self, monkeypatch):
        """The point of the change: prove more than one image is in flight at
        once, so this can't silently regress to serial and still pass."""
        count = 8
        images = self._images(count)

        in_flight = 0
        max_in_flight = 0

        async def _slow_detect(self, model_name, image_bytes, conf_thresh):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            try:
                await asyncio.sleep(0.01)  # a yield point, standing in for the HTTP round-trip
                return []
            finally:
                in_flight -= 1

        task = _make_task(
            _base_params(
                image_uuids=[f"img-{i}" for i in range(count)],
                bbox_uuids=[f"bbox-{i}" for i in range(count)],
                concurrency=4,
            )
        )
        await task._init(task._params, None)
        with self._patched_run(task, images, _slow_detect, monkeypatch, batch_size=8):
            result = await task._run()

        assert result["images_processed"] == count
        assert max_in_flight > 1, "images ran serially — the concurrency change has regressed"
        assert max_in_flight <= 4, f"semaphore breached: {max_in_flight} in flight with concurrency=4"

    async def test_concurrency_is_clamped_to_at_least_one(self, monkeypatch):
        """A 0 or negative param would make asyncio.Semaphore(0) block forever,
        hanging the batch loop with no way out."""
        images = self._images(2)

        async def _detect(self, model_name, image_bytes, conf_thresh):
            return []

        task = _make_task(_base_params(image_uuids=["img-0", "img-1"], bbox_uuids=["bbox-0", "bbox-1"], concurrency=0))
        await task._init(task._params, None)
        assert task._concurrency == 1
        with self._patched_run(task, images, _detect, monkeypatch):
            result = await asyncio.wait_for(task._run(), timeout=10)

        assert result["images_processed"] == 2

    async def test_defaults_to_the_shared_concurrency(self):
        from apps.tasks_server.tasks.PreviewModelRelabelTask import _DEFAULT_CONCURRENCY

        task = _make_task(_base_params())
        await task._init(task._params, None)
        assert task._concurrency == _DEFAULT_CONCURRENCY

    async def test_one_failing_image_does_not_abort_its_batch(self, monkeypatch):
        """gather() would propagate the first exception and lose the rest of the
        batch — each image has to catch its own."""
        count = 6
        images = self._images(count)

        async def _detect(self, model_name, image_bytes, conf_thresh):
            return []

        task = _make_task(
            _base_params(
                image_uuids=[f"img-{i}" for i in range(count)],
                bbox_uuids=[f"bbox-{i}" for i in range(count)],
                concurrency=4,
            )
        )
        await task._init(task._params, None)

        real_load = task._load_image_bytes

        async def _load(filename, images_api):
            if filename == "img_3.jpg":
                raise FileNotFoundError(filename)
            return b"fake-jpeg"

        with self._patched_run(task, images, _detect, monkeypatch, batch_size=6):
            task._load_image_bytes = _load
            result = await task._run()

        assert result["images_failed"] == 1
        # Still visited (and checkpointed) — a failure is not a reason to retry
        # the image forever on every resume.
        assert result["images_processed"] == count
        assert any("img_3.jpg" in e for e in result["sample_errors"])

    async def test_cancel_mid_batch_leaves_unvisited_images_out_of_the_checkpoint(self, monkeypatch):
        """Images skipped by the cancel check must not be marked processed, or a
        resumed run would silently never look at them."""
        count = 8
        images = self._images(count)

        started = 0

        async def _detect(self, model_name, image_bytes, conf_thresh):
            nonlocal started
            started += 1
            if started == 1:
                task._cancel_flag.set()
            await asyncio.sleep(0)
            return []

        task = _make_task(
            _base_params(
                image_uuids=[f"img-{i}" for i in range(count)],
                bbox_uuids=[f"bbox-{i}" for i in range(count)],
                concurrency=2,
            )
        )
        await task._init(task._params, None)
        with self._patched_run(task, images, _detect, monkeypatch, batch_size=8):
            result = await task._run()

        assert result["images_processed"] < count, "cancel did not stop the batch"
        # The two invariants a resume depends on: the counter and the visited set
        # agree, and nothing unvisited was checkpointed.
        assert result["images_processed"] == len(task._processed_image_uuids)
        resume = task._build_resume_data()
        assert len(resume["processed_image_uuids"]) == result["images_processed"]


class TestInferenceBackendWiring:
    """`_run()` builds its own `_HttpInferenceBackend`, and nothing above covers
    that construction: every other test injects a `_FakeBackend` or patches
    `_HttpInferenceBackend.detect` outright, so the backend's own plumbing is
    never entered.

    It was broken, and completely. `_HttpInferenceBackend` takes a *factory*
    returning a client — `ApplyFilterChainTask` passes
    `self._manager.get_inference_api_client`, uncalled — while this task passed
    the already-called client. `detect()` evaluates `self._client_provider()`,
    which on an `ApiClient` instance raises `TypeError: 'ApiClient' object is
    not callable`, for every image on every run. `_run`'s per-image handler
    catches it into `images_failed`, so the task still reported success and the
    preview modal simply said there were no candidates: the "Run Model on
    Results always returns 0" report.

    This test therefore patches strictly *below* the backend — the generated
    `InferenceApi` that `_HttpInferenceBackend` resolves — leaving `detect` and
    the client-provider call real, which is the only way the defect is visible.
    """

    @staticmethod
    def _detection(label_uuid, x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9):
        """The generated client's detection shape (attribute access, not dicts)."""
        bbox = SimpleNamespace(x=x, y=y, width=width, height=height, label_uuid=label_uuid, class_str=None)
        return SimpleNamespace(bbox=bbox, confidence=confidence)

    @staticmethod
    def _make_task(params):
        """Like the module-level `_make_task`, but with a manager that hands out a
        client which is *not callable*.

        This is the whole reason the defect could hide. `_MOCK_MANAGER` is a
        `MagicMock`, so `get_inference_api_client()` returns a `MagicMock` — and a
        `MagicMock` is callable, which makes the bad `self._client_provider()` call
        succeed and produce a second mock. The real `inference_client.ApiClient` is
        not callable, so only a stand-in that shares that property reproduces
        production behaviour.
        """

        class _ApiClientLike:
            """Non-callable, exactly like the generated `ApiClient`."""

        manager = MagicMock()
        manager.update_task_heartbeat = AsyncMock()
        manager.get_inference_api_client.return_value = _ApiClientLike()
        task = PreviewModelRelabelTask(task_config_uuid="cfg-uuid", params=params, manager=manager)
        task.set_instance_uuid("11111111-1111-1111-1111-111111111111")
        return task

    def _patch_edges(self, task, images, detections, monkeypatch):
        """Fake the DB and the inference HTTP call, and nothing in between."""
        by_uuid = {img["uuid"]: img for img in images}

        class _FakeImagesApi:
            def __init__(self, _client):
                pass

            def batch_images_by_uuids(self, payload):
                return MagicMock(images=[by_uuid[u] for u in payload.image_uuids if u in by_uuid])

        class _FakeLabelsApi:
            def __init__(self, _client):
                pass

            def list_labels(self):
                return MagicMock(labels=[MagicMock(uuid="cat", name="Cat"), MagicMock(uuid="dog", name="Dog")])

        monkeypatch.setattr("apps.tasks_server.tasks.PreviewModelRelabelTask.ImagesApi", _FakeImagesApi)
        monkeypatch.setattr("apps.tasks_server.tasks.PreviewModelRelabelTask.LabelsApi", _FakeLabelsApi)
        monkeypatch.setattr(
            "apps.tasks_server.tasks.PreviewModelRelabelTask._read_model_kind", lambda _n: "object_detector"
        )
        task._pin_model = AsyncMock()
        task._load_image_bytes = AsyncMock(return_value=b"fake-jpeg")

        # Patched where `_HttpInferenceBackend` looks it up, so the backend's
        # `InferenceApi(self._client_provider())` line still runs for real.
        def _fake_inference_api(_client):
            api = MagicMock()
            api.detect.return_value = SimpleNamespace(status="success", result=SimpleNamespace(detections=detections))
            return api

        monkeypatch.setattr("apps.tasks_server.tasks.ApplyFilterChainTask.InferenceApi", _fake_inference_api)

    async def test_run_reaches_inference_through_the_real_backend(self, monkeypatch):
        task = self._make_task(_base_params())
        await task._init(task._params, None)
        images = [_image(uuid="img-1", filename="img_1.jpg", bboxes=[_bbox(uuid="bbox-1", label_uuid="cat")])]
        self._patch_edges(task, images, [self._detection("dog")], monkeypatch)

        result = await task._run()

        # The assertion that fails on the bug: the run "succeeds" either way, so
        # only the failure counter and the sample error say what went wrong.
        assert result["images_failed"] == 0, f"inference backend never ran: {result['sample_errors']}"
        assert result["candidate_count"] == 1
        assert task._candidates[0]["proposedLabelUuid"] == "dog"

    async def test_add_new_run_over_a_boxless_image_proposes_a_new_box(self, monkeypatch):
        """The user-facing shape of the same bug: 'add boxes' over images with no
        bboxes at all, which can only ever produce new-box candidates."""
        task = self._make_task(_base_params(bbox_uuids=[], add_new=True, relabel_existing=False))
        await task._init(task._params, None)
        images = [_image(uuid="img-1", filename="img_1.jpg", bboxes=[])]
        self._patch_edges(task, images, [self._detection("dog")], monkeypatch)

        result = await task._run()

        assert result["images_failed"] == 0, f"inference backend never ran: {result['sample_errors']}"
        assert result["new_boxes_found"] == 1
        assert task._candidates[0]["isNew"] is True
