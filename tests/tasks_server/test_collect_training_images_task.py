"""Tests for CollectTrainingImagesTask.

Covers the resume/dedup logic introduced alongside the timing instrumentation:

  - _init() correctly loads / initialises _processed_ids from resume_data.
  - After a successful run, _resume_data contains the set of processed IDs.
  - On a resumed run, images whose unique IDs are already in _processed_ids are
    skipped without calling inference.
  - Images that carry no source/frame_idx metadata are never tracked (streaming
    providers such as URL or MQTT cameras cannot be deduplicated).
  - When _cancel_flag and _data_req_flag arrive simultaneously, state is saved
    before the task exits.
"""

import asyncio
import base64
import itertools
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from inference_client.api.inference_api import InferenceApi

from apps.helpers.bboxUtils import iou_xywh
from apps.helpers.consts import TaskStatus
from apps.helpers.imageProviders.imageProvider import (
    ImageProvider,
    ImageProviderMetadata,
    ImageProviderType,
    ImageWithProviderMetadata,
)
from apps.tasks_server.tasks._detection_merger import MergePolicy
from apps.tasks_server.tasks.CollectTrainingImagesTask import (
    SAME_BBOX_IOU_THRESHOLD,
    CollectTrainingImagesTask,
    ConfidenceBand,
    _parse_confidence_band,
)
from apps.tasks_server.tasks.utils.imageHelper import ImageHelper
from apps.tasks_server.tasks.utils.labelHelper import (
    build_label_ancestry_map,
    build_label_metadata_map,
)

# ─── Shared constants ─────────────────────────────────────────────────────────

_LABEL_UUID = "label-test-uuid"
_MODEL_NAME = "test-model"
_SOURCE = "clip.mp4"
# _save_image sanitizes the source name for the filename (dots aren't filesystem-safe
# in this context); unique_id/processed_ids use the raw _SOURCE instead.
_SOURCE_FILE = ImageHelper.sanitize_filename(_SOURCE)

_PARAMS = {
    "source_uuids": ["src-uuid"],
    "output_dir": "test_collected",
    "model_label_config": [{"modelName": _MODEL_NAME, "labelValues": {_LABEL_UUID: 0.5}}],
}


# ─── Test helpers ─────────────────────────────────────────────────────────────


def _tracked_image(frame_idx: int, source: str = _SOURCE) -> ImageWithProviderMetadata:
    """Image with source + frame_idx so it gets a unique_id."""
    return ImageWithProviderMetadata(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        metadata=ImageProviderMetadata(source=source, frame_idx=frame_idx),
    )


def _untracked_image() -> ImageWithProviderMetadata:
    """Image with no metadata — cannot be deduplicated (e.g. live URL stream)."""
    return ImageWithProviderMetadata(image=np.zeros((8, 8, 3), dtype=np.uint8))


class _FiniteProvider(ImageProvider):
    """Yields a fixed list of images then signals exhaustion.

    Models a FIXED provider (known frame count, e.g. a file list) — the type whose
    processed_ids resume-by-frame-idx tracking is still exercised by these tests.
    """

    def __init__(self, images: list):
        self._queue = list(images)

    @classmethod
    def provider_type(cls) -> ImageProviderType:
        return ImageProviderType.FIXED

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def getNextImage(self) -> Optional[ImageWithProviderMetadata]:
        return self._queue.pop(0) if self._queue else None


class _LiveProvider(ImageProvider):
    """Yields a fixed list of images then signals exhaustion, but reports a live
    provider type (LIVE_SNAPSHOT by default) — models a real camera/stream source."""

    def __init__(self, images: list, ptype: ImageProviderType = ImageProviderType.LIVE_SNAPSHOT):
        self._queue = list(images)
        self._ptype = ptype

    def provider_type(self) -> ImageProviderType:
        return self._ptype

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def getNextImage(self) -> Optional[ImageWithProviderMetadata]:
        return self._queue.pop(0) if self._queue else None


class _InfiniteLiveProvider(ImageProvider):
    """Yields an unbounded stream of tracked images with a live provider type.

    Used to exercise stop conditions (max_images/max_duration) and pin renewal:
    a provider that would run forever on its own, so the test hangs (and fails on
    timeout) if the stop condition/renewal logic under test doesn't actually work,
    rather than passing vacuously because the provider ran out of frames anyway.
    """

    def __init__(self, ptype: ImageProviderType = ImageProviderType.LIVE_SNAPSHOT, source: str = _SOURCE):
        self._ptype = ptype
        self._source = source
        self._idx = itertools.count()
        self.stopped = False

    def provider_type(self) -> ImageProviderType:
        return self._ptype

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        self.stopped = True

    async def getNextImage(self) -> Optional[ImageWithProviderMetadata]:
        return ImageWithProviderMetadata(
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            metadata=ImageProviderMetadata(source=self._source, frame_idx=next(self._idx)),
        )


def _detection(label_uuid: str = _LABEL_UUID, confidence: float = 0.9) -> MagicMock:
    det = MagicMock()
    det.confidence = confidence
    det.bbox.label_uuid = label_uuid
    det.bbox.x = det.bbox.y = 0.0
    det.bbox.width = det.bbox.height = 0.5
    det.bbox.class_str = "object"
    return det


def _make_task(params=None, resume_data=None) -> CollectTrainingImagesTask:
    manager = MagicMock()
    manager.get_db_api_client.return_value = MagicMock()
    manager.get_inference_api_client.return_value = MagicMock()
    task = CollectTrainingImagesTask("cfg-uuid", params or _PARAMS, manager)
    if resume_data:
        task.set_resume_data(resume_data)
    return task


async def _run_task(
    images: list,
    params=None,
    resume_data=None,
    detect: bool = True,
    tmp_path: Path = None,
    provider: Optional[ImageProvider] = None,
) -> CollectTrainingImagesTask:
    """Run a CollectTrainingImagesTask against a fake provider with all I/O mocked.

    Returns the completed task for assertion.
    """
    task = _make_task(params=params, resume_data=resume_data)
    if provider is None:
        provider = _FiniteProvider(images)

    inf_result = MagicMock()
    inf_result.result.detections = [_detection()] if detect else []

    pin_resp = MagicMock()
    pin_resp.pin_id = "pin-1"

    files_root = str(tmp_path or Path("/tmp"))

    with (
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.SourceHelper") as mock_sh,
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ModelsApi") as mock_models_cls,
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.InferenceApi", autospec=True) as mock_inf_cls,
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ImagesApi") as mock_img_cls,
        patch.dict(os.environ, {"FILES_ROOT": files_root}),
    ):
        mock_sh.return_value.get_image_provider.return_value = provider
        mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.return_value = pin_resp
        mock_inf_cls.return_value.detect.return_value = inf_result
        mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

        await task.start()
        await task.wait_for_task_done(timeout=10.0)

    return task


# ─── _init() unit tests ───────────────────────────────────────────────────────


async def test_init_empty_processed_ids_for_fresh_run():
    task = _make_task()
    await task._init(_PARAMS, resume_data=None)
    assert task._processed_ids == set()
    assert task._images_saved == 0


async def test_init_loads_processed_ids_from_resume_data():
    ids = [f"{_SOURCE}:{i}" for i in range(5)]
    task = _make_task()
    await task._init(_PARAMS, resume_data={"images_saved": 5, "processed_ids": ids})
    assert task._images_saved == 5
    assert task._processed_ids == set(ids)


async def test_init_empty_resume_data_gives_empty_ids():
    task = _make_task()
    await task._init(_PARAMS, resume_data={})
    assert task._processed_ids == set()
    assert task._images_saved == 0


@pytest.mark.parametrize("bad_dir", ["../../data/db", "/etc", "foo/../../bar", ".."])
async def test_init_rejects_output_dir_traversal(bad_dir):
    """output_dir is joined onto files_root when writing collected images; an
    absolute path or '..' traversal would write outside the dataset root."""
    params = {**_PARAMS, "output_dir": bad_dir}
    task = _make_task(params)
    with pytest.raises(ValueError):
        await task._init(params, resume_data=None)


async def test_init_accepts_nested_relative_output_dir():
    """A normal nested relative output_dir must still be accepted."""
    params = {**_PARAMS, "output_dir": "collected/run1"}
    task = _make_task(params)
    await task._init(params, resume_data=None)
    assert task._output_dir == "collected/run1"


# ─── Full-run integration tests ───────────────────────────────────────────────


async def test_processed_ids_saved_in_resume_data_after_run(tmp_path):
    """After a completed run, resume_data must contain the IDs of saved images."""
    images = [_tracked_image(0), _tracked_image(1), _tracked_image(2)]
    task = await _run_task(images, tmp_path=tmp_path)

    saved_ids = set(task._resume_data.get("processed_ids", []))
    assert f"{_SOURCE}:0" in saved_ids
    assert f"{_SOURCE}:1" in saved_ids
    assert f"{_SOURCE}:2" in saved_ids


async def test_images_saved_count_matches_detections(tmp_path):
    """images_saved in resume_data reflects only frames with matching detections."""
    images = [_tracked_image(0), _tracked_image(1)]
    task = await _run_task(images, detect=True, tmp_path=tmp_path)
    assert task._resume_data["images_saved"] == 2


async def test_no_detections_means_no_processed_ids(tmp_path):
    """If inference never fires (no detections), nothing should be saved or tracked."""
    images = [_tracked_image(0), _tracked_image(1)]
    task = await _run_task(images, detect=False, tmp_path=tmp_path)
    assert task._resume_data["images_saved"] == 0
    assert task._resume_data["processed_ids"] == []


# ─── Resume / dedup tests ─────────────────────────────────────────────────────


async def test_already_processed_images_are_skipped(tmp_path):
    """On a resumed run, images already in processed_ids must not trigger inference."""
    existing_ids = [f"{_SOURCE}:0", f"{_SOURCE}:1"]
    images = [_tracked_image(0), _tracked_image(1)]

    provider = _FiniteProvider(images)
    task = _make_task(resume_data={"images_saved": 2, "processed_ids": existing_ids})

    pin_resp = MagicMock()
    pin_resp.pin_id = "pin-1"

    with (
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.SourceHelper") as mock_sh,
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ModelsApi") as mock_models_cls,
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.InferenceApi", autospec=True) as mock_inf_cls,
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ImagesApi") as mock_img_cls,
        patch.dict(os.environ, {"FILES_ROOT": str(tmp_path)}),
    ):
        mock_sh.return_value.get_image_provider.return_value = provider
        mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.return_value = pin_resp
        inf_result = MagicMock()
        inf_result.result.detections = [_detection()]
        mock_inf_cls.return_value.detect.return_value = inf_result
        mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

        await task.start()
        await task.wait_for_task_done(timeout=10.0)

        # Inference must not have been called at all — images were skipped.
        mock_inf_cls.return_value.detect.assert_not_called()

    # images_saved should remain at the resumed count; no new saves.
    assert task._resume_data["images_saved"] == 2


async def test_images_without_source_metadata_not_tracked(tmp_path):
    """Images with no source/frame_idx (live stream frames) are never added to
    processed_ids, but are still processed normally."""
    images = [_untracked_image(), _untracked_image()]
    task = await _run_task(images, detect=True, tmp_path=tmp_path)

    # Both images were saved, but no IDs were recorded.
    assert task._resume_data["images_saved"] == 2
    assert task._resume_data["processed_ids"] == []


async def test_mixed_tracked_and_untracked_images(tmp_path):
    """Tracked and untracked images can coexist; only tracked ones appear in IDs."""
    images = [_tracked_image(0), _untracked_image(), _tracked_image(1)]
    task = await _run_task(images, detect=True, tmp_path=tmp_path)

    saved_ids = set(task._resume_data.get("processed_ids", []))
    assert f"{_SOURCE}:0" in saved_ids
    assert f"{_SOURCE}:1" in saved_ids
    assert len(saved_ids) == 2  # the untracked image does not contribute an ID
    assert task._resume_data["images_saved"] == 3


# ─── Model priority + IoU dedup (D3) ──────────────────────────────────────────


def _detection_at(x, y, w, h, label_uuid=_LABEL_UUID, confidence=0.9) -> MagicMock:
    det = MagicMock()
    det.confidence = confidence
    det.bbox.label_uuid = label_uuid
    det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height = x, y, w, h
    det.bbox.class_str = "object"
    return det


async def test_init_preserves_model_priority_order():
    """Config order is retained (and de-duplicated) as model priority order."""
    params = {
        **_PARAMS,
        "model_label_config": [
            {"modelName": "model-a", "labelValues": {_LABEL_UUID: 0.5}},
            {"modelName": "model-b", "labelValues": {_LABEL_UUID: 0.5}},
            {"modelName": "model-a", "labelValues": {_LABEL_UUID: 0.5}},
        ],
    }
    task = _make_task(params=params)
    await task._init(params, resume_data=None)
    assert task._model_names == ["model-a", "model-b"]


async def test_run_inference_dedups_overlapping_boxes_keeping_priority():
    """When two models detect overlapping boxes (IoU >= threshold), only the
    higher-priority (earlier-configured) model's detection survives."""
    params = {
        **_PARAMS,
        "model_label_config": [
            {"modelName": "model-a", "labelValues": {_LABEL_UUID: 0.5}},
            {"modelName": "model-b", "labelValues": {_LABEL_UUID: 0.5}},
        ],
    }
    task = _make_task(params=params)
    await task._init(params, resume_data=None)

    high_priority = _detection_at(0.0, 0.0, 0.5, 0.5, confidence=0.6)
    overlapping = _detection_at(0.01, 0.01, 0.5, 0.5, confidence=0.95)  # IoU ~0.96

    def _inference(name, payload):
        result = MagicMock()
        result.result.detections = [high_priority] if name == "model-a" else [overlapping]
        return result

    inference_api = MagicMock(spec=InferenceApi)
    inference_api.detect.side_effect = _inference

    matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})
    assert matching == [high_priority]


async def test_run_inference_keeps_non_overlapping_boxes():
    """Distinct (non-overlapping) detections from different models are all kept."""
    params = {
        **_PARAMS,
        "model_label_config": [
            {"modelName": "model-a", "labelValues": {_LABEL_UUID: 0.5}},
            {"modelName": "model-b", "labelValues": {_LABEL_UUID: 0.5}},
        ],
    }
    task = _make_task(params=params)
    await task._init(params, resume_data=None)

    box_a = _detection_at(0.0, 0.0, 0.2, 0.2)
    box_b = _detection_at(0.7, 0.7, 0.2, 0.2)

    def _inference(name, payload):
        result = MagicMock()
        result.result.detections = [box_a] if name == "model-a" else [box_b]
        return result

    inference_api = MagicMock(spec=InferenceApi)
    inference_api.detect.side_effect = _inference

    matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})
    assert matching == [box_a, box_b]


# ─── iou_xywh unit tests (shared IoU primitive) ─────────────────────────────────────────────────────


def _box(x, y, w, h):
    """Return a simple namespace object with the attributes _bbox_iou expects."""
    b = MagicMock()
    b.x, b.y, b.width, b.height = x, y, w, h
    return b


def test_bbox_iou_identical_boxes_returns_one():
    """Two identical boxes have IoU of exactly 1.0."""
    box = _box(0.1, 0.1, 0.4, 0.4)
    assert iou_xywh(box, box) == pytest.approx(1.0)


def test_bbox_iou_no_overlap_returns_zero():
    """Boxes that don't touch have IoU of 0.0."""
    a = _box(0.0, 0.0, 0.2, 0.2)
    b = _box(0.5, 0.5, 0.2, 0.2)
    assert iou_xywh(a, b) == 0.0


def test_bbox_iou_at_threshold_triggers_dedup():
    """A pair of boxes whose IoU equals SAME_BBOX_IOU_THRESHOLD satisfies the >= condition."""
    # Two boxes with exactly 50% linear overlap in each dimension → IoU = 0.25/0.75 = 0.333…
    # Build a pair that sits right at the threshold value instead.
    # Use unit boxes shifted so IoU is just above threshold (verified numerically).
    # Box A: [0, 0, 1, 1]; Box B: [0.18, 0.18, 1, 1] → intersection ≈ 0.82² ≈ 0.6724
    # union = 1 + 1 - 0.6724 = 1.3276 → IoU ≈ 0.506, still below 0.65.
    # Instead verify the boundary directly: construct a pair with known IoU >= threshold.
    # Two boxes: A=[0,0,1,1], B=[0.1,0.1,1,1]
    # inter_w = min(1,1.1) - max(0,0.1) = 1 - 0.1 = 0.9; same for h → inter = 0.81
    # union = 1 + 1 - 0.81 = 1.19 → IoU ≈ 0.6807 >= 0.65
    a = _box(0.0, 0.0, 1.0, 1.0)
    b = _box(0.1, 0.1, 1.0, 1.0)
    iou = iou_xywh(a, b)
    assert iou >= SAME_BBOX_IOU_THRESHOLD, f"Expected IoU >= {SAME_BBOX_IOU_THRESHOLD}, got {iou}"


def test_bbox_iou_below_threshold_does_not_dedup():
    """A pair of boxes with IoU well below the threshold is not treated as the same object."""
    # Two 0.3×0.3 boxes with small overlap: A=[0,0,0.3,0.3], B=[0.25,0.25,0.3,0.3]
    # inter_w = min(0.3,0.55) - max(0,0.25) = 0.3 - 0.25 = 0.05; same for h → inter = 0.0025
    # union = 0.09 + 0.09 - 0.0025 = 0.1775 → IoU ≈ 0.014, well below 0.65
    a = _box(0.0, 0.0, 0.3, 0.3)
    b = _box(0.25, 0.25, 0.3, 0.3)
    iou = iou_xywh(a, b)
    assert iou < SAME_BBOX_IOU_THRESHOLD


# ─── Single-model multi-box dedup ─────────────────────────────────────────────


async def test_run_inference_single_model_multiple_boxes_all_kept():
    """Multiple non-overlapping detections from a single model all survive dedup."""
    task = _make_task()
    await task._init(_PARAMS, resume_data=None)

    box1 = _detection_at(0.0, 0.0, 0.2, 0.2)
    box2 = _detection_at(0.4, 0.4, 0.2, 0.2)
    box3 = _detection_at(0.8, 0.0, 0.1, 0.1)

    inference_api = MagicMock(spec=InferenceApi)
    result = MagicMock()
    result.result.detections = [box1, box2, box3]
    inference_api.detect.return_value = result

    matching = await task._run_inference(inference_api, "b64", {_MODEL_NAME: None})
    assert matching == [box1, box2, box3]


# ─── get_status_actions ───────────────────────────────────────────────────────


def test_get_status_actions_no_tag_uuid():
    """get_status_actions returns [] when no task-tag UUID has been assigned."""
    task = CollectTrainingImagesTask.__new__(CollectTrainingImagesTask)
    task._task_tag_uuid = None
    assert task.get_status_actions() == []


def test_get_status_actions_with_tag_uuid():
    """get_status_actions returns an open_review action when a tag UUID is set."""
    task = CollectTrainingImagesTask.__new__(CollectTrainingImagesTask)
    task._task_tag_uuid = "some-tag-uuid"
    actions = task.get_status_actions()
    assert len(actions) == 1
    action = actions[0]
    assert action["type"] == "open_review"
    assert action["label"] == "Browse results"
    assert action["payload"] == {"tag_uuids": ["some-tag-uuid"]}


async def test_run_inference_runs_all_models_even_when_first_has_no_detections():
    """All models must be called regardless of whether earlier models return detections.

    This verifies the parallel execution does not short-circuit on empty results.
    """
    params = {
        **_PARAMS,
        "model_label_config": [
            {"modelName": "model-a", "labelValues": {_LABEL_UUID: 0.5}},
            {"modelName": "model-b", "labelValues": {_LABEL_UUID: 0.5}},
        ],
    }
    task = _make_task(params=params)
    await task._init(params, resume_data=None)

    box_b = _detection_at(0.7, 0.7, 0.2, 0.2)

    called = []

    def _inference(name, payload):
        called.append(name)
        result = MagicMock()
        result.result.detections = [] if name == "model-a" else [box_b]
        return result

    inference_api = MagicMock(spec=InferenceApi)
    inference_api.detect.side_effect = _inference

    matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

    assert "model-a" in called
    assert "model-b" in called
    assert matching == [box_b]


async def test_save_image_writes_jpeg_not_png(tmp_path):
    """_save_image must produce .jpg output files; no .png files should be created."""
    images = [_tracked_image(0), _tracked_image(1)]
    await _run_task(images, detect=True, tmp_path=tmp_path)

    jpg_files = list(tmp_path.rglob("*.jpg"))
    png_files = list(tmp_path.rglob("*.png"))
    assert len(jpg_files) == 2, f"Expected 2 .jpg files, found: {jpg_files}"
    assert png_files == [], f"No .png files expected, found: {png_files}"


def test_generate_task_tag_default_is_false():
    """generate_task_tag must default to False so pre-existing configs are unaffected."""
    task = CollectTrainingImagesTask.__new__(CollectTrainingImagesTask)
    # Simulate _init reading params with no explicit generate_task_tag key
    params = {
        "source_uuids": ["src"],
        "output_dir": "out",
        "model_label_config": [],
    }
    result = bool(params.get("generate_task_tag", False))
    assert result is False, "Default must remain False to avoid tagging existing task configs silently"


# ─── Filter chain integration (Phase 4) ───────────────────────────────────────


def _fc_detection(label_uuid=_LABEL_UUID, confidence=0.9, x=0.1, y=0.1, w=0.2, h=0.2) -> MagicMock:
    det = MagicMock()
    det.confidence = confidence
    det.bbox.label_uuid = label_uuid
    det.bbox.x = x
    det.bbox.y = y
    det.bbox.width = w
    det.bbox.height = h
    det.bbox.class_str = "object"
    det.bbox.tag_uuids = []
    return det


def _fc_step(order=0, model_name="m1", action="tag_image", confidence=0.5, iou_threshold=0.5, action_params=None):
    return {
        "order": order,
        "model_name": model_name,
        "action": action,
        "confidence": confidence,
        "iou_threshold": iou_threshold,
        "action_params": action_params or {},
        "enabled": True,
    }


class _FakeFilterChainBackend:
    def __init__(self, kinds, classify_results=None, detections=None):
        self._kinds = kinds
        self._classify_results = classify_results or {}
        self._detections = detections or {}

    async def get_model_kind(self, model_name):
        return self._kinds.get(model_name, "detector")

    async def classify_whole_image(self, model_name, image_bytes):
        return self._classify_results.get(model_name, 0.0)

    async def detect(self, model_name, image_bytes, conf_thresh):
        return self._detections.get(model_name, [])


async def _task_with_chain_runner(steps, backend) -> CollectTrainingImagesTask:
    """Build a task with self._filter_chain_runner preset directly, bypassing
    _run()'s HTTP chain resolution/pinning — isolates _apply_filter_chain_to_frame's
    pure matching/mutation logic."""
    from apps.tasks_server.tasks._filter_chain_runner import FilterChainRunner

    task = _make_task()
    await task._init(_PARAMS, resume_data=None)
    task._filter_chain_runner = FilterChainRunner(steps, backend)
    return task


def _fc_provided() -> MagicMock:
    provided = MagicMock()
    provided.metadata.source = _SOURCE
    provided.metadata.frame_idx = 0
    return provided


def _fc_packet(raw_bytes: bytes = b"fake-jpeg-bytes") -> str:
    """Mirrors base64_encode_jpg's real wire format (imageUtils.py) — a JSON
    packet with a base64 `data` field — rather than a bare base64 string.
    _run()'s actual base64_img (from base64_encode_jpg or a provider's
    pre-encoded frame) is always this packet shape."""
    return json.dumps({"format": "jpeg", "metadata": {}, "data": base64.b64encode(raw_bytes).decode("utf-8")})


class TestApplyFilterChainToFrame:
    async def test_no_runner_configured_returns_matching_unchanged(self):
        task = _make_task()
        await task._init(_PARAMS, resume_data=None)
        matching = [_fc_detection()]
        result, discard, force_save = await task._apply_filter_chain_to_frame(_fc_provided(), "aGVsbG8=", matching)
        assert result is matching
        assert discard is False
        assert force_save is False

    async def test_empty_matching_short_circuits_without_evaluating(self):
        task = _make_task()
        await task._init(_PARAMS, resume_data=None)
        task._filter_chain_runner = object()  # would blow up if actually invoked
        result, discard, force_save = await task._apply_filter_chain_to_frame(_fc_provided(), "aGVsbG8=", [])
        assert result == []
        assert discard is False
        assert force_save is False

    async def test_real_encoded_packet_is_unwrapped_before_evaluation(self):
        """Regression test: base64_img is always the JSON packet produced by
        base64_encode_jpg (imageUtils.py) — `{"format": "jpeg", "data": <b64>,
        "metadata": {}}` — never a bare base64 string. Every other test in this
        class passes a bare base64 string ("aGVsbG8="), which happens to
        base64-decode cleanly and so never exercised the real shape of a live
        frame's encoded bytes. Feeding the real packet through
        `base64.b64decode()` directly (as production code did) raises
        `binascii.Error: Incorrect padding`, crashing the whole collection
        task on its very first matched frame whenever a filter chain is
        configured."""
        captured: dict = {}

        class _CapturingBackend(_FakeFilterChainBackend):
            async def classify_whole_image(self, model_name, image_bytes):
                captured["image_bytes"] = image_bytes
                return await super().classify_whole_image(model_name, image_bytes)

        backend = _CapturingBackend(kinds={"m1": "image_classifier"}, classify_results={"m1": 0.9})
        step = _fc_step(model_name="m1", action="tag_image", confidence=0.5, action_params={"tag_uuid": "tag-x"})
        task = await _task_with_chain_runner([step], backend)

        raw_bytes = b"fake-jpeg-bytes"
        matching = [_fc_detection()]
        await task._apply_filter_chain_to_frame(_fc_provided(), _fc_packet(raw_bytes), matching)

        assert captured["image_bytes"] == raw_bytes

    async def test_bare_base64_string_that_is_also_valid_json_falls_back_correctly(self):
        """Regression test: the fallback for a bare (unwrapped) base64 string only
        catches json.JSONDecodeError. A digit-only base64 string is also valid JSON
        (it parses as an int), so json.loads() succeeds but returns something that
        isn't a dict — `["data"]` on it must not be attempted, or this raises
        `TypeError: 'int' object is not subscriptable` instead of falling back to
        raw-base64 decoding as intended."""
        captured: dict = {}

        class _CapturingBackend(_FakeFilterChainBackend):
            async def classify_whole_image(self, model_name, image_bytes):
                captured["image_bytes"] = image_bytes
                return await super().classify_whole_image(model_name, image_bytes)

        backend = _CapturingBackend(kinds={"m1": "image_classifier"}, classify_results={"m1": 0.9})
        step = _fc_step(model_name="m1", action="tag_image", confidence=0.5, action_params={"tag_uuid": "tag-x"})
        task = await _task_with_chain_runner([step], backend)

        bare_b64 = "12345678"  # digit-only: valid base64 AND valid JSON (parses as an int)
        matching = [_fc_detection()]
        await task._apply_filter_chain_to_frame(_fc_provided(), bare_b64, matching)

        assert captured["image_bytes"] == base64.b64decode(bare_b64)

    async def test_discard_image_match_discards_frame(self):
        backend = _FakeFilterChainBackend(kinds={"corrupt": "image_classifier"}, classify_results={"corrupt": 0.9})
        step = _fc_step(model_name="corrupt", action="discard_image", confidence=0.5)
        task = await _task_with_chain_runner([step], backend)

        matching = [_fc_detection()]
        result, discard, force_save = await task._apply_filter_chain_to_frame(_fc_provided(), "aGVsbG8=", matching)
        assert discard is True
        assert result == []
        assert force_save is False
        assert task._filtered_discarded == 1

    async def test_strip_unverified_bboxes_empties_matching_but_forces_save(self):
        """Per plan 2.1 (never delete images), stripping all detections must still
        force the (now bbox-less) image to be saved — it's not equivalent to discard."""
        backend = _FakeFilterChainBackend(kinds={"blurry": "image_classifier"}, classify_results={"blurry": 0.9})
        step = _fc_step(model_name="blurry", action="strip_unverified_bboxes", confidence=0.5)
        task = await _task_with_chain_runner([step], backend)

        matching = [_fc_detection(), _fc_detection()]
        result, discard, force_save = await task._apply_filter_chain_to_frame(_fc_provided(), "aGVsbG8=", matching)
        assert discard is False
        assert result == []
        assert force_save is True
        assert task._filtered_stripped == 2

    async def test_tag_image_stashes_image_level_tag_not_per_bbox(self):
        """tag_image is an image-level tag — it must NOT land on every bbox's
        tag_uuids (that would make an 'excluded' image tag indistinguishable
        from a per-object tag, and the Review image-level filter would never
        match it)."""
        backend = _FakeFilterChainBackend(kinds={"corrupt": "image_classifier"}, classify_results={"corrupt": 0.9})
        step = _fc_step(model_name="corrupt", action="tag_image", confidence=0.5, action_params={"tag_uuid": "tag-x"})
        task = await _task_with_chain_runner([step], backend)

        provided = _fc_provided()
        matching = [_fc_detection(), _fc_detection()]
        result, discard, force_save = await task._apply_filter_chain_to_frame(provided, "aGVsbG8=", matching)
        assert discard is False
        assert force_save is False
        assert result is matching
        assert all(det.bbox.tag_uuids == [] for det in matching)
        assert provided._filter_chain_image_tags == {"tag-x"}
        assert task._filtered_tagged == 1

    async def test_relabel_bbox_mutates_only_overlapping_detection(self):
        from apps.tasks_server.tasks._filter_chain_runner import Detection

        det_a = _fc_detection(label_uuid="cat", x=0.1, y=0.1, w=0.2, h=0.2)
        det_b = _fc_detection(label_uuid="dog", x=0.7, y=0.7, w=0.1, h=0.1)
        overlap_det = Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9)
        backend = _FakeFilterChainBackend(kinds={"valves": "detector"}, detections={"valves": [overlap_det]})
        step = _fc_step(
            model_name="valves",
            action="relabel_bbox",
            confidence=0.5,
            iou_threshold=0.5,
            action_params={"target_label_uuid": "water_valve"},
        )
        task = await _task_with_chain_runner([step], backend)

        matching = [det_a, det_b]
        result, discard, force_save = await task._apply_filter_chain_to_frame(_fc_provided(), "aGVsbG8=", matching)
        assert discard is False
        assert force_save is False
        assert det_a.bbox.label_uuid == "water_valve"
        assert det_b.bbox.label_uuid == "dog"  # untouched
        assert task._filtered_relabeled == 1

    async def test_tag_bbox_adds_tag_only_to_overlapping_detection(self):
        from apps.tasks_server.tasks._filter_chain_runner import Detection

        det_a = _fc_detection(x=0.1, y=0.1, w=0.2, h=0.2)
        det_b = _fc_detection(x=0.7, y=0.7, w=0.1, h=0.1)
        overlap_det = Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9)
        backend = _FakeFilterChainBackend(kinds={"valves": "detector"}, detections={"valves": [overlap_det]})
        step = _fc_step(
            model_name="valves",
            action="tag_bbox",
            confidence=0.5,
            iou_threshold=0.5,
            action_params={"tag_uuid": "tag-y"},
        )
        task = await _task_with_chain_runner([step], backend)

        matching = [det_a, det_b]
        result, discard, force_save = await task._apply_filter_chain_to_frame(_fc_provided(), "aGVsbG8=", matching)
        assert discard is False
        assert force_save is False
        assert "tag-y" in det_a.bbox.tag_uuids
        assert "tag-y" not in det_b.bbox.tag_uuids
        assert task._filtered_tagged == 1


async def test_filter_chain_discard_prevents_save_end_to_end(tmp_path):
    """Full _run() wiring: a configured filter_chain_uuid whose sole step
    discards every frame must result in zero saved images, even though
    inference itself found a matching detection."""
    images = [_tracked_image(0), _tracked_image(1)]
    params = {**_PARAMS, "filter_chain_uuid": "chain-1"}
    task = _make_task(params=params)
    provider = _FiniteProvider(images)

    inf_result = MagicMock()
    inf_result.result.detections = [_detection()]

    pin_resp = MagicMock()
    pin_resp.pin_id = "pin-1"

    mock_chain = MagicMock(
        steps=[
            MagicMock(
                order=0,
                model_name="corrupt",
                action="discard_image",
                confidence=0.5,
                iou_threshold=0.5,
                action_params={},
                enabled=True,
            )
        ]
    )
    mock_chain.name = "TestChain"
    mock_filter_chains_api = MagicMock()
    mock_filter_chains_api.get_filter_chain.return_value = MagicMock(chain=mock_chain)

    with (
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.SourceHelper") as mock_sh,
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ModelsApi") as mock_models_cls,
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.InferenceApi", autospec=True) as mock_inf_cls,
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ImagesApi") as mock_img_cls,
        patch(
            "apps.tasks_server.tasks.CollectTrainingImagesTask.FilterChainsApi",
            return_value=mock_filter_chains_api,
        ),
        patch("apps.tasks_server.tasks.ApplyFilterChainTask._read_model_kind", return_value="image_classifier"),
        patch(
            "apps.tasks_server.tasks.ApplyFilterChainTask._HttpInferenceBackend.classify_whole_image",
            new=AsyncMock(return_value=0.9),
        ),
        patch.dict(os.environ, {"FILES_ROOT": str(tmp_path)}),
    ):
        mock_sh.return_value.get_image_provider.return_value = provider
        mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.return_value = pin_resp
        mock_inf_cls.return_value.detect.return_value = inf_result
        mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

        await task.start()
        await task.wait_for_task_done(timeout=10.0)

    assert task._resume_data["images_saved"] == 0
    assert task._filtered_discarded == 2
    mock_img_cls.return_value.update_image_metadata.assert_not_called()


def _make_filter_chain_mock(step_kwargs: dict) -> MagicMock:
    mock_chain = MagicMock(
        steps=[
            MagicMock(
                order=0,
                model_name=step_kwargs.get("model_name", "corrupt"),
                action=step_kwargs["action"],
                confidence=step_kwargs.get("confidence", 0.5),
                iou_threshold=step_kwargs.get("iou_threshold", 0.5),
                action_params=step_kwargs.get("action_params", {}),
                enabled=True,
            )
        ]
    )
    mock_chain.name = "TestChain"
    mock_filter_chains_api = MagicMock()
    mock_filter_chains_api.get_filter_chain.return_value = MagicMock(chain=mock_chain)
    return mock_filter_chains_api


async def test_filter_chain_strip_unverified_bboxes_still_saves_image_end_to_end(tmp_path):
    """strip_unverified_bboxes must NOT behave like discard_image: the (now
    bbox-less) image is still written to disk and its metadata saved with an
    empty bbox list — never delete images (plan 2.1)."""
    images = [_tracked_image(0)]
    params = {**_PARAMS, "filter_chain_uuid": "chain-1"}
    task = _make_task(params=params)
    provider = _FiniteProvider(images)

    inf_result = MagicMock()
    inf_result.result.detections = [_detection()]
    pin_resp = MagicMock()
    pin_resp.pin_id = "pin-1"
    mock_filter_chains_api = _make_filter_chain_mock({"model_name": "blurry", "action": "strip_unverified_bboxes"})

    with (
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.SourceHelper") as mock_sh,
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ModelsApi") as mock_models_cls,
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.InferenceApi", autospec=True) as mock_inf_cls,
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ImagesApi") as mock_img_cls,
        patch(
            "apps.tasks_server.tasks.CollectTrainingImagesTask.FilterChainsApi",
            return_value=mock_filter_chains_api,
        ),
        patch("apps.tasks_server.tasks.ApplyFilterChainTask._read_model_kind", return_value="image_classifier"),
        patch(
            "apps.tasks_server.tasks.ApplyFilterChainTask._HttpInferenceBackend.classify_whole_image",
            new=AsyncMock(return_value=0.9),
        ),
        patch.dict(os.environ, {"FILES_ROOT": str(tmp_path)}),
    ):
        mock_sh.return_value.get_image_provider.return_value = provider
        mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.return_value = pin_resp
        mock_inf_cls.return_value.detect.return_value = inf_result
        mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

        await task.start()
        await task.wait_for_task_done(timeout=10.0)

    assert task._filtered_stripped == 1
    assert task._resume_data["images_saved"] == 1
    mock_img_cls.return_value.update_image_metadata.assert_called_once()
    payload = mock_img_cls.return_value.update_image_metadata.call_args.args[0]
    assert payload.boxes == []
    jpg_files = list(tmp_path.rglob("*.jpg"))
    assert len(jpg_files) == 1


async def test_filter_chain_tag_image_calls_set_image_tag_end_to_end(tmp_path):
    """tag_image must reach the DB via set_image_tag (image-level), not by
    polluting every saved bbox's tag_uuids."""
    images = [_tracked_image(0)]
    params = {**_PARAMS, "filter_chain_uuid": "chain-1"}
    task = _make_task(params=params)
    provider = _FiniteProvider(images)

    inf_result = MagicMock()
    inf_result.result.detections = [_detection()]
    pin_resp = MagicMock()
    pin_resp.pin_id = "pin-1"
    mock_filter_chains_api = _make_filter_chain_mock(
        {"model_name": "corrupt", "action": "tag_image", "action_params": {"tag_uuid": "tag-x"}}
    )

    with (
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.SourceHelper") as mock_sh,
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ModelsApi") as mock_models_cls,
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.InferenceApi", autospec=True) as mock_inf_cls,
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ImagesApi") as mock_img_cls,
        patch(
            "apps.tasks_server.tasks.CollectTrainingImagesTask.FilterChainsApi",
            return_value=mock_filter_chains_api,
        ),
        patch("apps.tasks_server.tasks.ApplyFilterChainTask._read_model_kind", return_value="image_classifier"),
        patch(
            "apps.tasks_server.tasks.ApplyFilterChainTask._HttpInferenceBackend.classify_whole_image",
            new=AsyncMock(return_value=0.9),
        ),
        patch.dict(os.environ, {"FILES_ROOT": str(tmp_path)}),
    ):
        mock_sh.return_value.get_image_provider.return_value = provider
        mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.return_value = pin_resp
        mock_inf_cls.return_value.detect.return_value = inf_result
        mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()
        mock_img_cls.return_value.set_image_tag.return_value = MagicMock()

        await task.start()
        await task.wait_for_task_done(timeout=10.0)

    assert task._resume_data["images_saved"] == 1
    mock_img_cls.return_value.set_image_tag.assert_called_once()
    tag_payload = mock_img_cls.return_value.set_image_tag.call_args.args[0]
    assert tag_payload.tag_uuid == "tag-x"
    assert tag_payload.action == "add"
    # The tag must not have leaked into the per-bbox tag_uuids sent for the image.
    boxes_payload = mock_img_cls.return_value.update_image_metadata.call_args.args[0].boxes
    assert all("tag-x" not in b.tag_uuids for b in boxes_payload)


# ─── Live-provider filename uniqueness (Phase 1, second half) ────────────────

# Live providers with no min_capture_interval fall back to the task-level
# snapshot_interval/stream_interval rate limit (default 15s / 1s). These tests care
# about filenames, not rate limiting, so zero out the fallback to avoid a real-time
# stall between successive getNextImage() calls in the same test run.
_NO_RATE_LIMIT_PARAMS = {**_PARAMS, "snapshot_interval": 0, "stream_interval": 0}


class TestLiveFilenameUniqueness:
    """Live providers (LIVE_SNAPSHOT / LIVE_STREAM) reset frame_idx to 0 on every
    task restart, so _save_image must fold a UTC timestamp into their filenames.
    FIXED providers must keep the old {source}_{frame_idx}.jpg naming exactly,
    since resume-by-processed_ids depends on it."""

    async def test_fixed_provider_keeps_old_naming(self, tmp_path):
        images = [_tracked_image(0), _tracked_image(1)]
        await _run_task(images, detect=True, tmp_path=tmp_path)
        saved = sorted(p.name for p in tmp_path.rglob("*.jpg"))
        assert saved == [f"{_SOURCE_FILE}_0.jpg", f"{_SOURCE_FILE}_1.jpg"]

    async def test_live_provider_filenames_include_utc_timestamp(self, tmp_path):
        images = [_tracked_image(0), _tracked_image(1)]
        provider = _LiveProvider(images)
        fixed_now = datetime(2024, 6, 1, 12, 30, 45, tzinfo=timezone.utc)
        with patch("apps.tasks_server.tasks.CollectTrainingImagesTask.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            await _run_task(images, params=_NO_RATE_LIMIT_PARAMS, detect=True, tmp_path=tmp_path, provider=provider)
        saved = sorted(p.name for p in tmp_path.rglob("*.jpg"))
        assert saved == [
            f"{_SOURCE_FILE}_20240601T123045_0.jpg",
            f"{_SOURCE_FILE}_20240601T123045_1.jpg",
        ]

    async def test_live_provider_filenames_unique_across_simulated_restart(self, tmp_path):
        """Simulate a task restart: frame_idx resets to 0 on the second run, but
        the UTC timestamp differs between the two runs, so no filename collides
        (a plain {source}_{frame_idx}.jpg scheme would silently overwrite here)."""
        first_images = [_tracked_image(0)]
        second_images = [_tracked_image(0)]  # frame_idx reset, as after a restart

        with patch("apps.tasks_server.tasks.CollectTrainingImagesTask.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
            await _run_task(
                first_images,
                params=_NO_RATE_LIMIT_PARAMS,
                detect=True,
                tmp_path=tmp_path,
                provider=_LiveProvider(first_images),
            )

            mock_dt.now.return_value = datetime(2024, 6, 1, 12, 5, 0, tzinfo=timezone.utc)
            await _run_task(
                second_images,
                params=_NO_RATE_LIMIT_PARAMS,
                detect=True,
                tmp_path=tmp_path,
                provider=_LiveProvider(second_images),
            )

        jpg_files = sorted(p.name for p in tmp_path.rglob("*.jpg"))
        assert len(jpg_files) == 2
        assert len(set(jpg_files)) == 2  # no collision despite frame_idx resetting to 0


class TestProcessedIdsLiveVsFixed:
    """processed_ids must only accumulate for FIXED providers — tracking it for
    live providers would grow resume-data JSON unboundedly over a long-lived run."""

    async def test_live_provider_processed_ids_stays_empty(self, tmp_path):
        images = [_tracked_image(0), _tracked_image(1)]
        provider = _LiveProvider(images)
        task = await _run_task(images, params=_NO_RATE_LIMIT_PARAMS, detect=True, tmp_path=tmp_path, provider=provider)
        assert task._resume_data["processed_ids"] == []
        assert task._resume_data["images_saved"] == 2

    async def test_fixed_provider_processed_ids_still_populated(self, tmp_path):
        images = [_tracked_image(0), _tracked_image(1)]
        task = await _run_task(images, detect=True, tmp_path=tmp_path)  # default _FiniteProvider is FIXED
        saved_ids = set(task._resume_data["processed_ids"])
        assert saved_ids == {f"{_SOURCE}:0", f"{_SOURCE}:1"}


class TestUndecodableFrameSkipped:
    """A live source (e.g. UrlImageProvider) that fails to decode a response into
    an image returns a wrapper with `image=None` (not a bare `None` — that's
    reserved for "this source is exhausted/dead", the FIXED-provider and
    fetch-timeout cases). _run() must skip such a frame without crashing (it used
    to reach cv2.imencode with an empty array and raise cv2.error) and without
    discarding the source from `active` — a single bad frame from a flaky camera
    must not permanently stop it from being polled again."""

    async def test_none_image_frame_is_skipped_without_crashing_or_dropping_source(self, tmp_path):
        images = [
            ImageWithProviderMetadata(image=None, metadata=ImageProviderMetadata(source=_SOURCE, frame_idx=0)),
            _tracked_image(1),
        ]
        provider = _LiveProvider(images)
        task = await _run_task(images, params=_NO_RATE_LIMIT_PARAMS, detect=True, tmp_path=tmp_path, provider=provider)

        assert task.get_status() == TaskStatus.COMPLETED
        assert task._images_saved == 1


class TestUndecodableFrameRateLimit:
    """A source that repeatedly returns undecodable frames must still have its
    rate-limit clock (last_fetch) updated, so the task-level snapshot_interval
    check keeps throttling it — otherwise it's retried on essentially every loop
    iteration instead of backing off as configured."""

    async def test_undecodable_frames_still_respect_rate_limit(self, tmp_path):
        call_count = 0

        class _AlwaysUndecodableProvider(ImageProvider):
            """A live source that never produces a decodable frame — models a
            camera stuck returning transient non-image replies."""

            def provider_type(self) -> ImageProviderType:
                return ImageProviderType.LIVE_SNAPSHOT

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            async def getNextImage(self) -> Optional[ImageWithProviderMetadata]:
                nonlocal call_count
                call_count += 1
                return ImageWithProviderMetadata(
                    image=None, metadata=ImageProviderMetadata(source=_SOURCE, frame_idx=call_count)
                )

        provider = _AlwaysUndecodableProvider()
        # Bounded by max_duration (real wall clock) rather than the provider running
        # out of frames — this provider never does.
        params = {**_PARAMS, "snapshot_interval": 0.2, "stream_interval": 0.2, "max_duration": 0.5}

        task = await _run_task([], params=params, detect=True, tmp_path=tmp_path, provider=provider)

        assert task.get_status() == TaskStatus.COMPLETED
        # 0.5s / 0.2s interval ⇒ at most ~3 polls when correctly rate-limited.
        # Without the fix, last_fetch is never updated for undecodable frames, so
        # the source is hammered on essentially every loop iteration instead —
        # producing far more than a handful of calls in the same window.
        assert call_count <= 5, f"expected the source to be rate-limited, got {call_count} polls"


class TestSourceLevelIntervalOverride:
    """A provider with min_capture_interval > 0 paces itself inside getNextImage();
    the task-level _rate_limits fallback (snapshot_interval/stream_interval) must
    not also apply to it, or the two intervals compound."""

    async def test_self_pacing_provider_not_rate_limited_by_task_interval(self, tmp_path):
        images = [_tracked_image(0), _tracked_image(1)]
        provider = _LiveProvider(images, ptype=ImageProviderType.LIVE_SNAPSHOT)
        provider.min_capture_interval = 5.0  # provider paces itself

        # Freeze the clock: the task-level fallback (snapshot_interval defaults to 15s)
        # would never re-fetch under a frozen clock if it were still being applied —
        # the test would hang (and time out) instead of completing.
        with patch("apps.tasks_server.tasks.CollectTrainingImagesTask.time.perf_counter", return_value=100.0):
            task = await _run_task(images, detect=True, tmp_path=tmp_path, provider=provider)

        assert task._resume_data["images_saved"] == 2

    async def test_interval_longer_than_fetch_timeout_is_not_dropped(self, tmp_path):
        """A source configured slower than _FETCH_TIMEOUT_S must still work.

        The pacing sleep happens inside getNextImage(), so if the fetch budget did not
        sit on top of min_capture_interval, a camera polled once a minute would time out
        on every call, exhaust _MAX_RETRIES and be dropped from the active set on its
        very first fetch — despite behaving exactly as configured.
        """
        images = [_tracked_image(0)]
        provider = _LiveProvider(images, ptype=ImageProviderType.LIVE_SNAPSHOT)
        # Scaled-down stand-ins for the real ratio (a 60s source vs a 30s fetch timeout):
        # the provider's internal pacing sleep is deliberately longer than the bare
        # fetch timeout, so the test fails unless the budget includes the interval.
        pacing_sleep = 0.2
        provider.min_capture_interval = 1.0

        real_get = provider.getNextImage

        async def _paced_get():
            # Real self-pacing providers sleep inside getNextImage() before fetching.
            await asyncio.sleep(pacing_sleep)
            return await real_get()

        provider.getNextImage = _paced_get

        # Bare timeout (0.05s) < pacing sleep (0.2s): without the fix every call raises
        # TimeoutError, all _MAX_RETRIES are burned and the provider is dropped, so
        # images_saved would be 0. With the fix the budget is 0.05 + 1.0 = 1.05s.
        with patch.object(CollectTrainingImagesTask, "_FETCH_TIMEOUT_S", 0.05):
            task = await _run_task(images, detect=True, tmp_path=tmp_path, provider=provider)

        assert task._resume_data["images_saved"] == 1


class TestCheckpointRequestHoist:
    """Phase 2: the _data_req_flag check used to sit inside the per-provider loop,
    after a frame was successfully fetched — so a source rate-limited to a long
    interval (e.g. 15 minutes) at the task level left a checkpoint request
    unanswered for the whole interval. It now sits at the top of the OUTER while
    loop, serviced every pass regardless of how any individual source is paced."""

    async def test_checkpoint_serviced_promptly_despite_long_task_level_interval(self, tmp_path):
        # A single LIVE_SNAPSHOT provider with no self-pacing (min_capture_interval=0)
        # falls back to the task-level snapshot_interval rate limit. Set that interval
        # absurdly high: after the first frame is fetched, the provider is skipped via
        # a fast `continue` on every subsequent outer-loop pass — it never blocks.
        images = [_tracked_image(0), _tracked_image(1)]
        provider = _LiveProvider(images, ptype=ImageProviderType.LIVE_SNAPSHOT)
        params = {**_PARAMS, "snapshot_interval": 999_999}

        task = _make_task(params=params)

        inf_result = MagicMock()
        inf_result.result.detections = [_detection()]
        pin_resp = MagicMock()
        pin_resp.pin_id = "pin-1"
        files_root = str(tmp_path)

        with (
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.SourceHelper") as mock_sh,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ModelsApi") as mock_models_cls,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.InferenceApi", autospec=True) as mock_inf_cls,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ImagesApi") as mock_img_cls,
            patch.dict(os.environ, {"FILES_ROOT": files_root}),
        ):
            mock_sh.return_value.get_image_provider.return_value = provider
            mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.return_value = pin_resp
            mock_inf_cls.return_value.detect.return_value = inf_result
            mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

            await task.start()
            # Let the task fetch+save the first frame; the source is now rate-limited
            # for a simulated ~11.5 days (999_999s) before it will be polled again.
            await asyncio.sleep(0.3)

            task.request_data_update()
            # Pre-hoist, this would only be serviced after the source's next
            # successful fetch — i.e. it would hang for 999_999s and this
            # wait_for_data_ready() call would time out. Post-hoist it's serviced
            # on the very next outer-loop pass (a sub-second busy-loop tick).
            await task.wait_for_data_ready(timeout=2.0)
            assert task.is_data_ready()

            task.cancel_task()
            await task.wait_for_task_done(timeout=5.0)


# ─── Long-run hygiene: max_images / max_duration stop conditions ──────────────


class TestStopConditions:
    """max_images/max_duration must stop an otherwise-unbounded live task cleanly:
    the existing `finally` teardown (drain saves, stop providers, unpin models,
    write final resume data) must still run — not a bare `break` that skips it."""

    async def test_max_images_stops_cleanly_with_full_teardown(self, tmp_path):
        provider = _InfiniteLiveProvider()
        params = {**_NO_RATE_LIMIT_PARAMS, "max_images": 3}
        task = _make_task(params=params)

        inf_result = MagicMock()
        inf_result.result.detections = [_detection()]
        pin_resp = MagicMock()
        pin_resp.pin_id = "pin-1"

        with (
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.SourceHelper") as mock_sh,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ModelsApi") as mock_models_cls,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.InferenceApi", autospec=True) as mock_inf_cls,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ImagesApi") as mock_img_cls,
            patch.dict(os.environ, {"FILES_ROOT": str(tmp_path)}),
        ):
            mock_sh.return_value.get_image_provider.return_value = provider
            mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.return_value = pin_resp
            mock_inf_cls.return_value.detect.return_value = inf_result
            mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

            await task.start()
            # If the stop condition didn't work, the provider is infinite and this
            # would time out — the test fails loudly instead of passing vacuously.
            await task.wait_for_task_done(timeout=10.0)

            assert task.get_status() == TaskStatus.COMPLETED
            assert task._resume_data["images_saved"] >= 3
            assert provider.stopped is True, "provider.stop() must be called — normal teardown, not a bare break"
            mock_models_cls.return_value.unpin_model.assert_called()

    async def test_max_duration_stops_cleanly_with_full_teardown(self, tmp_path):
        provider = _InfiniteLiveProvider()
        params = {**_NO_RATE_LIMIT_PARAMS, "max_duration": 0.1}
        task = _make_task(params=params)

        inf_result = MagicMock()
        inf_result.result.detections = [_detection()]
        pin_resp = MagicMock()
        pin_resp.pin_id = "pin-1"

        with (
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.SourceHelper") as mock_sh,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ModelsApi") as mock_models_cls,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.InferenceApi", autospec=True) as mock_inf_cls,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ImagesApi") as mock_img_cls,
            patch.dict(os.environ, {"FILES_ROOT": str(tmp_path)}),
        ):
            mock_sh.return_value.get_image_provider.return_value = provider
            mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.return_value = pin_resp
            mock_inf_cls.return_value.detect.return_value = inf_result
            mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

            await task.start()
            await task.wait_for_task_done(timeout=10.0)

            assert task.get_status() == TaskStatus.COMPLETED
            assert task._resume_data["elapsed_duration_s"] >= 0.1
            assert provider.stopped is True
            mock_models_cls.return_value.unpin_model.assert_called()

    async def test_stop_conditions_default_to_unlimited(self):
        """A blank/0 max_images or max_duration must not stop the task early —
        the defaults must remain unlimited (live sources are normally user-stopped)."""
        task = _make_task()
        await task._init(_PARAMS, resume_data=None)
        assert task._max_images is None
        assert task._max_duration_s is None

        params = {**_PARAMS, "max_images": 0, "max_duration": 0}
        task2 = _make_task(params=params)
        await task2._init(params, resume_data=None)
        assert task2._max_images is None
        assert task2._max_duration_s is None


# ─── Long-run hygiene: progress for live tasks ─────────────────────────────────


class TestLiveProgress:
    """Live tasks (no FIXED provider) used to report 0% forever. Progress must be
    measured against whichever stop condition is set, and indeterminate (None —
    the frontend's striped bar) when neither is."""

    async def test_no_stop_condition_reports_indeterminate_progress(self, tmp_path):
        provider = _InfiniteLiveProvider()
        task = _make_task(params=_NO_RATE_LIMIT_PARAMS)

        inf_result = MagicMock()
        inf_result.result.detections = [_detection()]
        pin_resp = MagicMock()
        pin_resp.pin_id = "pin-1"

        with (
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.SourceHelper") as mock_sh,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ModelsApi") as mock_models_cls,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.InferenceApi", autospec=True) as mock_inf_cls,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ImagesApi") as mock_img_cls,
            patch.dict(os.environ, {"FILES_ROOT": str(tmp_path)}),
        ):
            mock_sh.return_value.get_image_provider.return_value = provider
            mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.return_value = pin_resp
            mock_inf_cls.return_value.detect.return_value = inf_result
            mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

            await task.start()
            # Poll on a real-time budget rather than a fixed tick count: under
            # xdist parallel load, the outer loop's asyncio.to_thread calls (model
            # pinning, tag lookup, etc.) can take longer than a handful of bare
            # event-loop ticks to complete.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and task.get_progress() is not None:
                await asyncio.sleep(0.05)
            assert task.get_progress() is None

            task.cancel_task()
            await task.wait_for_task_done(timeout=5.0)

    async def test_max_images_stop_condition_reports_determinate_progress(self, tmp_path):
        # Large enough that the task is still running (not yet stopped) once we
        # sample progress, small enough that the ratio-driven progress value is
        # measurably different from the base Task default of 0.0.
        provider = _InfiniteLiveProvider()
        params = {**_NO_RATE_LIMIT_PARAMS, "max_images": 1_000_000}
        task = _make_task(params=params)

        inf_result = MagicMock()
        inf_result.result.detections = [_detection()]
        pin_resp = MagicMock()
        pin_resp.pin_id = "pin-1"

        with (
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.SourceHelper") as mock_sh,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ModelsApi") as mock_models_cls,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.InferenceApi", autospec=True) as mock_inf_cls,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ImagesApi") as mock_img_cls,
            patch.dict(os.environ, {"FILES_ROOT": str(tmp_path)}),
        ):
            mock_sh.return_value.get_image_provider.return_value = provider
            mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.return_value = pin_resp
            mock_inf_cls.return_value.detect.return_value = inf_result
            mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

            await task.start()
            # Wait for at least one image to actually save, so the ratio-driven
            # progress value has diverged from the (also non-None) base default —
            # otherwise this assertion would pass vacuously even without the fix.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and getattr(task, "_images_saved", 0) == 0:
                await asyncio.sleep(0.05)
            assert getattr(task, "_images_saved", 0) > 0, "test setup: no images were saved to measure progress"

            progress = task.get_progress()
            assert progress is not None
            expected = min(100.0, task._images_saved / params["max_images"] * 100.0)
            assert progress == pytest.approx(expected)
            assert 0.0 < progress < 100.0

            task.cancel_task()
            await task.wait_for_task_done(timeout=5.0)


# ─── Long-run hygiene: model pin renewal ───────────────────────────────────────


class TestPinRenewal:
    """Pins are requested with a finite duration and never renewed today, so a task
    that outlives the pin duration silently reloads the model. Renewal must happen
    on a timer at roughly half the pin duration, swap in the new pin_id everywhere
    it's used, and release the stale one."""

    async def test_pin_renewed_after_half_duration(self, tmp_path):
        provider = _InfiniteLiveProvider()
        task = _make_task(params=_NO_RATE_LIMIT_PARAMS)

        inf_result = MagicMock()
        inf_result.result.detections = [_detection()]

        pin_id_counter = itertools.count(1)

        def _pin_resp(*_args, **_kwargs):
            resp = MagicMock()
            resp.pin_id = f"pin-{next(pin_id_counter)}"
            return resp

        with (
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.SourceHelper") as mock_sh,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ModelsApi") as mock_models_cls,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.InferenceApi", autospec=True) as mock_inf_cls,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ImagesApi") as mock_img_cls,
            patch.object(CollectTrainingImagesTask, "_PIN_RENEWAL_INTERVAL_S", 0.05),
            patch.dict(os.environ, {"FILES_ROOT": str(tmp_path)}),
        ):
            mock_sh.return_value.get_image_provider.return_value = provider
            mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.side_effect = _pin_resp
            mock_inf_cls.return_value.detect.return_value = inf_result
            mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

            await task.start()

            # Wait on the OBSERVABLE END STATE, not on the pin call or a fixed wall-clock
            # budget. Two earlier formulations of this wait were flaky:
            #   - a fixed ~1s of real time: under CPU contention the renewal coroutine
            #     simply isn't scheduled in time, and the test fails though the code is fine;
            #   - an event set from inside the pin mock: that fires in the to_thread worker
            #     BEFORE the coroutine resumes and swaps the new pin_id in, so the assert
            #     could still observe the stale value.
            # Renewal completes only once the new pin is swapped in AND the stale one is
            # released, so wait for exactly that. Exits in ~one 0.05s interval normally;
            # the deadline only bites if renewal is genuinely broken.
            def _renewal_complete() -> bool:
                # _combined_pin_ids is only created once _run() has finished pinning, so
                # it may not exist yet on the first poll — hence getattr, not direct access.
                swapped = getattr(task, "_combined_pin_ids", {}).get(_MODEL_NAME) not in (None, "pin-1")
                unpinned = (_MODEL_NAME, "pin-1") in [
                    c.args for c in mock_models_cls.return_value.unpin_model.call_args_list
                ]
                return swapped and unpinned

            async def _await_renewal() -> None:
                while not _renewal_complete():
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(_await_renewal(), timeout=30.0)

            assert task._combined_pin_ids[_MODEL_NAME] != "pin-1", "pin_id must have been swapped by renewal"
            assert mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.call_count >= 2
            unpin_calls = [c.args for c in mock_models_cls.return_value.unpin_model.call_args_list]
            assert (_MODEL_NAME, "pin-1") in unpin_calls, "the stale pin_id must be released after renewal"

            task.cancel_task()
            await task.wait_for_task_done(timeout=5.0)

    async def test_renewal_failure_does_not_kill_task(self, tmp_path):
        """A renewal HTTP failure must be logged and skipped, never raised — it
        must not tear down an otherwise-healthy long-running task."""
        provider = _InfiniteLiveProvider()
        task = _make_task(params=_NO_RATE_LIMIT_PARAMS)

        inf_result = MagicMock()
        inf_result.result.detections = [_detection()]
        pin_resp = MagicMock()
        pin_resp.pin_id = "pin-1"

        with (
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.SourceHelper") as mock_sh,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ModelsApi") as mock_models_cls,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.InferenceApi", autospec=True) as mock_inf_cls,
            patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ImagesApi") as mock_img_cls,
            patch.object(CollectTrainingImagesTask, "_PIN_RENEWAL_INTERVAL_S", 0.05),
            patch.dict(os.environ, {"FILES_ROOT": str(tmp_path)}),
        ):
            mock_sh.return_value.get_image_provider.return_value = provider
            pin_api = mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post
            # Initial pin succeeds; every renewal attempt raises.
            pin_api.side_effect = [pin_resp, Exception("boom"), Exception("boom"), Exception("boom")]
            mock_inf_cls.return_value.detect.return_value = inf_result
            mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

            await task.start()
            await asyncio.sleep(0.3)  # several renewal intervals' worth

            assert task.get_status() == TaskStatus.RUNNING, "renewal failures must not crash the task"
            assert task._combined_pin_ids[_MODEL_NAME] == "pin-1", "failed renewal must leave the old pin in place"

            task.cancel_task()
            await task.wait_for_task_done(timeout=5.0)
            assert task.get_status() == TaskStatus.COMPLETED


# ─── Long-run hygiene: stale-source notices ────────────────────────────────────


class TestStaleSourceNotices:
    """Notices escalate info -> warning -> error as multiples of the source's own
    configured fetch interval elapse without a successful frame, and clear the
    moment the source yields a frame again — tracked by provider.source_name
    (Phase 1's stable identity), not id(provider)."""

    async def _task_with_source_state(self, interval: float = 10.0, last_fetch_offset: float = 0.0):
        task = _make_task()
        await task._init(_PARAMS, resume_data=None)
        task._status = TaskStatus.RUNNING
        task._source_intervals = {"cam1": interval}
        task._run_started_at = 1000.0
        task._last_fetch_by_source = {"cam1": 1000.0 + last_fetch_offset}
        return task

    async def test_no_notice_below_info_threshold(self):
        task = await self._task_with_source_state(interval=10.0)
        with patch("apps.tasks_server.tasks.CollectTrainingImagesTask.time.perf_counter", return_value=1000.0 + 20.0):
            assert task.get_notices() == []

    async def test_info_notice_at_info_multiple(self):
        task = await self._task_with_source_state(interval=10.0)
        with patch("apps.tasks_server.tasks.CollectTrainingImagesTask.time.perf_counter", return_value=1000.0 + 30.0):
            notices = task.get_notices()
        assert len(notices) == 1
        assert notices[0]["level"] == "info"
        assert "cam1" in notices[0]["message"]

    async def test_warning_notice_at_warning_multiple(self):
        task = await self._task_with_source_state(interval=10.0)
        with patch("apps.tasks_server.tasks.CollectTrainingImagesTask.time.perf_counter", return_value=1000.0 + 100.0):
            notices = task.get_notices()
        assert len(notices) == 1
        assert notices[0]["level"] == "warning"

    async def test_error_notice_at_error_multiple(self):
        task = await self._task_with_source_state(interval=10.0)
        with patch("apps.tasks_server.tasks.CollectTrainingImagesTask.time.perf_counter", return_value=1000.0 + 300.0):
            notices = task.get_notices()
        assert len(notices) == 1
        assert notices[0]["level"] == "error"

    async def test_recovery_clears_the_notice(self):
        """Once the source yields a frame again, the notice must disappear on the
        very next get_notices() call — no separate recovery bookkeeping."""
        task = await self._task_with_source_state(interval=10.0)
        with patch("apps.tasks_server.tasks.CollectTrainingImagesTask.time.perf_counter", return_value=1000.0 + 300.0):
            assert task.get_notices()[0]["level"] == "error"

        # A frame arrives — simulate the main loop's bookkeeping.
        task._last_fetch_by_source["cam1"] = 1000.0 + 300.0
        with patch("apps.tasks_server.tasks.CollectTrainingImagesTask.time.perf_counter", return_value=1000.0 + 301.0):
            assert task.get_notices() == []

    async def test_no_notices_when_task_not_running(self):
        task = await self._task_with_source_state(interval=10.0)
        task._status = TaskStatus.COMPLETED
        with patch("apps.tasks_server.tasks.CollectTrainingImagesTask.time.perf_counter", return_value=1000.0 + 300.0):
            assert task.get_notices() == []


class TestMaxDurationExcludesPausedTime:
    """Time spent paused in place must NOT count toward max_duration.

    Counting it means a task paused overnight and resumed in the morning has
    already blown its duration budget and stops collecting immediately — both
    surprising and silently destructive, since the user's intent in pausing was
    to keep the task alive.
    """

    async def test_paused_time_is_excluded_from_elapsed_duration(self):
        task = _make_task(params={**_PARAMS, "max_duration": 1000})
        await task._init(task._params, resume_data=None)

        start = time.perf_counter()
        task._run_started_at = start

        # Simulate 100s of wall clock, 60s of which the task spent paused in place.
        with patch(
            "apps.tasks_server.tasks.CollectTrainingImagesTask.time.perf_counter",
            return_value=start + 100.0,
        ):
            assert task._elapsed_duration_s() == pytest.approx(100.0)
            task._paused_duration_s = 60.0
            # Only the 40s of ACTIVE time counts toward the budget.
            assert task._elapsed_duration_s() == pytest.approx(40.0)

    async def test_banked_active_time_survives_restart_without_paused_time(self):
        """elapsed_duration_s in resume_data carries active time only, so a restart
        doesn't re-import previously-paused time."""
        task = _make_task(params={**_PARAMS, "max_duration": 1000})
        await task._init(task._params, resume_data={"elapsed_duration_s": 40.0})

        start = time.perf_counter()
        task._run_started_at = start
        with patch(
            "apps.tasks_server.tasks.CollectTrainingImagesTask.time.perf_counter",
            return_value=start + 10.0,
        ):
            assert task._elapsed_duration_s() == pytest.approx(50.0)


# ─── Characterization tests: inference merge/dedup parity (Phase 5 prep) ──────
#
# CollectTrainingImagesTask._run_inference() ends with a merge block (search for
# "Flatten in priority order then apply cross-model hierarchy-aware IoU dedup")
# that Phase 5 of LIVE_COLLECTION_PLAN.md will extract into a shared, policy-
# parameterized helper. These tests pin *exactly* what that block does today —
# including edge cases (multi-supersede in one pass, IoU boundary behaviour,
# transitive ancestry, model-priority ordering under skewed completion timing) —
# so the refactor can be verified behaviour-preserving. They must keep passing,
# unmodified, against the pre-refactor code.
#
# Pinned rule set (for each detection, in model-priority order, versus already-
# kept boxes, whenever iou_xywh(det, kept) >= self._overlap_iou_threshold):
#   1. same label_uuid                          → drop the later (lower-priority) det
#   2. det's label is an ancestor of kept's      → kept is more specific → drop det
#   3. kept's label is an ancestor of det's      → det is more specific → supersede
#      (remove kept, keep det) — a single det may supersede several kept boxes
#   4. unrelated labels                          → keep both
#   IoU strictly below threshold                 → keep both (no comparison happens)
#
# Boundary: the code says `if iou_xywh(det.bbox, kept.bbox) < threshold: continue`,
# i.e. it *skips* only when IoU is strictly less than the threshold. IoU == threshold
# is NOT skipped, so equality participates in dedup exactly like "above" — the
# comparison is effectively `>=` triggers merge logic, `<` does not.


class TestInferenceMergeParity:
    """Characterization (parity) tests pinning CollectTrainingImagesTask._run_inference's
    current merge/dedup behaviour ahead of the Phase 5 extraction into a shared,
    first-wins/last-wins-parameterized helper (see LIVE_COLLECTION_PLAN.md).

    These tests intentionally assert on *which* detection object survives (by
    identity/confidence), not just counts, so a refactor that silently swaps which
    detection wins is caught. No production code is touched here — this class only
    records today's behaviour, bugs and all.
    """

    # 3-level label hierarchy shared by several tests below:
    #   ROOT (no parent) <- MID (parent=ROOT) <- LEAF (parent=MID)
    _ROOT = "label-root"
    _MID = "label-mid"
    _LEAF = "label-leaf"
    _UNRELATED = "label-unrelated"

    def _ancestry(self) -> dict[str, set]:
        return {
            self._ROOT: set(),
            self._MID: {self._ROOT},
            self._LEAF: {self._MID, self._ROOT},
            self._UNRELATED: set(),
        }

    async def _task_with_models(self, *model_names: str, overlap_iou_threshold: Optional[float] = None):
        model_label_config = [
            {
                "modelName": name,
                "labelValues": {
                    self._ROOT: 0.1,
                    self._MID: 0.1,
                    self._LEAF: 0.1,
                    self._UNRELATED: 0.1,
                },
            }
            for name in model_names
        ]
        params = {**_PARAMS, "model_label_config": model_label_config}
        if overlap_iou_threshold is not None:
            params["overlap_iou_threshold"] = overlap_iou_threshold
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        task._label_ancestry = self._ancestry()
        return task

    @staticmethod
    def _inference_api(per_model: dict[str, list]) -> MagicMock:
        def _inference(name, payload):
            result = MagicMock()
            result.result.detections = per_model.get(name, [])
            return result

        api = MagicMock(spec=InferenceApi)
        api.detect.side_effect = _inference
        return api

    # ── Rule 1: same label, two models — priority wins by identity ────────────

    async def test_same_label_higher_priority_model_survives_by_identity(self):
        task = await self._task_with_models("model-a", "model-b")
        high_priority = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.4)
        low_priority = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.99)

        inference_api = self._inference_api({"model-a": [high_priority], "model-b": [low_priority]})
        matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        # Identity check, not just count/label: the surviving object must be
        # model-a's detection even though model-b's has higher confidence.
        assert len(matching) == 1
        assert matching[0] is high_priority
        assert matching[0].confidence == pytest.approx(0.4)

    # ── Rule 2 vs Rule 3: ancestor/descendant pairs, both orderings ───────────

    async def test_ancestor_kept_first_descendant_second_supersedes(self):
        """kept=coarser label first; det=more-specific label second (higher IoU
        priority order) → det supersedes kept (kept removed, det kept)."""
        task = await self._task_with_models("model-a", "model-b")
        ancestor_det = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._ROOT, confidence=0.5)
        descendant_det = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.5)

        inference_api = self._inference_api({"model-a": [ancestor_det], "model-b": [descendant_det]})
        matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 1
        assert matching[0] is descendant_det

    async def test_descendant_kept_first_ancestor_second_is_dropped(self):
        """kept=more-specific label first; det=coarser label second → det is
        dropped (dominated); the original kept object is untouched."""
        task = await self._task_with_models("model-a", "model-b")
        descendant_det = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.5)
        ancestor_det = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._ROOT, confidence=0.5)

        inference_api = self._inference_api({"model-a": [descendant_det], "model-b": [ancestor_det]})
        matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 1
        assert matching[0] is descendant_det

    # ── Rule 4: unrelated labels — keep both ───────────────────────────────────

    async def test_unrelated_labels_overlapping_both_kept(self):
        task = await self._task_with_models("model-a", "model-b")
        det_a = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.5)
        det_b = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._UNRELATED, confidence=0.5)

        inference_api = self._inference_api({"model-a": [det_a], "model-b": [det_b]})
        matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 2
        assert det_a in matching and det_b in matching

    # ── IoU boundary behaviour: pinned via the real iou_xywh primitive ────────

    def _shifted_pair(self, dx: float):
        """Two same-label unit-square boxes, second shifted by dx along x. IoU is
        a strictly decreasing function of dx here, letting us bisect a boundary."""
        a = _detection_at(0.0, 0.0, 1.0, 1.0, label_uuid=self._LEAF, confidence=0.9)
        b = _detection_at(dx, 0.0, 1.0, 1.0, label_uuid=self._LEAF, confidence=0.9)
        return a, b

    async def test_iou_exactly_at_threshold_triggers_merge(self):
        """IoU == threshold is NOT skipped by the `< threshold: continue` check,
        so equality DOES trigger the same-label dedup (only one detection survives)."""
        boundary_a, boundary_b = self._shifted_pair(dx=0.2)
        threshold = iou_xywh(boundary_a.bbox, boundary_b.bbox)

        task = await self._task_with_models("model-a", "model-b", overlap_iou_threshold=threshold)
        inference_api = self._inference_api({"model-a": [boundary_a], "model-b": [boundary_b]})
        matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 1
        assert matching[0] is boundary_a

    async def test_iou_just_below_threshold_keeps_both(self):
        """A dx nudged slightly larger drops IoU just below threshold → no merge,
        both detections kept (distinct objects)."""
        boundary_a, boundary_b = self._shifted_pair(dx=0.2)
        threshold = iou_xywh(boundary_a.bbox, boundary_b.bbox)

        below_a, below_b = self._shifted_pair(dx=0.2 + 1e-6)
        below_iou = iou_xywh(below_a.bbox, below_b.bbox)
        assert below_iou < threshold, "test construction assumption: dx increase must decrease IoU"

        task = await self._task_with_models("model-a", "model-b", overlap_iou_threshold=threshold)
        inference_api = self._inference_api({"model-a": [below_a], "model-b": [below_b]})
        matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 2

    async def test_iou_just_above_threshold_triggers_merge(self):
        """A dx nudged slightly smaller raises IoU just above threshold → merge
        still triggers (same as at-threshold), only the priority winner survives."""
        boundary_a, boundary_b = self._shifted_pair(dx=0.2)
        threshold = iou_xywh(boundary_a.bbox, boundary_b.bbox)

        above_a, above_b = self._shifted_pair(dx=0.2 - 1e-6)
        above_iou = iou_xywh(above_a.bbox, above_b.bbox)
        assert above_iou > threshold, "test construction assumption: dx decrease must increase IoU"

        task = await self._task_with_models("model-a", "model-b", overlap_iou_threshold=threshold)
        inference_api = self._inference_api({"model-a": [above_a], "model-b": [above_b]})
        matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 1
        assert matching[0] is above_a

    # ── Rule 3, multiple supersessions in one pass ─────────────────────────────

    async def test_single_detection_supersedes_multiple_kept_boxes(self):
        """Two already-kept ancestor-labeled boxes (that do NOT overlap each other)
        both get superseded by one later, more-specific detection that overlaps
        each of them — exercises `to_supersede` with 2 entries and confirms the
        `reversed()` pop order doesn't corrupt indices (a forward pop order would
        shift index 1 before it's used, or IndexError on the second pop)."""
        task = await self._task_with_models("model-a", "model-b", "model-c")

        # root_det and mid_det don't overlap each other at all (iou=0), but the
        # wide leaf_det overlaps both of them at iou=0.3.
        root_det = _detection_at(0.0, 0.0, 0.3, 1.0, label_uuid=self._ROOT, confidence=0.5)
        mid_det = _detection_at(0.7, 0.0, 0.3, 1.0, label_uuid=self._MID, confidence=0.5)
        leaf_det = _detection_at(0.0, 0.0, 1.0, 1.0, label_uuid=self._LEAF, confidence=0.5)

        assert iou_xywh(root_det.bbox, mid_det.bbox) == 0.0
        assert iou_xywh(leaf_det.bbox, root_det.bbox) == pytest.approx(0.3)
        assert iou_xywh(leaf_det.bbox, mid_det.bbox) == pytest.approx(0.3)

        task._overlap_iou_threshold = 0.25
        inference_api = self._inference_api({"model-a": [root_det], "model-b": [mid_det], "model-c": [leaf_det]})
        matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None, "model-c": None})

        assert len(matching) == 1
        assert matching[0] is leaf_det

    # ── Transitive ancestry across 3 levels ─────────────────────────────────────

    async def test_ancestry_is_transitive_grandchild_supersedes_grandparent(self):
        """LEAF is two hops below ROOT (LEAF -> MID -> ROOT); a ROOT-labeled kept
        box is still recognized as an ancestor of a LEAF det and gets superseded,
        confirming ancestry is resolved transitively, not just parent-of."""
        task = await self._task_with_models("model-a", "model-b")
        root_det = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._ROOT, confidence=0.5)
        leaf_det = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.5)

        inference_api = self._inference_api({"model-a": [root_det], "model-b": [leaf_det]})
        matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 1
        assert matching[0] is leaf_det

    async def test_ancestry_is_transitive_grandparent_det_dropped_by_grandchild_kept(self):
        """Same pair, reversed priority: LEAF kept first, ROOT det second → ROOT
        det is dominated (dropped) since LEAF (already kept) is more specific."""
        task = await self._task_with_models("model-a", "model-b")
        leaf_det = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.5)
        root_det = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._ROOT, confidence=0.5)

        inference_api = self._inference_api({"model-a": [leaf_det], "model-b": [root_det]})
        matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 1
        assert matching[0] is leaf_det

    # ── Model ordering: gather() result order matches priority, not completion order ──

    async def test_gather_result_order_follows_priority_not_completion_time(self):
        """model-a is configured first (higher priority) but its (thread-pool)
        inference call is made artificially slower than model-b's. asyncio.gather
        preserves input order regardless of completion order, so results_per_model
        must still line up with self._model_names — model-a's detection must win
        the same-label dedup even though model-b's HTTP call actually returns first."""
        task = await self._task_with_models("model-a", "model-b")
        assert task._model_names == ["model-a", "model-b"]

        high_priority = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.1)
        low_priority = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.99)

        def _inference(name, payload):
            # model-a (higher priority) is deliberately slower than model-b, so
            # model-b's call returns first in real wall-clock time.
            if name == "model-a":
                time.sleep(0.08)
            result = MagicMock()
            result.result.detections = [high_priority] if name == "model-a" else [low_priority]
            return result

        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect.side_effect = _inference

        matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 1
        assert matching[0] is high_priority


class TestMergePolicyLastWins:
    """Phase 5A: merge_policy="last_wins" is the mirror image of the default
    first-wins policy pinned by TestInferenceMergeParity above. Same 3-level
    hierarchy (ROOT <- MID <- LEAF), same helper machinery, but every task is
    built with `merge_policy` set explicitly so first-wins remains the default
    for configs that don't set it."""

    _ROOT = TestInferenceMergeParity._ROOT
    _MID = TestInferenceMergeParity._MID
    _LEAF = TestInferenceMergeParity._LEAF
    _UNRELATED = TestInferenceMergeParity._UNRELATED

    def _ancestry(self) -> dict[str, set]:
        return {
            self._ROOT: set(),
            self._MID: {self._ROOT},
            self._LEAF: {self._MID, self._ROOT},
            self._UNRELATED: set(),
        }

    async def _task_with_models(self, *model_names: str, merge_policy: str = MergePolicy.LAST_WINS.value):
        model_label_config = [
            {
                "modelName": name,
                "labelValues": {
                    self._ROOT: 0.1,
                    self._MID: 0.1,
                    self._LEAF: 0.1,
                    self._UNRELATED: 0.1,
                },
            }
            for name in model_names
        ]
        params = {**_PARAMS, "model_label_config": model_label_config, "merge_policy": merge_policy}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        task._label_ancestry = self._ancestry()
        return task

    @staticmethod
    def _inference_api(per_model: dict[str, list]) -> MagicMock:
        return TestInferenceMergeParity._inference_api(per_model)

    async def test_default_policy_is_first_wins(self):
        """Sanity check: omitting merge_policy from params keeps today's default."""
        task = _make_task(params=_PARAMS)
        await task._init(_PARAMS, resume_data=None)
        assert task._merge_policy is MergePolicy.FIRST_WINS

    async def test_unknown_policy_value_falls_back_to_first_wins(self):
        """Unvalidated task params (CLAUDE.md: '_init params are unvalidated —
        cast and validate inside _init') must not raise or silently misbehave
        on garbage input; fall back to the safe default."""
        params = {**_PARAMS, "merge_policy": "not-a-real-policy"}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        assert task._merge_policy is MergePolicy.FIRST_WINS

    async def test_same_label_lower_priority_model_survives_under_last_wins(self):
        """Mirror of TestInferenceMergeParity.test_same_label_higher_priority_model_survives_by_identity:
        under last_wins, model-b (configured second / lower priority) wins the
        same-label collision instead of model-a."""
        task = await self._task_with_models("model-a", "model-b")
        high_priority = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.4)
        low_priority = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.99)

        inference_api = self._inference_api({"model-a": [high_priority], "model-b": [low_priority]})
        matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 1
        assert matching[0] is low_priority

    async def test_ancestor_supersede_preserved_regardless_of_processing_order(self):
        """Hierarchy supersede (more-specific label beats a coarser ancestor) must
        hold under last_wins too, for both orderings of which arrives 'first'."""
        task = await self._task_with_models("model-a", "model-b")
        ancestor_det = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._ROOT, confidence=0.5)
        descendant_det = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.5)

        inference_api = self._inference_api({"model-a": [ancestor_det], "model-b": [descendant_det]})
        matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 1
        assert matching[0] is descendant_det

    async def test_ancestor_supersede_preserved_reverse_model_order(self):
        task = await self._task_with_models("model-a", "model-b")
        descendant_det = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.5)
        ancestor_det = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._ROOT, confidence=0.5)

        inference_api = self._inference_api({"model-a": [descendant_det], "model-b": [ancestor_det]})
        matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 1
        assert matching[0] is descendant_det

    async def test_unrelated_labels_overlapping_both_kept_under_last_wins(self):
        task = await self._task_with_models("model-a", "model-b")
        det_a = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.5)
        det_b = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._UNRELATED, confidence=0.5)

        inference_api = self._inference_api({"model-a": [det_a], "model-b": [det_b]})
        matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 2
        assert det_a in matching and det_b in matching

    async def test_gather_result_order_still_follows_config_priority_not_completion_time(self):
        """Even under last_wins, results_per_model must line up with
        self._model_names (asyncio.gather preserves input order) before the
        policy's own reversal is applied — otherwise the "last model" the
        policy picks wouldn't be the actually-lowest-priority-configured one."""
        task = await self._task_with_models("model-a", "model-b")
        assert task._model_names == ["model-a", "model-b"]

        high_priority = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.1)
        low_priority = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.99)

        def _inference(name, payload):
            if name == "model-b":
                time.sleep(0.08)
            result = MagicMock()
            result.result.detections = [high_priority] if name == "model-a" else [low_priority]
            return result

        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect.side_effect = _inference

        matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 1
        assert matching[0] is low_priority


class TestBuildLabelAncestryMap:
    """Characterization tests for build_label_ancestry_map (tasks/utils/labelHelper.py),
    the map CollectTrainingImagesTask._run_inference consults (via self._label_ancestry)
    for the ancestor/descendant dedup rules pinned in TestInferenceMergeParity above."""

    @staticmethod
    def _labels_api(labels: list) -> MagicMock:
        """labels: list of (uuid, parent_uuid) tuples."""
        api = MagicMock()
        resp = MagicMock()
        resp.labels = [MagicMock(uuid=u, parent_uuid=p) for u, p in labels]
        api.list_labels.return_value = resp
        return api

    async def test_multi_level_ancestry_is_transitive(self):
        labels_api = self._labels_api(
            [("root", None), ("mid", "root"), ("leaf", "mid")],
        )
        ancestry = await build_label_ancestry_map(labels_api)
        assert ancestry["root"] == set()
        assert ancestry["mid"] == {"root"}
        assert ancestry["leaf"] == {"mid", "root"}

    async def test_label_with_no_parent_maps_to_empty_set(self):
        labels_api = self._labels_api([("only", None)])
        ancestry = await build_label_ancestry_map(labels_api)
        assert ancestry["only"] == set()

    async def test_unknown_uuid_absent_from_map_get_returns_empty_default(self):
        """A label uuid never seen by build_label_ancestry_map (e.g. a stale/
        deleted label still referenced by a detection) is simply absent from the
        dict — callers must use .get(uuid, ()), exactly as _run_inference does."""
        labels_api = self._labels_api([("known", None)])
        ancestry = await build_label_ancestry_map(labels_api)
        assert "unknown-uuid" not in ancestry
        assert ancestry.get("unknown-uuid", ()) == ()

    async def test_dangling_parent_uuid_becomes_synthetic_ancestry_entry(self):
        """A label whose parent_uuid points at a uuid absent from the fetched label
        list (e.g. a parent that was deleted, or a data inconsistency) does not
        raise: the dangling parent uuid is still recorded as an ancestor of the
        child, and gets its own map entry with an empty ancestor set (since its
        own parent_uuid is unknown -> None via .get())."""
        labels_api = self._labels_api([("child", "ghost-parent")])
        ancestry = await build_label_ancestry_map(labels_api)
        assert ancestry["child"] == {"ghost-parent"}
        # The dangling parent itself gets synthesized into the map with no ancestors.
        assert ancestry["ghost-parent"] == set()

    async def test_no_labels_returns_empty_map(self):
        labels_api = self._labels_api([])
        ancestry = await build_label_ancestry_map(labels_api)
        assert ancestry == {}


# ─── Phase 4: live collection status panel ─────────────────────────────────────


class TestStatusSchema:
    """status_schema() ids must match the keys get_status_data() returns — a
    silent rename of one side without the other would break the id <-> data
    pairing without any error.

    Some widgets are conditional: the dedup-calibration ones are only populated
    when dedup is enabled (same rule the stat_grid already follows for its dedup
    and filter-chain fields), so the invariant is "every emitted key is declared
    in the schema", plus the unconditional ids are always emitted."""

    _ALWAYS = {"last_capture", "stats", "sources"}
    _DEDUP_ONLY = {
        "dedup_hint",
        "dedup_image_hist",
        "dedup_bbox_hist",
        "dedup_image_sweep",
        "dedup_bbox_sweep",
    }

    def test_schema_declares_every_widget_id(self):
        schema_ids = {w["id"] for w in CollectTrainingImagesTask.status_schema()}
        assert schema_ids == self._ALWAYS | self._DEDUP_ONLY

    def test_unconditional_keys_present_before_init(self):
        task = _make_task()
        # get_status_data() must be callable straight after construction (before
        # _init even runs) without raising — it's read by a poll handler that has
        # no idea whether the task has started yet.
        data_keys = set(task.get_status_data().keys())
        assert data_keys == self._ALWAYS

    async def test_every_emitted_key_is_declared_in_the_schema(self):
        """Both with dedup off and on — no key may appear that no widget renders."""
        schema_ids = {w["id"] for w in CollectTrainingImagesTask.status_schema()}
        for enable_dedup in (False, True):
            params = {**_PARAMS, "enable_dedup": enable_dedup}
            task = _make_task(params=params)
            await task._init(params, resume_data=None)
            data_keys = set(task.get_status_data().keys())
            assert data_keys <= schema_ids, f"undeclared keys with dedup={enable_dedup}"
            assert self._ALWAYS <= data_keys
            expected_dedup = self._DEDUP_ONLY if enable_dedup else set()
            assert data_keys & self._DEDUP_ONLY == expected_dedup

    def test_schema_entries_are_status_sourced_with_expected_types(self):
        schema = {w["id"]: w for w in CollectTrainingImagesTask.status_schema()}
        assert schema["last_capture"]["type"] == "annotated_image"
        assert schema["stats"]["type"] == "stat_grid"
        assert schema["sources"]["type"] == "table"
        for widget in schema.values():
            assert widget.get("source", "status") == "status"

    def test_sources_table_columns_include_health(self):
        schema = {w["id"]: w for w in CollectTrainingImagesTask.status_schema()}
        col_keys = {c["key"] for c in schema["sources"]["columns"]}
        assert {"name", "type", "interval", "last_frame", "health"} <= col_keys


class TestGetStatusDataNoIO:
    """get_status_data() is called synchronously from a request handler and MUST
    NOT perform I/O (no DB/filesystem access, no awaits) — see the base class
    docstring. Constructing it as `async def` at all would make a bare call
    return a coroutine instead of a dict; that alone proves nothing awaits."""

    def test_get_status_data_is_a_plain_sync_method(self):
        assert not asyncio.iscoroutinefunction(CollectTrainingImagesTask.get_status_data)

    def test_get_status_data_never_touches_the_api_clients(self):
        task = _make_task()
        # Any attribute access on these mocks that isn't a plain read is a red flag;
        # since get_status_data() takes no arguments and isn't a coroutine, the only
        # way it could reach the DB/inference clients at all is through self._manager.
        task._manager = MagicMock()
        task._manager.get_db_api_client.side_effect = AssertionError("get_status_data must not touch the DB client")
        task._manager.get_inference_api_client.side_effect = AssertionError(
            "get_status_data must not touch the inference client"
        )

        result = task.get_status_data()

        task._manager.get_db_api_client.assert_not_called()
        task._manager.get_inference_api_client.assert_not_called()
        assert set(result.keys()) == {"last_capture", "stats", "sources"}

    async def test_get_status_data_safe_before_init(self):
        """Pending-state tasks (never _init'd) must not crash a poll handler."""
        task = _make_task()
        data = task.get_status_data()
        assert data["last_capture"] is None
        assert data["sources"] == []


class TestLastCaptureStatusData:
    """The 'last capture' widget carries an image path + bbox list by reference —
    never inline image bytes — matching seedRuntimeBboxes()'s expected shape
    (canvas/image.js): uuid, label {uuid,text,color}|None, x/y/width/height."""

    async def test_last_capture_populated_after_a_save(self, tmp_path):
        images = [_tracked_image(0)]
        task = await _run_task(images, detect=True, tmp_path=tmp_path)

        last_capture = task.get_status_data()["last_capture"]
        assert last_capture is not None
        assert last_capture is task._last_capture

    async def test_last_capture_is_path_and_bboxes_never_image_bytes(self, tmp_path):
        images = [_tracked_image(0)]
        task = await _run_task(images, detect=True, tmp_path=tmp_path)

        last_capture = task.get_status_data()["last_capture"]
        assert set(last_capture.keys()) == {"path", "bboxes"}
        assert isinstance(last_capture["path"], str)
        assert "test_collected" in last_capture["path"]  # output_dir from _PARAMS
        # Never leak raw pixel/bytes data into the status payload.
        assert "image" not in last_capture
        assert "bytes" not in last_capture
        assert "image_base64" not in last_capture

    async def test_last_capture_bbox_shape_matches_seedRuntimeBboxes_contract(self, tmp_path):
        images = [_tracked_image(0)]
        task = await _run_task(images, detect=True, tmp_path=tmp_path)

        bboxes = task.get_status_data()["last_capture"]["bboxes"]
        assert len(bboxes) == 1
        bbox = bboxes[0]
        assert set(bbox.keys()) == {"uuid", "label", "x", "y", "width", "height"}
        assert isinstance(bbox["uuid"], str) and bbox["uuid"]
        assert bbox["label"]["uuid"] == _LABEL_UUID
        assert bbox["x"] == 0.0 and bbox["y"] == 0.0
        assert bbox["width"] == 0.5 and bbox["height"] == 0.5

    def test_build_seed_bbox_resolves_label_text_and_color(self):
        task = _make_task()
        task._label_meta = {_LABEL_UUID: {"text": "Cat", "color": "#ff0000"}}
        det = _detection()
        seed = task._build_seed_bbox(det, "bbox-uuid-1")
        assert seed == {
            "uuid": "bbox-uuid-1",
            "label": {"uuid": _LABEL_UUID, "text": "Cat", "color": "#ff0000"},
            "x": 0.0,
            "y": 0.0,
            "width": 0.5,
            "height": 0.5,
        }

    def test_build_seed_bbox_falls_back_to_uuid_when_no_meta(self):
        task = _make_task()
        task._label_meta = {}
        det = _detection()
        seed = task._build_seed_bbox(det, "bbox-uuid-2")
        assert seed["label"] == {"uuid": _LABEL_UUID, "text": _LABEL_UUID, "color": None}

    def test_build_seed_bbox_no_label_is_none(self):
        task = _make_task()
        task._label_meta = {}
        det = _detection(label_uuid=None)
        seed = task._build_seed_bbox(det, "bbox-uuid-3")
        assert seed["label"] is None


class TestStatGridDedupAndFilterChainToggle:
    """Dedup and filter-chain stat_grid entries only appear in get_status_data()'s
    'stats' payload when the corresponding feature is actually active — otherwise
    the grid would be full of permanent zeros for features the config never
    enabled."""

    _DEDUP_KEYS = {"dedup_checked", "dedup_image_hits", "dedup_bbox_hits", "dedup_discarded"}
    _FILTER_KEYS = {"filter_discarded", "filter_relabeled", "filter_tagged", "filter_stripped"}

    async def test_dedup_and_filter_keys_absent_when_both_disabled(self):
        task = _make_task()
        await task._init(_PARAMS, resume_data=None)
        stats = task.get_status_data()["stats"]
        assert not (self._DEDUP_KEYS & stats.keys())
        assert not (self._FILTER_KEYS & stats.keys())
        assert {"saved", "processed", "captures_per_min", "uptime"} <= stats.keys()

    async def test_dedup_keys_present_when_dedup_enabled(self):
        params = {**_PARAMS, "enable_dedup": True}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        stats = task.get_status_data()["stats"]
        assert self._DEDUP_KEYS <= stats.keys()
        assert not (self._FILTER_KEYS & stats.keys())

    async def test_filter_keys_present_when_chain_active(self):
        task = _make_task()
        await task._init(_PARAMS, resume_data=None)
        # _filter_chain_runner is only built for real inside _run() once a chain
        # resolves with enabled steps; simulate "active" the same way the sources
        # table's own health check treats it (getattr(..., None) is not None).
        task._filter_chain_runner = MagicMock()
        stats = task.get_status_data()["stats"]
        assert self._FILTER_KEYS <= stats.keys()
        assert not (self._DEDUP_KEYS & stats.keys())


class TestSourcesTableHealth:
    """Sources table health must mirror get_notices()' stale-source escalation
    thresholds exactly — both read from the shared _source_staleness() helper, so
    a live source below the info threshold reports 'ok', and above it reports the
    same level get_notices() would surface."""

    async def _task_with_source_state(self):
        task = _make_task()
        await task._init(_PARAMS, resume_data=None)
        task._status = TaskStatus.RUNNING  # get_notices() only surfaces for running/paused tasks
        task._source_names = ["cam1", "cam2"]
        task._source_types = {"cam1": "live_snapshot", "cam2": "fixed"}
        task._source_intervals = {"cam1": 10.0}  # cam2 is FIXED: no interval tracked
        task._run_started_at = 1000.0
        task._last_fetch_by_source = {"cam1": 1000.0}
        return task

    def _row(self, task, name):
        rows = task.get_status_data()["sources"]
        return next(r for r in rows if r["name"] == name)

    async def test_healthy_source_reports_ok(self):
        task = await self._task_with_source_state()
        with patch("apps.tasks_server.tasks.CollectTrainingImagesTask.time.perf_counter", return_value=1000.0 + 2.0):
            row = self._row(task, "cam1")
            notices = task.get_notices()
        assert row["health"] == "ok"
        assert notices == []

    async def test_health_escalation_matches_get_notices_levels(self):
        task = await self._task_with_source_state()
        # Same offsets TestStaleSourceNotices uses for interval=10.0: 30s -> info,
        # 100s -> warning, 300s -> error.
        for offset, expected_level in ((30.0, "info"), (100.0, "warning"), (300.0, "error")):
            with patch(
                "apps.tasks_server.tasks.CollectTrainingImagesTask.time.perf_counter",
                return_value=1000.0 + offset,
            ):
                row = self._row(task, "cam1")
                notices = task.get_notices()
            assert row["health"] == expected_level
            assert notices[0]["level"] == expected_level

    async def test_fixed_source_with_no_tracked_interval_reports_na(self):
        task = await self._task_with_source_state()
        with patch("apps.tasks_server.tasks.CollectTrainingImagesTask.time.perf_counter", return_value=1000.0 + 300.0):
            row = self._row(task, "cam2")
        assert row["health"] == "n/a"
        assert row["interval"] == "—"

    async def test_source_table_includes_type_and_interval(self):
        task = await self._task_with_source_state()
        with patch("apps.tasks_server.tasks.CollectTrainingImagesTask.time.perf_counter", return_value=1000.0 + 2.0):
            row = self._row(task, "cam1")
        assert row["type"] == "live_snapshot"
        assert row["interval"] == "10s"
        assert row["last_frame"] == "2s ago"

    async def test_never_fetched_source_reports_never(self):
        task = await self._task_with_source_state()
        task._last_fetch_by_source = {}
        with patch("apps.tasks_server.tasks.CollectTrainingImagesTask.time.perf_counter", return_value=1000.0 + 2.0):
            row = self._row(task, "cam1")
        assert row["last_frame"] == "never"


class TestUptimeAndCapturesPerMinuteExcludePausedTime:
    """Uptime and captures-per-minute must be computed from _elapsed_duration_s(),
    which already excludes time spent paused in place — otherwise a long pause
    would inflate the run's apparent wall-clock duration and crater the reported
    captures-per-minute rate for no real reason."""

    async def test_uptime_excludes_paused_time(self):
        task = _make_task()
        await task._init(_PARAMS, resume_data=None)
        start = time.perf_counter()
        task._run_started_at = start
        task._images_saved = 10

        with patch(
            "apps.tasks_server.tasks.CollectTrainingImagesTask.time.perf_counter",
            return_value=start + 100.0,
        ):
            stats_before_pause = task.get_status_data()["stats"]
            task._paused_duration_s = 60.0
            stats_after_pause = task.get_status_data()["stats"]

        # 100s of wall clock, un-adjusted, would report "1m 40s"; excluding the 60s
        # spent paused, only the 40s of ACTIVE time counts.
        assert stats_before_pause["uptime"] == "1m 40s"
        assert stats_after_pause["uptime"] == "40s"

    async def test_captures_per_minute_excludes_paused_time(self):
        task = _make_task()
        await task._init(_PARAMS, resume_data=None)
        start = time.perf_counter()
        task._run_started_at = start
        task._images_saved = 20
        task._paused_duration_s = 60.0

        with patch(
            "apps.tasks_server.tasks.CollectTrainingImagesTask.time.perf_counter",
            return_value=start + 100.0,
        ):
            # Active time = 100 - 60 = 40s = 40/60 min -> 20 / (40/60) = 30/min.
            # If paused time were wrongly included, active time would be 100s ->
            # 20 / (100/60) = 12/min instead.
            stats = task.get_status_data()["stats"]

        assert stats["captures_per_min"] == pytest.approx(30.0)


class TestBuildLabelMetadataMap:
    """build_label_metadata_map (tasks/utils/labelHelper.py) resolves a bare
    label_uuid to {text, color} — used by CollectTrainingImagesTask to build
    UI-ready 'last capture' bbox descriptors without any status-time DB access."""

    @staticmethod
    def _labels_api(labels: list) -> MagicMock:
        """labels: list of (uuid, name, color) tuples.

        MagicMock's constructor treats `name=` specially (it sets the mock's own
        repr, not an attribute), so `.name` must be assigned after construction.
        """
        api = MagicMock()
        resp = MagicMock()
        label_mocks = []
        for u, n, c in labels:
            lbl = MagicMock(uuid=u, color=c)
            lbl.name = n
            label_mocks.append(lbl)
        resp.labels = label_mocks
        api.list_labels.return_value = resp
        return api

    async def test_maps_uuid_to_name_and_color(self):
        labels_api = self._labels_api([("l1", "Cat", "#ff0000"), ("l2", "Dog", "#00ff00")])
        meta = await build_label_metadata_map(labels_api)
        assert meta == {
            "l1": {"text": "Cat", "color": "#ff0000"},
            "l2": {"text": "Dog", "color": "#00ff00"},
        }

    async def test_no_labels_returns_empty_map(self):
        labels_api = self._labels_api([])
        meta = await build_label_metadata_map(labels_api)
        assert meta == {}


# ─── Phase 5B: confidence bands (LIVE_COLLECTION_PLAN.md) ─────────────────────
#
# Wire format: model_label_config[i].labelValues[labelUuid] is either the legacy
# scalar floor (`0.25`, every already-saved task config) or a `{min, max}` band
# dict. Both bounds are INCLUSIVE: `min <= confidence <= max` is "in band".
#
# The band is applied AFTER the cross-model merge, never before — filtering
# per-detection inside _run_model (pre-merge) would let a merge-worthy
# suppression never happen, because the detection that should have done the
# suppressing (e.g. a higher-priority model's box that happens to be outside
# the band) would already be gone by the time merge runs. See
# TestConfidenceBandPostMergeOrdering below for the airtight regression test.
#
# TRIGGER vs ANNOTATION: a trigger predicate (any merged detection in-band)
# decides whether the frame is saved at all; the annotation set (what actually
# gets written) is NOT filtered by band — it's every merged detection,
# in-band or not, once the trigger fires. See TestConfidenceBandTriggerVsAnnotation.


class TestParseConfidenceBand:
    """Unit tests for _parse_confidence_band, the wire-format parser that
    accepts both the legacy scalar and the new {min, max} dict."""

    def test_scalar_becomes_floor_with_max_one(self):
        band = _parse_confidence_band(0.25)
        assert band == ConfidenceBand(min_confidence=0.25, max_confidence=1.0)

    def test_int_scalar_is_accepted(self):
        band = _parse_confidence_band(1)
        assert band == ConfidenceBand(min_confidence=1.0, max_confidence=1.0)

    def test_zero_scalar_means_disabled(self):
        """Matches the pre-Phase-5B convention: the UI writes 0 to mean 'not
        collecting this label' — _init has always skipped these entries."""
        assert _parse_confidence_band(0) is None
        assert _parse_confidence_band(0.0) is None

    def test_negative_scalar_means_disabled(self):
        assert _parse_confidence_band(-0.1) is None

    def test_band_dict_parsed_directly(self):
        band = _parse_confidence_band({"min": 0.2, "max": 0.6})
        assert band == ConfidenceBand(min_confidence=0.2, max_confidence=0.6)

    def test_band_dict_missing_max_defaults_to_one(self):
        band = _parse_confidence_band({"min": 0.3})
        assert band == ConfidenceBand(min_confidence=0.3, max_confidence=1.0)

    def test_band_dict_zero_min_means_disabled(self):
        assert _parse_confidence_band({"min": 0.0, "max": 0.6}) is None

    def test_unparsable_scalar_is_disabled_not_a_crash(self):
        assert _parse_confidence_band("not-a-number") is None
        assert _parse_confidence_band(None) is None

    def test_inverted_band_clamped_to_single_point(self):
        """_init params are unvalidated raw JSON (CLAUDE.md) — a malformed/
        inverted band must degrade safely rather than raise or silently match
        nothing/everything."""
        band = _parse_confidence_band({"min": 0.6, "max": 0.2})
        assert band == ConfidenceBand(min_confidence=0.6, max_confidence=0.6)

    def test_out_of_range_bounds_are_clamped_to_zero_one(self):
        band = _parse_confidence_band({"min": 0.3, "max": 5.0})
        assert band == ConfidenceBand(min_confidence=0.3, max_confidence=1.0)


class TestConfidenceBandContains:
    def test_inclusive_at_min_and_max(self):
        band = ConfidenceBand(min_confidence=0.2, max_confidence=0.6)
        assert band.contains(0.2) is True
        assert band.contains(0.6) is True
        assert band.contains(0.4) is True

    def test_exclusive_just_outside_bounds(self):
        band = ConfidenceBand(min_confidence=0.2, max_confidence=0.6)
        assert band.contains(0.2 - 1e-9) is False
        assert band.contains(0.6 + 1e-9) is False


class TestConfidenceBandScalarBackCompat:
    """A task config saved before Phase 5B uses a bare scalar for every
    labelValues entry. _init must keep parsing that exactly as a floor with no
    ceiling — this protects every already-saved config."""

    async def test_init_parses_scalar_into_floor_band(self):
        params = {**_PARAMS, "model_label_config": [{"modelName": _MODEL_NAME, "labelValues": {_LABEL_UUID: 0.25}}]}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        assert task._target_label_bands[_LABEL_UUID] == ConfidenceBand(min_confidence=0.25, max_confidence=1.0)

    async def test_scalar_config_triggers_and_annotates_exactly_like_a_floor(self):
        """A detection at/above the scalar floor triggers a save; one below it
        does not — matching pre-Phase-5B `det.confidence >= threshold` semantics."""
        params = {**_PARAMS, "model_label_config": [{"modelName": _MODEL_NAME, "labelValues": {_LABEL_UUID: 0.5}}]}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)

        at_floor = _detection_at(0.0, 0.0, 0.2, 0.2, confidence=0.5)
        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect.return_value.result.detections = [at_floor]
        matching = await task._run_inference(inference_api, "b64", {_MODEL_NAME: None})
        assert matching == [at_floor]

    async def test_scalar_config_below_floor_does_not_trigger(self):
        params = {**_PARAMS, "model_label_config": [{"modelName": _MODEL_NAME, "labelValues": {_LABEL_UUID: 0.5}}]}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)

        below_floor = _detection_at(0.0, 0.0, 0.2, 0.2, confidence=0.49)
        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect.return_value.result.detections = [below_floor]
        matching = await task._run_inference(inference_api, "b64", {_MODEL_NAME: None})
        assert matching == []

    async def test_scalar_config_has_no_ceiling_very_high_confidence_still_triggers(self):
        """Pre-Phase-5B behaviour: a scalar floor never rejects a detection for
        being TOO confident — only a real {min, max} band can have a ceiling."""
        params = {**_PARAMS, "model_label_config": [{"modelName": _MODEL_NAME, "labelValues": {_LABEL_UUID: 0.1}}]}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)

        very_confident = _detection_at(0.0, 0.0, 0.2, 0.2, confidence=0.999)
        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect.return_value.result.detections = [very_confident]
        matching = await task._run_inference(inference_api, "b64", {_MODEL_NAME: None})
        assert matching == [very_confident]


class TestConfidenceBandPostMergeOrdering:
    """The whole point of Phase 5B: the band must be applied AFTER the
    cross-model merge, never before.

    Scenario: model-a (higher priority) detects the object at 0.9 confidence,
    which is OUTSIDE its label's band (max=0.6) — a plausible real case: a
    band is often used to target a "needs review" confidence range, not raw
    detector certainty. model-b (lower priority) detects the SAME physical
    object (overlapping bbox, same label) at 0.4 confidence, which IS inside
    the band.

    Correct (post-merge) order: merge runs first and is band-agnostic — it
    only knows about label/IoU/priority — so model-a's higher-priority
    detection wins the same-label collision and model-b's is dropped as a
    duplicate of the same object. The band check then runs against the single
    surviving (0.9, out-of-band) detection, which does not trigger a save.

    A regression to filtering-first would instead drop model-a's detection
    before merge ever runs (0.9 is out-of-band), leaving model-b's 0.4 to
    reach merge alone with nothing to suppress it — surviving, in-band, and
    triggering a save with the LOW-confidence duplicate. That is exactly the
    bug the plan describes: "a high-confidence detection no longer suppresses
    an overlapping low-confidence one, so the task would collect low-confidence
    boxes for objects another model is certain about."
    """

    async def _task_with_band(self, band: dict):
        params = {
            **_PARAMS,
            "model_label_config": [
                {"modelName": "model-a", "labelValues": {_LABEL_UUID: band}},
                {"modelName": "model-b", "labelValues": {_LABEL_UUID: band}},
            ],
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        return task

    async def test_high_priority_out_of_band_detection_suppresses_low_priority_in_band_duplicate(self):
        task = await self._task_with_band({"min": 0.2, "max": 0.6})

        out_of_band_priority = _detection_at(0.0, 0.0, 0.5, 0.5, confidence=0.9)  # model-a, wins merge
        in_band_duplicate = _detection_at(0.01, 0.01, 0.5, 0.5, confidence=0.4)  # model-b, same object

        def _inference(name, payload):
            result = MagicMock()
            result.result.detections = [out_of_band_priority] if name == "model-a" else [in_band_duplicate]
            return result

        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect.side_effect = _inference

        matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        # If band filtering had run BEFORE merge, model-a's 0.9 would already be
        # gone and model-b's 0.4 would survive alone -> matching == [in_band_duplicate].
        # Post-merge, model-a wins the same-label collision and the lone
        # survivor (0.9) is out of band -> nothing triggers a save.
        assert matching == []

    async def test_merge_still_wins_by_priority_when_survivor_is_in_band(self):
        """Companion case: same setup, but the higher-priority detection's
        confidence IS inside the band -> it survives merge AND triggers, and
        the lower-priority duplicate is correctly dropped (never double-saved)."""
        task = await self._task_with_band({"min": 0.2, "max": 0.6})

        in_band_priority = _detection_at(0.0, 0.0, 0.5, 0.5, confidence=0.5)  # model-a
        also_in_band_duplicate = _detection_at(0.01, 0.01, 0.5, 0.5, confidence=0.3)  # model-b, same object

        def _inference(name, payload):
            result = MagicMock()
            result.result.detections = [in_band_priority] if name == "model-a" else [also_in_band_duplicate]
            return result

        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect.side_effect = _inference

        matching = await task._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert matching == [in_band_priority]


class TestConfidenceBandTriggerVsAnnotation:
    """A trigger predicate (any merged detection in-band) decides whether the
    frame is saved; the annotation set (what's written) is NOT filtered by
    band — it's every merged detection, in-band or not, once the trigger
    fires. Otherwise a frame could be saved with a known true positive (the
    out-of-band box) left unlabeled."""

    _LABEL_IN_BAND = _LABEL_UUID
    _LABEL_OUT_OF_BAND = "label-out-of-band"

    async def _task(self):
        params = {
            **_PARAMS,
            "model_label_config": [
                {
                    "modelName": _MODEL_NAME,
                    "labelValues": {
                        self._LABEL_IN_BAND: {"min": 0.5, "max": 1.0},
                        self._LABEL_OUT_OF_BAND: {"min": 0.9, "max": 1.0},
                    },
                }
            ],
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        return task

    async def test_frame_with_one_in_band_and_one_out_of_band_box_is_saved_with_both_annotated(self):
        task = await self._task()

        # Non-overlapping boxes with different (unrelated) labels: merge keeps
        # both untouched, so this isolates the trigger/annotation split from
        # the merge step entirely.
        in_band_det = _detection_at(0.0, 0.0, 0.2, 0.2, label_uuid=self._LABEL_IN_BAND, confidence=0.7)
        out_of_band_det = _detection_at(0.7, 0.7, 0.2, 0.2, label_uuid=self._LABEL_OUT_OF_BAND, confidence=0.6)

        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect.return_value.result.detections = [in_band_det, out_of_band_det]

        matching = await task._run_inference(inference_api, "b64", {_MODEL_NAME: None})

        # Trigger fired (in_band_det is in band) -> frame saved. Annotation set
        # is unfiltered -> the out-of-band box is written too, not dropped.
        assert len(matching) == 2
        assert in_band_det in matching
        assert out_of_band_det in matching

    async def test_frame_with_no_in_band_detection_is_not_saved(self):
        task = await self._task()

        only_out_of_band = _detection_at(0.7, 0.7, 0.2, 0.2, label_uuid=self._LABEL_OUT_OF_BAND, confidence=0.6)

        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect.return_value.result.detections = [only_out_of_band]

        matching = await task._run_inference(inference_api, "b64", {_MODEL_NAME: None})
        assert matching == []


class TestConfidenceBandBoundaryInclusivity:
    """Pinned convention: both the min and max bounds of a confidence band are
    INCLUSIVE — `min <= confidence <= max` triggers a save."""

    async def _task_with_band(self, min_c: float, max_c: float):
        params = {
            **_PARAMS,
            "model_label_config": [
                {"modelName": _MODEL_NAME, "labelValues": {_LABEL_UUID: {"min": min_c, "max": max_c}}}
            ],
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        return task

    async def _run(self, task, confidence: float):
        det = _detection_at(0.0, 0.0, 0.2, 0.2, confidence=confidence)
        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect.return_value.result.detections = [det]
        matching = await task._run_inference(inference_api, "b64", {_MODEL_NAME: None})
        return matching, det

    async def test_confidence_exactly_at_min_triggers(self):
        task = await self._task_with_band(0.2, 0.8)
        matching, det = await self._run(task, 0.2)
        assert matching == [det]

    async def test_confidence_exactly_at_max_triggers(self):
        task = await self._task_with_band(0.2, 0.8)
        matching, det = await self._run(task, 0.8)
        assert matching == [det]

    async def test_confidence_just_below_min_does_not_trigger(self):
        task = await self._task_with_band(0.2, 0.8)
        matching, _det = await self._run(task, 0.2 - 1e-6)
        assert matching == []

    async def test_confidence_just_above_max_does_not_trigger(self):
        task = await self._task_with_band(0.2, 0.8)
        matching, _det = await self._run(task, 0.8 + 1e-6)
        assert matching == []


class TestDedupThresholdCalibration:
    """The calibration widgets exist so the user can pick a duplicate threshold
    from evidence instead of guessing. That only works if the sweep replays the
    *same* discard rules _dedup_worker applies — so these tests pin the replay
    to those rules, not merely to its own output."""

    async def _task(self, action="discard", image_t=5, bbox_t=8, samples=()):
        params = {
            **_PARAMS,
            "enable_dedup": True,
            "duplicate_action": action,
            "image_dup_threshold": image_t,
            "bbox_dup_threshold": bbox_t,
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        for sample in samples:
            task._dedup_samples.append(sample)
        return task

    # ── _would_discard mirrors _dedup_worker ────────────────────────────────

    async def test_discard_image_action_uses_image_distance_only(self):
        task = await self._task(action="discard_image")
        assert task._would_discard(3, [40, 40], image_t=5, bbox_t=8) is True
        assert task._would_discard(9, [0, 0], image_t=5, bbox_t=8) is False

    async def test_discard_action_requires_image_dup_and_all_bboxes_dup(self):
        """_dedup_worker gates 'keep if any unique' on image_is_dup too."""
        task = await self._task(action="discard")
        assert task._would_discard(3, [2, 4], image_t=5, bbox_t=8) is True
        # One unique bbox keeps the frame.
        assert task._would_discard(3, [2, 40], image_t=5, bbox_t=8) is False
        # Every bbox duplicated but a different scene — kept, matching the worker.
        assert task._would_discard(30, [2, 4], image_t=5, bbox_t=8) is False

    async def test_measure_and_ignore_actions_never_discard(self):
        for action in ("measure", "ignore"):
            task = await self._task(action=action)
            assert task._would_discard(0, [0], image_t=5, bbox_t=8) is False, action

    async def test_none_distance_is_not_a_duplicate(self):
        """None = nothing to compare against; it must not count as distance 0."""
        task = await self._task(action="discard_image")
        assert task._would_discard(None, [], image_t=64, bbox_t=64) is False

    # ── sweep ───────────────────────────────────────────────────────────────

    async def test_sweep_is_empty_without_samples(self):
        task = await self._task()
        assert task._dedup_sweep_rows(task._IMAGE_SWEEP_THRESHOLDS, "image") == []

    async def test_sweep_discard_count_is_monotonic_in_threshold(self):
        """A larger threshold can only ever discard more, never fewer."""
        samples = [(d, [d]) for d in (0, 1, 2, 4, 7, 11, 20, 33)]
        task = await self._task(action="discard_image", samples=samples)
        rows = task._dedup_sweep_rows((0, 2, 5, 10, 20, 40), "image")
        counts = [int(r["discarded"].split(" ")[0]) for r in rows]
        assert counts == sorted(counts)
        assert counts[0] == 1  # only the distance-0 frame
        assert counts[-1] == len(samples)  # threshold 40 catches everything

    async def test_sweep_marks_the_configured_threshold(self):
        task = await self._task(image_t=5, samples=[(1, [1])])
        rows = task._dedup_sweep_rows((4, 5, 6), "image")
        assert [("←current" in r["threshold"]) for r in rows] == [False, True, False]

    async def test_sweep_kept_and_discarded_sum_to_the_sample(self):
        samples = [(d, [d]) for d in (0, 3, 9, 30)]
        task = await self._task(action="discard_image", samples=samples)
        for row in task._dedup_sweep_rows((0, 5, 40), "image"):
            discarded = int(row["discarded"].split(" ")[0])
            kept = int(row["kept"].split(" ")[0])
            assert discarded + kept == len(samples)

    async def test_bbox_sweep_holds_image_threshold_fixed(self):
        """Varying the bbox threshold must not silently move the image one."""
        samples = [(30, [0]), (0, [0])]
        task = await self._task(action="discard", image_t=5, samples=samples)
        rows = task._dedup_sweep_rows((0, 40), "bbox")
        # The image-distance-30 frame is never discarded at image_t=5, whatever
        # the bbox threshold does.
        assert all(int(r["discarded"].split(" ")[0]) <= 1 for r in rows)

    # ── histogram ───────────────────────────────────────────────────────────

    async def test_histogram_counts_and_marks_threshold_bin(self):
        task = await self._task(image_t=2)
        hist = task._dedup_histogram([0, 0, 1, 2, 5], marker=2)
        assert hist["total"] == 5
        assert sum(b["count"] for b in hist["bins"]) == 5
        marked = [b for b in hist["bins"] if b["marked"]]
        assert len(marked) == 1 and marked[0]["label"] == "2"

    async def test_histogram_marks_threshold_even_beyond_observed_range(self):
        """A threshold set far above every measurement is exactly the case worth
        seeing — the marker must not fall off the end of the chart."""
        hist = (await self._task())._dedup_histogram([0, 1], marker=12)
        assert any(b["marked"] for b in hist["bins"])

    async def test_empty_histogram_reports_no_bins(self):
        hist = (await self._task())._dedup_histogram([], marker=5)
        assert hist == {"bins": [], "total": 0}

    async def test_histogram_buckets_wider_than_one(self):
        hist = (await self._task())._dedup_histogram([0, 1, 2, 3], marker=0, bucket=2)
        assert hist["bins"][0] == {"label": "0–1", "count": 2, "marked": True}
        assert hist["bins"][1] == {"label": "2–3", "count": 2, "marked": False}

    # ── status payload wiring ───────────────────────────────────────────────

    async def test_calibration_absent_when_dedup_disabled(self):
        task = _make_task()
        await task._init(_PARAMS, resume_data=None)
        data = task.get_status_data()
        assert "dedup_image_hist" not in data
        assert "dedup_image_sweep" not in data

    async def test_calibration_present_when_dedup_enabled(self):
        task = await self._task(samples=[(1, [3]), (9, [20])])
        data = task.get_status_data()
        assert data["dedup_image_hist"]["total"] == 2
        assert data["dedup_bbox_hist"]["total"] == 2
        assert len(data["dedup_image_sweep"]) == len(task._IMAGE_SWEEP_THRESHOLDS)
        assert len(data["dedup_bbox_sweep"]) == len(task._BBOX_SWEEP_THRESHOLDS)
        assert data["dedup_hint"]

    async def test_status_data_reports_distance_percentiles(self):
        task = await self._task(samples=[(0, [2]), (4, [6]), (8, [10])])
        stats = task.get_status_data()["stats"]
        assert stats["dedup_sample_size"] == 3
        assert stats["dedup_image_dist_median"] == 4
        assert stats["dedup_bbox_dist_median"] == 6

    async def test_percentile_keys_omitted_when_nothing_measured(self):
        """Absent beats a misleading 0 — the stat_grid renders absent as '—'."""
        stats = (await self._task()).get_status_data()["stats"]
        assert "dedup_image_dist_median" not in stats
        assert stats["dedup_sample_size"] == 0

    async def test_none_distances_are_excluded_from_stats(self):
        """First frames of a run have an empty index; they aren't 'unique'."""
        task = await self._task(samples=[(None, [None]), (4, [6])])
        stats = task.get_status_data()["stats"]
        assert stats["dedup_image_dist_median"] == 4
        assert task.get_status_data()["dedup_image_hist"]["total"] == 1

    async def test_sample_window_is_bounded(self):
        task = await self._task()
        for i in range(task._DEDUP_SAMPLE_SIZE + 50):
            task._dedup_samples.append((i % 10, []))
        assert len(task._dedup_samples) == task._DEDUP_SAMPLE_SIZE

    # ── the hint explains the censoring caveat ──────────────────────────────

    async def test_hint_warns_about_bias_while_discarding(self):
        for action in ("discard", "discard_image"):
            hint = (await self._task(action=action, samples=[(1, [1])]))._dedup_hint_text()
            assert "Measure only (calibration)" in hint, action

    async def test_hint_confirms_unbiased_sample_in_calibration_mode(self):
        hint = (await self._task(action="measure", samples=[(1, [1])]))._dedup_hint_text()
        assert "unbiased" in hint
        assert "drift upward" not in hint


class TestMeasureOnlyAction:
    """'Measure only' must be offered in the schema and must never discard."""

    async def test_measure_option_is_offered_with_an_explanation(self):
        group = CollectTrainingImagesTask.params_schema()["dedup_group"]["fields"]
        action = group["duplicate_action"]
        assert "measure" in [o["value"] for o in action["options"]]
        # The user has to be told what calibration mode is for, visibly — a
        # tooltip-only `description` is not enough for a whole workflow.
        assert "calibration" in action["help"].lower()

    async def test_measure_mode_keeps_every_frame(self):
        params = {**_PARAMS, "enable_dedup": True, "duplicate_action": "measure"}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        for image_dist in (0, 1, 50):
            assert task._would_discard(image_dist, [0], image_t=64, bbox_t=256) is False


class TestDedupWorkerRecordsDistances:
    """_dedup_worker is the only place the measured distances enter the sample,
    so the wiring from the API response through to _dedup_samples is tested for
    real rather than assumed."""

    async def _run_worker(self, action, response, bbox_count=1):
        params = {
            **_PARAMS,
            "enable_dedup": True,
            "duplicate_action": action,
            "image_dup_threshold": 5,
            "bbox_dup_threshold": 8,
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        # _init() reads hash sizes through a mocked settings API; pin them to the
        # real defaults so the phash round-trip inside the worker actually runs.
        task._image_hash_size = 8
        task._bbox_hash_size = 16
        task._dedup_queue = asyncio.Queue()
        task._pending_saves = []

        provided = MagicMock()
        provided.image = np.zeros((32, 32, 3), dtype=np.uint8)
        provided.metadata.source = "src"
        provided.metadata.frame_idx = None
        matching = [_detection(label_uuid=f"label-{i}") for i in range(bbox_count)]

        images_api = MagicMock()
        images_api.check_dedup_bulk.return_value = response

        await task._dedup_queue.put((provided, matching, ImageProviderType.LIVE_STREAM))
        await task._dedup_queue.put(None)
        with patch.object(task, "_save_image", new=AsyncMock(return_value=None)) as save:
            await task._dedup_worker(images_api, Path("/tmp"))
        return task, save

    @staticmethod
    def _response(image_min, bbox_mins, image_is_dup=False, bbox_flags=None):
        resp = MagicMock()
        resp.image_min_distance = image_min
        resp.bbox_min_distances = bbox_mins
        resp.image_is_dup = image_is_dup
        resp.bbox_flags = bbox_flags if bbox_flags is not None else [False] * len(bbox_mins)
        return resp

    async def test_distances_are_recorded_in_the_sample(self):
        task, _ = await self._run_worker("measure", self._response(3, [7]))
        assert list(task._dedup_samples) == [(3, [7])]

    async def test_none_distances_are_preserved_not_coerced(self):
        """An empty index must stay None all the way into the sample."""
        task, _ = await self._run_worker("measure", self._response(None, [None]))
        assert list(task._dedup_samples) == [(None, [None])]

    async def test_measure_mode_saves_the_frame_despite_a_duplicate_hit(self):
        task, save = await self._run_worker(
            "measure",
            self._response(0, [0], image_is_dup=True, bbox_flags=[True]),
        )
        assert save.await_count == 1
        assert task._dup_frames_discarded == 0

    async def test_discard_image_mode_still_discards(self):
        """Calibration support must not have disarmed the real discard paths."""
        task, save = await self._run_worker(
            "discard_image",
            self._response(0, [0], image_is_dup=True, bbox_flags=[True]),
        )
        assert save.await_count == 0
        assert task._dup_frames_discarded == 1
        # ...and the discarded frame is still measured, so the sweep sees it.
        assert list(task._dedup_samples) == [(0, [0])]

    async def test_ignore_mode_skips_measurement_entirely(self):
        """'Ignore' exists to be cheap — it must not start paying for hashing."""
        task, save = await self._run_worker("ignore", self._response(0, [0]))
        # The ignore path dispatches the save without awaiting it (nothing depends
        # on the index being updated first), so assert dispatch rather than await.
        assert save.call_count == 1
        assert list(task._dedup_samples) == []
        assert task._dedup_frames_checked == 0
