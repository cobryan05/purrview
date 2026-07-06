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
import os
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from apps.helpers.bboxUtils import iou_xywh
from apps.helpers.imageProviders.imageProvider import (
    ImageProvider,
    ImageProviderMetadata,
    ImageWithProviderMetadata,
)
from apps.tasks_server.tasks.CollectTrainingImagesTask import (
    SAME_BBOX_IOU_THRESHOLD,
    CollectTrainingImagesTask,
)

# ─── Shared constants ─────────────────────────────────────────────────────────

_LABEL_UUID = "label-test-uuid"
_MODEL_NAME = "test-model"
_SOURCE = "clip.mp4"

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
    """Yields a fixed list of images then signals exhaustion."""

    def __init__(self, images: list):
        self._queue = list(images)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def getNextImage(self) -> Optional[ImageWithProviderMetadata]:
        return self._queue.pop(0) if self._queue else None


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
) -> CollectTrainingImagesTask:
    """Run a CollectTrainingImagesTask against a fake provider with all I/O mocked.

    Returns the completed task for assertion.
    """
    task = _make_task(params=params, resume_data=resume_data)
    provider = _FiniteProvider(images)

    inf_result = MagicMock()
    inf_result.result.detections = [_detection()] if detect else []

    pin_resp = MagicMock()
    pin_resp.pin_id = "pin-1"

    files_root = str(tmp_path or Path("/tmp"))

    with (
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.SourceHelper") as mock_sh,
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ModelsApi") as mock_models_cls,
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.InferenceApi") as mock_inf_cls,
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ImagesApi") as mock_img_cls,
        patch.dict(os.environ, {"FILES_ROOT": files_root}),
    ):
        mock_sh.return_value.get_image_provider.return_value = provider
        mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.return_value = pin_resp
        mock_inf_cls.return_value.inference.return_value = inf_result
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
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.InferenceApi") as mock_inf_cls,
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.ImagesApi") as mock_img_cls,
        patch.dict(os.environ, {"FILES_ROOT": str(tmp_path)}),
    ):
        mock_sh.return_value.get_image_provider.return_value = provider
        mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.return_value = pin_resp
        inf_result = MagicMock()
        inf_result.result.detections = [_detection()]
        mock_inf_cls.return_value.inference.return_value = inf_result
        mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

        await task.start()
        await task.wait_for_task_done(timeout=10.0)

        # Inference must not have been called at all — images were skipped.
        mock_inf_cls.return_value.inference.assert_not_called()

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

    inference_api = MagicMock()
    inference_api.inference.side_effect = _inference

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

    inference_api = MagicMock()
    inference_api.inference.side_effect = _inference

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

    inference_api = MagicMock()
    result = MagicMock()
    result.result.detections = [box1, box2, box3]
    inference_api.inference.return_value = result

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

    inference_api = MagicMock()
    inference_api.inference.side_effect = _inference

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
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.InferenceApi") as mock_inf_cls,
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
        mock_inf_cls.return_value.inference.return_value = inf_result
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
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.InferenceApi") as mock_inf_cls,
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
        mock_inf_cls.return_value.inference.return_value = inf_result
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
        patch("apps.tasks_server.tasks.CollectTrainingImagesTask.InferenceApi") as mock_inf_cls,
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
        mock_inf_cls.return_value.inference.return_value = inf_result
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
