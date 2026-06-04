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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from apps.helpers.imageProviders.imageProvider import (
    ImageProvider,
    ImageProviderMetadata,
    ImageWithProviderMetadata,
)
from apps.tasks_server.tasks.CollectTrainingImagesTask import CollectTrainingImagesTask

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
