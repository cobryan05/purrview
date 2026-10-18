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
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.error import URLError

import db_client
import numpy as np
import pytest
from inference_client.api.inference_api import InferenceApi

from apps.helpers.actions.collectTrainingImagesAction import _DetectorMissBBox, _DetectorMissDetection
from apps.helpers.bboxUtils import iou_xywh
from apps.helpers.consts import TaskStatus
from apps.helpers.db.db_client import CONFUSABLE_TAG_UUID, DETECTOR_MISS_TAG_UUID, UNVERIFIED_TAG_UUID
from apps.helpers.imageProviders.imageProvider import (
    ImageProvider,
    ImageProviderMetadata,
    ImageProviderType,
    ImageWithProviderMetadata,
)
from apps.helpers.training.collector import (
    SAVE_REASON_BAND,
    SAVE_REASON_DETECTOR_MISS,
    SAVE_REASON_TRACK_WORST,
    CollectorHost,
    FrameOffer,
    TrackedCrop,
    TrainingCollector,
)
from apps.helpers.training.collectorSpec import CollectorSpec, collector_params_schema
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
from tests.waits import await_condition

# ─── Shared constants ─────────────────────────────────────────────────────────


def _detached_collector(params: Optional[dict] = None) -> TrainingCollector:
    """A collector wired to a host that does no I/O.

    Enough for the pure paths — review actions, the calibration sweep, seed
    bboxes — which is exactly what the tasks built with `__new__` exercise. A
    host whose factories return None fails loudly rather than quietly reaching a
    real client if one of those paths ever starts doing I/O.
    """
    return TrainingCollector(
        CollectorSpec.from_params(params or {"output_dir": "out"}),
        CollectorHost(
            log=lambda _message, _level=logging.INFO: None,
            db_api_client=lambda: None,
            inference_api_client=lambda: None,
            stop_event=asyncio.Event(),
        ),
        files_root=Path("/tmp"),
    )


def _detached_task(params: Optional[dict] = None) -> CollectTrainingImagesTask:
    """A task built the way the status-poll paths see it in the PENDING window:
    constructed but never `_init`ed, with only a collector attached."""
    task = CollectTrainingImagesTask.__new__(CollectTrainingImagesTask)
    task._collector = _detached_collector(params)
    return task


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


@pytest.fixture(autouse=True)
def _collector_params_from_the_params_dict(request):
    """`_init` reads the collect knobs off a `training_collectors` row, not the
    params it is handed. These tests are about what the collector does with those
    knobs, not about the DB round trip, so the fetch resolves to the params dict
    under test. `TestCollectorResolution` opts out and covers the round trip."""
    if request.node.get_closest_marker("real_collector_fetch"):
        yield
        return
    with patch.object(
        CollectTrainingImagesTask,
        "_fetch_collector_params",
        new=AsyncMock(side_effect=lambda params: dict(params)),
    ):
        yield


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


def _batch_response(per_model: dict) -> MagicMock:
    """A detect_unit response carrying one result per model.

    The task sends its whole model set in one request now (one image decoded
    once instead of once per model), so the double has to answer in that shape:
    ``results`` keyed by model name, plus ``errors`` for models that failed.
    """
    resp = MagicMock()
    resp.results = per_model
    resp.errors = {}
    return resp


def _result_with(detections) -> MagicMock:
    """A single model's result carrying *detections*."""
    r = MagicMock()
    r.detections = detections
    return r


def _batch_of(result) -> "callable":
    """detect_unit side_effect giving every requested model the same result."""

    def _call(payload):
        return _batch_response({name: result for name in payload.model_names})

    return _call


def _batch_per_model(fn) -> "callable":
    """detect_unit side_effect delegating to ``fn(model_name)`` per model.

    Replaces the old per-model ``detect(name, payload)`` side effects. Order is
    preserved by construction: the task rebuilds its list from its own
    ``_model_names``, so these fakes do not need to model response ordering.
    """

    def _call(payload):
        return _batch_response({name: fn(name) for name in payload.model_names})

    return _call


def _make_task(params=None, resume_data=None) -> CollectTrainingImagesTask:
    manager = MagicMock()
    manager.get_db_api_client.return_value = MagicMock()
    manager.get_inference_api_client.return_value = MagicMock()
    task = CollectTrainingImagesTask("cfg-uuid", params or _PARAMS, manager)
    # A no-I/O collector so tests that poke collector state without running _init
    # have something to poke. _init builds the real one and replaces this. Params
    # the spec rejects (the output_dir traversal cases) fall back to a default —
    # the rejection is _init's job to demonstrate, not this helper's.
    try:
        task._collector = _detached_collector(params or _PARAMS)
    except ValueError:
        task._collector = _detached_collector()
    if resume_data:
        task.set_resume_data(resume_data)
    return task


async def _await_task_done(task: CollectTrainingImagesTask, timeout: float = 60.0) -> None:
    """Wait for a task to finish — and kill it if it doesn't.

    `wait_for_task_done` raises on timeout but leaves the task *running*, and many
    tests here drive `_InfiniteLiveProvider`, a source that never ends unless a stop
    condition fires. A bare wait therefore leaked a live coroutine on every timeout:
    the test failed, and then the xdist worker hung on a loop nobody would ever stop,
    for eleven minutes, until it was finally reported as

        worker 'gwN' crashed while running <whichever test timed out first>

    That is the whole explanation for this file's load-dependent failures naming a
    different, innocent test each time. The multi-source tests patch
    `_SOURCE_RETRY_BACKOFF_S` to 0.0, so the leaked loop is a *hot* one that starves
    the event loop it is waiting on.

    The ceiling is generous for the same reason playwright_tests/conftest.py uses 60s:
    a loaded box should not be mistaken for a broken stop condition. It still fails —
    loudly and quickly enough to diagnose — rather than hanging the worker.
    """
    try:
        await task.wait_for_task_done(timeout=timeout)
    except asyncio.TimeoutError:
        await task.kill()
        raise


async def _run_task(
    images: list,
    params=None,
    resume_data=None,
    detect: bool = True,
    tmp_path: Path = None,
    provider: Optional[ImageProvider] = None,
    providers: Optional[list] = None,
) -> CollectTrainingImagesTask:
    """Run a CollectTrainingImagesTask against a fake provider with all I/O mocked.

    Pass `providers` (and matching source_uuids in `params`) to run against more than
    one source — needed to show that one failing source doesn't take the run with it.

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
        patch("apps.helpers.training.collector.ModelsApi") as mock_models_cls,
        patch("apps.helpers.training.collector.InferenceApi", autospec=True) as mock_inf_cls,
        patch("apps.helpers.training.collector.ImagesApi") as mock_img_cls,
        patch.dict(os.environ, {"FILES_ROOT": files_root}),
    ):
        if providers is not None:
            mock_sh.return_value.get_image_provider.side_effect = list(providers)
        else:
            mock_sh.return_value.get_image_provider.return_value = provider
        mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.return_value = pin_resp
        mock_inf_cls.return_value.detect_unit.side_effect = _batch_of(inf_result.result)
        mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

        await task.start()
        await _await_task_done(task)
        # Stashed so tests can assert on the mocked collaborators (e.g. that inference
        # was never called) after the patch context has exited.
        task._test_mocks = {
            "inference_api": mock_inf_cls.return_value,
            "images_api": mock_img_cls.return_value,
            "models_api": mock_models_cls.return_value,
        }

    return task


# ─── _init() unit tests ───────────────────────────────────────────────────────


async def test_init_empty_processed_ids_for_fresh_run():
    task = _make_task()
    await task._init(_PARAMS, resume_data=None)
    assert task._processed_ids == set()
    assert task._collector._images_saved == 0


async def test_init_loads_processed_ids_from_resume_data():
    ids = [f"{_SOURCE}:{i}" for i in range(5)]
    task = _make_task()
    await task._init(_PARAMS, resume_data={"images_saved": 5, "processed_ids": ids})
    assert task._collector._images_saved == 5
    assert task._processed_ids == set(ids)


async def test_init_empty_resume_data_gives_empty_ids():
    task = _make_task()
    await task._init(_PARAMS, resume_data={})
    assert task._processed_ids == set()
    assert task._collector._images_saved == 0


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
    assert task._spec.output_dir == "collected/run1"


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
        patch("apps.helpers.training.collector.ModelsApi") as mock_models_cls,
        patch("apps.helpers.training.collector.InferenceApi", autospec=True) as mock_inf_cls,
        patch("apps.helpers.training.collector.ImagesApi") as mock_img_cls,
        patch.dict(os.environ, {"FILES_ROOT": str(tmp_path)}),
    ):
        mock_sh.return_value.get_image_provider.return_value = provider
        mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.return_value = pin_resp
        inf_result = MagicMock()
        inf_result.result.detections = [_detection()]
        mock_inf_cls.return_value.detect_unit.side_effect = _batch_of(inf_result.result)
        mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

        await task.start()
        await _await_task_done(task)

        # Inference must not have been called at all — images were skipped.
        mock_inf_cls.return_value.detect_unit.assert_not_called()

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
    assert task._spec.model_names == ["model-a", "model-b"]


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
    inference_api.detect_unit.side_effect = _batch_per_model(lambda n: _inference(n, None).result)

    matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})
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
    inference_api.detect_unit.side_effect = _batch_per_model(lambda n: _inference(n, None).result)

    matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})
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
    inference_api.detect_unit.side_effect = _batch_of(result.result)

    matching = await task._collector._run_inference(inference_api, "b64", {_MODEL_NAME: None})
    assert matching == [box1, box2, box3]


# ─── get_status_actions ───────────────────────────────────────────────────────


def test_get_status_actions_no_tag_uuid():
    """get_status_actions returns [] when no task-tag UUID has been assigned."""
    task = _detached_task()
    task._collector._task_tag_uuid = None
    assert task.get_status_actions() == []


def test_get_status_actions_offers_all_and_unverified():
    """Two ways in, not one.

    The collector auto-annotates with its models and tags what it draws
    UNVERIFIED, so after a long run "what did I collect" and "what still needs
    me" are different questions, and usually very different counts.
    """
    task = _detached_task()
    task._collector._task_tag_uuid = "some-tag-uuid"
    task._collector._spec.collect_all = False

    actions = task.get_status_actions()

    assert [a["label"] for a in actions] == ["Browse all results", "Browse unverified"]
    assert {a["type"] for a in actions} == {"open_review"}
    assert actions[0]["payload"] == {
        "tag_uuids": ["some-tag-uuid"],
        "image_tag_uuids": ["some-tag-uuid"],
        "bbox_verification": "any",
    }
    assert actions[1]["payload"] == {
        "tag_uuids": ["some-tag-uuid"],
        "image_tag_uuids": ["some-tag-uuid"],
        "bbox_verification": "unverified",
    }


def test_browse_all_states_its_filter_rather_than_omitting_it():
    """ "Any" is sent explicitly so it *clears* the unverified filter left behind
    by the button beside it. Omitting it would leave Review still narrowed, and
    "Browse all results" would silently show a subset."""
    task = _detached_task()
    task._collector._task_tag_uuid = "some-tag-uuid"
    task._collector._spec.collect_all = False

    assert task.get_status_actions()[0]["payload"]["bbox_verification"] == "any"


def test_status_and_result_actions_do_not_drift():
    """The live card and the completion result offer the same two entry points,
    from one definition."""
    task = _detached_task()
    task._collector._task_tag_uuid = "some-tag-uuid"
    task._collector._spec.collect_all = False

    assert task.get_status_actions() == task._review_actions()


def test_get_status_actions_in_collect_all_mode_filters_on_the_image_tag_only():
    """Collect-everything runs save unannotated frames, so a bbox-tag filter would
    match nothing and would exclude every frame the button is meant to show."""
    task = _detached_task()
    task._collector._task_tag_uuid = "some-tag-uuid"
    task._collector._spec.collect_all = True

    for action in task.get_status_actions():
        assert action["payload"]["tag_uuids"] == []
        assert action["payload"]["image_tag_uuids"] == ["some-tag-uuid"]


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
    inference_api.detect_unit.side_effect = _batch_per_model(lambda n: _inference(n, None).result)

    matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

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
    task._collector._filter_chain_runner = FilterChainRunner(steps, backend)
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
        result, discard, force_save = await task._collector._apply_filter_chain_to_frame(
            _fc_provided(), "aGVsbG8=", matching
        )
        assert result is matching
        assert discard is False
        assert force_save is False

    async def test_empty_matching_short_circuits_without_evaluating(self):
        task = _make_task()
        await task._init(_PARAMS, resume_data=None)
        task._collector._filter_chain_runner = object()  # would blow up if actually invoked
        result, discard, force_save = await task._collector._apply_filter_chain_to_frame(_fc_provided(), "aGVsbG8=", [])
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
        await task._collector._apply_filter_chain_to_frame(_fc_provided(), _fc_packet(raw_bytes), matching)

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
        await task._collector._apply_filter_chain_to_frame(_fc_provided(), bare_b64, matching)

        assert captured["image_bytes"] == base64.b64decode(bare_b64)

    async def test_discard_image_match_discards_frame(self):
        backend = _FakeFilterChainBackend(kinds={"corrupt": "image_classifier"}, classify_results={"corrupt": 0.9})
        step = _fc_step(model_name="corrupt", action="discard_image", confidence=0.5)
        task = await _task_with_chain_runner([step], backend)

        matching = [_fc_detection()]
        result, discard, force_save = await task._collector._apply_filter_chain_to_frame(
            _fc_provided(), "aGVsbG8=", matching
        )
        assert discard is True
        assert result == []
        assert force_save is False
        assert task._collector._filtered_discarded == 1

    async def test_strip_unverified_bboxes_empties_matching_but_forces_save(self):
        """Per plan 2.1 (never delete images), stripping all detections must still
        force the (now bbox-less) image to be saved — it's not equivalent to discard."""
        backend = _FakeFilterChainBackend(kinds={"blurry": "image_classifier"}, classify_results={"blurry": 0.9})
        step = _fc_step(model_name="blurry", action="strip_unverified_bboxes", confidence=0.5)
        task = await _task_with_chain_runner([step], backend)

        matching = [_fc_detection(), _fc_detection()]
        result, discard, force_save = await task._collector._apply_filter_chain_to_frame(
            _fc_provided(), "aGVsbG8=", matching
        )
        assert discard is False
        assert result == []
        assert force_save is True
        assert task._collector._filtered_stripped == 2

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
        result, discard, force_save = await task._collector._apply_filter_chain_to_frame(provided, "aGVsbG8=", matching)
        assert discard is False
        assert force_save is False
        assert result is matching
        assert all(det.bbox.tag_uuids == [] for det in matching)
        assert provided._filter_chain_image_tags == {"tag-x"}
        assert task._collector._filtered_tagged == 1

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
        result, discard, force_save = await task._collector._apply_filter_chain_to_frame(
            _fc_provided(), "aGVsbG8=", matching
        )
        assert discard is False
        assert force_save is False
        assert det_a.bbox.label_uuid == "water_valve"
        assert det_b.bbox.label_uuid == "dog"  # untouched
        assert task._collector._filtered_relabeled == 1

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
        result, discard, force_save = await task._collector._apply_filter_chain_to_frame(
            _fc_provided(), "aGVsbG8=", matching
        )
        assert discard is False
        assert force_save is False
        assert "tag-y" in det_a.bbox.tag_uuids
        assert "tag-y" not in det_b.bbox.tag_uuids
        assert task._collector._filtered_tagged == 1


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
        patch("apps.helpers.training.collector.ModelsApi") as mock_models_cls,
        patch("apps.helpers.training.collector.InferenceApi", autospec=True) as mock_inf_cls,
        patch("apps.helpers.training.collector.ImagesApi") as mock_img_cls,
        patch(
            "apps.helpers.training.collector.FilterChainsApi",
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
        mock_inf_cls.return_value.detect_unit.side_effect = _batch_of(inf_result.result)
        mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

        await task.start()
        await _await_task_done(task)

    assert task._resume_data["images_saved"] == 0
    assert task._collector._filtered_discarded == 2
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
        patch("apps.helpers.training.collector.ModelsApi") as mock_models_cls,
        patch("apps.helpers.training.collector.InferenceApi", autospec=True) as mock_inf_cls,
        patch("apps.helpers.training.collector.ImagesApi") as mock_img_cls,
        patch(
            "apps.helpers.training.collector.FilterChainsApi",
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
        mock_inf_cls.return_value.detect_unit.side_effect = _batch_of(inf_result.result)
        mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

        await task.start()
        await _await_task_done(task)

    assert task._collector._filtered_stripped == 1
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
        patch("apps.helpers.training.collector.ModelsApi") as mock_models_cls,
        patch("apps.helpers.training.collector.InferenceApi", autospec=True) as mock_inf_cls,
        patch("apps.helpers.training.collector.ImagesApi") as mock_img_cls,
        patch(
            "apps.helpers.training.collector.FilterChainsApi",
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
        mock_inf_cls.return_value.detect_unit.side_effect = _batch_of(inf_result.result)
        mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()
        mock_img_cls.return_value.set_image_tag.return_value = MagicMock()

        await task.start()
        await _await_task_done(task)

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
        with patch("apps.helpers.training.collector.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            await _run_task(images, params=_NO_RATE_LIMIT_PARAMS, detect=True, tmp_path=tmp_path, provider=provider)
        saved = sorted(p.name for p in tmp_path.rglob("*.jpg"))
        # The trailing counter is the within-run uniqueness guarantee; frame_idx
        # beside it is provenance. See _save_image.
        assert saved == [
            f"{_SOURCE_FILE}_20240601T123045_0_1.jpg",
            f"{_SOURCE_FILE}_20240601T123045_1_2.jpg",
        ]

    async def test_two_saves_of_one_frame_do_not_collide(self, tmp_path):
        """A watcher can save the same analysed frame twice — once from the
        periodic offer and again when a detector-miss gap commits the frame it
        was holding. Both carry the same frame_idx, and the UTC stamp is only
        second-resolution, so frame_idx + timestamp alone would put them on one
        path and let one set of boxes overwrite the other."""
        collector = _detached_collector({"output_dir": "out"})
        collector._files_root = tmp_path
        collector._meta_write_lock = asyncio.Lock()
        images_api = MagicMock()
        frame = _frame_offer_with(np.zeros((32, 32, 3), dtype=np.uint8))
        frame.provided.metadata.frame_idx = 7

        first = await collector._save_image(images_api, frame, [])
        second = await collector._save_image(images_api, frame, [])

        assert first != second
        assert len(list(tmp_path.rglob("*.jpg"))) == 2

    async def test_live_provider_filenames_unique_across_simulated_restart(self, tmp_path):
        """Simulate a task restart: frame_idx resets to 0 on the second run, but
        the UTC timestamp differs between the two runs, so no filename collides
        (a plain {source}_{frame_idx}.jpg scheme would silently overwrite here)."""
        first_images = [_tracked_image(0)]
        second_images = [_tracked_image(0)]  # frame_idx reset, as after a restart

        with patch("apps.helpers.training.collector.datetime") as mock_dt:
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
        assert task._collector._images_saved == 1


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
            patch("apps.helpers.training.collector.ModelsApi") as mock_models_cls,
            patch("apps.helpers.training.collector.InferenceApi", autospec=True) as mock_inf_cls,
            patch("apps.helpers.training.collector.ImagesApi") as mock_img_cls,
            patch.dict(os.environ, {"FILES_ROOT": files_root}),
        ):
            mock_sh.return_value.get_image_provider.return_value = provider
            mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.return_value = pin_resp
            mock_inf_cls.return_value.detect_unit.side_effect = _batch_of(inf_result.result)
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
            await _await_task_done(task)


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
            patch("apps.helpers.training.collector.ModelsApi") as mock_models_cls,
            patch("apps.helpers.training.collector.InferenceApi", autospec=True) as mock_inf_cls,
            patch("apps.helpers.training.collector.ImagesApi") as mock_img_cls,
            patch.dict(os.environ, {"FILES_ROOT": str(tmp_path)}),
        ):
            mock_sh.return_value.get_image_provider.return_value = provider
            mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.return_value = pin_resp
            mock_inf_cls.return_value.detect_unit.side_effect = _batch_of(inf_result.result)
            mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

            await task.start()
            # If the stop condition didn't work, the provider is infinite and this
            # would time out — the test fails loudly instead of passing vacuously.
            await _await_task_done(task)

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
            patch("apps.helpers.training.collector.ModelsApi") as mock_models_cls,
            patch("apps.helpers.training.collector.InferenceApi", autospec=True) as mock_inf_cls,
            patch("apps.helpers.training.collector.ImagesApi") as mock_img_cls,
            patch.dict(os.environ, {"FILES_ROOT": str(tmp_path)}),
        ):
            mock_sh.return_value.get_image_provider.return_value = provider
            mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.return_value = pin_resp
            mock_inf_cls.return_value.detect_unit.side_effect = _batch_of(inf_result.result)
            mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

            await task.start()
            await _await_task_done(task)

            assert task.get_status() == TaskStatus.COMPLETED
            assert task._resume_data["elapsed_duration_s"] >= 0.1
            assert provider.stopped is True
            mock_models_cls.return_value.unpin_model.assert_called()

    async def test_stop_conditions_default_to_unlimited(self):
        """A blank/0 max_images or max_duration must not stop the task early —
        the defaults must remain unlimited (live sources are normally user-stopped)."""
        task = _make_task()
        await task._init(_PARAMS, resume_data=None)
        assert task._spec.max_images is None
        assert task._spec.max_duration_s is None

        params = {**_PARAMS, "max_images": 0, "max_duration": 0}
        task2 = _make_task(params=params)
        await task2._init(params, resume_data=None)
        assert task2._spec.max_images is None
        assert task2._spec.max_duration_s is None


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
            patch("apps.helpers.training.collector.ModelsApi") as mock_models_cls,
            patch("apps.helpers.training.collector.InferenceApi", autospec=True) as mock_inf_cls,
            patch("apps.helpers.training.collector.ImagesApi") as mock_img_cls,
            patch.dict(os.environ, {"FILES_ROOT": str(tmp_path)}),
        ):
            mock_sh.return_value.get_image_provider.return_value = provider
            mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.return_value = pin_resp
            mock_inf_cls.return_value.detect_unit.side_effect = _batch_of(inf_result.result)
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
            await _await_task_done(task)

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
            patch("apps.helpers.training.collector.ModelsApi") as mock_models_cls,
            patch("apps.helpers.training.collector.InferenceApi", autospec=True) as mock_inf_cls,
            patch("apps.helpers.training.collector.ImagesApi") as mock_img_cls,
            patch.dict(os.environ, {"FILES_ROOT": str(tmp_path)}),
        ):
            mock_sh.return_value.get_image_provider.return_value = provider
            mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.return_value = pin_resp
            mock_inf_cls.return_value.detect_unit.side_effect = _batch_of(inf_result.result)
            mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

            await task.start()
            # Wait for at least one image to actually save, so the ratio-driven
            # progress value has diverged from the (also non-None) base default —
            # otherwise this assertion would pass vacuously even without the fix.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and task._collector.images_saved == 0:
                await asyncio.sleep(0.05)
            assert task._collector.images_saved > 0, "test setup: no images were saved to measure progress"

            progress = task.get_progress()
            assert progress is not None
            expected = min(100.0, task._collector._images_saved / params["max_images"] * 100.0)
            assert progress == pytest.approx(expected)
            assert 0.0 < progress < 100.0

            task.cancel_task()
            await _await_task_done(task)


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
            patch("apps.helpers.training.collector.ModelsApi") as mock_models_cls,
            patch("apps.helpers.training.collector.InferenceApi", autospec=True) as mock_inf_cls,
            patch("apps.helpers.training.collector.ImagesApi") as mock_img_cls,
            patch.object(TrainingCollector, "PIN_RENEWAL_INTERVAL_S", 0.05),
            patch.dict(os.environ, {"FILES_ROOT": str(tmp_path)}),
        ):
            mock_sh.return_value.get_image_provider.return_value = provider
            mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.side_effect = _pin_resp
            mock_inf_cls.return_value.detect_unit.side_effect = _batch_of(inf_result.result)
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
                # _combined_pin_ids is only populated once the collector has finished
                # pinning, so it may still be empty on the first poll.
                swapped = task._collector._combined_pin_ids.get(_MODEL_NAME) not in (None, "pin-1")
                unpinned = (_MODEL_NAME, "pin-1") in [
                    c.args for c in mock_models_cls.return_value.unpin_model.call_args_list
                ]
                return swapped and unpinned

            async def _await_renewal() -> None:
                while not _renewal_complete():
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(_await_renewal(), timeout=30.0)

            assert task._collector._combined_pin_ids[_MODEL_NAME] != "pin-1", "pin_id must have been swapped by renewal"
            assert mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post.call_count >= 2
            unpin_calls = [c.args for c in mock_models_cls.return_value.unpin_model.call_args_list]
            assert (_MODEL_NAME, "pin-1") in unpin_calls, "the stale pin_id must be released after renewal"

            task.cancel_task()
            await _await_task_done(task)

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
            patch("apps.helpers.training.collector.ModelsApi") as mock_models_cls,
            patch("apps.helpers.training.collector.InferenceApi", autospec=True) as mock_inf_cls,
            patch("apps.helpers.training.collector.ImagesApi") as mock_img_cls,
            patch.object(TrainingCollector, "PIN_RENEWAL_INTERVAL_S", 0.05),
            patch.dict(os.environ, {"FILES_ROOT": str(tmp_path)}),
        ):
            mock_sh.return_value.get_image_provider.return_value = provider
            pin_api = mock_models_cls.return_value.pin_model_api_api_models_model_name_pin_post
            # Initial pin succeeds; every renewal attempt raises.
            pin_api.side_effect = [pin_resp, Exception("boom"), Exception("boom"), Exception("boom")]
            mock_inf_cls.return_value.detect_unit.side_effect = _batch_of(inf_result.result)
            mock_img_cls.return_value.update_image_metadata.return_value = MagicMock()

            await task.start()
            # Wait for renewal to have actually been *attempted* (and raised) rather
            # than for 300ms of wall clock. The sleep made this vacuous in both
            # directions: on a slow box no renewal had been tried yet, so "did not
            # crash" said nothing; on a fast one it was luck that any had.
            await await_condition(
                lambda: pin_api.call_count >= 2,
                lambda: f"no renewal was attempted, so nothing was proved (pin calls: {pin_api.call_count})",
            )

            assert task.get_status() == TaskStatus.RUNNING, "renewal failures must not crash the task"
            assert (
                task._collector._combined_pin_ids[_MODEL_NAME] == "pin-1"
            ), "failed renewal must leave the old pin in place"

            task.cancel_task()
            await _await_task_done(task)
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
# TrainingCollector._run_inference() ends with a merge block (search for
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
    """Characterization (parity) tests pinning TrainingCollector._run_inference's
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
        task._collector._label_ancestry = self._ancestry()
        return task

    @staticmethod
    def _inference_api(per_model: dict[str, list]) -> MagicMock:
        def _inference(name, payload):
            result = MagicMock()
            result.result.detections = per_model.get(name, [])
            return result

        api = MagicMock(spec=InferenceApi)
        api.detect_unit.side_effect = _batch_per_model(lambda n: _inference(n, None).result)
        return api

    # ── Rule 1: same label, two models — priority wins by identity ────────────

    async def test_same_label_higher_priority_model_survives_by_identity(self):
        task = await self._task_with_models("model-a", "model-b")
        high_priority = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.4)
        low_priority = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.99)

        inference_api = self._inference_api({"model-a": [high_priority], "model-b": [low_priority]})
        matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

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
        matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 1
        assert matching[0] is descendant_det

    async def test_descendant_kept_first_ancestor_second_is_dropped(self):
        """kept=more-specific label first; det=coarser label second → det is
        dropped (dominated); the original kept object is untouched."""
        task = await self._task_with_models("model-a", "model-b")
        descendant_det = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.5)
        ancestor_det = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._ROOT, confidence=0.5)

        inference_api = self._inference_api({"model-a": [descendant_det], "model-b": [ancestor_det]})
        matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 1
        assert matching[0] is descendant_det

    # ── Rule 4: unrelated labels — keep both ───────────────────────────────────

    async def test_unrelated_labels_overlapping_both_kept(self):
        task = await self._task_with_models("model-a", "model-b")
        det_a = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.5)
        det_b = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._UNRELATED, confidence=0.5)

        inference_api = self._inference_api({"model-a": [det_a], "model-b": [det_b]})
        matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

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
        matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

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
        matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

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
        matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

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

        task._spec.overlap_iou_threshold = 0.25
        inference_api = self._inference_api({"model-a": [root_det], "model-b": [mid_det], "model-c": [leaf_det]})
        matching = await task._collector._run_inference(
            inference_api, "b64", {"model-a": None, "model-b": None, "model-c": None}
        )

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
        matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 1
        assert matching[0] is leaf_det

    async def test_ancestry_is_transitive_grandparent_det_dropped_by_grandchild_kept(self):
        """Same pair, reversed priority: LEAF kept first, ROOT det second → ROOT
        det is dominated (dropped) since LEAF (already kept) is more specific."""
        task = await self._task_with_models("model-a", "model-b")
        leaf_det = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.5)
        root_det = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._ROOT, confidence=0.5)

        inference_api = self._inference_api({"model-a": [leaf_det], "model-b": [root_det]})
        matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

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
        assert task._spec.model_names == ["model-a", "model-b"]

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
        inference_api.detect_unit.side_effect = _batch_per_model(lambda n: _inference(n, None).result)

        matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

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
        task._collector._label_ancestry = self._ancestry()
        return task

    @staticmethod
    def _inference_api(per_model: dict[str, list]) -> MagicMock:
        return TestInferenceMergeParity._inference_api(per_model)

    async def test_default_policy_is_first_wins(self):
        """Sanity check: omitting merge_policy from params keeps today's default."""
        task = _make_task(params=_PARAMS)
        await task._init(_PARAMS, resume_data=None)
        assert task._spec.merge_policy is MergePolicy.FIRST_WINS

    async def test_unknown_policy_value_falls_back_to_first_wins(self):
        """Unvalidated task params (CLAUDE.md: '_init params are unvalidated —
        cast and validate inside _init') must not raise or silently misbehave
        on garbage input; fall back to the safe default."""
        params = {**_PARAMS, "merge_policy": "not-a-real-policy"}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        assert task._spec.merge_policy is MergePolicy.FIRST_WINS

    async def test_same_label_lower_priority_model_survives_under_last_wins(self):
        """Mirror of TestInferenceMergeParity.test_same_label_higher_priority_model_survives_by_identity:
        under last_wins, model-b (configured second / lower priority) wins the
        same-label collision instead of model-a."""
        task = await self._task_with_models("model-a", "model-b")
        high_priority = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.4)
        low_priority = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.99)

        inference_api = self._inference_api({"model-a": [high_priority], "model-b": [low_priority]})
        matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 1
        assert matching[0] is low_priority

    async def test_ancestor_supersede_preserved_regardless_of_processing_order(self):
        """Hierarchy supersede (more-specific label beats a coarser ancestor) must
        hold under last_wins too, for both orderings of which arrives 'first'."""
        task = await self._task_with_models("model-a", "model-b")
        ancestor_det = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._ROOT, confidence=0.5)
        descendant_det = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.5)

        inference_api = self._inference_api({"model-a": [ancestor_det], "model-b": [descendant_det]})
        matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 1
        assert matching[0] is descendant_det

    async def test_ancestor_supersede_preserved_reverse_model_order(self):
        task = await self._task_with_models("model-a", "model-b")
        descendant_det = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.5)
        ancestor_det = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._ROOT, confidence=0.5)

        inference_api = self._inference_api({"model-a": [descendant_det], "model-b": [ancestor_det]})
        matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 1
        assert matching[0] is descendant_det

    async def test_unrelated_labels_overlapping_both_kept_under_last_wins(self):
        task = await self._task_with_models("model-a", "model-b")
        det_a = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.5)
        det_b = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._UNRELATED, confidence=0.5)

        inference_api = self._inference_api({"model-a": [det_a], "model-b": [det_b]})
        matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 2
        assert det_a in matching and det_b in matching

    async def test_gather_result_order_still_follows_config_priority_not_completion_time(self):
        """Even under last_wins, results_per_model must line up with
        self._model_names (asyncio.gather preserves input order) before the
        policy's own reversal is applied — otherwise the "last model" the
        policy picks wouldn't be the actually-lowest-priority-configured one."""
        task = await self._task_with_models("model-a", "model-b")
        assert task._spec.model_names == ["model-a", "model-b"]

        high_priority = _detection_at(0.0, 0.0, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.1)
        low_priority = _detection_at(0.01, 0.01, 0.5, 0.5, label_uuid=self._LEAF, confidence=0.99)

        def _inference(name, payload):
            if name == "model-b":
                time.sleep(0.08)
            result = MagicMock()
            result.result.detections = [high_priority] if name == "model-a" else [low_priority]
            return result

        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect_unit.side_effect = _batch_per_model(lambda n: _inference(n, None).result)

        matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert len(matching) == 1
        assert matching[0] is low_priority


class TestBuildLabelAncestryMap:
    """Characterization tests for build_label_ancestry_map (tasks/utils/labelHelper.py),
    the map TrainingCollector._run_inference consults (via self._label_ancestry)
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


def _leaf_widgets(widgets):
    """Flatten `group` widgets away.

    A group is layout, not a data view: it holds no value of its own and never
    appears in get_status_data(). Every id <-> data invariant below is therefore
    about the leaves, at whatever depth the schema chooses to nest them."""
    leaves = []
    for widget in widgets:
        if widget.get("type") == "group":
            leaves.extend(_leaf_widgets(widget.get("widgets", [])))
        else:
            leaves.append(widget)
    return leaves


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
        schema_ids = {w["id"] for w in _leaf_widgets(CollectTrainingImagesTask.status_schema())}
        assert schema_ids == self._ALWAYS | self._DEDUP_ONLY

    def test_dedup_calibration_is_a_collapsed_nested_group(self):
        """Four charts and tables is the right detail while reading a threshold
        off them and far too much to scroll past otherwise, so the block is one
        collapsed group with the image and bbox halves nested inside it."""
        top = {w["id"]: w for w in CollectTrainingImagesTask.status_schema()}
        group = top["dedup_calibration"]
        assert group["type"] == "group"
        assert group["collapsed"] is True
        # A group carries no data, so it must not claim a source.
        assert "source" not in group

        inner = {w["id"]: w for w in group["widgets"] if w.get("type") == "group"}
        assert set(inner) == {"dedup_hint_group", "dedup_image_group", "dedup_bbox_group"}
        assert {w["id"] for w in inner["dedup_image_group"]["widgets"]} == {
            "dedup_image_hist",
            "dedup_image_sweep",
        }
        assert {w["id"] for w in inner["dedup_bbox_group"]["widgets"]} == {
            "dedup_bbox_hist",
            "dedup_bbox_sweep",
        }
        # The inner distance groups open with their parent — a second click to
        # see anything would defeat the point of opening the outer group. The
        # explanatory hint is the exception: it's the same paragraph every
        # time, so it starts collapsed even once the calibration block itself
        # is open.
        assert inner["dedup_image_group"].get("collapsed", False) is False
        assert inner["dedup_bbox_group"].get("collapsed", False) is False
        assert inner["dedup_hint_group"]["collapsed"] is True
        assert {w["id"] for w in inner["dedup_hint_group"]["widgets"]} == {"dedup_hint"}

    def test_unconditional_keys_present_before_init(self):
        task = _make_task()
        # get_status_data() must be callable straight after construction (before
        # _init even runs) without raising — it's read by a poll handler that has
        # no idea whether the task has started yet.
        data_keys = set(task.get_status_data().keys())
        assert data_keys == self._ALWAYS

    async def test_every_emitted_key_is_declared_in_the_schema(self):
        """Both with dedup off and on — no key may appear that no widget renders."""
        schema_ids = {w["id"] for w in _leaf_widgets(CollectTrainingImagesTask.status_schema())}
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
        schema = {w["id"]: w for w in _leaf_widgets(CollectTrainingImagesTask.status_schema())}
        assert schema["last_capture"]["type"] == "annotated_image"
        assert schema["stats"]["type"] == "stat_grid"
        assert schema["sources"]["type"] == "table"
        for widget in schema.values():
            assert widget.get("source", "status") == "status"

    def test_sources_table_columns_include_health(self):
        schema = {w["id"]: w for w in _leaf_widgets(CollectTrainingImagesTask.status_schema())}
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
        assert last_capture is task._collector._last_capture

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
        task._collector._label_meta = {_LABEL_UUID: {"text": "Cat", "color": "#ff0000"}}
        det = _detection()
        seed = task._collector._build_seed_bbox(det, "bbox-uuid-1")
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
        task._collector._label_meta = {}
        det = _detection()
        seed = task._collector._build_seed_bbox(det, "bbox-uuid-2")
        assert seed["label"] == {"uuid": _LABEL_UUID, "text": _LABEL_UUID, "color": None}

    def test_build_seed_bbox_no_label_is_none(self):
        task = _make_task()
        task._collector._label_meta = {}
        det = _detection(label_uuid=None)
        seed = task._collector._build_seed_bbox(det, "bbox-uuid-3")
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
        task._collector._filter_chain_runner = MagicMock()
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
        # An exact base, not a real `perf_counter()` reading. `_format_duration`
        # truncates (`int(seconds)`), so this assertion needs elapsed to be exactly
        # 100.0 — and `(start + 100.0) - start` is only exact while both values sit in
        # the same binade. A real reading in the 60–200s range (a freshly booted CI
        # runner, minutes into a job) crosses one about 25% of the time, loses an ulp,
        # and truncates 99.999… to "1m 39s". A long-uptime dev box reads ~3e6 and can
        # never reproduce it, which is how this reached CI green-locally twice.
        task._run_started_at = 0.0
        task._collector._images_saved = 10

        with patch(
            "apps.tasks_server.tasks.CollectTrainingImagesTask.time.perf_counter",
            return_value=100.0,
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
        task._collector._images_saved = 20
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

    def test_max_zero_means_review_disabled_not_a_single_point(self):
        """Unlike a generic inverted band (max < min), an explicit max of 0 is
        the UI's 'Review up to' = 0 convention: tag the label, but never let
        it trigger a review on its own."""
        band = _parse_confidence_band({"min": 0.6, "max": 0})
        assert band.min_confidence == 0.6
        assert band.review_disabled is True

    def test_max_zero_with_low_min_is_also_review_disabled(self):
        band = _parse_confidence_band({"min": 0.1, "max": 0.0})
        assert band.review_disabled is True


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

    def test_review_disabled_never_contains_anything(self):
        band = ConfidenceBand(min_confidence=0.2, max_confidence=1.0, review_disabled=True)
        assert band.contains(0.2) is False
        assert band.contains(0.5) is False
        assert band.contains(1.0) is False


class TestConfidenceBandScalarBackCompat:
    """A task config saved before Phase 5B uses a bare scalar for every
    labelValues entry. _init must keep parsing that exactly as a floor with no
    ceiling — this protects every already-saved config."""

    async def test_init_parses_scalar_into_floor_band(self):
        params = {**_PARAMS, "model_label_config": [{"modelName": _MODEL_NAME, "labelValues": {_LABEL_UUID: 0.25}}]}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        assert task._spec.target_label_bands[_LABEL_UUID] == ConfidenceBand(min_confidence=0.25, max_confidence=1.0)

    async def test_scalar_config_triggers_and_annotates_exactly_like_a_floor(self):
        """A detection at/above the scalar floor triggers a save; one below it
        does not — matching pre-Phase-5B `det.confidence >= threshold` semantics."""
        params = {**_PARAMS, "model_label_config": [{"modelName": _MODEL_NAME, "labelValues": {_LABEL_UUID: 0.5}}]}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)

        at_floor = _detection_at(0.0, 0.0, 0.2, 0.2, confidence=0.5)
        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect_unit.side_effect = _batch_of(_result_with([at_floor]))
        matching = await task._collector._run_inference(inference_api, "b64", {_MODEL_NAME: None})
        assert matching == [at_floor]

    async def test_scalar_config_below_floor_does_not_trigger(self):
        params = {**_PARAMS, "model_label_config": [{"modelName": _MODEL_NAME, "labelValues": {_LABEL_UUID: 0.5}}]}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)

        below_floor = _detection_at(0.0, 0.0, 0.2, 0.2, confidence=0.49)
        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect_unit.side_effect = _batch_of(_result_with([below_floor]))
        matching = await task._collector._run_inference(inference_api, "b64", {_MODEL_NAME: None})
        assert matching == []

    async def test_scalar_config_has_no_ceiling_very_high_confidence_still_triggers(self):
        """Pre-Phase-5B behaviour: a scalar floor never rejects a detection for
        being TOO confident — only a real {min, max} band can have a ceiling."""
        params = {**_PARAMS, "model_label_config": [{"modelName": _MODEL_NAME, "labelValues": {_LABEL_UUID: 0.1}}]}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)

        very_confident = _detection_at(0.0, 0.0, 0.2, 0.2, confidence=0.999)
        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect_unit.side_effect = _batch_of(_result_with([very_confident]))
        matching = await task._collector._run_inference(inference_api, "b64", {_MODEL_NAME: None})
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
        inference_api.detect_unit.side_effect = _batch_per_model(lambda n: _inference(n, None).result)

        matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

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
        inference_api.detect_unit.side_effect = _batch_per_model(lambda n: _inference(n, None).result)

        matching = await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None})

        assert matching == [in_band_priority]


class TestConfidenceBandTriggerVsAnnotation:
    """A trigger predicate (any merged detection in-band) decides whether the
    frame is saved. The annotation set (what's written) is then every merged
    detection at or above its OWN label's min — the max is review-only and
    never suppresses a box, so a frame is never saved with a known true
    positive left unlabeled.

    "Out of band" therefore splits into two halves that behave differently:
    above max is still annotated, below min is not.
    """

    _LABEL_IN_BAND = _LABEL_UUID
    _LABEL_OUT_OF_BAND = "label-out-of-band"

    async def _task(self):
        params = {
            **_PARAMS,
            "model_label_config": [
                {
                    "modelName": _MODEL_NAME,
                    "labelValues": {
                        self._LABEL_IN_BAND: {"min": 0.5, "max": 0.8},
                        self._LABEL_OUT_OF_BAND: {"min": 0.2, "max": 0.4},
                    },
                }
            ],
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        return task

    async def test_frame_saved_for_one_label_annotates_another_label_above_its_max(self):
        task = await self._task()

        # Non-overlapping boxes with different (unrelated) labels: merge keeps
        # both untouched, so this isolates the trigger/annotation split from
        # the merge step entirely.
        in_band_det = _detection_at(0.0, 0.0, 0.2, 0.2, label_uuid=self._LABEL_IN_BAND, confidence=0.7)
        # Above its label's max (0.4) but well above its min (0.2): too confident
        # to be worth reviewing on its own, still worth boxing on a saved frame.
        above_max_det = _detection_at(0.7, 0.7, 0.2, 0.2, label_uuid=self._LABEL_OUT_OF_BAND, confidence=0.95)

        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect_unit.side_effect = _batch_of(_result_with([in_band_det, above_max_det]))

        matching = await task._collector._run_inference(inference_api, "b64", {_MODEL_NAME: None})

        # Trigger fired (in_band_det is in band) -> frame saved. The max is
        # review-only -> the above-max box is written too, not dropped.
        assert len(matching) == 2
        assert in_band_det in matching
        assert above_max_det in matching

    async def test_detection_below_its_own_label_min_is_not_annotated(self):
        """The network conf_thresh is the loosest min across ALL labels
        (_min_all_conf = 0.2 here), so a 0.3 detection of the min-0.5 label
        reaches the task. It must not be boxed: its own label's floor says 0.5.
        Without a per-label floor it would ride in on the other label's 0.2."""
        task = await self._task()

        trigger_det = _detection_at(0.7, 0.7, 0.2, 0.2, label_uuid=self._LABEL_OUT_OF_BAND, confidence=0.3)
        below_own_min_det = _detection_at(0.0, 0.0, 0.2, 0.2, label_uuid=self._LABEL_IN_BAND, confidence=0.3)

        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect_unit.side_effect = _batch_of(_result_with([trigger_det, below_own_min_det]))

        matching = await task._collector._run_inference(inference_api, "b64", {_MODEL_NAME: None})

        assert matching == [trigger_det]

    async def test_annotation_floor_is_per_label_not_the_global_loosest_min(self):
        """Same frame, same confidence, two labels with different floors: the
        0.3 detection is kept for the min-0.2 label and dropped for the
        min-0.5 one."""
        task = await self._task()

        kept = _detection_at(0.7, 0.7, 0.2, 0.2, label_uuid=self._LABEL_OUT_OF_BAND, confidence=0.3)
        dropped = _detection_at(0.0, 0.0, 0.2, 0.2, label_uuid=self._LABEL_IN_BAND, confidence=0.3)

        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect_unit.side_effect = _batch_of(_result_with([kept, dropped]))

        matching = await task._collector._run_inference(inference_api, "b64", {_MODEL_NAME: None})

        assert kept in matching
        assert dropped not in matching

    async def test_frame_with_no_in_band_detection_is_not_saved(self):
        task = await self._task()

        only_out_of_band = _detection_at(0.7, 0.7, 0.2, 0.2, label_uuid=self._LABEL_OUT_OF_BAND, confidence=0.6)

        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect_unit.side_effect = _batch_of(_result_with([only_out_of_band]))

        matching = await task._collector._run_inference(inference_api, "b64", {_MODEL_NAME: None})
        assert matching == []


class TestConfidenceBandReviewDisabled:
    """'Review up to' = 0 tags a label without letting it trigger a review on
    its own, mirroring 'Tag from' = 0 disabling the label outright."""

    _LABEL_REVIEW_DISABLED = _LABEL_UUID
    _LABEL_NORMAL = "label-normal-band"

    async def test_in_band_detection_for_review_disabled_label_does_not_trigger_alone(self):
        params = {
            **_PARAMS,
            "model_label_config": [
                {
                    "modelName": _MODEL_NAME,
                    "labelValues": {self._LABEL_REVIEW_DISABLED: {"min": 0.3, "max": 0}},
                }
            ],
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)

        det = _detection_at(0.0, 0.0, 0.2, 0.2, label_uuid=self._LABEL_REVIEW_DISABLED, confidence=0.9)
        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect_unit.side_effect = _batch_of(_result_with([det]))

        matching = await task._collector._run_inference(inference_api, "b64", {_MODEL_NAME: None})

        assert matching == []

    async def test_review_disabled_label_is_still_annotated_when_another_label_triggers(self):
        params = {
            **_PARAMS,
            "model_label_config": [
                {
                    "modelName": _MODEL_NAME,
                    "labelValues": {
                        self._LABEL_REVIEW_DISABLED: {"min": 0.3, "max": 0},
                        self._LABEL_NORMAL: {"min": 0.2, "max": 0.8},
                    },
                }
            ],
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)

        review_disabled_det = _detection_at(0.0, 0.0, 0.2, 0.2, label_uuid=self._LABEL_REVIEW_DISABLED, confidence=0.9)
        triggering_det = _detection_at(0.7, 0.7, 0.2, 0.2, label_uuid=self._LABEL_NORMAL, confidence=0.5)

        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect_unit.side_effect = _batch_of(_result_with([review_disabled_det, triggering_det]))

        matching = await task._collector._run_inference(inference_api, "b64", {_MODEL_NAME: None})

        assert review_disabled_det in matching
        assert triggering_det in matching

    async def test_below_min_review_disabled_detection_is_not_annotated(self):
        params = {
            **_PARAMS,
            "model_label_config": [
                {
                    "modelName": _MODEL_NAME,
                    "labelValues": {
                        self._LABEL_REVIEW_DISABLED: {"min": 0.3, "max": 0},
                        self._LABEL_NORMAL: {"min": 0.2, "max": 0.8},
                    },
                }
            ],
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)

        below_min_det = _detection_at(0.0, 0.0, 0.2, 0.2, label_uuid=self._LABEL_REVIEW_DISABLED, confidence=0.1)
        triggering_det = _detection_at(0.7, 0.7, 0.2, 0.2, label_uuid=self._LABEL_NORMAL, confidence=0.5)

        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect_unit.side_effect = _batch_of(_result_with([below_min_det, triggering_det]))

        matching = await task._collector._run_inference(inference_api, "b64", {_MODEL_NAME: None})

        assert matching == [triggering_det]

    async def test_all_labels_review_disabled_never_saves_a_frame(self):
        params = {
            **_PARAMS,
            "model_label_config": [
                {
                    "modelName": _MODEL_NAME,
                    "labelValues": {self._LABEL_REVIEW_DISABLED: {"min": 0.3, "max": 0}},
                }
            ],
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)

        det = _detection_at(0.0, 0.0, 0.2, 0.2, label_uuid=self._LABEL_REVIEW_DISABLED, confidence=1.0)
        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect_unit.side_effect = _batch_of(_result_with([det]))

        matching = await task._collector._run_inference(inference_api, "b64", {_MODEL_NAME: None})

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
        inference_api.detect_unit.side_effect = _batch_of(_result_with([det]))
        matching = await task._collector._run_inference(inference_api, "b64", {_MODEL_NAME: None})
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
            task._collector._dedup_samples.append(sample)
        return task

    # ── _would_discard mirrors _dedup_worker ────────────────────────────────

    async def test_discard_image_action_uses_image_distance_only(self):
        task = await self._task(action="discard_image")
        assert task._collector._would_discard(3, [40, 40], image_t=5, bbox_t=8) is True
        assert task._collector._would_discard(9, [0, 0], image_t=5, bbox_t=8) is False

    async def test_discard_action_requires_image_dup_and_all_bboxes_dup(self):
        """_dedup_worker gates 'keep if any unique' on image_is_dup too."""
        task = await self._task(action="discard")
        assert task._collector._would_discard(3, [2, 4], image_t=5, bbox_t=8) is True
        # One unique bbox keeps the frame.
        assert task._collector._would_discard(3, [2, 40], image_t=5, bbox_t=8) is False
        # Every bbox duplicated but a different scene — kept, matching the worker.
        assert task._collector._would_discard(30, [2, 4], image_t=5, bbox_t=8) is False

    async def test_measure_and_ignore_actions_never_discard(self):
        for action in ("measure", "ignore"):
            task = await self._task(action=action)
            assert task._collector._would_discard(0, [0], image_t=5, bbox_t=8) is False, action

    async def test_none_distance_is_not_a_duplicate(self):
        """None = nothing to compare against; it must not count as distance 0."""
        task = await self._task(action="discard_image")
        assert task._collector._would_discard(None, [], image_t=64, bbox_t=64) is False

    # ── sweep ───────────────────────────────────────────────────────────────

    async def test_sweep_is_empty_without_samples(self):
        task = await self._task()
        assert task._collector._dedup_sweep_rows(task._IMAGE_SWEEP_THRESHOLDS, "image") == []

    async def test_sweep_discard_count_is_monotonic_in_threshold(self):
        """A larger threshold can only ever discard more, never fewer."""
        samples = [(d, [d]) for d in (0, 1, 2, 4, 7, 11, 20, 33)]
        task = await self._task(action="discard_image", samples=samples)
        rows = task._collector._dedup_sweep_rows((0, 2, 5, 10, 20, 40), "image")
        counts = [int(r["discarded"].split(" ")[0]) for r in rows]
        assert counts == sorted(counts)
        assert counts[0] == 1  # only the distance-0 frame
        assert counts[-1] == len(samples)  # threshold 40 catches everything

    async def test_sweep_marks_the_configured_threshold(self):
        task = await self._task(image_t=5, samples=[(1, [1])])
        rows = task._collector._dedup_sweep_rows((4, 5, 6), "image")
        assert [("←current" in r["threshold"]) for r in rows] == [False, True, False]

    async def test_sweep_kept_and_discarded_sum_to_the_sample(self):
        samples = [(d, [d]) for d in (0, 3, 9, 30)]
        task = await self._task(action="discard_image", samples=samples)
        for row in task._collector._dedup_sweep_rows((0, 5, 40), "image"):
            discarded = int(row["discarded"].split(" ")[0])
            kept = int(row["kept"].split(" ")[0])
            assert discarded + kept == len(samples)

    async def test_bbox_sweep_holds_image_threshold_fixed(self):
        """Varying the bbox threshold must not silently move the image one."""
        samples = [(30, [0]), (0, [0])]
        task = await self._task(action="discard", image_t=5, samples=samples)
        rows = task._collector._dedup_sweep_rows((0, 40), "bbox")
        # The image-distance-30 frame is never discarded at image_t=5, whatever
        # the bbox threshold does.
        assert all(int(r["discarded"].split(" ")[0]) <= 1 for r in rows)

    async def test_sweep_ladders_run_past_any_plausible_threshold(self):
        """A sweep that stops at the current setting can't answer "what if I
        raised it?", which is the only question the table exists to answer."""
        assert max(TrainingCollector.BBOX_SWEEP_THRESHOLDS) >= 100
        assert max(TrainingCollector.IMAGE_SWEEP_THRESHOLDS) >= 64

    async def test_sweep_includes_a_configured_threshold_off_the_ladder(self):
        task = await self._task(image_t=7, samples=[(1, [1])])
        rows = task._collector._dedup_sweep_rows((0, 5, 10), "image")
        assert [r["threshold"] for r in rows] == ["0", "5", "7 ←current", "10"]

    async def test_off_ladder_bbox_threshold_is_inserted_in_order(self):
        task = await self._task(bbox_t=33, samples=[(1, [1])])
        rows = task._collector._dedup_sweep_rows((0, 32, 40), "bbox")
        assert [r["threshold"] for r in rows] == ["0", "32", "33 ←current", "40"]

    # ── histogram ───────────────────────────────────────────────────────────

    async def test_histogram_counts_and_marks_threshold_bin(self):
        task = await self._task(image_t=2)
        hist = task._collector._dedup_histogram([0, 0, 1, 2, 5], marker=2)
        assert hist["total"] == 5
        assert sum(b["count"] for b in hist["bins"]) == 5
        marked = [b for b in hist["bins"] if b["marked"]]
        assert len(marked) == 1 and marked[0]["label"] == "2"

    async def test_histogram_marks_threshold_even_beyond_observed_range(self):
        """A threshold set far above every measurement is exactly the case worth
        seeing — the marker must not fall off the end of the chart."""
        hist = (await self._task())._collector._dedup_histogram([0, 1], marker=12)
        assert any(b["marked"] for b in hist["bins"])

    async def test_empty_histogram_reports_no_bins(self):
        hist = (await self._task())._collector._dedup_histogram([], marker=5)
        assert hist == {"bins": [], "total": 0}

    async def test_histogram_buckets_wider_than_one(self):
        hist = (await self._task())._collector._dedup_histogram([0, 1, 2, 3], marker=0, bucket=2)
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
            task._collector._dedup_samples.append((i % 10, []))
        assert len(task._collector._dedup_samples) == task._DEDUP_SAMPLE_SIZE

    # ── the hint explains the censoring caveat ──────────────────────────────

    async def test_hint_warns_about_bias_while_discarding(self):
        for action in ("discard", "discard_image"):
            hint = (await self._task(action=action, samples=[(1, [1])]))._collector._dedup_hint_text()
            assert "Measure only (calibration)" in hint, action

    async def test_hint_confirms_unbiased_sample_in_calibration_mode(self):
        hint = (await self._task(action="measure", samples=[(1, [1])]))._collector._dedup_hint_text()
        assert "unbiased" in hint
        assert "drift upward" not in hint


def _frame_offer_with(image: np.ndarray, tracks=(), dead_track_ids=()) -> FrameOffer:
    provided = MagicMock()
    provided.image = image
    provided.metadata.source = "src"
    provided.metadata.source_uuid = None
    provided.metadata.frame_idx = None
    provided.metadata.additional_info = {}
    # A MagicMock auto-vivifies every attribute, so `getattr(provided,
    # "_det_meta", None)` in `_save_image` never sees the default — it must be
    # set explicitly to the "nothing stashed" value real code produces.
    provided._det_meta = None
    return FrameOffer(
        provided=provided,
        base64="b64",
        provider_type=ImageProviderType.LIVE_STREAM,
        tracks=tuple(tracks),
        dead_track_ids=tuple(dead_track_ids),
    )


def _novelty(collector, frame, dets, *, saved: bool = True) -> list:
    """The prefilter as `offer()` actually drives it: decide, then record the
    poses kept — but only once the frame has really been dispatched or queued.
    Pass `saved=False` for the shed-frame case (a full dedup queue), where
    nothing about this frame may enter the track's pose history.
    """
    kept, pending = collector._apply_track_novelty(frame, dets)
    if saved:
        collector._commit_track_novelty(pending, frame)
    return kept


class TestPerTrackNoveltySampling:
    """Plan §4B: within one track's visit, keep a crop only if it's a
    meaningfully different pose than what's already been kept for that SAME
    track — complementary to (not a replacement for) the global phash dedup,
    which asks a different question (cross-visit duplication).
    """

    _TRACK = TrackedCrop(track_id=5, x=0.1, y=0.1, width=0.3, height=0.3)

    @staticmethod
    def _solid_image(value: int) -> np.ndarray:
        return np.full((64, 64, 3), value, dtype=np.uint8)

    @staticmethod
    def _noisy_image(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

    def test_a_detection_matching_no_live_track_is_left_untouched(self):
        collector = _detached_collector()
        det = _detection_at(0.1, 0.1, 0.3, 0.3)
        kept = _novelty(collector, _frame_offer_with(self._solid_image(0), tracks=[self._TRACK]), [det])
        # No track overlaps this bbox (the fixture track is at a different
        # location), so novelty sampling has nothing to say about it.
        assert kept == [det]

    def test_the_first_crop_offered_for_a_track_is_always_kept(self):
        collector = _detached_collector()
        det = _detection_at(0.1, 0.1, 0.3, 0.3)
        kept = _novelty(collector, _frame_offer_with(self._noisy_image(1), tracks=[self._TRACK]), [det])
        assert kept == [det]
        assert collector._track_novelty_kept == 1
        assert collector._track_novelty_dropped == 0

    def test_a_near_identical_second_pose_is_dropped(self):
        collector = _detached_collector()
        image = self._solid_image(128)
        det_a = _detection_at(0.1, 0.1, 0.3, 0.3)
        det_b = _detection_at(0.1, 0.1, 0.3, 0.3)

        _novelty(collector, _frame_offer_with(image, tracks=[self._TRACK]), [det_a])
        kept = _novelty(collector, _frame_offer_with(image, tracks=[self._TRACK]), [det_b])

        assert kept == []
        assert collector._track_novelty_dropped == 1

    def test_a_visually_distinct_second_pose_is_kept(self):
        """A cat that shifts pose/orientation must not be discarded — only the
        stillness in between should be. Distinct random crops stand in for
        'the animal moved enough to look different'."""
        collector = _detached_collector()
        det_a = _detection_at(0.1, 0.1, 0.3, 0.3)
        det_b = _detection_at(0.1, 0.1, 0.3, 0.3)

        _novelty(collector, _frame_offer_with(self._noisy_image(1), tracks=[self._TRACK]), [det_a])
        kept = _novelty(collector, _frame_offer_with(self._noisy_image(2), tracks=[self._TRACK]), [det_b])

        assert kept == [det_b]
        assert collector._track_novelty_kept == 2

    def test_max_frames_per_track_is_a_safety_valve_not_the_rule(self):
        """The bound exists for a pathologically ever-moving object; it must not
        fire on an ordinary distinct-pose sequence well under it."""
        collector = _detached_collector()
        assert collector.MAX_FRAMES_PER_TRACK > 5
        for seed in range(5):
            det = _detection_at(0.1, 0.1, 0.3, 0.3)
            kept = _novelty(collector, _frame_offer_with(self._noisy_image(seed), tracks=[self._TRACK]), [det])
            assert kept == [det]
        assert collector._track_novelty_kept == 5

    def test_the_cap_is_configurable_per_collector(self):
        """A collector whose scene genuinely produces more distinct poses than the
        default allows must be tunable without a code change — plan §4B keeps the
        cap as a knob precisely because the right value is scene-dependent."""
        collector = _detached_collector({"output_dir": "out", "max_frames_per_track": 3})

        for seed in range(5):
            _novelty(
                collector,
                _frame_offer_with(self._noisy_image(seed), tracks=[self._TRACK]),
                [_detection_at(0.1, 0.1, 0.3, 0.3)],
            )

        # Only the most recent 3 poses are still compared against — the deque is
        # bounded by the configured value, not by MAX_FRAMES_PER_TRACK.
        assert len(collector._track_hashes[self._TRACK.track_id]) == 3

    def test_the_cap_actually_stops_a_track_that_keeps_looking_new(self):
        """The case the knob's help text promises to bound: an animal that never
        stops moving, so every crop is genuinely novel. The pose history alone
        cannot do this — it EVICTS at its maxlen, which frees an early pose to
        read as novel again and lets the track contribute forever.
        """
        collector = _detached_collector({"output_dir": "out", "max_frames_per_track": 3})
        kept_total = 0

        for seed in range(12):
            kept_total += len(
                _novelty(
                    collector,
                    _frame_offer_with(self._noisy_image(seed), tracks=[self._TRACK]),
                    [_detection_at(0.1, 0.1, 0.3, 0.3)],
                )
            )

        assert kept_total == 3
        assert collector._track_novelty_dropped == 9

    async def test_a_shed_frame_contributes_nothing_to_the_pose_history(self):
        """A full dedup queue drops the frame outright (the watcher's
        non-blocking offer). Recording its pose anyway would measure every later
        crop of that track against an image that reached no disk, so a collector
        that is shedding frames would progressively stop collecting novel ones.
        """
        collector = _detached_collector({"output_dir": "out", "enable_dedup": True})
        collector._blocking_offer = False
        collector._dedup_queue = asyncio.Queue(maxsize=1)
        collector._dedup_queue.put_nowait(("already full", [], SAVE_REASON_BAND))
        det = _detection_at(0.1, 0.1, 0.3, 0.3)

        with (
            patch("apps.helpers.training.collector.InferenceApi"),
            patch.object(collector, "_run_inference", new=AsyncMock(return_value=[det])),
        ):
            await collector.offer(_frame_offer_with(self._noisy_image(1), tracks=[self._TRACK]))

        assert collector._frames_dropped_behind == 1
        assert self._TRACK.track_id not in collector._track_hashes
        assert collector._track_novelty_kept == 0

    def test_a_zero_cap_cannot_silently_disable_the_prefilter(self):
        """maxlen=0 would keep no history at all, so every crop would compare as
        novel and per-track sampling would quietly stop happening."""
        collector = _detached_collector({"output_dir": "out", "max_frames_per_track": 0})
        image = self._solid_image(128)

        _novelty(collector, _frame_offer_with(image, tracks=[self._TRACK]), [_detection_at(0.1, 0.1, 0.3, 0.3)])
        kept = _novelty(collector, _frame_offer_with(image, tracks=[self._TRACK]), [_detection_at(0.1, 0.1, 0.3, 0.3)])

        assert kept == []


class TestTrackDeathAndAKeptCropInOneOffer:
    """A track can contribute a kept crop *and* die within the same `offer()`.

    The death cleanup pops the track's per-track state; committing novelty writes it
    back. Run in the wrong order — cleanup first, novelty second — the entries are
    resurrected for a track_id that is already gone, and a track only ever dies once, so
    nothing will ever pop them again. That is an unbounded leak on a long-running
    watcher, and it is silent: every count stays right and every crop still saves.
    """

    _TRACK = TrackedCrop(track_id=11, x=0.1, y=0.1, width=0.3, height=0.3)

    @staticmethod
    def _noisy_image(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

    async def test_state_for_a_track_that_dies_in_the_same_offer_is_not_left_behind(self):
        # Needs a model configured: with none, the spec is `collect_all`, `offer()`
        # skips inference entirely and the novelty prefilter never runs.
        collector = _detached_collector(_PARAMS)
        det = _detection_at(0.1, 0.1, 0.3, 0.3)
        frame = _frame_offer_with(
            self._noisy_image(1),
            tracks=[self._TRACK],
            dead_track_ids=[self._TRACK.track_id],
        )

        with (
            patch("apps.helpers.training.collector.InferenceApi"),
            patch.object(collector, "_run_inference", new=AsyncMock(return_value=[det])),
        ):
            await collector.offer(frame)

        # The crop was kept — this is not the "nothing happened" case.
        assert collector._track_novelty_kept == 1
        # …and none of the track's per-track state outlived its death.
        assert self._TRACK.track_id not in collector._track_hashes
        assert self._TRACK.track_id not in collector._track_kept
        assert self._TRACK.track_id not in collector._track_worst


class TestTrackDeathCommitsTheWorstHeldCrop:
    """Plan §4B: 'always keep the track's lowest-confidence frame ... commit at
    track end.' A crop the novelty filter dropped as a redundant pose must not
    be lost forever if it also happened to be the hardest example of the visit.
    """

    _TRACK = TrackedCrop(track_id=5, x=0.1, y=0.1, width=0.3, height=0.3)

    @staticmethod
    def _noisy_image(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

    async def test_a_dropped_low_confidence_pose_is_saved_on_track_death(self):
        collector = _detached_collector({"output_dir": "out", "enable_dedup": False})
        image = np.full((64, 64, 3), 200, dtype=np.uint8)
        det_high = _detection_at(0.1, 0.1, 0.3, 0.3, confidence=0.9)
        det_low = _detection_at(0.1, 0.1, 0.3, 0.3, confidence=0.2)

        _novelty(collector, _frame_offer_with(image, tracks=[self._TRACK]), [det_high])
        kept = _novelty(collector, _frame_offer_with(image, tracks=[self._TRACK]), [det_low])
        assert kept == []  # near-duplicate pose, correctly dropped

        with patch.object(collector, "_dispatch_save") as dispatch:
            await collector._commit_dead_track_worst((5,))

        dispatch.assert_called_once()
        saved_frame, saved_matching, reason = dispatch.call_args.args
        assert saved_matching == [det_low]
        # Attributed to the track-worst commit, not to the ordinary band path, so
        # the status card can say whether this rule is producing anything.
        assert reason == SAVE_REASON_TRACK_WORST
        assert collector._track_worst_committed == 1
        # The track is fully forgotten — nothing left to double-commit later.
        assert 5 not in collector._track_worst
        assert 5 not in collector._track_hashes

    async def test_the_held_crop_bypasses_dedup_entirely(self):
        """The one save that must not be routed through dedup. This crop was
        held BECAUSE its bbox phash sat within `bbox_dup_threshold` of crops
        already saved for the track, and the dedup worker queries the same index
        at the same threshold — so under duplicate_action="discard" the commit
        would be thrown away nearly every time while `_track_worst_committed`
        went on counting it as a success.
        """
        collector = _detached_collector({"output_dir": "out", "enable_dedup": True})
        image = np.full((64, 64, 3), 200, dtype=np.uint8)
        det_high = _detection_at(0.1, 0.1, 0.3, 0.3, confidence=0.9)
        det_low = _detection_at(0.1, 0.1, 0.3, 0.3, confidence=0.2)

        _novelty(collector, _frame_offer_with(image, tracks=[self._TRACK]), [det_high])
        assert _novelty(collector, _frame_offer_with(image, tracks=[self._TRACK]), [det_low]) == []

        with (
            patch.object(collector, "_dispatch_save") as dispatch,
            patch.object(collector, "_enqueue_for_dedup", new=AsyncMock()) as enqueue,
        ):
            await collector._commit_dead_track_worst((5,))

        enqueue.assert_not_awaited()
        assert dispatch.call_args.args[1] == [det_low]

    async def test_a_naturally_saved_crop_needs_no_extra_commit(self):
        """When the lowest-confidence pose was novel and already went through the
        ordinary save path, track death must not re-save anything."""
        collector = _detached_collector({"output_dir": "out", "enable_dedup": False})
        det = _detection_at(0.1, 0.1, 0.3, 0.3, confidence=0.2)

        kept = _novelty(collector, _frame_offer_with(self._noisy_image(3), tracks=[self._TRACK]), [det])
        assert kept == [det]

        with patch.object(collector, "_dispatch_save") as dispatch:
            await collector._commit_dead_track_worst((5,))

        dispatch.assert_not_called()

    async def test_death_of_an_untouched_track_is_a_no_op(self):
        collector = _detached_collector({"output_dir": "out", "enable_dedup": False})
        with patch.object(collector, "_dispatch_save") as dispatch:
            await collector._commit_dead_track_worst((999,))
        dispatch.assert_not_called()
        assert collector._track_worst_committed == 0


def _detector_miss_det(label_uuid=_LABEL_UUID) -> _DetectorMissDetection:
    return _DetectorMissDetection(
        bbox=_DetectorMissBBox(
            x=0.1, y=0.1, width=0.2, height=0.2, label_uuid=label_uuid, tag_uuids=[DETECTOR_MISS_TAG_UUID]
        )
    )


class TestOfferDetectorMiss:
    """Plan §4A: a tracker-supplied box on a detector-miss frame. `offer_detector_miss`
    is the collector's side of it — the watch action decides WHEN a gap resolves;
    this is what happens once it calls in.
    """

    async def test_dispatches_a_save_directly_when_dedup_is_off(self):
        collector = _detached_collector({"output_dir": "out", "enable_dedup": False})
        frame = _frame_offer_with(np.zeros((32, 32, 3), dtype=np.uint8))
        det = _detector_miss_det()

        with patch.object(collector, "_dispatch_save") as dispatch:
            await collector.offer_detector_miss(frame, det)

        dispatch.assert_called_once_with(frame, [det], SAVE_REASON_DETECTOR_MISS)
        assert collector._detector_miss_offered == 1

    async def test_routes_through_dedup_when_enabled(self):
        collector = _detached_collector({"output_dir": "out", "enable_dedup": True})
        frame = _frame_offer_with(np.zeros((32, 32, 3), dtype=np.uint8))
        det = _detector_miss_det()

        with patch.object(collector, "_enqueue_for_dedup", new=AsyncMock()) as enqueue:
            await collector.offer_detector_miss(frame, det)

        enqueue.assert_awaited_once_with(frame, [det], SAVE_REASON_DETECTOR_MISS)

    async def test_review_disabled_label_is_never_seeded(self):
        """'Review up to = 0' is a statement about the review queue, not about one
        trigger. A tracker gap on such a label must not save a frame either."""
        collector = _detached_collector(
            {
                "output_dir": "out",
                "enable_dedup": False,
                "model_label_config": [
                    {"modelName": _MODEL_NAME, "labelValues": {_LABEL_UUID: {"min": 0.2, "max": 0}}}
                ],
            }
        )
        frame = _frame_offer_with(np.zeros((32, 32, 3), dtype=np.uint8))
        det = _detector_miss_det()

        with patch.object(collector, "_dispatch_save") as dispatch:
            await collector.offer_detector_miss(frame, det)

        dispatch.assert_not_called()
        # Still counted as offered: the driver did observe the gap, and the
        # offered-vs-saved gap is what makes the refusal visible in the status.
        assert collector._detector_miss_offered == 1

    async def test_bypasses_per_track_novelty_and_the_filter_chain(self):
        """Neither reasons about model-derived detections; this box is neither."""
        collector = _detached_collector({"output_dir": "out", "enable_dedup": False})
        frame = _frame_offer_with(np.zeros((32, 32, 3), dtype=np.uint8))
        det = _detector_miss_det()

        with (
            patch.object(collector, "_dispatch_save") as dispatch,
            patch.object(collector, "_apply_track_novelty") as novelty,
        ):
            await collector.offer_detector_miss(frame, det)

        novelty.assert_not_called()
        dispatch.assert_called_once()


class TestDetectorMissBboxMetadata:
    """A detector-miss box has no model score behind it — `_save_image` must
    write neither confidence metadata key (plan §5), unlike every other saved
    box, which always gets `confidence`."""

    async def test_no_confidence_keys_are_written(self, tmp_path):
        task = _detached_task({"output_dir": "out"})
        task._collector._files_root = tmp_path
        task._collector._meta_write_lock = asyncio.Lock()
        images_api = MagicMock()
        frame = _frame_offer_with(np.zeros((32, 32, 3), dtype=np.uint8))
        det = _detector_miss_det()

        await task._collector._save_image(images_api, frame, [det])

        payload = images_api.update_image_metadata.call_args.args[0]
        (box,) = payload.boxes
        assert "confidence" not in box.bbox_meta
        assert "source_confidence" not in box.bbox_meta
        assert DETECTOR_MISS_TAG_UUID in box.tag_uuids

    async def test_a_real_detection_still_gets_a_confidence_key(self, tmp_path):
        """Guards against the None-guard swallowing the ordinary case too."""
        task = _detached_task({"output_dir": "out"})
        task._collector._files_root = tmp_path
        task._collector._meta_write_lock = asyncio.Lock()
        images_api = MagicMock()
        frame = _frame_offer_with(np.zeros((32, 32, 3), dtype=np.uint8))
        det = _detection_at(0.1, 0.1, 0.2, 0.2, confidence=0.77)

        await task._collector._save_image(images_api, frame, [det])

        payload = images_api.update_image_metadata.call_args.args[0]
        (box,) = payload.boxes
        assert box.bbox_meta["confidence"] == "0.77"


@pytest.mark.real_collector_fetch
class TestCollectorResolution:
    """`_init` gets its knobs from the named `training_collectors` row.

    The autouse fixture stubs this out for every other test here, so these are the
    only ones exercising the round trip.
    """

    def _task_with_collector_api(self, params, api):
        task = _make_task(params=params)
        return task, patch.object(db_client, "TrainingCollectorsApi", return_value=api)

    async def test_stored_params_win_over_anything_inline(self):
        """A config that still carries stale inline knobs must not use them —
        that is the two-shapes fork the migration exists to close."""
        api = MagicMock()
        api.get_training_collector.return_value = MagicMock(
            collector=MagicMock(params={"output_dir": "from_collector"}),
        )
        params = {"source_uuids": ["src-uuid"], "collector_uuid": "col-1", "output_dir": "stale_inline"}
        task, patcher = self._task_with_collector_api(params, api)
        with patcher:
            await task._init(params, resume_data=None)
        assert task._spec.output_dir == "from_collector"

    async def test_missing_collector_fails_the_run(self):
        """Falling back to defaults would collect into some other directory with
        no models configured, and look like a working run while doing it."""
        api = MagicMock()
        api.get_training_collector.return_value = MagicMock(collector=None, message="gone")
        params = {"source_uuids": ["src-uuid"], "collector_uuid": "col-missing"}
        task, patcher = self._task_with_collector_api(params, api)
        with patcher, pytest.raises(ValueError, match="col-missing"):
            await task._init(params, resume_data=None)

    async def test_no_collector_selected_fails_the_run(self):
        params = {"source_uuids": ["src-uuid"]}
        task = _make_task(params=params)
        with pytest.raises(ValueError, match="No collector selected"):
            await task._init(params, resume_data=None)


class TestMeasureOnlyAction:
    """'Measure only' must be offered in the schema and must never discard."""

    async def test_measure_option_is_offered_with_an_explanation(self):
        group = collector_params_schema()["dedup_group"]["fields"]
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
            assert task._collector._would_discard(image_dist, [0], image_t=64, bbox_t=256) is False


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
        task._collector._image_hash_size = 8
        task._collector._bbox_hash_size = 16
        task._collector._dedup_queue = asyncio.Queue()
        task._collector._pending_saves = []

        provided = MagicMock()
        provided.image = np.zeros((32, 32, 3), dtype=np.uint8)
        provided.metadata.source = "src"
        provided.metadata.frame_idx = None
        matching = [_detection(label_uuid=f"label-{i}") for i in range(bbox_count)]

        images_api = MagicMock()
        images_api.check_dedup_bulk.return_value = response

        await task._collector._dedup_queue.put(
            (
                FrameOffer(provided=provided, base64="b64", provider_type=ImageProviderType.LIVE_STREAM),
                matching,
                SAVE_REASON_BAND,
            )
        )
        await task._collector._dedup_queue.put(None)
        with patch.object(task._collector, "_save_image", new=AsyncMock(return_value=None)) as save:
            await task._collector._dedup_worker(images_api)
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
        assert list(task._collector._dedup_samples) == [(3, [7])]

    async def test_none_distances_are_preserved_not_coerced(self):
        """An empty index must stay None all the way into the sample."""
        task, _ = await self._run_worker("measure", self._response(None, [None]))
        assert list(task._collector._dedup_samples) == [(None, [None])]

    async def test_measure_mode_saves_the_frame_despite_a_duplicate_hit(self):
        task, save = await self._run_worker(
            "measure",
            self._response(0, [0], image_is_dup=True, bbox_flags=[True]),
        )
        assert save.await_count == 1
        assert task._collector._dup_frames_discarded == 0

    async def test_discard_image_mode_still_discards(self):
        """Calibration support must not have disarmed the real discard paths."""
        task, save = await self._run_worker(
            "discard_image",
            self._response(0, [0], image_is_dup=True, bbox_flags=[True]),
        )
        assert save.await_count == 0
        assert task._collector._dup_frames_discarded == 1
        # ...and the discarded frame is still measured, so the sweep sees it.
        assert list(task._collector._dedup_samples) == [(0, [0])]

    async def test_ignore_mode_skips_measurement_entirely(self):
        """'Ignore' exists to be cheap — it must not start paying for hashing."""
        task, save = await self._run_worker("ignore", self._response(0, [0]))
        # The ignore path dispatches the save without awaiting it (nothing depends
        # on the index being updated first), so assert dispatch rather than await.
        assert save.call_count == 1
        assert list(task._collector._dedup_samples) == []
        assert task._collector._dedup_frames_checked == 0


class TestSweepModelsADiscardAction:
    """The sweep answers "what would threshold N have discarded?" — so it must
    model a *discard* action even when the task is running in calibration mode.

    Reading the configured action directly meant "Measure only" (which never
    discards) pinned the whole "Would discard" column at 0, in precisely the mode
    the sweep exists to serve."""

    async def _task(self, action, samples):
        params = {
            **_PARAMS,
            "enable_dedup": True,
            "duplicate_action": action,
            "image_dup_threshold": 5,
            "bbox_dup_threshold": 8,
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        for s in samples:
            task._collector._dedup_samples.append(s)
        return task

    @staticmethod
    def _discarded(row):
        return int(row["discarded"].split(" ")[0])

    async def test_measure_mode_still_reports_would_be_discards(self):
        """The reported regression: 25 frames match at threshold 0, 0 discarded."""
        samples = [(0, [0])] * 25 + [(30, [30])] * 7
        task = await self._task("measure", samples)
        rows = task._collector._dedup_sweep_rows((0, 40), "image")
        assert self._discarded(rows[0]) == 25, "matching frames must be counted as discards"
        # At image threshold 40 all 32 frames match on the image, but the modelled
        # rule is "keep if any unique": the 7 frames whose bbox distance is 30 sit
        # above the bbox threshold of 8, so they survive. Discards track the rule,
        # not the image match count.
        assert self._discarded(rows[-1]) == 25

    async def test_ignore_mode_also_reports_would_be_discards(self):
        task = await self._task("ignore", [(0, [0])] * 4)
        assert self._discarded(task._collector._dedup_sweep_rows((0,), "image")[0]) == 4

    async def test_discard_image_matched_equals_discarded(self):
        """Under discard_image the user's intuition holds exactly."""
        samples = [(0, [99])] * 5 + [(30, [0])] * 3
        task = await self._task("discard_image", samples)
        for row in task._collector._dedup_sweep_rows((0, 5, 40), "image"):
            assert row["matched"] == str(self._discarded(row))

    async def test_configured_discard_action_is_preserved(self):
        """A task already set to 'discard' must keep that rule, not be coerced."""
        # Image is a duplicate but one bbox is unique → 'discard' keeps the frame,
        # while 'discard_image' would drop it.
        samples = [(0, [0, 99])]
        keep = await self._task("discard", samples)
        drop = await self._task("discard_image", samples)
        assert self._discarded(keep._collector._dedup_sweep_rows((0,), "image")[0]) == 0
        assert self._discarded(drop._collector._dedup_sweep_rows((0,), "image")[0]) == 1

    async def test_hint_names_the_action_being_modelled(self):
        """Under 'discard', matched and discarded legitimately differ — the user
        must be told which rule the table is applying rather than left to guess."""
        task = await self._task("measure", [(0, [0])])
        hint = task._collector._dedup_hint_text()
        assert "keep if any unique" in hint.lower() or "discard" in hint.lower()


class TestHintDirectionIsUnambiguous:
    """The histogram lists distance 0 first and counts upward, so "below" is
    ambiguous in the one sentence that matters: a row further down the list is a
    *bigger* number. The guidance must not lean on that word."""

    async def _hint(self, action="measure"):
        params = {**_PARAMS, "enable_dedup": True, "duplicate_action": action}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        task._collector._dedup_samples.append((1, [1]))
        return task._collector._dedup_hint_text()

    async def test_threshold_advice_avoids_directional_words(self):
        hint = await self._hint()
        advice = hint.split("To choose one,")[1].split(rest := "The sweep applies")[0]
        for word in (" below ", " above ", " down ", " up the "):
            assert word not in advice.lower(), f"ambiguous direction word {word!r} in: {advice}"

    async def test_threshold_advice_names_the_row_to_pick(self):
        """Ambiguity is only removed if it says *which* row concretely."""
        hint = await self._hint()
        assert "largest distance" in hint
        assert "last populated row" in hint

    async def test_hint_covers_the_no_gap_case(self):
        """A continuously-changing scene has no gap; silence there reads as
        'the tool is broken' rather than 'this is a judgement call'."""
        assert "no clear empty stretch" in await self._hint()


# ─── Per-source failure isolation ─────────────────────────────────────────────


class _FailingProvider(ImageProvider):
    """A live source that raises on every fetch, like a camera whose TLS handshake
    times out (urllib raises URLError — NOT an asyncio.TimeoutError, so wait_for
    passes it straight through)."""

    def __init__(self, name: str = "dead-cam", exc: Optional[Exception] = None, fail_times: Optional[int] = None):
        self._ptype = ImageProviderType.LIVE_SNAPSHOT
        self.source_name = name
        self._exc = exc or URLError("_ssl.c:999: The handshake operation timed out")
        self._fail_times = fail_times
        self.calls = 0
        self._idx = itertools.count()

    def provider_type(self) -> ImageProviderType:
        return self._ptype

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def getNextImage(self) -> Optional[ImageWithProviderMetadata]:
        self.calls += 1
        if self._fail_times is not None and self.calls > self._fail_times:
            return ImageWithProviderMetadata(
                image=np.zeros((8, 8, 3), dtype=np.uint8),
                metadata=ImageProviderMetadata(source=self.source_name, frame_idx=next(self._idx)),
            )
        raise self._exc


class TestLongDownSourceDoesNotOverflowTheBackoff:
    """A camera that stays down must not eventually kill the whole run.

    `2 ** (consecutive - 1)` is an arbitrary-precision int, and it is evaluated
    *before* the multiply — so it keeps doubling long after `min()` has pinned the
    backoff itself at `_SOURCE_RETRY_BACKOFF_MAX_S`. At 1024 consecutive failures
    the product no longer fits in a float and `_record_source_failure` raises

        OverflowError: int too large to convert to float

    which escapes the fetch loop, propagates to Task._execute and ends the
    collection — discarding every healthy source's frames. That is precisely the
    outcome TestOneFailingSourceDoesNotKillTheRun exists to prevent; the retry
    logic added to prevent it reintroduces it on a longer timescale.

    With the backoff capped at 300s, 1024 consecutive failures is a little over
    three days of one dead camera on an otherwise healthy multi-day run — well
    inside what this task is built for (the give-up timer only fires when *every*
    source is failing, so one dead camera never ends the run on its own).
    """

    async def test_a_source_down_for_days_keeps_a_finite_backoff(self):
        task = _make_task()
        await task._init(_PARAMS, resume_data=None)

        # Comfortably past 1024, the point where 2**(n-1) leaves float range.
        for _ in range(1200):
            task._record_source_failure("dead-cam", "handshake timed out", 0.0)

        entry = task._source_failures["dead-cam"]
        assert entry["consecutive"] == 1200
        # Still pinned at the ceiling, and still a real number.
        assert entry["retry_at"] == pytest.approx(CollectTrainingImagesTask._SOURCE_RETRY_BACKOFF_MAX_S)


class TestOneFailingSourceDoesNotKillTheRun:
    """A raising provider used to escape the fetch loop entirely (only
    asyncio.TimeoutError was caught), propagate to Task._execute and end the whole
    collection — throwing away every other source's frames. One camera going down
    is a routine event on a multi-hour run; the run must survive it."""

    async def test_run_continues_and_saves_from_the_healthy_source(self, tmp_path):
        good = _InfiniteLiveProvider(source="good-cam")
        good.source_name = "good-cam"
        bad = _FailingProvider(name="bad-cam")
        params = {**_NO_RATE_LIMIT_PARAMS, "source_uuids": ["src-good", "src-bad"], "max_images": 3}

        with patch.object(CollectTrainingImagesTask, "_SOURCE_RETRY_BACKOFF_S", 0.0):
            task = await _run_task([], params=params, detect=True, tmp_path=tmp_path, providers=[good, bad])

        # Stopped on max_images, not on the failure.
        assert task.get_status() == TaskStatus.COMPLETED
        assert task.get_error_message() is None
        # >= because saves are pipelined: the stop condition can be crossed while a
        # frame is already in flight (same reason the max_images test uses >=).
        assert task._collector._images_saved >= 3
        # The bad source was actually exercised, and is remembered as failing.
        assert bad.calls > 0
        assert "bad-cam" in task._source_failures
        assert "handshake operation timed out" in task._source_failures["bad-cam"]["error"]

    async def test_failing_source_is_retried_not_dropped(self, tmp_path):
        """The old timeout path discarded a source for good after _MAX_RETRIES, so a
        camera that blipped was lost for the rest of the run even once it came back."""
        good = _InfiniteLiveProvider(source="good-cam")
        good.source_name = "good-cam"
        # Fails its first burst of attempts, then serves frames normally.
        flaky = _FailingProvider(name="flaky-cam", fail_times=CollectTrainingImagesTask._MAX_RETRIES)
        params = {**_NO_RATE_LIMIT_PARAMS, "source_uuids": ["src-good", "src-flaky"], "max_images": 12}

        with patch.object(CollectTrainingImagesTask, "_SOURCE_RETRY_BACKOFF_S", 0.0):
            task = await _run_task([], params=params, detect=True, tmp_path=tmp_path, providers=[good, flaky])

        # It was tried again after its failures and recovered, clearing its entry.
        assert flaky.calls > CollectTrainingImagesTask._MAX_RETRIES
        assert "flaky-cam" not in task._source_failures

    async def test_a_failing_source_is_reported_as_a_notice_with_its_error(self, tmp_path):
        good = _InfiniteLiveProvider(source="good-cam")
        good.source_name = "good-cam"
        bad = _FailingProvider(name="bad-cam")
        params = {**_NO_RATE_LIMIT_PARAMS, "source_uuids": ["src-good", "src-bad"], "max_images": 2}

        with patch.object(CollectTrainingImagesTask, "_SOURCE_RETRY_BACKOFF_S", 0.0):
            task = await _run_task([], params=params, detect=True, tmp_path=tmp_path, providers=[good, bad])

        # get_notices() only speaks while running/paused, so assert against the
        # state it reads rather than restarting the task.
        task._status = TaskStatus.RUNNING
        notices = task.get_notices()
        failing = [n for n in notices if n["key"] == "failing_source_bad-cam"]
        assert failing, f"expected a notice for the failing source, got {notices}"
        assert failing[0]["level"] == "error"
        assert "handshake operation timed out" in failing[0]["message"]

    async def test_sources_table_shows_the_failure(self, tmp_path):
        good = _InfiniteLiveProvider(source="good-cam")
        good.source_name = "good-cam"
        bad = _FailingProvider(name="bad-cam")
        params = {**_NO_RATE_LIMIT_PARAMS, "source_uuids": ["src-good", "src-bad"], "max_images": 2}

        with patch.object(CollectTrainingImagesTask, "_SOURCE_RETRY_BACKOFF_S", 0.0):
            task = await _run_task([], params=params, detect=True, tmp_path=tmp_path, providers=[good, bad])

        rows = {r["name"]: r for r in task.get_status_data()["sources"]}
        assert rows["bad-cam"]["health"] == "failing"
        assert "handshake operation timed out" in rows["bad-cam"]["last_error"]
        assert rows["good-cam"]["health"] != "failing"
        assert rows["good-cam"]["last_error"] == "—"


class TestAllSourcesFailingStopsTheRun:
    """The other half of the contract: with nothing left that can produce a frame,
    spinning forever would be worse than stopping. It stops with an error that names
    what failed (persisted to task_instances.error_message)."""

    async def test_every_source_failing_past_the_grace_ends_the_task(self, tmp_path):
        bad_a = _FailingProvider(name="cam-a")
        bad_b = _FailingProvider(name="cam-b", exc=OSError("Network is unreachable"))
        params = {**_NO_RATE_LIMIT_PARAMS, "source_uuids": ["src-a", "src-b"]}

        with (
            patch.object(CollectTrainingImagesTask, "_SOURCE_RETRY_BACKOFF_S", 0.0),
            patch.object(CollectTrainingImagesTask, "_ALL_SOURCES_FAILING_GRACE_S", 0.0),
        ):
            task = await _run_task([], params=params, detect=True, tmp_path=tmp_path, providers=[bad_a, bad_b])

        assert task.get_status() == TaskStatus.ERROR
        message = task.get_error_message()
        assert message is not None
        assert "All 2 capture source(s) have been failing" in message
        # Names each source and its own reason, so the card explains itself.
        assert "cam-a" in message and "handshake operation timed out" in message
        assert "cam-b" in message and "Network is unreachable" in message

    async def test_does_not_stop_while_one_source_still_works(self, tmp_path):
        """Same zero grace, but one healthy source — the run must NOT end in error."""
        good = _InfiniteLiveProvider(source="good-cam")
        good.source_name = "good-cam"
        bad = _FailingProvider(name="bad-cam")
        params = {**_NO_RATE_LIMIT_PARAMS, "source_uuids": ["src-good", "src-bad"], "max_images": 3}

        with (
            patch.object(CollectTrainingImagesTask, "_SOURCE_RETRY_BACKOFF_S", 0.0),
            patch.object(CollectTrainingImagesTask, "_ALL_SOURCES_FAILING_GRACE_S", 0.0),
        ):
            task = await _run_task([], params=params, detect=True, tmp_path=tmp_path, providers=[good, bad])

        assert task.get_status() == TaskStatus.COMPLETED
        assert task._collector._images_saved >= 3


# ─── Collect-everything mode (no models configured) ───────────────────────────


_NO_MODEL_PARAMS = {**_PARAMS, "model_label_config": []}


class TestCollectAllWhenNoModelsConfigured:
    """With no models configured there is nothing to match against, so the task
    collects every frame the providers hand over — unannotated, and still subject
    to the filter chain and deduplication."""

    async def test_init_sets_collect_all_when_model_label_config_is_empty(self):
        task = _make_task(params=_NO_MODEL_PARAMS)
        await task._init(_NO_MODEL_PARAMS, resume_data=None)
        assert task._spec.collect_all is True
        assert task._spec.model_names == []

    async def test_init_does_not_set_collect_all_when_a_model_is_configured(self):
        task = _make_task()
        await task._init(_PARAMS, resume_data=None)
        assert task._spec.collect_all is False

    async def test_a_model_with_only_disabled_labels_is_not_collect_all(self):
        """Zero confidence is the UI's "not collecting this label" convention. That's
        an explicitly disabled model, not an unconfigured task — it must keep its
        existing behaviour (collect nothing) rather than silently collecting
        everything."""
        params = {**_PARAMS, "model_label_config": [{"modelName": _MODEL_NAME, "labelValues": {_LABEL_UUID: 0}}]}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        assert task._spec.collect_all is False
        assert task._spec.target_labels == []

    async def test_every_frame_is_saved_without_running_inference(self, tmp_path):
        images = [_tracked_image(0), _tracked_image(1), _tracked_image(2)]
        task = await _run_task(images, params=_NO_MODEL_PARAMS, detect=False, tmp_path=tmp_path)

        assert task.get_status() == TaskStatus.COMPLETED
        assert task._collector._images_saved == 3
        assert len(list(tmp_path.rglob("*.jpg"))) == 3
        task._test_mocks["inference_api"].detect_unit.assert_not_called()

    async def test_saved_frames_carry_no_bboxes(self, tmp_path):
        """Nothing detected anything, so the images land unannotated — the point of
        the mode is the raw frames, not empty boxes."""
        task = await _run_task([_tracked_image(0)], params=_NO_MODEL_PARAMS, detect=False, tmp_path=tmp_path)

        payloads = [c.args[0] for c in task._test_mocks["images_api"].update_image_metadata.call_args_list]
        assert len(payloads) == 1
        assert payloads[0].boxes == []

    async def test_no_models_are_pinned(self, tmp_path):
        task = await _run_task([_tracked_image(0)], params=_NO_MODEL_PARAMS, detect=False, tmp_path=tmp_path)
        task._test_mocks["models_api"].pin_model_api_api_models_model_name_pin_post.assert_not_called()

    async def test_frames_are_still_tracked_for_resume(self, tmp_path):
        images = [_tracked_image(0), _tracked_image(1)]
        task = await _run_task(images, params=_NO_MODEL_PARAMS, detect=False, tmp_path=tmp_path)
        assert set(task._resume_data["processed_ids"]) == {f"{_SOURCE}:0", f"{_SOURCE}:1"}


class TestCollectAllStillRunsTheFilterChain:
    """An empty detection list normally means "frame not saved anyway", so the chain
    is skipped. In collect-everything mode an empty list is the normal case for a
    frame that IS about to be saved, so its image-level steps must still run — they
    are the only filtering such a run has."""

    async def _collect_all_task_with_chain(self, steps, backend):
        from apps.tasks_server.tasks._filter_chain_runner import FilterChainRunner

        task = _make_task(params=_NO_MODEL_PARAMS)
        await task._init(_NO_MODEL_PARAMS, resume_data=None)
        task._collector._filter_chain_runner = FilterChainRunner(steps, backend)
        return task

    async def test_discard_image_still_discards_a_frame_with_no_detections(self):
        backend = _FakeFilterChainBackend(kinds={"corrupt": "image_classifier"}, classify_results={"corrupt": 0.9})
        step = _fc_step(model_name="corrupt", action="discard_image", confidence=0.5)
        task = await self._collect_all_task_with_chain([step], backend)

        result, discard, force_save = await task._collector._apply_filter_chain_to_frame(
            _fc_provided(), _fc_packet(), []
        )
        assert discard is True
        assert result == []
        assert task._collector._filtered_discarded == 1

    async def test_tag_image_still_tags_a_frame_with_no_detections(self):
        backend = _FakeFilterChainBackend(kinds={"night": "image_classifier"}, classify_results={"night": 0.9})
        step = _fc_step(model_name="night", action="tag_image", confidence=0.5, action_params={"tag_uuid": "tag-x"})
        task = await self._collect_all_task_with_chain([step], backend)

        provided = _fc_provided()
        _result, discard, _force_save = await task._collector._apply_filter_chain_to_frame(provided, _fc_packet(), [])
        assert discard is False
        assert provided._filter_chain_image_tags == {"tag-x"}

    async def test_a_discarded_frame_is_not_saved(self, tmp_path):
        """End-to-end counterpart: `matching` is empty either way, so the discard flag
        is the only thing that can stop the save. _run() only consults the chain when
        _filter_chain_runner is set, so _init is wrapped to install one."""
        original_init = CollectTrainingImagesTask._init

        async def _init_with_runner(self, params, resume_data):
            await original_init(self, params, resume_data)
            self._collector._filter_chain_runner = object()  # truthiness is all _run() checks

        async def _discard_everything(self, provided, base64_img, matching):
            return [], True, False

        with (
            patch.object(CollectTrainingImagesTask, "_init", _init_with_runner),
            patch.object(TrainingCollector, "_apply_filter_chain_to_frame", _discard_everything),
        ):
            task = await _run_task(
                [_tracked_image(0), _tracked_image(1)],
                params=_NO_MODEL_PARAMS,
                detect=False,
                tmp_path=tmp_path,
            )

        assert task._collector._images_saved == 0
        assert list(tmp_path.rglob("*.jpg")) == []

    async def test_a_kept_frame_is_still_saved(self, tmp_path):
        """The mirror image: a chain that matches nothing must not stop collection."""
        original_init = CollectTrainingImagesTask._init

        async def _init_with_runner(self, params, resume_data):
            await original_init(self, params, resume_data)
            self._collector._filter_chain_runner = object()

        async def _keep_everything(self, provided, base64_img, matching):
            return matching, False, False

        with (
            patch.object(CollectTrainingImagesTask, "_init", _init_with_runner),
            patch.object(TrainingCollector, "_apply_filter_chain_to_frame", _keep_everything),
        ):
            task = await _run_task(
                [_tracked_image(0), _tracked_image(1)],
                params=_NO_MODEL_PARAMS,
                detect=False,
                tmp_path=tmp_path,
            )

        assert task._collector._images_saved == 2


class TestCollectAllDedup:
    """With no bboxes the worker's bbox checks are vacuous, so dedup degrades to an
    image-level comparison — which is the right rule when there is nothing else to
    compare."""

    async def _run_worker(self, action, response):
        params = {**_NO_MODEL_PARAMS, "enable_dedup": True, "duplicate_action": action}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        task._collector._image_hash_size = 8
        task._collector._bbox_hash_size = 16
        task._collector._dedup_queue = asyncio.Queue()
        task._collector._pending_saves = []

        provided = MagicMock()
        provided.image = np.zeros((32, 32, 3), dtype=np.uint8)
        provided.metadata.source = "src"
        provided.metadata.frame_idx = None

        images_api = MagicMock()
        images_api.check_dedup_bulk.return_value = response

        await task._collector._dedup_queue.put(
            (
                FrameOffer(provided=provided, base64="b64", provider_type=ImageProviderType.LIVE_STREAM),
                [],
                SAVE_REASON_BAND,
            )
        )
        await task._collector._dedup_queue.put(None)
        with patch.object(task._collector, "_save_image", new=AsyncMock(return_value="f.jpg")) as save:
            await task._collector._dedup_worker(images_api)
        return task, save

    @staticmethod
    def _response(image_min, image_is_dup):
        resp = MagicMock()
        resp.image_min_distance = image_min
        resp.bbox_min_distances = []
        resp.image_is_dup = image_is_dup
        resp.bbox_flags = []
        return resp

    async def test_duplicate_frame_is_discarded(self):
        task, save = await self._run_worker("discard", self._response(0, image_is_dup=True))
        save.assert_not_called()
        assert task._collector._dup_frames_discarded == 1

    async def test_novel_frame_is_saved(self):
        task, save = await self._run_worker("discard", self._response(30, image_is_dup=False))
        save.assert_called_once()
        assert task._collector._dup_frames_discarded == 0


# ─── Task-level tags land on the image record, not only on bboxes ─────────────


class TestTaskTagsAreAppliedAtImageLevel:
    """A task tag ("collected by this run") is an image-level fact. It was only ever
    written into each saved bbox's tag_uuids, so a collect-everything run — no models,
    therefore no bboxes — wrote it nowhere at all and its "See results" filter found
    nothing. The tag must reach the image record in both modes, while still staying on
    the bboxes for runs that have them."""

    @staticmethod
    def _image_tags(task) -> list[str]:
        return [c.args[0].tag_uuid for c in task._test_mocks["images_api"].set_image_tag.call_args_list]

    @staticmethod
    def _bbox_tags(task) -> list[str]:
        payloads = [c.args[0] for c in task._test_mocks["images_api"].update_image_metadata.call_args_list]
        return [t for p in payloads for b in p.boxes for t in b.tag_uuids]

    async def test_apply_tags_reach_the_image_record(self, tmp_path):
        params = {**_PARAMS, "apply_tags": ["tag-a", "tag-b"]}
        task = await _run_task([_tracked_image(0)], params=params, detect=True, tmp_path=tmp_path)

        assert task._collector._images_saved == 1
        assert self._image_tags(task) == ["tag-a", "tag-b"]
        assert all(c.args[0].action == "add" for c in task._test_mocks["images_api"].set_image_tag.call_args_list)

    async def test_apply_tags_still_reach_the_bboxes(self, tmp_path):
        """Image-level tagging is additive — the existing per-bbox tags stay put so
        already-collected data keeps matching the bbox-tag filter."""
        params = {**_PARAMS, "apply_tags": ["tag-a"]}
        task = await _run_task([_tracked_image(0)], params=params, detect=True, tmp_path=tmp_path)

        assert "tag-a" in self._bbox_tags(task)

    async def test_apply_tags_reach_the_image_when_there_are_no_bboxes(self, tmp_path):
        """The reported bug: collect-everything mode saves unannotated frames, so a
        bbox-only tag write is a no-op."""
        params = {**_NO_MODEL_PARAMS, "apply_tags": ["tag-a"]}
        task = await _run_task([_tracked_image(0)], params=params, detect=False, tmp_path=tmp_path)

        assert task._collector._images_saved == 1
        assert self._bbox_tags(task) == []
        assert self._image_tags(task) == ["tag-a"]

    async def test_generated_task_tag_reaches_the_image_record(self, tmp_path):
        """The generated task tag takes the same path as apply_tags (both land in
        _extra_tag_uuids), but it is the one the task's own "See results" action
        filters on, so assert it end-to-end."""
        params = {**_NO_MODEL_PARAMS, "generate_task_tag": True}
        tags_api = MagicMock()
        tags_api.list_tags.return_value = MagicMock(tags=[])

        with patch("apps.helpers.training.collector.TagsApi", return_value=tags_api):
            task = await _run_task([_tracked_image(0)], params=params, detect=False, tmp_path=tmp_path)

        assert task._collector._task_tag_uuid is not None
        assert self._image_tags(task) == [task._collector._task_tag_uuid]

    async def test_filter_chain_image_tags_are_kept_alongside_task_tags(self, tmp_path):
        """The filter chain's own tag_image writes share this call site — neither set
        may displace the other."""
        params = {**_NO_MODEL_PARAMS, "apply_tags": ["tag-a"]}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        task._collector._meta_write_lock = asyncio.Lock()  # normally created by _run()

        provided = ImageWithProviderMetadata(
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            metadata=ImageProviderMetadata(source=_SOURCE, frame_idx=0),
        )
        provided._filter_chain_image_tags = {"tag-chain"}
        images_api = MagicMock()

        task._collector._files_root = tmp_path
        await task._collector._save_image(images_api, FrameOffer(provided=provided, base64="b64"), [])

        assert {c.args[0].tag_uuid for c in images_api.set_image_tag.call_args_list} == {"tag-a", "tag-chain"}

    async def test_renaming_the_task_tag_creates_a_new_tag_instead_of_renaming(self, tmp_path):
        """Reported bug: the generated task tag's UUID was derived only from the task
        config UUID, so a name change reused the same DB row and renamed it — every
        image already collected under the old name silently became the new name.
        A new name must mint a NEW tag and leave the old row untouched."""
        from db_client.models.tag_read import TagRead

        params = {
            **_NO_MODEL_PARAMS,
            "generate_task_tag": True,
            "task_tag_name": "second_collect_run",
        }
        old_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, "cfg-uuid"))
        tags_api = MagicMock()
        tags_api.list_tags.return_value = MagicMock(
            tags=[TagRead(uuid=old_uuid, name="first_collect_run", kind="task_generated")]
        )

        with patch("apps.helpers.training.collector.TagsApi", return_value=tags_api):
            task = await _run_task([_tracked_image(0)], params=params, detect=False, tmp_path=tmp_path)

        tags_api.update_tag.assert_not_called()
        assert task._collector._task_tag_uuid != old_uuid
        tags_api.add_tag.assert_called_once()
        created = tags_api.add_tag.call_args.args[0]
        assert created.name == "second_collect_run"
        assert created.uuid == task._collector._task_tag_uuid
        # The new tag — not the old one — is what this run's images get.
        assert self._image_tags(task) == [task._collector._task_tag_uuid]

    async def test_unchanged_task_tag_name_reuses_the_legacy_row(self, tmp_path):
        """Configs that predate the name-keyed UUID must keep their original tag row
        while the configured name is unchanged, so their collected history stays on
        one tag instead of splitting across two."""
        from db_client.models.tag_read import TagRead

        params = {
            **_NO_MODEL_PARAMS,
            "generate_task_tag": True,
            "task_tag_name": "first_collect_run",
        }
        legacy_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, "cfg-uuid"))
        tags_api = MagicMock()
        tags_api.list_tags.return_value = MagicMock(
            tags=[TagRead(uuid=legacy_uuid, name="first_collect_run", kind="task_generated")]
        )

        with patch("apps.helpers.training.collector.TagsApi", return_value=tags_api):
            task = await _run_task([_tracked_image(0)], params=params, detect=False, tmp_path=tmp_path)

        assert task._collector._task_tag_uuid == legacy_uuid
        tags_api.add_tag.assert_not_called()
        tags_api.update_tag.assert_not_called()

    async def test_reverting_the_task_tag_name_returns_to_the_original_tag(self, tmp_path):
        """Name-keyed UUIDs are stable, so switching back to a previously used name
        re-attaches to that name's existing tag rather than making a third one."""
        from db_client.models.tag_read import TagRead

        params = {
            **_NO_MODEL_PARAMS,
            "generate_task_tag": True,
            "task_tag_name": "second_collect_run",
        }
        second_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, "cfg-uuid:second_collect_run"))
        tags_api = MagicMock()
        tags_api.list_tags.return_value = MagicMock(
            tags=[TagRead(uuid=second_uuid, name="second_collect_run", kind="task_generated")]
        )

        with patch("apps.helpers.training.collector.TagsApi", return_value=tags_api):
            task = await _run_task([_tracked_image(0)], params=params, detect=False, tmp_path=tmp_path)

        assert task._collector._task_tag_uuid == second_uuid
        tags_api.add_tag.assert_not_called()
        tags_api.update_tag.assert_not_called()

    async def test_a_tag_in_both_sets_is_written_once(self, tmp_path):
        """A filter chain configured to apply the same tag the task already applies
        must not produce a duplicate write."""
        params = {**_NO_MODEL_PARAMS, "apply_tags": ["tag-a"]}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        task._collector._meta_write_lock = asyncio.Lock()  # normally created by _run()

        provided = ImageWithProviderMetadata(
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            metadata=ImageProviderMetadata(source=_SOURCE, frame_idx=0),
        )
        provided._filter_chain_image_tags = {"tag-a"}
        images_api = MagicMock()

        task._collector._files_root = tmp_path
        await task._collector._save_image(images_api, FrameOffer(provided=provided, base64="b64"), [])

        assert [c.args[0].tag_uuid for c in images_api.set_image_tag.call_args_list] == ["tag-a"]


class TestBboxConfidenceReachesTheWire:
    """The confidence CollectTrainingImagesTask records must survive
    serialization by the **generated** db_client.

    ⚠️ The generated model's field is `bbox_meta`, not `extra` — the server-side
    model declares `AliasChoices("bbox_meta", "extra")` and the OpenAPI generator
    takes the *first* choice as the field name. Pydantic ignores unknown kwargs,
    so passing `extra=` was accepted, silently dropped, and every bbox this task
    wrote landed with `metadata_json = {}`. Nothing raised, so no test that
    stopped at the model object could have caught it — this one asserts on the
    serialized wire payload.
    """

    @staticmethod
    def _box(**overrides):
        from db_client.models.bounding_box_metadata_model import BoundingBoxMetadataModel

        kwargs = {
            "bbox_uuid": "bbox-1",
            "x": 0.1,
            "y": 0.1,
            "width": 0.2,
            "height": 0.2,
            "label_uuid": "lbl-1",
            "tag_uuids": ["tag-1"],
        }
        kwargs.update(overrides)
        return BoundingBoxMetadataModel(**kwargs)

    def test_confidence_is_present_in_the_serialized_payload(self):
        import json

        from db_client.models.update_metadata_payload import UpdateMetadataPayload

        box = self._box(bbox_meta={"confidence": "0.82"})
        payload = UpdateMetadataPayload(image_path="x.jpg", boxes=[box])
        wire = json.loads(payload.to_json())["boxes"][0]

        assert "bbox_meta" in wire, f"confidence never reached the wire: {wire}"
        assert wire["bbox_meta"] == {"confidence": "0.82"}

    def test_server_model_parses_that_payload_back_into_extra(self):
        """Closes the loop: the generated client's `bbox_meta` must land on the
        server model's `extra`, which web.py maps onto metadata_json."""
        import json

        from db_client.models.update_metadata_payload import UpdateMetadataPayload

        from apps.helpers.types import BoundingBoxMetadataModel as ServerModel

        payload = UpdateMetadataPayload(image_path="x.jpg", boxes=[self._box(bbox_meta={"confidence": "0.82"})])
        parsed = ServerModel.model_validate(json.loads(payload.to_json())["boxes"][0])

        assert parsed.extra == {"confidence": "0.82"}

    def test_the_extra_kwarg_is_silently_ignored(self):
        """Documents the trap itself, so nobody "tidies" the call site back to
        `extra=` — it does not error, it just quietly loses the data."""
        box = self._box(extra={"confidence": "0.82"})
        assert box.bbox_meta is None
        assert not hasattr(box, "extra")


# ─── Confusable detections / ambiguity (AMBIGUITY_REVIEW_PLAN.md Phase A) ─────

_RIVAL_UUID = "label-rival-uuid"

_AMBIGUITY_PARAMS = {
    **_PARAMS,
    "enable_ambiguity": True,
    "ambiguity_metric": "ratio",
    "ambiguity_threshold": 0.7,
    "ambiguity_scope": "any",
}


def _alt(label_uuid: str | None, confidence: float, class_str: str = "rival") -> MagicMock:
    alt = MagicMock()
    alt.label_uuid = label_uuid
    alt.confidence = confidence
    alt.class_str = class_str
    return alt


def _detection_with_alts(confidence: float, alternates: list, label_uuid: str = _LABEL_UUID) -> MagicMock:
    """A detection carrying runner-up classes, as the inference server returns
    them when `top_k` is requested."""
    det = _detection_at(0.0, 0.0, 0.2, 0.2, label_uuid=label_uuid, confidence=confidence)
    det.alternates = alternates
    det.bbox.uuid = str(uuid.uuid4())
    return det


class TestAmbiguityDetection:
    """The 'confident but confused' trigger: a detection whose runner-up class
    scored close to the winner, which the confidence band structurally cannot
    see (the winner is high-confidence, so no ceiling catches it)."""

    async def _task(self, **overrides):
        params = {**_AMBIGUITY_PARAMS, **overrides}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        return task

    async def test_close_runner_up_is_flagged(self):
        task = await self._task()
        det = _detection_with_alts(0.91, [_alt(_RIVAL_UUID, 0.88)])
        info = task._collector._detect_ambiguity(det, _MODEL_NAME)
        assert info is not None
        assert info.model == _MODEL_NAME
        assert info.rival_label_uuid == _RIVAL_UUID
        assert info.value == pytest.approx(0.88 / 0.91)

    async def test_distant_runner_up_is_not_flagged(self):
        task = await self._task()
        det = _detection_with_alts(0.91, [_alt(_RIVAL_UUID, 0.10)])
        assert task._collector._detect_ambiguity(det, _MODEL_NAME) is None

    async def test_detection_with_no_alternates_is_never_flagged(self):
        """A model that returned no runners-up (or one that cannot produce them)
        must not be read as 'not confused' by accident — it simply yields None."""
        task = await self._task()
        det = _detection_with_alts(0.91, [])
        assert task._collector._detect_ambiguity(det, _MODEL_NAME) is None

    async def test_disabled_by_default_flags_nothing(self):
        params = {**_PARAMS}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        det = _detection_with_alts(0.91, [_alt(_RIVAL_UUID, 0.90)])
        assert task._collector._detect_ambiguity(det, _MODEL_NAME) is None

    async def test_margin_metric_uses_absolute_gap_and_inverted_direction(self):
        """Margin fires when the gap is at or BELOW the threshold — the opposite
        direction from ratio. Getting this backwards would flag every confident
        detection instead of the confusable ones."""
        task = await self._task(ambiguity_metric="margin", ambiguity_threshold=0.05)
        close = _detection_with_alts(0.91, [_alt(_RIVAL_UUID, 0.88)])  # gap 0.03
        far = _detection_with_alts(0.91, [_alt(_RIVAL_UUID, 0.50)])  # gap 0.41
        assert task._collector._detect_ambiguity(close, _MODEL_NAME) is not None
        assert task._collector._detect_ambiguity(far, _MODEL_NAME) is None

    async def test_review_disabled_label_never_triggers_ambiguity(self):
        """'Review up to = 0' means 'tag it, but never let it pull a frame into
        review'. Ambiguity must not become a back door around that."""
        params = {
            **_AMBIGUITY_PARAMS,
            "model_label_config": [{"modelName": _MODEL_NAME, "labelValues": {_LABEL_UUID: {"min": 0.2, "max": 0}}}],
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        assert task._spec.target_label_bands[_LABEL_UUID].review_disabled is True

        det = _detection_with_alts(0.91, [_alt(_RIVAL_UUID, 0.90)])
        assert task._collector._detect_ambiguity(det, _MODEL_NAME) is None

    async def test_review_disabled_rival_is_not_a_reason_to_review(self):
        """The mirror of the test above. 'Review up to = 0' on a label means it
        never pulls a frame into review — as the winner OR as the runner-up.

        The reported bug: a label set to 0 kept filling the review queue because
        another label's box was flagged confusable *with it*, so the frame was
        saved anyway and arrived carrying the disabled label's box.
        """
        params = {
            **_AMBIGUITY_PARAMS,
            "ambiguity_scope": "any",
            "model_label_config": [
                {
                    "modelName": _MODEL_NAME,
                    "labelValues": {_LABEL_UUID: 0.2, _RIVAL_UUID: {"min": 0.2, "max": 0}},
                }
            ],
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)

        det = _detection_with_alts(0.91, [_alt(_RIVAL_UUID, 0.90)])
        assert task._collector._detect_ambiguity(det, _MODEL_NAME) is None
        # Not merely unflagged — never measured, so it can't skew the calibration
        # stats the status panel shows.
        assert task._collector._ambiguity_evaluated == 0

    async def test_next_alternate_is_still_considered_after_a_disabled_rival(self):
        """Skipping a review-disabled rival must not skip the whole detection:
        the next runner-up is then the strongest eligible one."""
        other = "label-other-uuid"
        params = {
            **_AMBIGUITY_PARAMS,
            "ambiguity_scope": "any",
            "model_label_config": [
                {
                    "modelName": _MODEL_NAME,
                    "labelValues": {_LABEL_UUID: 0.2, _RIVAL_UUID: {"min": 0.2, "max": 0}},
                }
            ],
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)

        det = _detection_with_alts(0.91, [_alt(_RIVAL_UUID, 0.90), _alt(other, 0.89, class_str="other")])
        info = task._collector._detect_ambiguity(det, _MODEL_NAME)
        assert info is not None
        assert info.rival_label_uuid == other

    async def test_min_confidence_floor_skips_weak_detections(self):
        """Weak detections are already collected by the confidence band; the floor
        stops the same frame being flagged twice for two different reasons."""
        task = await self._task(ambiguity_min_confidence=0.6)
        weak = _detection_with_alts(0.4, [_alt(_RIVAL_UUID, 0.39)])
        strong = _detection_with_alts(0.9, [_alt(_RIVAL_UUID, 0.89)])
        assert task._collector._detect_ambiguity(weak, _MODEL_NAME) is None
        assert task._collector._detect_ambiguity(strong, _MODEL_NAME) is not None

    async def test_scope_configured_ignores_rival_outside_target_labels(self):
        task = await self._task(ambiguity_scope="configured")
        # _RIVAL_UUID is not in model_label_config, so under 'configured' it does
        # not count, however close it scored.
        det = _detection_with_alts(0.91, [_alt(_RIVAL_UUID, 0.90)])
        assert task._collector._detect_ambiguity(det, _MODEL_NAME) is None

    async def test_scope_any_counts_rival_outside_target_labels(self):
        task = await self._task(ambiguity_scope="any")
        det = _detection_with_alts(0.91, [_alt(_RIVAL_UUID, 0.90)])
        assert task._collector._detect_ambiguity(det, _MODEL_NAME) is not None

    async def test_scope_configured_accepts_a_configured_rival(self):
        params = {
            **_AMBIGUITY_PARAMS,
            "ambiguity_scope": "configured",
            "model_label_config": [
                {"modelName": _MODEL_NAME, "labelValues": {_LABEL_UUID: 0.5, _RIVAL_UUID: 0.5}},
            ],
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        det = _detection_with_alts(0.91, [_alt(_RIVAL_UUID, 0.90)])
        assert task._collector._detect_ambiguity(det, _MODEL_NAME) is not None

    async def test_zero_confidence_winner_does_not_divide_by_zero(self):
        """A 0.0 winner would make ratio inf/NaN, which compares True against any
        threshold and would flag everything."""
        task = await self._task()
        det = _detection_with_alts(0.0, [_alt(_RIVAL_UUID, 0.0)])
        info = task._collector._detect_ambiguity(det, _MODEL_NAME)
        assert info is None or info.value == 0.0


class TestAmbiguityAsAnIndependentSaveTrigger:
    """The point of the feature: a high-confidence detection nothing in the
    confidence band could ever collect still gets the frame saved."""

    async def test_confident_but_confused_frame_is_saved(self):
        """Band is min 0.5 / max 1.0, so a 0.91 detection is in-band anyway. Use a
        band that would NOT trigger (max 0.6) and show ambiguity saves it."""
        params = {
            **_AMBIGUITY_PARAMS,
            "model_label_config": [{"modelName": _MODEL_NAME, "labelValues": {_LABEL_UUID: {"min": 0.2, "max": 0.6}}}],
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)

        confident_and_confused = _detection_with_alts(0.91, [_alt(_RIVAL_UUID, 0.88)])
        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect_unit.side_effect = _batch_of(_result_with([confident_and_confused]))

        frame = _untracked_image()
        matching = await task._collector._run_inference(inference_api, "b64", {_MODEL_NAME: None}, provided=frame)

        # Band alone would have rejected it: 0.91 > max 0.6.
        assert task._collector._frame_triggers_save([confident_and_confused]) is False
        assert matching == [confident_and_confused]
        assert frame._det_meta[confident_and_confused.bbox.uuid]["ambiguity"] is not None

    async def test_confident_and_unconfused_frame_is_still_not_saved(self):
        """The complement — ambiguity must not turn into 'save everything'."""
        params = {
            **_AMBIGUITY_PARAMS,
            "model_label_config": [{"modelName": _MODEL_NAME, "labelValues": {_LABEL_UUID: {"min": 0.2, "max": 0.6}}}],
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)

        confident_and_sure = _detection_with_alts(0.91, [_alt(_RIVAL_UUID, 0.05)])
        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect_unit.side_effect = _batch_of(_result_with([confident_and_sure]))

        frame = _untracked_image()
        matching = await task._collector._run_inference(inference_api, "b64", {_MODEL_NAME: None}, provided=frame)
        assert matching == []

    async def test_top_k_is_only_requested_when_the_feature_is_on(self):
        """Every other run must keep today's response size."""
        task = _make_task(params=_PARAMS)
        await task._init(_PARAMS, resume_data=None)
        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect_unit.side_effect = _batch_of(_result_with([]))
        await task._collector._run_inference(inference_api, "b64", {_MODEL_NAME: None})
        # No requester asked for top_k > 0 on any model, so the union carries
        # none — same "don't ask" contract as the old scalar 0, expressed as
        # an empty per-model dict now that top_k is unioned across requesters
        # (apps/helpers/frameInference.py).
        assert inference_api.detect_unit.call_args.args[0].top_k.actual_instance == {}

    async def test_top_k_is_requested_when_the_feature_is_on(self):
        task = _make_task(params=_AMBIGUITY_PARAMS)
        await task._init(_AMBIGUITY_PARAMS, resume_data=None)
        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect_unit.side_effect = _batch_of(_result_with([]))
        await task._collector._run_inference(inference_api, "b64", {_MODEL_NAME: None})
        assert inference_api.detect_unit.call_args.args[0].top_k.actual_instance == {_MODEL_NAME: 3}


class TestAmbiguityAttributionSurvivesTheMerge:
    """The headline regression: ambiguity is computed per model BEFORE the
    cross-model merge, and the surviving detection must still name the model that
    was actually confused. Computing it after the merge cannot answer that at all
    — the merge drops losers and blends models together."""

    async def _two_model_task(self, **overrides):
        params = {
            **_AMBIGUITY_PARAMS,
            "model_label_config": [
                {"modelName": "model-a", "labelValues": {_LABEL_UUID: 0.5}},
                {"modelName": "model-b", "labelValues": {_LABEL_UUID: 0.5}},
            ],
            **overrides,
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        return task

    async def test_surviving_detection_keeps_its_own_models_attribution(self):
        """model-a wins the same-label collision; the metadata must say model-a,
        not model-b, and must carry model-a's ambiguity."""
        task = await self._two_model_task()

        a_det = _detection_with_alts(0.9, [_alt(_RIVAL_UUID, 0.88)])
        b_det = _detection_with_alts(0.7, [_alt(_RIVAL_UUID, 0.10)])  # overlapping, same label

        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect_unit.side_effect = _batch_per_model(
            lambda n: MagicMock(detections=[a_det] if n == "model-a" else [b_det], alternates_available=True)
        )

        frame = _untracked_image()
        matching = await task._collector._run_inference(
            inference_api, "b64", {"model-a": None, "model-b": None}, provided=frame
        )

        assert matching == [a_det]
        meta = frame._det_meta[a_det.bbox.uuid]
        assert meta["model"] == "model-a"
        assert meta["ambiguity"].model == "model-a"
        # The dropped detection's metadata must not linger — it describes a box
        # that will never be written.
        assert b_det.bbox.uuid not in frame._det_meta

    async def test_each_model_is_attributed_separately_when_both_survive(self):
        """Non-overlapping boxes: both survive, each keeps its own model name."""
        task = await self._two_model_task()

        a_det = _detection_with_alts(0.9, [_alt(_RIVAL_UUID, 0.88)])
        b_det = _detection_with_alts(0.9, [_alt(_RIVAL_UUID, 0.88)])
        b_det.bbox.x, b_det.bbox.y = 0.7, 0.7  # far away, no IoU overlap

        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect_unit.side_effect = _batch_per_model(
            lambda n: MagicMock(detections=[a_det] if n == "model-a" else [b_det], alternates_available=True)
        )

        frame = _untracked_image()
        await task._collector._run_inference(inference_api, "b64", {"model-a": None, "model-b": None}, provided=frame)

        assert frame._det_meta[a_det.bbox.uuid]["model"] == "model-a"
        assert frame._det_meta[b_det.bbox.uuid]["model"] == "model-b"


class TestAmbiguityUnsupportedModels:
    """An NMS-free/E2E export argmaxes inside its own graph and can never report a
    runner-up. The failure mode to avoid is silence: a multi-day run that flags
    nothing looks exactly like a run where nothing was confusable."""

    async def test_unavailable_model_raises_a_notice(self):
        task = _make_task(params=_AMBIGUITY_PARAMS)
        await task._init(_AMBIGUITY_PARAMS, resume_data=None)

        inference_api = MagicMock(spec=InferenceApi)
        inference_api.detect_unit.side_effect = _batch_of(MagicMock(detections=[], alternates_available=False))
        await task._collector._run_inference(inference_api, "b64", {_MODEL_NAME: None})

        assert _MODEL_NAME in task._collector._ambiguity_unavailable_models
        task._status = TaskStatus.RUNNING
        messages = [n["message"] for n in task.get_notices()]
        assert any(_MODEL_NAME in m and "runner-up" in m for m in messages)


class TestAmbiguityPersistence:
    """What lands on the bbox: the tag Review filters on, and the metadata that
    names which model was confused and with what."""

    async def test_flagged_bbox_gets_tag_metadata_and_source_model(self, tmp_path):
        params = {
            **_AMBIGUITY_PARAMS,
            "model_label_config": [{"modelName": _MODEL_NAME, "labelValues": {_LABEL_UUID: {"min": 0.2, "max": 0.6}}}],
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        task._collector._unverified_tag_uuid = UNVERIFIED_TAG_UUID

        det = _detection_with_alts(0.91, [_alt(_RIVAL_UUID, 0.88)])
        frame = _untracked_image()
        frame._det_meta = {
            det.bbox.uuid: {
                "model": _MODEL_NAME,
                "ambiguity": task._collector._detect_ambiguity(det, _MODEL_NAME),
            }
        }

        images_api = MagicMock()
        task._collector._meta_write_lock = asyncio.Lock()
        task._collector._files_root = tmp_path
        await task._collector._save_image(
            images_api, FrameOffer(provided=frame, base64="b64", provider_type=ImageProviderType.FIXED), [det]
        )

        box = images_api.update_image_metadata.call_args.args[0].boxes[0]
        assert CONFUSABLE_TAG_UUID in box.tag_uuids
        assert box.bbox_meta["source_model"] == _MODEL_NAME
        assert box.bbox_meta["ambiguity_model"] == _MODEL_NAME
        assert box.bbox_meta["ambiguity_rival_label_uuid"] == _RIVAL_UUID
        assert json.loads(box.bbox_meta["alternates"])[0]["label_uuid"] == _RIVAL_UUID
        # The confidence key every reader already depends on must be untouched.
        assert box.bbox_meta["confidence"] == "0.91"


class TestFrameProvenance:
    """Phase 5, COLLECTOR_UNIFICATION_PLAN.md: a watcher's event-scoped
    collect_training_images action stashes which rule/event produced a frame
    in `ImageProviderMetadata.additional_info` — _save_image must fold it
    into every box's bbox_meta, the same way source_model already is, so the
    question survives even on a frame with no ambiguity info."""

    async def test_frame_provenance_lands_on_every_box(self, tmp_path):
        task = _make_task(params=_PARAMS)
        await task._init(_PARAMS, resume_data=None)

        det = _detection()
        provided = ImageWithProviderMetadata(
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            metadata=ImageProviderMetadata(
                additional_info={"watch_rule_uuid": "rule-a", "watch_event_started_at": "123.0"}
            ),
        )

        images_api = MagicMock()
        task._collector._meta_write_lock = asyncio.Lock()
        task._collector._files_root = tmp_path
        await task._collector._save_image(
            images_api, FrameOffer(provided=provided, base64="b64", provider_type=ImageProviderType.FIXED), [det]
        )

        box = images_api.update_image_metadata.call_args.args[0].boxes[0]
        assert box.bbox_meta["watch_rule_uuid"] == "rule-a"
        assert box.bbox_meta["watch_event_started_at"] == "123.0"

    async def test_no_additional_info_means_no_provenance_keys(self, tmp_path):
        """Continuous-mode frames (and the standalone task, which never sets
        additional_info) must not gain empty/spurious keys."""
        task = _make_task(params=_PARAMS)
        await task._init(_PARAMS, resume_data=None)

        det = _detection()
        images_api = MagicMock()
        task._collector._meta_write_lock = asyncio.Lock()
        task._collector._files_root = tmp_path
        await task._collector._save_image(
            images_api,
            FrameOffer(provided=_untracked_image(), base64="b64", provider_type=ImageProviderType.FIXED),
            [det],
        )

        box = images_api.update_image_metadata.call_args.args[0].boxes[0]
        assert "watch_rule_uuid" not in box.bbox_meta
        assert "watch_event_started_at" not in box.bbox_meta

    async def test_metadata_values_are_all_strings(self, tmp_path):
        """metadata_json is typed Dict[str, str] end to end (RawJsonDict). A
        nested dict or a bare float is rejected by the generated client's
        validation, so this pins the whole payload flat and stringly-typed."""
        task = _make_task(params=_AMBIGUITY_PARAMS)
        await task._init(_AMBIGUITY_PARAMS, resume_data=None)
        task._collector._unverified_tag_uuid = UNVERIFIED_TAG_UUID

        det = _detection_with_alts(0.91, [_alt(_RIVAL_UUID, 0.88)])
        frame = _untracked_image()
        frame._det_meta = {
            det.bbox.uuid: {"model": _MODEL_NAME, "ambiguity": task._collector._detect_ambiguity(det, _MODEL_NAME)}
        }

        images_api = MagicMock()
        task._collector._meta_write_lock = asyncio.Lock()
        task._collector._files_root = tmp_path
        await task._collector._save_image(
            images_api, FrameOffer(provided=frame, base64="b64", provider_type=ImageProviderType.FIXED), [det]
        )

        box = images_api.update_image_metadata.call_args.args[0].boxes[0]
        assert all(isinstance(v, str) for v in box.bbox_meta.values())

    async def test_unflagged_bbox_gets_source_model_but_no_tag_or_alternates(self, tmp_path):
        """source_model is written on EVERY collected box — it is the precondition
        for asking 'which model is confused?' — but the tag and the alternates
        payload are only for boxes that actually fired."""
        task = _make_task(params=_AMBIGUITY_PARAMS)
        await task._init(_AMBIGUITY_PARAMS, resume_data=None)
        task._collector._unverified_tag_uuid = UNVERIFIED_TAG_UUID

        det = _detection_with_alts(0.91, [_alt(_RIVAL_UUID, 0.05)])
        frame = _untracked_image()
        frame._det_meta = {det.bbox.uuid: {"model": _MODEL_NAME, "ambiguity": None}}

        images_api = MagicMock()
        task._collector._meta_write_lock = asyncio.Lock()
        task._collector._files_root = tmp_path
        await task._collector._save_image(
            images_api, FrameOffer(provided=frame, base64="b64", provider_type=ImageProviderType.FIXED), [det]
        )

        box = images_api.update_image_metadata.call_args.args[0].boxes[0]
        assert CONFUSABLE_TAG_UUID not in box.tag_uuids
        assert box.bbox_meta["source_model"] == _MODEL_NAME
        assert "ambiguity_model" not in box.bbox_meta
        assert "alternates" not in box.bbox_meta

    async def test_browse_confusable_action_only_appears_when_enabled(self):
        task = _make_task(params=_AMBIGUITY_PARAMS)
        await task._init(_AMBIGUITY_PARAMS, resume_data=None)
        task._collector._task_tag_uuid = "task-tag"
        actions = task._review_actions()
        labels = [a["label"] for a in actions]
        assert "Browse confusable" in labels
        action = next(a for a in actions if a["label"] == "Browse confusable")
        assert action["payload"]["bbox_verification"] == "unverified"

        off = _make_task(params=_PARAMS)
        await off._init(_PARAMS, resume_data=None)
        off._collector._task_tag_uuid = "task-tag"
        assert "Browse confusable" not in [a["label"] for a in off._review_actions()]

    async def test_browse_detector_misses_action_only_appears_when_enabled(self):
        params = {**_PARAMS, "enable_detector_miss": True}
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        task._collector._task_tag_uuid = "task-tag"
        actions = task._review_actions()
        labels = [a["label"] for a in actions]
        assert "Browse detector misses" in labels
        action = next(a for a in actions if a["label"] == "Browse detector misses")
        assert action["payload"]["tag_uuids"] == [task._collector._task_tag_uuid, DETECTOR_MISS_TAG_UUID]
        assert action["payload"]["bbox_verification"] == "unverified"

        off = _make_task(params=_PARAMS)
        await off._init(_PARAMS, resume_data=None)
        off._collector._task_tag_uuid = "task-tag"
        assert "Browse detector misses" not in [a["label"] for a in off._review_actions()]


class TestCrossModelAmbiguity:
    """Two models putting incompatible labels on the same object. Needs no
    runner-up scores, so it is the only confusion signal available for
    NMS-free/E2E models — and it must name BOTH models, or it says nothing more
    than 'the merged set was inconsistent'."""

    _DOG_UUID = "label-dog-uuid"

    async def _task(self, ancestry=None, **overrides):
        params = {
            **_AMBIGUITY_PARAMS,
            "model_label_config": [
                {"modelName": "model-a", "labelValues": {_LABEL_UUID: 0.5}},
                {"modelName": "model-b", "labelValues": {self._DOG_UUID: 0.5}},
            ],
            **overrides,
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        task._collector._label_ancestry = ancestry or {}
        return task

    def _api(self, a_dets, b_dets):
        api = MagicMock(spec=InferenceApi)
        api.detect_unit.side_effect = _batch_per_model(
            lambda n: MagicMock(detections=a_dets if n == "model-a" else b_dets, alternates_available=True)
        )
        return api

    async def _run(self, task, a_dets, b_dets):
        frame = _untracked_image()
        await task._collector._run_inference(
            self._api(a_dets, b_dets), "b64", {"model-a": None, "model-b": None}, provided=frame
        )
        return frame

    async def test_disagreeing_models_flag_both_boxes_naming_each_other(self):
        task = await self._task()
        cat = _detection_with_alts(0.9, [], label_uuid=_LABEL_UUID)
        dog = _detection_with_alts(0.85, [], label_uuid=self._DOG_UUID)  # same place, IoU 1.0

        frame = await self._run(task, [cat], [dog])

        cat_info = frame._det_meta[cat.bbox.uuid]["ambiguity"]
        dog_info = frame._det_meta[dog.bbox.uuid]["ambiguity"]
        assert cat_info.kind == "cross_model"
        assert cat_info.model == "model-a" and cat_info.rival_model == "model-b"
        assert cat_info.rival_label_uuid == self._DOG_UUID
        # The mirror image, so opening either box explains itself.
        assert dog_info.model == "model-b" and dog_info.rival_model == "model-a"
        assert dog_info.rival_label_uuid == _LABEL_UUID

    async def test_one_disagreement_counts_once(self):
        """Both boxes carry the verdict, but it is one disagreement.

        Counting per box made `confusable_flagged` twice `confusable_measured` —
        the status panel read "2 flagged of 1 measured" — and listed the pair
        twice in the top-pairs summary, once in each direction.
        """
        task = await self._task()
        cat = _detection_with_alts(0.9, [], label_uuid=_LABEL_UUID)
        dog = _detection_with_alts(0.85, [], label_uuid=self._DOG_UUID)

        await self._run(task, [cat], [dog])

        assert task._collector._ambiguity_evaluated == 1
        assert task._collector._ambiguity_flagged == 1
        assert sum(task._collector._ambiguity_pairs.values()) == 1
        # Attributed to the higher-confidence side: model-a said cat, over model-b's dog.
        assert list(task._collector._ambiguity_pairs) == [("model-a", _LABEL_UUID, self._DOG_UUID)]

    async def test_a_pair_gated_out_on_both_sides_counts_as_measured_not_flagged(self):
        """`ambiguity_min_confidence` rejecting both boxes means the pair was looked
        at and found not worth reporting — it must not inflate the flagged tally."""
        task = await self._task()
        task._spec.ambiguity_min_confidence = 0.99
        cat = _detection_with_alts(0.9, [], label_uuid=_LABEL_UUID)
        dog = _detection_with_alts(0.85, [], label_uuid=self._DOG_UUID)

        await self._run(task, [cat], [dog])

        assert task._collector._ambiguity_evaluated == 1
        assert task._collector._ambiguity_flagged == 0
        assert task._collector._ambiguity_pairs == {}

    async def test_review_disabled_label_is_flagged_on_neither_side(self):
        """A cross-model pair where one side's label is set to 'Review up to = 0'
        is not a reason to review the frame — not for the disabled label, and not
        for the model that disagreed with it."""
        task = await self._task(
            model_label_config=[
                {"modelName": "model-a", "labelValues": {_LABEL_UUID: 0.5}},
                {"modelName": "model-b", "labelValues": {self._DOG_UUID: {"min": 0.5, "max": 0}}},
            ]
        )
        cat = _detection_with_alts(0.9, [], label_uuid=_LABEL_UUID)
        dog = _detection_with_alts(0.85, [], label_uuid=self._DOG_UUID)

        frame = await self._run(task, [cat], [dog])

        assert all(m["ambiguity"] is None for m in frame._det_meta.values())
        assert task._collector._ambiguity_flagged == 0
        assert task._collector._ambiguity_evaluated == 0

    async def test_agreeing_models_are_not_flagged(self):
        """Same label on the same object is agreement, not confusion."""
        task = await self._task()
        a = _detection_with_alts(0.9, [], label_uuid=_LABEL_UUID)
        b = _detection_with_alts(0.85, [], label_uuid=_LABEL_UUID)
        frame = await self._run(task, [a], [b])
        assert all(m["ambiguity"] is None for m in frame._det_meta.values())

    async def test_parent_child_labels_are_hierarchy_not_disagreement(self):
        """'Animal' over 'Cat' is a refinement — the merge already treats it that
        way, and flagging it would fire on every working hierarchical model."""
        task = await self._task(ancestry={_LABEL_UUID: {self._DOG_UUID}})
        child = _detection_with_alts(0.9, [], label_uuid=_LABEL_UUID)
        parent = _detection_with_alts(0.85, [], label_uuid=self._DOG_UUID)
        frame = await self._run(task, [child], [parent])
        assert all(m["ambiguity"] is None for m in frame._det_meta.values())

    async def test_non_overlapping_boxes_are_not_flagged(self):
        """Two different objects, not one contested object."""
        task = await self._task()
        cat = _detection_with_alts(0.9, [], label_uuid=_LABEL_UUID)
        dog = _detection_with_alts(0.85, [], label_uuid=self._DOG_UUID)
        dog.bbox.x, dog.bbox.y = 0.8, 0.8
        frame = await self._run(task, [cat], [dog])
        assert all(m["ambiguity"] is None for m in frame._det_meta.values())

    async def test_confident_disagreement_below_threshold_is_not_flagged(self):
        """A confident model beating an unconfident one is not a close call."""
        task = await self._task()
        cat = _detection_with_alts(0.95, [], label_uuid=_LABEL_UUID)
        dog = _detection_with_alts(0.51, [], label_uuid=self._DOG_UUID)  # ratio 0.54 < 0.7
        frame = await self._run(task, [cat], [dog])
        assert all(m["ambiguity"] is None for m in frame._det_meta.values())

    async def test_same_model_verdict_wins_over_cross_model(self):
        """Same-model confusion is the more direct statement about a specific
        model's decision boundary, so it must not be overwritten."""
        task = await self._task()
        cat = _detection_with_alts(0.9, [_alt(_RIVAL_UUID, 0.89)], label_uuid=_LABEL_UUID)
        dog = _detection_with_alts(0.85, [], label_uuid=self._DOG_UUID)
        frame = await self._run(task, [cat], [dog])
        assert frame._det_meta[cat.bbox.uuid]["ambiguity"].kind == "same_model"
        # The other box still gets its cross-model verdict.
        assert frame._det_meta[dog.bbox.uuid]["ambiguity"].kind == "cross_model"

    async def test_can_be_disabled_independently(self):
        task = await self._task(ambiguity_cross_model=False)
        cat = _detection_with_alts(0.9, [], label_uuid=_LABEL_UUID)
        dog = _detection_with_alts(0.85, [], label_uuid=self._DOG_UUID)
        frame = await self._run(task, [cat], [dog])
        assert all(m["ambiguity"] is None for m in frame._det_meta.values())

    async def test_cross_model_works_without_any_runner_up_scores(self):
        """The E2E fallback: neither detection carries alternates, and the flag
        still fires. This is what makes the feature usable for models that
        cannot report runner-up classes at all."""
        task = await self._task()
        cat = _detection_with_alts(0.9, [], label_uuid=_LABEL_UUID)
        dog = _detection_with_alts(0.88, [], label_uuid=self._DOG_UUID)
        assert cat.alternates == [] and dog.alternates == []
        frame = await self._run(task, [cat], [dog])
        assert frame._det_meta[cat.bbox.uuid]["ambiguity"] is not None

    async def test_cross_model_metadata_names_both_models_and_stays_stringly_typed(self, tmp_path):
        task = await self._task()
        task._collector._unverified_tag_uuid = UNVERIFIED_TAG_UUID
        cat = _detection_with_alts(0.9, [], label_uuid=_LABEL_UUID)
        dog = _detection_with_alts(0.85, [], label_uuid=self._DOG_UUID)
        frame = await self._run(task, [cat], [dog])

        images_api = MagicMock()
        task._collector._meta_write_lock = asyncio.Lock()
        task._collector._files_root = tmp_path
        await task._collector._save_image(
            images_api, FrameOffer(provided=frame, base64="b64", provider_type=ImageProviderType.FIXED), [cat]
        )

        box = images_api.update_image_metadata.call_args.args[0].boxes[0]
        assert CONFUSABLE_TAG_UUID in box.tag_uuids
        assert box.bbox_meta["ambiguity_kind"] == "cross_model"
        assert box.bbox_meta["ambiguity_model"] == "model-a"
        assert box.bbox_meta["ambiguity_rival_model"] == "model-b"
        assert all(isinstance(v, str) for v in box.bbox_meta.values())

    async def test_rival_below_its_own_label_floor_is_not_a_disagreement(self):
        """The scan only considers detections that will actually be written.

        `conf_thresh` sent to the network is the LOOSEST floor across all
        configured labels, so the raw per-model lists contain detections well
        below their own label's minimum. Those are dropped by _annotatable and
        never become boxes — letting one act as a rival would produce a flag
        naming a box the reviewer cannot see, and would treat a detection the
        config calls uncollectable as evidence. It is also what keeps this
        quadratic scan proportional to what the run collects rather than to what
        the network returned.
        """
        params = {
            **_AMBIGUITY_PARAMS,
            "model_label_config": [
                {"modelName": "model-a", "labelValues": {_LABEL_UUID: 0.5}},
                {"modelName": "model-b", "labelValues": {self._DOG_UUID: 0.5}},
            ],
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        task._collector._label_ancestry = {}

        # ratio 0.45/0.6 = 0.75, over the 0.7 threshold — it would fire on the
        # numbers alone, but the dog is below the dog label's own 0.5 floor.
        cat = _detection_with_alts(0.6, [], label_uuid=_LABEL_UUID)
        dog = _detection_with_alts(0.45, [], label_uuid=self._DOG_UUID)
        frame = await self._run(task, [cat], [dog])

        assert frame._det_meta[cat.bbox.uuid]["ambiguity"] is None

    async def test_rival_at_its_floor_still_counts(self):
        """The complement — the prune must cut below the floor, not at it."""
        params = {
            **_AMBIGUITY_PARAMS,
            "model_label_config": [
                {"modelName": "model-a", "labelValues": {_LABEL_UUID: 0.5}},
                {"modelName": "model-b", "labelValues": {self._DOG_UUID: 0.5}},
            ],
        }
        task = _make_task(params=params)
        await task._init(params, resume_data=None)
        task._collector._label_ancestry = {}

        cat = _detection_with_alts(0.6, [], label_uuid=_LABEL_UUID)
        dog = _detection_with_alts(0.5, [], label_uuid=self._DOG_UUID)  # exactly at floor
        frame = await self._run(task, [cat], [dog])

        assert frame._det_meta[cat.bbox.uuid]["ambiguity"] is not None


class TestConcurrentFetch:
    """One quiet source must not stall the others.

    The fetch loop used to be sequential, which quietly turned each source's timeout
    into a budget for *everyone*: a Ring camera idling on ~30s cloud snapshots held up
    the whole rotation, and its peers went stale behind it while nothing was actually
    failing. That reads as several broken cameras rather than one slow one, which is
    the expensive part — you go looking in the wrong place.
    """

    class _Provider:
        min_capture_interval = 0.0
        source_name = "s"

        def __init__(self, delay, fail=False):
            self._delay, self._fail = delay, fail
            self.calls = 0

        async def getNextImage(self):
            self.calls += 1
            await asyncio.sleep(self._delay)
            if self._fail:
                raise RuntimeError("camera is down")
            return object()

    def _task(self):
        task = CollectTrainingImagesTask.__new__(CollectTrainingImagesTask)
        task._FETCH_TIMEOUT_S = 5.0
        task._MAX_RETRIES = 1
        return task

    async def test_a_slow_source_does_not_delay_a_fast_one(self):
        task = self._task()
        slow, fast = self._Provider(0.30), self._Provider(0.0)

        started = time.perf_counter()
        results = await asyncio.gather(
            task._fetch_with_retries(slow, "slow"),
            task._fetch_with_retries(fast, "fast"),
        )
        elapsed = time.perf_counter() - started

        assert all(r[1] is None for r in results), "neither fetch should have errored"
        # Sequentially this is >= 0.30s *per* slow source; concurrently it is one 0.30s.
        assert elapsed < 0.30 + 0.20, f"fetches did not overlap (took {elapsed:.2f}s)"

    async def test_one_failing_source_does_not_abandon_its_siblings(self):
        """The reason _fetch_with_retries returns errors instead of raising: a raise
        inside gather() takes down every sibling fetch in the same pass."""
        task = self._task()
        broken, healthy = self._Provider(0.0, fail=True), self._Provider(0.0)

        (broken_img, broken_err, _, _), (ok_img, ok_err, _, _) = await asyncio.gather(
            task._fetch_with_retries(broken, "broken"),
            task._fetch_with_retries(healthy, "healthy"),
        )

        assert broken_err is not None and "camera is down" in broken_err
        assert broken_img is None
        assert ok_err is None and ok_img is not None, "the healthy source must still have been fetched"

    async def test_a_timeout_is_reported_not_raised(self):
        task = self._task()
        task._FETCH_TIMEOUT_S = 0.05
        stuck = self._Provider(5.0)

        provided, error, _, _ = await task._fetch_with_retries(stuck, "stuck")

        assert provided is None
        assert error is not None and "timed out" in error


class TestSaveReasonAttribution:
    """Saves are split by *why* the frame was kept (plan §6).

    A single `saved` total cannot answer "is this knob producing anything",
    because the interesting reasons are the rare ones. Attribution happens where
    the save COMPLETES, not where it is dispatched, so the subtotals always sum
    to `saved` rather than counting frames dedup went on to discard.
    """

    @staticmethod
    def _collector():
        collector = _detached_collector({"output_dir": "out"})
        collector._save_image = AsyncMock(return_value="f.jpg")
        return collector

    @staticmethod
    def _offer():
        provided = MagicMock()
        provided.metadata.source = "src"
        return FrameOffer(provided=provided, base64="b64", unique_id=None)

    async def test_each_reason_lands_in_its_own_subtotal(self):
        collector = self._collector()

        collector._dispatch_save(self._offer(), [], SAVE_REASON_BAND)
        collector._dispatch_save(self._offer(), [], SAVE_REASON_DETECTOR_MISS)
        collector._dispatch_save(self._offer(), [], SAVE_REASON_TRACK_WORST)
        await collector.drain_saves()

        stats, _ = collector.status_stats()
        assert (stats["saved_band"], stats["saved_detector_miss"], stats["saved_track_worst"]) == (1, 1, 1)

    async def test_subtotals_sum_to_the_saved_total(self):
        collector = self._collector()
        for reason in (SAVE_REASON_BAND, SAVE_REASON_BAND, SAVE_REASON_DETECTOR_MISS):
            collector._dispatch_save(self._offer(), [], reason)
        await collector.drain_saves()

        stats, _ = collector.status_stats()
        assert stats["saved_band"] + stats["saved_detector_miss"] + stats["saved_track_worst"] == stats["saved"]

    async def test_a_save_that_returns_nothing_is_not_attributed(self):
        """`_save_image` returning None means the write did not happen — counting it
        would make the subtotals overstate what is on disk."""
        collector = self._collector()
        collector._save_image = AsyncMock(return_value=None)

        collector._dispatch_save(self._offer(), [], SAVE_REASON_DETECTOR_MISS)
        await collector.drain_saves()

        stats, _ = collector.status_stats()
        assert stats["saved"] == 0 and "saved_detector_miss" not in stats

    async def test_band_only_runs_do_not_show_the_split(self):
        """The standalone task saves for one reason only; three rows restating the
        same number is noise on its status panel."""
        collector = self._collector()
        collector._dispatch_save(self._offer(), [], SAVE_REASON_BAND)
        await collector.drain_saves()

        stats, _ = collector.status_stats()
        assert stats["saved"] == 1 and "saved_band" not in stats


class TestSendWidth:
    """`send_width = min(source_native_width, model_input_width)` (plan §2.3b) —
    arithmetic the system does itself rather than a user-facing toggle."""

    @staticmethod
    def _collector(model_input_width):
        collector = _detached_collector({"output_dir": "out"})
        collector._model_input_width = model_input_width
        return collector

    def test_capped_at_the_models_own_input(self):
        """Above the model's input you pay to ship pixels the server resizes away —
        the mistake the deleted full-res inference mode would have made every frame."""
        assert self._collector(640).send_width(1920) == 640

    def test_a_source_narrower_than_the_model_is_sent_as_is(self):
        assert self._collector(1280).send_width(704) == 704

    def test_unknown_model_width_means_change_nothing(self):
        """An unreadable ONNX graph must never make a working collector start
        re-encoding frames at a guessed size."""
        assert self._collector(None).send_width(1920) is None


class _FakeBBox:
    def __init__(self, label_uuid):
        self.label_uuid = label_uuid


class _FakeImage:
    def __init__(self, uuid, created_at, label_uuids=()):
        self.uuid = uuid
        self.filename = f"{uuid}.jpg"
        self.created_at = created_at
        self.bboxes = [_FakeBBox(label_uuid) for label_uuid in label_uuids]


class _FakeReviewQueueResponse:
    def __init__(self, images, total_count):
        self.images = images
        self.total_count = total_count


class TestReviewQueueEviction:
    """`TrainingCollector._enforce_review_queue_cap` — the part of the review-queue
    cap feature that actually deletes something. `select_for_eviction` itself
    (which strategy picks which candidate) has its own pure tests; these guard
    the surrounding wiring: does it skip when nothing is configured, does it ask
    for the right verification state, does it delete the right count and no more.
    """

    @staticmethod
    def _collector(params):
        collector = _detached_collector({"output_dir": "out", **params})
        # Normally set by _resolve_tags() during start(), which needs a real DB
        # round-trip this test doesn't want. Eviction only needs the identity,
        # not how it was resolved.
        collector._task_tag_uuid = "collector-tag-uuid"
        return collector

    async def test_no_op_when_no_caps_configured(self):
        collector = self._collector({})
        with patch("apps.helpers.training.collector.ImagesApi") as fake_api_cls:
            await collector._enforce_review_queue_cap()
        fake_api_cls.assert_not_called()

    async def test_no_op_before_the_task_tag_is_resolved(self):
        """The PENDING-window state `start()` hasn't reached yet — nothing to
        scope eviction to, so it must not guess and delete something else's images."""
        collector = self._collector({"max_unverified_images": 1})
        collector._task_tag_uuid = None
        with patch("apps.helpers.training.collector.ImagesApi") as fake_api_cls:
            collector._maybe_enforce_review_queue_cap()
            await asyncio.sleep(0)  # let any (wrongly) scheduled task start
        fake_api_cls.assert_not_called()

    async def test_evicts_the_oldest_excess_under_fifo(self):
        collector = self._collector({"max_unverified_images": 2, "eviction_strategy": "fifo"})
        images = [
            _FakeImage("oldest", datetime(2026, 1, 1)),
            _FakeImage("middle", datetime(2026, 1, 2)),
            _FakeImage("newest", datetime(2026, 1, 3)),
        ]
        fake_api = MagicMock()
        fake_api.get_review_queue.side_effect = [
            _FakeReviewQueueResponse([], total_count=3),  # the cheap count-only check
            _FakeReviewQueueResponse(images, total_count=3),  # the full candidate page
        ]
        with patch("apps.helpers.training.collector.ImagesApi", return_value=fake_api):
            await collector._enforce_review_queue_cap()
        fake_api.delete_image.assert_called_once_with("oldest")
        assert collector._evicted_total == 1

    async def test_stays_under_the_ceiling_never_deletes_when_not_over(self):
        collector = self._collector({"max_unverified_images": 5})
        fake_api = MagicMock()
        fake_api.get_review_queue.return_value = _FakeReviewQueueResponse([], total_count=3)
        with patch("apps.helpers.training.collector.ImagesApi", return_value=fake_api):
            await collector._enforce_review_queue_cap()
        fake_api.delete_image.assert_not_called()
        # Only the cheap count check ran — no reason to fetch a candidate page
        # for a queue that isn't over the cap.
        assert fake_api.get_review_queue.call_count == 1

    async def test_label_balance_targets_the_over_represented_label(self):
        collector = self._collector({"max_unverified_images": 3, "eviction_strategy": "label_balance"})
        images = [
            _FakeImage("cat1", datetime(2026, 1, 1), ["cat"]),
            _FakeImage("cat2", datetime(2026, 1, 2), ["cat"]),
            _FakeImage("raccoon1", datetime(2026, 1, 3), ["raccoon"]),
            _FakeImage("cat3", datetime(2026, 1, 4), ["cat"]),
        ]
        fake_api = MagicMock()
        fake_api.get_review_queue.side_effect = [
            _FakeReviewQueueResponse([], total_count=4),
            _FakeReviewQueueResponse(images, total_count=4),
        ]
        with patch("apps.helpers.training.collector.ImagesApi", return_value=fake_api):
            await collector._enforce_review_queue_cap()
        fake_api.delete_image.assert_called_once()
        assert fake_api.delete_image.call_args[0][0] in ("cat1", "cat2", "cat3")

    async def test_collect_saves_fires_an_eviction_pass_in_the_background(self):
        """The actual wiring a driver relies on: harvesting a completed save
        triggers a cap check without collect_saves() itself ever awaiting it."""
        collector = self._collector({"max_unverified_images": 1})
        images = [_FakeImage("old", datetime(2026, 1, 1)), _FakeImage("new", datetime(2026, 1, 2))]
        fake_api = MagicMock()
        fake_api.get_review_queue.side_effect = [
            _FakeReviewQueueResponse([], total_count=2),
            _FakeReviewQueueResponse(images, total_count=2),
        ]
        with patch("apps.helpers.training.collector.ImagesApi", return_value=fake_api):
            done = asyncio.get_event_loop().create_future()
            done.set_result("saved.jpg")
            collector._pending_saves.append((done, None, SAVE_REASON_BAND))
            collector.collect_saves()
            assert collector._eviction_task is not None
            await await_condition(lambda: fake_api.delete_image.called, "eviction pass never ran")
            await collector._eviction_task
        fake_api.delete_image.assert_called_once_with("old")
