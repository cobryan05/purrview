"""Tests for PreviewClassifierMatchTask.

Exercises the candidate-building logic against a fake ClassifierRuntime
(returning (label_uuid, confidence) predictions) injected directly into
`_classify_batch` — no real ONNX session, inference server, or db_client
HTTP round-trip involved (mirrors PreviewModelRelabelTask's test pattern).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest

from apps.tasks_server.tasks.PreviewClassifierMatchTask import (
    PreviewClassifierMatchTask,
)

_MOCK_MANAGER = MagicMock()
_MOCK_MANAGER.update_task_heartbeat = AsyncMock()


def _make_task(params: dict) -> PreviewClassifierMatchTask:
    task = PreviewClassifierMatchTask(task_config_uuid="cfg-uuid", params=params, manager=_MOCK_MANAGER)
    task.set_instance_uuid("11111111-1111-1111-1111-111111111111")
    return task


def _base_params(**overrides):
    params = {
        "model_name": "my_classifier",
        "conf_thresh": 0.5,
        "image_uuids": ["img-1"],
    }
    params.update(overrides)
    return params


def _image(uuid="img-1", filename="/data/x.jpg"):
    return {"uuid": uuid, "filename": filename}


class _FakeRuntime:
    """Fake ClassifierRuntime — `predict_batch` returns a fixed per-image
    prediction list regardless of the tensors passed in."""

    def __init__(self, batch_predictions):
        self._batch_predictions = batch_predictions

    def preprocess(self, img_bytes):
        return np.zeros((3, 2, 2), dtype=np.float32)

    async def predict_batch(self, tensors):
        return self._batch_predictions


class TestGuards:
    async def test_missing_model_name_raises(self):
        task = _make_task(_base_params(model_name=""))
        with pytest.raises(ValueError, match="model_name"):
            await task._init(task._params, None)

    async def test_missing_image_uuids_raises(self):
        task = _make_task(_base_params(image_uuids=[]))
        with pytest.raises(ValueError, match="image_uuids"):
            await task._init(task._params, None)

    async def test_invalid_model_name_rejected(self):
        task = _make_task(_base_params(model_name="../escape"))
        with pytest.raises(ValueError, match="Invalid model_name"):
            await task._init(task._params, None)

    async def test_default_conf_thresh(self):
        task = _make_task(_base_params())
        await task._init(task._params, None)
        assert task._conf_thresh == 0.5


class TestMatching:
    """Exercises `_classify_batch` directly with a fake runtime, bypassing the
    ONNX session/db_client plumbing and the model-kind guard in `_run()`."""

    async def _run_classify_batch(self, task, predictions):
        await task._init(task._params, None)
        task._load_image_bytes = AsyncMock(return_value=b"fake-image-bytes")
        runtime = _FakeRuntime([predictions])
        img = _image()
        await task._classify_batch(
            images_api=MagicMock(),
            runtime=runtime,
            image_uuids=["img-1"],
            images_by_uuid={"img-1": img},
            label_names={"corrupt": "Corrupt"},
        )
        return task._candidates

    async def test_match_above_threshold_produces_candidate(self):
        task = _make_task(_base_params())
        predictions = [("corrupt", 0.9)]
        candidates = await self._run_classify_batch(task, predictions)
        assert len(candidates) == 1
        c = candidates[0]
        assert c["imageUuid"] == "img-1"
        assert c["imagePath"] == "/data/x.jpg"
        assert c["predictedLabelUuid"] == "corrupt"
        assert c["predictedLabelName"] == "Corrupt"
        assert c["confidence"] == 0.9

    async def test_below_threshold_excluded(self):
        task = _make_task(_base_params(conf_thresh=0.5))
        predictions = [("corrupt", 0.2)]
        candidates = await self._run_classify_batch(task, predictions)
        assert candidates == []
        assert task._no_match_count == 1

    async def test_no_predictions_excluded(self):
        task = _make_task(_base_params())
        candidates = await self._run_classify_batch(task, [])
        assert candidates == []
        assert task._no_match_count == 1

    async def test_unmapped_label_excluded(self):
        task = _make_task(_base_params())
        predictions = [(None, 0.95)]
        candidates = await self._run_classify_batch(task, predictions)
        assert candidates == []
        assert task._no_match_count == 1

    async def test_multi_label_classifier_picks_max_confidence(self):
        task = _make_task(_base_params())
        predictions = [("a", 0.4), ("b", 0.95), ("c", 0.6)]
        candidates = await self._run_classify_batch(task, predictions)
        assert len(candidates) == 1
        assert candidates[0]["predictedLabelUuid"] == "b"
        assert candidates[0]["confidence"] == 0.95

    async def test_image_load_failure_increments_failed_count(self):
        """Unlike the old per-image HTTP path, `_classify_batch` catches
        load/preprocess failures itself so one bad image in a batch doesn't
        abort inference for the rest of the batch."""
        task = _make_task(_base_params())
        await task._init(task._params, None)
        task._load_image_bytes = AsyncMock(side_effect=FileNotFoundError("/data/x.jpg"))
        runtime = _FakeRuntime([])
        img = _image()
        await task._classify_batch(
            images_api=MagicMock(),
            runtime=runtime,
            image_uuids=["img-1"],
            images_by_uuid={"img-1": img},
            label_names={"corrupt": "Corrupt"},
        )
        assert task._candidates == []
        assert task._images_failed == 1
        assert task._images_processed == 1
        assert task._sample_errors and "img-1" in task._sample_errors[0]


class TestModelKindGuard:
    async def test_non_classifier_model_rejected(self, monkeypatch):
        import apps.tasks_server.tasks.PreviewClassifierMatchTask as mod

        monkeypatch.setattr(mod, "_read_model_kind", lambda name: "detector")
        task = _make_task(_base_params())
        await task._init(task._params, None)
        with pytest.raises(ValueError, match="only supports classifier models"):
            await task._run()


class TestRun:
    """End-to-end `_run()` against a mocked ONNX session — proves the model is
    loaded exactly once (the "pin") and multiple images share a single
    `session.run` call (the batching), matching RunClassifierTask's approach."""

    async def test_run_loads_model_once_and_batches_inference(self, tmp_path, monkeypatch):
        result, mock_ctor, mock_session = await _run_two_image_task(tmp_path, monkeypatch, _base_params())

        assert mock_ctor.call_count == 1  # model loaded exactly once for the whole run
        assert mock_session.run.call_count == 1  # both images inferred in a single batched call
        assert result["candidate_count"] == 2
        assert {c["imageUuid"] for c in result["candidates"]} == {"img-a", "img-b"}
        assert all(c["predictedLabelUuid"] == "corrupt" for c in result["candidates"])


async def _run_two_image_task(tmp_path, monkeypatch, params):
    """Drive `_run()` over two images against a mocked ONNX session; both match "corrupt"."""
    import apps.tasks_server.tasks.PreviewClassifierMatchTask as mod

    model_name = "corrupt_clf"
    (tmp_path / f"{model_name}.onnx").write_bytes(b"fake")
    config = {
        "model_kind": "image_classifier",
        "input_size": [224, 224],
        "class_names": ["absent", "present"],
        "label_uuid": "corrupt",
    }
    (tmp_path / f"{model_name}.json").write_text(json.dumps(config))

    monkeypatch.setattr(mod, "_read_model_kind", lambda name: "image_classifier")

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input")]
    mock_session.get_providers.return_value = ["CPUExecutionProvider"]
    # Two images fetched in one batch -> one batched call, two rows of logits.
    mock_session.run.return_value = [np.array([[0.1, 5.0], [0.1, 5.0]])]

    img_a = {"uuid": "img-a", "filename": "/data/a.jpg"}
    img_b = {"uuid": "img-b", "filename": "/data/b.jpg"}

    mock_images_api = MagicMock()
    mock_images_api.batch_images_by_uuids.return_value = MagicMock(images=[img_a, img_b])

    mock_labels_api = MagicMock()
    label = MagicMock(uuid="corrupt", name="Corrupt")
    mock_labels_api.list_labels.return_value = MagicMock(labels=[label])

    params = {**params, "model_name": model_name, "image_uuids": ["img-a", "img-b"], "conf_thresh": 0.5}
    task = _make_task(params)

    with (
        patch("apps.tasks_server.tasks._classifier_runtime._MODELS_ROOT", tmp_path),
        patch("apps.tasks_server.tasks.PreviewClassifierMatchTask.ImagesApi", return_value=mock_images_api),
        patch("apps.tasks_server.tasks.PreviewClassifierMatchTask.LabelsApi", return_value=mock_labels_api),
        patch("onnxruntime.InferenceSession", return_value=mock_session) as mock_ctor,
    ):
        img_bytes = cv2.imencode(".jpg", np.zeros((10, 10, 3), dtype=np.uint8))[1].tobytes()

        await task._init(task._params, None)
        task._load_image_bytes = AsyncMock(return_value=img_bytes)
        result = await task._run()
        await task._deinit()

    return result, mock_ctor, mock_session


class _FakePreviewRunsApi:
    """Captures preview-run writes without an HTTP round-trip."""

    added = []
    updated = []
    fail_adds = False

    def __init__(self, _client):
        pass

    def add_preview_candidates(self, run_uuid, payload):
        if _FakePreviewRunsApi.fail_adds:
            raise RuntimeError("db down")
        _FakePreviewRunsApi.added.append((run_uuid, payload))

    def update_preview_run(self, run_uuid, payload):
        _FakePreviewRunsApi.updated.append((run_uuid, payload))


class TestPreviewRunPersistence:
    """`preview_run_uuid` makes the run outlive the tab that started it: the
    candidates and the stats both land in the saved run, written by the task."""

    @pytest.fixture(autouse=True)
    def fake_api(self, monkeypatch):
        _FakePreviewRunsApi.added = []
        _FakePreviewRunsApi.updated = []
        _FakePreviewRunsApi.fail_adds = False
        monkeypatch.setattr("apps.tasks_server.tasks.PreviewClassifierMatchTask.PreviewRunsApi", _FakePreviewRunsApi)

    async def test_run_writes_candidates_and_stats_into_the_run(self, tmp_path, monkeypatch):
        result, _, _ = await _run_two_image_task(tmp_path, monkeypatch, _base_params(preview_run_uuid="run-1"))

        written = [c for run_uuid, payload in _FakePreviewRunsApi.added for c in payload.candidates]
        assert all(run_uuid == "run-1" for run_uuid, _ in _FakePreviewRunsApi.added)
        assert {c.image_uuid for c in written} == {"img-a", "img-b"}
        # Whole-image proposals: no box to point at, and not a new box either.
        assert all(c.bbox_uuid is None and not c.is_new for c in written)
        assert all(c.proposed_label_uuid == "corrupt" and c.source_model == "corrupt_clf" for c in written)

        assert len(_FakePreviewRunsApi.updated) == 1
        run_uuid, update = _FakePreviewRunsApi.updated[0]
        assert run_uuid == "run-1"
        assert update.stats["images_processed"] == 2
        assert update.stats["candidate_count"] == 2
        assert "candidates" not in update.stats

        assert result["preview_run_uuid"] == "run-1"
        assert result["candidates"] == []
        assert result["candidate_count"] == 2

    async def test_without_a_run_uuid_nothing_is_written(self, tmp_path, monkeypatch):
        result, _, _ = await _run_two_image_task(tmp_path, monkeypatch, _base_params())
        assert _FakePreviewRunsApi.added == []
        assert _FakePreviewRunsApi.updated == []
        assert len(result["candidates"]) == 2

    async def test_flushed_candidates_are_not_reinserted_on_resume(self):
        task = _make_task(_base_params(preview_run_uuid="run-1"))
        await task._init(task._params, None)
        task._candidates = [{"imageUuid": "img-1", "predictedLabelUuid": "corrupt", "confidence": 0.9}]
        await task._flush_candidates(force=True)

        resumed = _make_task(_base_params(preview_run_uuid="run-1"))
        await resumed._init(resumed._params, task._build_resume_data())
        await resumed._flush_candidates(force=True)

        assert len(_FakePreviewRunsApi.added) == 1
        assert resumed._build_result()["candidate_count"] == 1

    async def test_a_failed_flush_keeps_the_buffer_for_retry(self):
        task = _make_task(_base_params(preview_run_uuid="run-1"))
        await task._init(task._params, None)
        task._candidates = [{"imageUuid": "img-1", "predictedLabelUuid": "corrupt", "confidence": 0.9}]

        _FakePreviewRunsApi.fail_adds = True
        with pytest.raises(RuntimeError):
            await task._flush_candidates(force=True)
        assert len(task._candidates) == 1

        _FakePreviewRunsApi.fail_adds = False
        await task._flush_candidates(force=True)
        assert task._candidates == []
        assert len(_FakePreviewRunsApi.added) == 1


class TestResume:
    def test_build_resume_data_and_result_shapes(self):
        task = _make_task(_base_params())
        task._images_processed = 3
        task._candidates = [{"imageUuid": "img-1"}]
        task._candidates_written = 0
        task._preview_run_uuid = None
        task._processed_image_uuids = {"img-1", "img-2"}
        task._images_failed = 1
        task._sample_errors = ["img-3: boom"]
        task._no_match_count = 2

        resume = task._build_resume_data()
        assert resume == {
            "images_processed": 3,
            "candidates": [{"imageUuid": "img-1"}],
            "candidates_written": 0,
            "processed_image_uuids": ["img-1", "img-2"],
            "images_failed": 1,
            "sample_errors": ["img-3: boom"],
            "no_match_count": 2,
        }

        result = task._build_result()
        assert result["images_processed"] == 3
        assert result["candidate_count"] == 1
        assert result["candidates"] == [{"imageUuid": "img-1"}]
        assert result["images_failed"] == 1
        assert result["no_match_count"] == 2

    async def test_resume_data_round_trips_into_init(self):
        resume_data = {
            "images_processed": 2,
            "candidates": [{"imageUuid": "img-1"}],
            "processed_image_uuids": ["img-1"],
            "images_failed": 0,
            "sample_errors": [],
            "no_match_count": 1,
        }
        task = _make_task(_base_params(image_uuids=["img-1", "img-2"]))
        await task._init(task._params, resume_data)
        assert task._images_processed == 2
        assert task._candidates == [{"imageUuid": "img-1"}]
        assert task._processed_image_uuids == {"img-1"}
        assert task._no_match_count == 1


class TestLoadImageBytesResolvesFilesRoot:
    async def test_relative_filename_resolved_against_files_root(self, tmp_path, monkeypatch):
        import apps.tasks_server.tasks.PreviewClassifierMatchTask as mod

        monkeypatch.setattr(mod, "FILES_ROOT", tmp_path)
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "img.jpg").write_bytes(b"fake-jpeg-bytes")

        task = _make_task(_base_params())
        await task._init(task._params, None)
        result = await task._load_image_bytes("subdir/img.jpg", images_api=MagicMock())
        assert result == b"fake-jpeg-bytes"

    async def test_path_escaping_files_root_raises(self, tmp_path, monkeypatch):
        import apps.tasks_server.tasks.PreviewClassifierMatchTask as mod

        monkeypatch.setattr(mod, "FILES_ROOT", tmp_path)
        task = _make_task(_base_params())
        await task._init(task._params, None)
        with pytest.raises(FileNotFoundError):
            await task._load_image_bytes("../../etc/passwd", images_api=MagicMock())
