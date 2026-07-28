"""Tests for PreviewClassifierMatchTask.

Exercises the candidate-building logic against a fake InferenceBackend
(returning ClassifyPrediction-like objects) injected directly into
`_process_image` — no real inference server or db_client HTTP round-trip
involved (mirrors PreviewModelRelabelTask's test pattern).
"""

from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from apps.tasks_server.tasks.PreviewClassifierMatchTask import PreviewClassifierMatchTask

_MOCK_MANAGER = MagicMock()
_MOCK_MANAGER.update_task_heartbeat = AsyncMock()


@dataclass
class _Prediction:
    label_uuid: Optional[str]
    confidence: float


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


class _FakeBackend:
    """Fake InferenceBackend — returns a fixed prediction list regardless of input."""

    def __init__(self, predictions):
        self._predictions = predictions

    async def classify(self, model_name, image_bytes):
        return self._predictions


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
    """Exercises `_process_image` directly with a fake backend, bypassing the
    HTTP/db_client plumbing and the model-kind guard in `_run()`."""

    async def _run_process_image(self, task, predictions):
        await task._init(task._params, None)
        task._load_image_bytes = AsyncMock(return_value=b"fake-image-bytes")
        backend = _FakeBackend(predictions)
        img = _image()
        await task._process_image(
            images_api=MagicMock(),
            backend=backend,
            img=img,
            label_names={"corrupt": "Corrupt"},
        )
        return task._candidates

    async def test_match_above_threshold_produces_candidate(self):
        task = _make_task(_base_params())
        predictions = [_Prediction(label_uuid="corrupt", confidence=0.9)]
        candidates = await self._run_process_image(task, predictions)
        assert len(candidates) == 1
        c = candidates[0]
        assert c["imageUuid"] == "img-1"
        assert c["imagePath"] == "/data/x.jpg"
        assert c["predictedLabelUuid"] == "corrupt"
        assert c["predictedLabelName"] == "Corrupt"
        assert c["confidence"] == 0.9

    async def test_below_threshold_excluded(self):
        task = _make_task(_base_params(conf_thresh=0.5))
        predictions = [_Prediction(label_uuid="corrupt", confidence=0.2)]
        candidates = await self._run_process_image(task, predictions)
        assert candidates == []
        assert task._no_match_count == 1

    async def test_no_predictions_excluded(self):
        task = _make_task(_base_params())
        candidates = await self._run_process_image(task, [])
        assert candidates == []
        assert task._no_match_count == 1

    async def test_unmapped_label_excluded(self):
        task = _make_task(_base_params())
        predictions = [_Prediction(label_uuid=None, confidence=0.95)]
        candidates = await self._run_process_image(task, predictions)
        assert candidates == []
        assert task._no_match_count == 1

    async def test_multi_label_classifier_picks_max_confidence(self):
        task = _make_task(_base_params())
        predictions = [
            _Prediction(label_uuid="a", confidence=0.4),
            _Prediction(label_uuid="b", confidence=0.95),
            _Prediction(label_uuid="c", confidence=0.6),
        ]
        candidates = await self._run_process_image(task, predictions)
        assert len(candidates) == 1
        assert candidates[0]["predictedLabelUuid"] == "b"
        assert candidates[0]["confidence"] == 0.95

    async def test_image_load_failure_propagates(self):
        task = _make_task(_base_params())
        await task._init(task._params, None)
        task._load_image_bytes = AsyncMock(side_effect=FileNotFoundError("/data/x.jpg"))
        backend = _FakeBackend([])
        img = _image()
        with pytest.raises(FileNotFoundError):
            await task._process_image(
                images_api=MagicMock(),
                backend=backend,
                img=img,
                label_names={"corrupt": "Corrupt"},
            )
        # _process_image itself propagates — it's the _run() batch loop (not
        # exercised here) that catches per-image failures and increments
        # _images_failed, mirroring PreviewModelRelabelTask.


class TestModelKindGuard:
    async def test_non_classifier_model_rejected(self, monkeypatch):
        import apps.tasks_server.tasks.PreviewClassifierMatchTask as mod

        monkeypatch.setattr(mod, "_read_model_kind", lambda name: "detector")
        task = _make_task(_base_params())
        await task._init(task._params, None)
        with pytest.raises(ValueError, match="only supports classifier models"):
            await task._run()


class TestResume:
    def test_build_resume_data_and_result_shapes(self):
        task = _make_task(_base_params())
        task._images_processed = 3
        task._candidates = [{"imageUuid": "img-1"}]
        task._processed_image_uuids = {"img-1", "img-2"}
        task._images_failed = 1
        task._sample_errors = ["img-3: boom"]
        task._no_match_count = 2

        resume = task._build_resume_data()
        assert resume == {
            "images_processed": 3,
            "candidates": [{"imageUuid": "img-1"}],
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
