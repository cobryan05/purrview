"""Tests for RunClassifierTask.

Tests directly instantiate the task and call _init/_run using mocked DB and
ONNX session — no actual model file or running server needed.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest

from apps.tasks_server.tasks.RunClassifierTask import (
    RunClassifierTask,
    _preprocess_image,
    _softmax,
)

_MOCK_MANAGER = MagicMock()
_MOCK_MANAGER.update_task_heartbeat = AsyncMock()

_BINARY_CONFIG = {"input_size": [224, 224], "class_names": ["absent", "present"], "model_kind": "image_classifier"}


def _make_task(params: dict) -> RunClassifierTask:
    return RunClassifierTask(
        task_config_uuid="cfg-uuid",
        params=params,
        manager=_MOCK_MANAGER,
    )


def _write_model_files(tmp_path: Path, model_name: str, label_uuid: str = "label-uuid-1", extra: dict = {}) -> None:
    (tmp_path / f"{model_name}.onnx").write_bytes(b"fake")
    config = {**_BINARY_CONFIG, "label_uuid": label_uuid, **extra}
    (tmp_path / f"{model_name}.json").write_text(json.dumps(config))


class TestParamsSchema:
    def test_required_params_present(self):
        schema = RunClassifierTask.params_schema()
        assert "model_name" in schema
        assert "confidence_threshold" in schema
        assert "source_uuids" in schema

    def test_tag_params_removed(self):
        schema = RunClassifierTask.params_schema()
        assert "present_tag_uuid" not in schema
        assert "absent_tag_uuid" not in schema


class TestPreprocessImage:
    def test_returns_correct_shape(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img_bytes = cv2.imencode(".jpg", img)[1].tobytes()
        result = _preprocess_image(img_bytes, (224, 224))
        assert result is not None
        assert result.shape == (3, 224, 224)
        assert result.dtype == np.float32

    def test_invalid_bytes_returns_none(self):
        result = _preprocess_image(b"not an image", (224, 224))
        assert result is None

    def test_normalises_channels(self):
        img = np.ones((10, 10, 3), dtype=np.uint8) * 255
        img_bytes = cv2.imencode(".png", img)[1].tobytes()
        result = _preprocess_image(img_bytes, (224, 224))
        assert result is not None
        # After normalization, channel 0 should be (1.0 - 0.485) / 0.229 ≈ 2.249
        assert abs(result[0, 0, 0] - (1.0 - 0.485) / 0.229) < 0.05


class TestSoftmax:
    def test_probabilities_sum_to_one(self):
        x = np.array([[1.0, 2.0, 3.0]])
        probs = _softmax(x, axis=1)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_higher_logit_gets_higher_probability(self):
        x = np.array([[1.0, 5.0]])
        probs = _softmax(x, axis=1)
        assert probs[0, 1] > probs[0, 0]


class TestInit:
    async def test_init_sets_model_name(self, tmp_path):
        _write_model_files(tmp_path, "test_clf", label_uuid="label-abc")
        task = _make_task({"model_name": "test_clf", "confidence_threshold": 0.8})
        with patch("apps.tasks_server.tasks.RunClassifierTask._MODELS_ROOT", tmp_path):
            await task._init(task._params, None)

        assert task._model_name == "test_clf"
        assert task._confidence_threshold == pytest.approx(0.8)
        assert task._label_uuid == "label-abc"

    async def test_init_raises_if_model_missing(self, tmp_path):
        task = _make_task({"model_name": "nonexistent"})
        with patch("apps.tasks_server.tasks.RunClassifierTask._MODELS_ROOT", tmp_path):
            with pytest.raises(FileNotFoundError):
                await task._init(task._params, None)

    async def test_init_raises_if_wrong_model_kind(self, tmp_path):
        model_name = "multi_clf"
        (tmp_path / f"{model_name}.onnx").write_bytes(b"fake")
        (tmp_path / f"{model_name}.json").write_text(
            json.dumps({"model_kind": "multi_label_classifier", "input_size": [224, 224], "label_uuids": ["a", "b"]})
        )
        task = _make_task({"model_name": model_name})
        with patch("apps.tasks_server.tasks.RunClassifierTask._MODELS_ROOT", tmp_path):
            with pytest.raises(ValueError, match="multi_label_classifier"):
                await task._init(task._params, None)

    @pytest.mark.parametrize("bad_name", ["../secret", "sub/model", "..", "/abs/model", "a/../../etc/passwd"])
    async def test_init_rejects_model_name_traversal(self, tmp_path, bad_name):
        """model_name is joined onto _MODELS_ROOT; a name with path separators or
        '..' could read arbitrary .onnx/.json files outside the models root."""
        task = _make_task({"model_name": bad_name})
        with patch("apps.tasks_server.tasks.RunClassifierTask._MODELS_ROOT", tmp_path):
            with pytest.raises(ValueError):
                await task._init(task._params, None)

    async def test_init_raises_if_label_uuid_missing(self, tmp_path):
        model_name = "clf"
        (tmp_path / f"{model_name}.onnx").write_bytes(b"fake")
        (tmp_path / f"{model_name}.json").write_text(
            json.dumps({"model_kind": "image_classifier", "input_size": [224, 224]})
        )
        task = _make_task({"model_name": model_name})
        with patch("apps.tasks_server.tasks.RunClassifierTask._MODELS_ROOT", tmp_path):
            with pytest.raises(ValueError, match="label_uuid"):
                await task._init(task._params, None)

    async def test_init_resumes_processed_count(self, tmp_path):
        _write_model_files(tmp_path, "clf")
        task = _make_task({"model_name": "clf"})
        with patch("apps.tasks_server.tasks.RunClassifierTask._MODELS_ROOT", tmp_path):
            await task._init(task._params, {"images_processed": 10, "images_labeled": 3})

        assert task._images_processed == 10
        assert task._images_labeled == 3


class TestRun:
    """Tests for _run using mocked ONNX session and DB API."""

    def _make_stub(self, uuid: str, filename: str):
        stub = MagicMock()
        stub.uuid = uuid
        stub.filename = filename
        return stub

    def _make_tiny_jpeg(self) -> bytes:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        return buf.tobytes()

    async def test_run_classifies_images_and_writes_present_label(self, tmp_path):
        _write_model_files(tmp_path, "clf", label_uuid="label-cat")

        img_bytes = self._make_tiny_jpeg()
        img_a = tmp_path / "a.jpg"
        img_a.write_bytes(img_bytes)
        img_b = tmp_path / "b.jpg"
        img_b.write_bytes(img_bytes)

        stubs = [
            self._make_stub("uuid-a", str(img_a)),
            self._make_stub("uuid-b", str(img_b)),
        ]

        mock_images_api = MagicMock()
        mock_images_api.list_image_stubs.return_value = MagicMock(stubs=stubs)
        mock_images_api.set_image_label_status.return_value = MagicMock()

        # High confidence for "present" (index 1).
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.array([[0.1, 5.0], [0.1, 5.0]])]

        task = _make_task(
            {"model_name": "clf", "confidence_threshold": 0.7, "batch_size": 10, "image_source": "filesystem"}
        )

        with (
            patch("apps.tasks_server.tasks.RunClassifierTask._MODELS_ROOT", tmp_path),
            patch("apps.tasks_server.tasks.RunClassifierTask.ImagesApi", return_value=mock_images_api),
            patch("apps.tasks_server.tasks.RunClassifierTask.SourcesApi"),
            patch("onnxruntime.InferenceSession", return_value=mock_session),
        ):
            await task._init(task._params, None)
            result = await task._run()

        assert result["images_processed"] == 2
        assert result["images_labeled"] == 2

        calls = mock_images_api.set_image_label_status.call_args_list
        assert len(calls) == 2
        for call in calls:
            payload = call.args[0]
            assert payload.label_uuid == "label-cat"
            assert payload.status.value == "present"

    async def test_run_below_threshold_writes_absent_label(self, tmp_path):
        _write_model_files(tmp_path, "clf", label_uuid="label-cat")

        img_bytes = self._make_tiny_jpeg()
        img_a = tmp_path / "a.jpg"
        img_a.write_bytes(img_bytes)
        stubs = [self._make_stub("uuid-a", str(img_a))]

        mock_images_api = MagicMock()
        mock_images_api.list_image_stubs.return_value = MagicMock(stubs=stubs)
        mock_images_api.set_image_label_status.return_value = MagicMock()

        # Low confidence for "present" → should write ABSENT.
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.array([[5.0, 0.1]])]

        task = _make_task(
            {"model_name": "clf", "confidence_threshold": 0.7, "batch_size": 10, "image_source": "filesystem"}
        )

        with (
            patch("apps.tasks_server.tasks.RunClassifierTask._MODELS_ROOT", tmp_path),
            patch("apps.tasks_server.tasks.RunClassifierTask.ImagesApi", return_value=mock_images_api),
            patch("apps.tasks_server.tasks.RunClassifierTask.SourcesApi"),
            patch("onnxruntime.InferenceSession", return_value=mock_session),
        ):
            await task._init(task._params, None)
            result = await task._run()

        assert result["images_labeled"] == 1
        payload = mock_images_api.set_image_label_status.call_args.args[0]
        assert payload.label_uuid == "label-cat"
        assert payload.status.value == "absent"

    async def test_run_cancel_stops_processing(self, tmp_path):
        _write_model_files(tmp_path, "clf")

        img_bytes = self._make_tiny_jpeg()
        stubs = []
        for i in range(5):
            p = tmp_path / f"img{i}.jpg"
            p.write_bytes(img_bytes)
            stubs.append(self._make_stub(f"uuid-{i}", str(p)))

        mock_images_api = MagicMock()
        mock_images_api.list_image_stubs.return_value = MagicMock(stubs=stubs)
        mock_images_api.set_image_label_status.return_value = MagicMock()

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.array([[0.1, 5.0]])]

        task = _make_task(
            {"model_name": "clf", "confidence_threshold": 0.5, "batch_size": 1, "image_source": "filesystem"}
        )

        original_sleep = asyncio.sleep
        call_count = 0

        async def _counting_sleep(t):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                task._cancel_flag.set()
            await original_sleep(0)

        with (
            patch("apps.tasks_server.tasks.RunClassifierTask._MODELS_ROOT", tmp_path),
            patch("apps.tasks_server.tasks.RunClassifierTask.ImagesApi", return_value=mock_images_api),
            patch("apps.tasks_server.tasks.RunClassifierTask.SourcesApi"),
            patch("onnxruntime.InferenceSession", return_value=mock_session),
            patch("apps.tasks_server.tasks.RunClassifierTask.asyncio.sleep", _counting_sleep),
        ):
            await task._init(task._params, None)
            result = await task._run()

        assert result["images_processed"] < 5

    async def test_run_skips_unreadable_images(self, tmp_path):
        _write_model_files(tmp_path, "clf")

        img_bytes = self._make_tiny_jpeg()
        valid = tmp_path / "good.jpg"
        valid.write_bytes(img_bytes)

        stubs = [
            self._make_stub("uuid-good", str(valid)),
            self._make_stub("uuid-bad", "/nonexistent/image.jpg"),
        ]

        mock_images_api = MagicMock()
        mock_images_api.list_image_stubs.return_value = MagicMock(stubs=stubs)
        mock_images_api.set_image_label_status.return_value = MagicMock()

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.array([[0.1, 5.0]])]

        task = _make_task(
            {"model_name": "clf", "confidence_threshold": 0.5, "batch_size": 10, "image_source": "filesystem"}
        )

        with (
            patch("apps.tasks_server.tasks.RunClassifierTask._MODELS_ROOT", tmp_path),
            patch("apps.tasks_server.tasks.RunClassifierTask.ImagesApi", return_value=mock_images_api),
            patch("apps.tasks_server.tasks.RunClassifierTask.SourcesApi"),
            patch("onnxruntime.InferenceSession", return_value=mock_session),
        ):
            await task._init(task._params, None)
            result = await task._run()

        # Only the valid image should be labeled.
        assert result["images_labeled"] == 1
