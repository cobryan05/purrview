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


def _make_task(params: dict) -> RunClassifierTask:
    return RunClassifierTask(
        task_config_uuid="cfg-uuid",
        params=params,
        manager=_MOCK_MANAGER,
    )


class TestPreprocessImage:
    def test_returns_correct_shape(self):
        # Create a small solid-colour PNG in memory.
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
        # White image → known normalized values.
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
        # Create dummy model files.
        model_name = "test_clf"
        (tmp_path / f"{model_name}.onnx").write_bytes(b"fake")
        config = {"input_size": [224, 224], "class_names": ["absent", "present"]}
        (tmp_path / f"{model_name}.json").write_text(json.dumps(config))

        task = _make_task(
            {
                "model_name": model_name,
                "present_tag_uuid": "tag-present",
                "confidence_threshold": 0.8,
            }
        )
        with patch("apps.tasks_server.tasks.RunClassifierTask._MODELS_ROOT", tmp_path):
            await task._init(task._params, None)

        assert task._model_name == model_name
        assert task._confidence_threshold == pytest.approx(0.8)
        assert task._present_tag_uuid == "tag-present"

    async def test_init_raises_if_model_missing(self, tmp_path):
        task = _make_task({"model_name": "nonexistent"})
        with patch("apps.tasks_server.tasks.RunClassifierTask._MODELS_ROOT", tmp_path):
            with pytest.raises(FileNotFoundError):
                await task._init(task._params, None)

    async def test_init_resumes_processed_count(self, tmp_path):
        model_name = "clf"
        (tmp_path / f"{model_name}.onnx").write_bytes(b"fake")
        (tmp_path / f"{model_name}.json").write_text(json.dumps({"input_size": [224, 224]}))

        task = _make_task({"model_name": model_name})
        with patch("apps.tasks_server.tasks.RunClassifierTask._MODELS_ROOT", tmp_path):
            await task._init(task._params, {"images_processed": 10, "images_tagged": 3})

        assert task._images_processed == 10
        assert task._images_tagged == 3


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

    async def test_run_classifies_images_and_applies_present_tag(self, tmp_path):
        model_name = "clf"
        (tmp_path / f"{model_name}.onnx").write_bytes(b"fake")
        config = {"input_size": [224, 224], "class_names": ["absent", "present"]}
        (tmp_path / f"{model_name}.json").write_text(json.dumps(config))

        # Two fake image files.
        img_bytes = self._make_tiny_jpeg()
        img_a = tmp_path / "a.jpg"
        img_a.write_bytes(img_bytes)
        img_b = tmp_path / "b.jpg"
        img_b.write_bytes(img_bytes)

        stubs = [
            self._make_stub("uuid-a", str(img_a)),
            self._make_stub("uuid-b", str(img_b)),
        ]

        # Mock DB: list_image_stubs returns our stubs; set_image_tag is a no-op.
        mock_images_api = MagicMock()
        mock_images_api.list_image_stubs.return_value = MagicMock(stubs=stubs)
        mock_images_api.set_image_tag.return_value = MagicMock()

        # Mock ONNX: always returns high confidence for "present" (index 1).
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.array([[0.1, 5.0], [0.1, 5.0]])]  # high present prob

        task = _make_task(
            {
                "model_name": model_name,
                "present_tag_uuid": "tag-present-uuid",
                "confidence_threshold": 0.7,
                "batch_size": 10,
                "image_source": "filesystem",
            }
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
        assert result["images_tagged"] == 2
        # Both images should have been tagged with present_tag_uuid.
        calls = mock_images_api.set_image_tag.call_args_list
        assert len(calls) == 2
        for call in calls:
            payload = call.args[0]
            assert payload.tag_uuid == "tag-present-uuid"
            assert payload.action == "add"

    async def test_run_below_threshold_uses_absent_tag(self, tmp_path):
        model_name = "clf"
        (tmp_path / f"{model_name}.onnx").write_bytes(b"fake")
        config = {"input_size": [224, 224], "class_names": ["absent", "present"]}
        (tmp_path / f"{model_name}.json").write_text(json.dumps(config))

        img_bytes = self._make_tiny_jpeg()
        img_a = tmp_path / "a.jpg"
        img_a.write_bytes(img_bytes)
        stubs = [self._make_stub("uuid-a", str(img_a))]

        mock_images_api = MagicMock()
        mock_images_api.list_image_stubs.return_value = MagicMock(stubs=stubs)
        mock_images_api.set_image_tag.return_value = MagicMock()

        # Low confidence for "present" → should get absent tag.
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.array([[5.0, 0.1]])]  # low present prob

        task = _make_task(
            {
                "model_name": model_name,
                "present_tag_uuid": "tag-present",
                "absent_tag_uuid": "tag-absent",
                "confidence_threshold": 0.7,
                "batch_size": 10,
                "image_source": "filesystem",
            }
        )

        with (
            patch("apps.tasks_server.tasks.RunClassifierTask._MODELS_ROOT", tmp_path),
            patch("apps.tasks_server.tasks.RunClassifierTask.ImagesApi", return_value=mock_images_api),
            patch("apps.tasks_server.tasks.RunClassifierTask.SourcesApi"),
            patch("onnxruntime.InferenceSession", return_value=mock_session),
        ):
            await task._init(task._params, None)
            result = await task._run()

        assert result["images_tagged"] == 1
        payload = mock_images_api.set_image_tag.call_args.args[0]
        assert payload.tag_uuid == "tag-absent"

    async def test_run_cancel_stops_processing(self, tmp_path):
        model_name = "clf"
        (tmp_path / f"{model_name}.onnx").write_bytes(b"fake")
        (tmp_path / f"{model_name}.json").write_text(json.dumps({"input_size": [224, 224]}))

        # Create 5 fake images.
        img_bytes = self._make_tiny_jpeg()
        stubs = []
        for i in range(5):
            p = tmp_path / f"img{i}.jpg"
            p.write_bytes(img_bytes)
            stubs.append(self._make_stub(f"uuid-{i}", str(p)))

        mock_images_api = MagicMock()
        mock_images_api.list_image_stubs.return_value = MagicMock(stubs=stubs)
        mock_images_api.set_image_tag.return_value = MagicMock()

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.array([[0.1, 5.0]])]  # 1 item batches

        task = _make_task(
            {
                "model_name": model_name,
                "present_tag_uuid": "tag-p",
                "confidence_threshold": 0.5,
                "batch_size": 1,  # one image per batch so cancel fires quickly
                "image_source": "filesystem",
            }
        )

        # Cancel after the first batch.
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

        # Should have processed fewer than all 5 images.
        assert result["images_processed"] < 5

    async def test_run_skips_unreadable_images(self, tmp_path):
        model_name = "clf"
        (tmp_path / f"{model_name}.onnx").write_bytes(b"fake")
        (tmp_path / f"{model_name}.json").write_text(json.dumps({"input_size": [224, 224]}))

        # One valid image and one path that doesn't exist.
        img_bytes = self._make_tiny_jpeg()
        valid = tmp_path / "good.jpg"
        valid.write_bytes(img_bytes)

        stubs = [
            self._make_stub("uuid-good", str(valid)),
            self._make_stub("uuid-bad", "/nonexistent/image.jpg"),
        ]

        mock_images_api = MagicMock()
        mock_images_api.list_image_stubs.return_value = MagicMock(stubs=stubs)
        mock_images_api.set_image_tag.return_value = MagicMock()

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.array([[0.1, 5.0]])]  # only valid image in batch

        task = _make_task(
            {
                "model_name": model_name,
                "present_tag_uuid": "tag-p",
                "confidence_threshold": 0.5,
                "batch_size": 10,
                "image_source": "filesystem",
            }
        )

        with (
            patch("apps.tasks_server.tasks.RunClassifierTask._MODELS_ROOT", tmp_path),
            patch("apps.tasks_server.tasks.RunClassifierTask.ImagesApi", return_value=mock_images_api),
            patch("apps.tasks_server.tasks.RunClassifierTask.SourcesApi"),
            patch("onnxruntime.InferenceSession", return_value=mock_session),
        ):
            await task._init(task._params, None)
            result = await task._run()

        # Only 1 image should be tagged (the valid one).
        assert result["images_tagged"] == 1
