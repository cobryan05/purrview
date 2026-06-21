"""Tests for the /classify and /meta endpoints added to inference_server."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.inference_server.conftest import make_png_b64

_BINARY_CONFIG = {
    "model_kind": "image_classifier",
    "input_size": [224, 224],
    "class_names": ["absent", "present"],
    "label_uuid": "label-cat-uuid",
}

_MULTI_CONFIG = {
    "model_kind": "multi_label_classifier",
    "input_size": [224, 224],
    "label_uuids": ["label-a", "label-b"],
}

_YOLO_CONFIG = {
    "yolo": 8,
    "classes": ["cat"],
    "class_map": {},
}


@pytest.fixture
def models_dir_with_classifiers(tmp_path):
    # YOLO detector (from base conftest)
    (tmp_path / "yolo_test.onnx").write_bytes(b"fake onnx")
    (tmp_path / "yolo_test.json").write_text(json.dumps(_YOLO_CONFIG))
    # Binary classifier
    (tmp_path / "binary_clf.onnx").write_bytes(b"fake onnx")
    (tmp_path / "binary_clf.json").write_text(json.dumps(_BINARY_CONFIG))
    # Multi-label classifier
    (tmp_path / "multi_clf.onnx").write_bytes(b"fake onnx")
    (tmp_path / "multi_clf.json").write_text(json.dumps(_MULTI_CONFIG))
    return tmp_path


@pytest.fixture
async def clf_client(models_dir_with_classifiers, mock_provider, monkeypatch):
    from apps.inference_server.manager import Manager
    from apps.inference_server.web import WebApp
    from httpx import ASGITransport, AsyncClient

    monkeypatch.setattr("apps.inference_server.manager.load_yolo_onnx", lambda path: mock_provider)
    mgr = Manager(models_path=str(models_dir_with_classifiers))
    await mgr.list_models()
    web_app = WebApp(app_name="test", manager=mgr)
    transport = ASGITransport(app=web_app.app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestModelMeta:
    async def test_yolo_model_returns_detector_kind(self, clf_client):
        resp = await clf_client.get("/api/models/yolo_test/meta")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["modelKind"] == "detector"

    async def test_binary_classifier_returns_correct_kind(self, clf_client):
        resp = await clf_client.get("/api/models/binary_clf/meta")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["modelKind"] == "image_classifier"
        assert data["labelUuid"] == "label-cat-uuid"

    async def test_multi_label_classifier_returns_label_uuids(self, clf_client):
        resp = await clf_client.get("/api/models/multi_clf/meta")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["modelKind"] == "multi_label_classifier"
        assert data["labelUuids"] == ["label-a", "label-b"]

    async def test_unknown_model_returns_failure(self, clf_client):
        resp = await clf_client.get("/api/models/nonexistent/meta")
        assert resp.status_code == 200
        assert resp.json()["status"] == "failure"


class TestClassify:
    def _make_mock_session(self, logits: list[float]):
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(name="input")]
        session.run.return_value = [np.array([logits], dtype=np.float32)]
        return session

    async def test_binary_classifier_high_confidence_is_present(self, clf_client):
        mock_session = self._make_mock_session([0.1, 5.0])  # high logit for "present"
        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            resp = await clf_client.post(
                "/api/models/binary_clf/classify",
                json={"imageBase64": make_png_b64()},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        preds = data["result"]["predictions"]
        assert len(preds) == 1
        assert preds[0]["labelUuid"] == "label-cat-uuid"
        assert preds[0]["isPresent"] is True
        assert preds[0]["confidence"] > 0.5

    async def test_binary_classifier_low_confidence_is_absent(self, clf_client):
        mock_session = self._make_mock_session([5.0, 0.1])  # high logit for "absent"
        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            resp = await clf_client.post(
                "/api/models/binary_clf/classify",
                json={"imageBase64": make_png_b64()},
            )
        assert resp.status_code == 200
        preds = resp.json()["result"]["predictions"]
        assert preds[0]["isPresent"] is False
        assert preds[0]["confidence"] < 0.5

    async def test_multi_label_classifier_returns_one_prediction_per_label(self, clf_client):
        mock_session = self._make_mock_session([3.0, -1.0])  # high sigmoid for label-a, low for label-b
        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            resp = await clf_client.post(
                "/api/models/multi_clf/classify",
                json={"imageBase64": make_png_b64()},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        preds = data["result"]["predictions"]
        assert len(preds) == 2
        assert preds[0]["labelUuid"] == "label-a"
        assert preds[0]["isPresent"] is True
        assert preds[1]["labelUuid"] == "label-b"
        assert preds[1]["isPresent"] is False

    async def test_classify_yolo_model_returns_failure(self, clf_client):
        resp = await clf_client.post(
            "/api/models/yolo_test/classify",
            json={"imageBase64": make_png_b64()},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failure"
        assert "not a classifier" in data["message"]

    async def test_classify_unknown_model_returns_failure(self, clf_client):
        resp = await clf_client.post(
            "/api/models/nonexistent/classify",
            json={"imageBase64": make_png_b64()},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "failure"

    async def test_classify_invalid_image_returns_failure(self, clf_client):
        mock_session = self._make_mock_session([0.1, 5.0])
        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            resp = await clf_client.post(
                "/api/models/binary_clf/classify",
                json={"imageBase64": "not-valid-base64!!!"},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "failure"
