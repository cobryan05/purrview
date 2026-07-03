"""Tests for the /classify and /meta endpoints added to inference_server."""

import json
from unittest.mock import AsyncMock

import pytest

from apps.helpers.types import BoundingBoxMetadata, DetectionResult, InferenceResult
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


def _det(label_uuid: str, confidence: float) -> DetectionResult:
    return DetectionResult(
        bbox=BoundingBoxMetadata(
            uuid="u",
            x=0.0,
            y=0.0,
            width=224.0,
            height=224.0,
            label_uuid=label_uuid,
            class_str=label_uuid,
        ),
        confidence=confidence,
    )


def _result(*detections: DetectionResult) -> InferenceResult:
    return InferenceResult(source_width=224, source_height=224, detections=list(detections), inference_time=1.0)


@pytest.fixture
def models_dir_with_classifiers(tmp_path):
    # YOLO detector
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
    from httpx import ASGITransport, AsyncClient

    from apps.inference_server.manager import Manager
    from apps.inference_server.web import WebApp

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
    async def test_binary_classifier_high_confidence_is_present(self, clf_client, mock_provider):
        mock_provider.processImage = AsyncMock(
            return_value=_result(_det("label-cat-uuid", 0.95))
        )
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

    async def test_binary_classifier_low_confidence_is_absent(self, clf_client, mock_provider):
        mock_provider.processImage = AsyncMock(
            return_value=_result(_det("label-cat-uuid", 0.2))
        )
        resp = await clf_client.post(
            "/api/models/binary_clf/classify",
            json={"imageBase64": make_png_b64()},
        )
        assert resp.status_code == 200
        preds = resp.json()["result"]["predictions"]
        assert preds[0]["isPresent"] is False
        assert preds[0]["confidence"] < 0.5

    async def test_multi_label_classifier_returns_one_prediction_per_label(self, clf_client, mock_provider):
        mock_provider.processImage = AsyncMock(
            return_value=_result(
                _det("label-a", 0.95),
                _det("label-b", 0.2),
            )
        )
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
        resp = await clf_client.post(
            "/api/models/binary_clf/classify",
            json={"imageBase64": "not-valid-base64!!!"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "failure"
