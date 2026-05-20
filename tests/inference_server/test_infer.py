import pytest

from tests.inference_server.conftest import make_png_b64


async def test_infer_returns_empty_detections(http_client):
    resp = await http_client.post(
        "/api/models/yolo_test/infer",
        json={"confThresh": 0.5, "imageBase64": make_png_b64()},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["result"]["detections"] == []


async def test_infer_unknown_model_returns_failure(http_client):
    resp = await http_client.post(
        "/api/models/nonexistent/infer",
        json={"confThresh": 0.5, "imageBase64": make_png_b64()},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "failure"


async def test_infer_returns_inference_time(http_client):
    resp = await http_client.post(
        "/api/models/yolo_test/infer",
        json={"confThresh": 0.5, "imageBase64": make_png_b64()},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"]["inferenceTime"] == pytest.approx(0.01, abs=1e-6)


async def test_infer_source_img_omitted_by_default(http_client):
    resp = await http_client.post(
        "/api/models/yolo_test/infer",
        json={"confThresh": 0.5, "imageBase64": make_png_b64()},
    )
    assert resp.status_code == 200
    assert resp.json()["result"]["sourceImage"] is None


async def test_infer_invalid_base64_returns_failure(http_client):
    resp = await http_client.post(
        "/api/models/yolo_test/infer",
        json={"confThresh": 0.5, "imageBase64": "not-valid-base64!!!"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "failure"
