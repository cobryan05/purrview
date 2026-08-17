import pytest

from tests.inference_server.conftest import make_png_b64


async def test_infer_returns_empty_detections(http_client):
    resp = await http_client.post(
        "/api/models/yolo_test/detect",
        json={"confThresh": 0.5, "imageBase64": make_png_b64()},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["result"]["detections"] == []


async def test_infer_unknown_model_returns_failure(http_client):
    resp = await http_client.post(
        "/api/models/nonexistent/detect",
        json={"confThresh": 0.5, "imageBase64": make_png_b64()},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "failure"


async def test_infer_returns_inference_time(http_client):
    resp = await http_client.post(
        "/api/models/yolo_test/detect",
        json={"confThresh": 0.5, "imageBase64": make_png_b64()},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"]["inferenceTime"] == pytest.approx(0.01, abs=1e-6)


async def test_infer_source_img_omitted_by_default(http_client):
    resp = await http_client.post(
        "/api/models/yolo_test/detect",
        json={"confThresh": 0.5, "imageBase64": make_png_b64()},
    )
    assert resp.status_code == 200
    assert resp.json()["result"]["sourceImage"] is None


async def test_infer_invalid_base64_returns_failure(http_client):
    resp = await http_client.post(
        "/api/models/yolo_test/detect",
        json={"confThresh": 0.5, "imageBase64": "not-valid-base64!!!"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "failure"


# ── /api/models/detect-batch ─────────────────────────────────────────────────
#
# One image, several models. Replaces one request per model, each carrying its
# own copy of the same image for the server to decode again.


async def test_detect_batch_returns_a_result_per_model(http_client):
    resp = await http_client.post(
        "/api/models/detect-batch",
        json={
            "confThresh": 0.5,
            "imageBase64": make_png_b64(),
            "modelNames": ["yolo_test", "yolo_test2"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert set(data["results"]) == {"yolo_test", "yolo_test2"}
    assert data["errors"] == {}


async def test_detect_batch_preserves_requested_model_order(http_client):
    """Callers merge overlapping detections first-model-wins, so the order the
    results come back in decides which model's label survives."""
    order = ["yolo_test2", "yolo_test"]
    resp = await http_client.post(
        "/api/models/detect-batch",
        json={"confThresh": 0.5, "imageBase64": make_png_b64(), "modelNames": order},
    )
    assert resp.status_code == 200
    assert list(resp.json()["results"]) == order


async def test_detect_batch_matches_the_single_model_endpoint(http_client):
    """The batch path must not change what a model detects — only how many
    times the image is decoded on the way in."""
    img = make_png_b64()
    single = await http_client.post("/api/models/yolo_test/detect", json={"confThresh": 0.5, "imageBase64": img})
    batch = await http_client.post(
        "/api/models/detect-batch",
        json={"confThresh": 0.5, "imageBase64": img, "modelNames": ["yolo_test"]},
    )

    assert single.json()["result"]["detections"] == batch.json()["results"]["yolo_test"]["detections"]


async def test_detect_batch_reports_a_bad_model_without_losing_the_others(http_client):
    """One broken model in a stack costs its own detections, not every other
    model's — a stack that silently returned nothing would be far worse."""
    resp = await http_client.post(
        "/api/models/detect-batch",
        json={
            "confThresh": 0.5,
            "imageBase64": make_png_b64(),
            "modelNames": ["yolo_test", "nonexistent"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "yolo_test" in data["results"]
    assert "nonexistent" in data["errors"]


async def test_detect_batch_rejects_an_empty_model_list(http_client):
    resp = await http_client.post(
        "/api/models/detect-batch",
        json={"confThresh": 0.5, "imageBase64": make_png_b64(), "modelNames": []},
    )
    assert resp.status_code == 422


async def test_detect_batch_route_is_not_shadowed_by_the_model_name_route(http_client):
    """/api/models/{model_name}/detect would happily match a model called
    "detect-batch"; the batch route is declared first so it cannot."""
    resp = await http_client.post(
        "/api/models/detect-batch",
        json={"confThresh": 0.5, "imageBase64": make_png_b64(), "modelNames": ["yolo_test"]},
    )
    assert resp.status_code == 200
    assert "results" in resp.json()
