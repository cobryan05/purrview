import numpy as np
import pytest

from apps.helpers.types import InferenceResult
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


@pytest.mark.parametrize(
    "top_k",
    [-5, 100000, {"yolo_test": -1}, {"yolo_test": 33}],
    ids=["negative-scalar", "huge-scalar", "negative-in-dict", "over-limit-in-dict"],
)
async def test_detect_batch_rejects_an_out_of_range_top_k(http_client, top_k):
    """The [0, 32] bound the single-image endpoint enforces applies to both
    shapes of the batch endpoint's top_k. Widening the field to a union dropped
    the Field(ge/le) bounds, which Pydantic cannot apply inside a dict anyway —
    an unbounded ask reached the ONNX top-k/NMS selection unchecked."""
    resp = await http_client.post(
        "/api/models/detect-batch",
        json={
            "confThresh": 0.5,
            "imageBase64": make_png_b64(),
            "modelNames": ["yolo_test"],
            "topK": top_k,
        },
    )
    assert resp.status_code == 422


async def test_detect_batch_accepts_a_top_k_at_the_limit(http_client):
    resp = await http_client.post(
        "/api/models/detect-batch",
        json={
            "confThresh": 0.5,
            "imageBase64": make_png_b64(),
            "modelNames": ["yolo_test"],
            "topK": {"yolo_test": 32},
        },
    )
    assert resp.status_code == 200


async def test_detect_batch_per_model_top_k_dict(manager, monkeypatch):
    """A dict top_k gives each model its own ask; a model missing from the
    dict gets 0 — today's scalar default — rather than inheriting a sibling's
    value or raising."""
    captured: dict = {}

    async def fake_recognize(model_name, image, conf_thresh, return_annotated_img=False, pin_id=None, **kwargs):
        captured[model_name] = kwargs.get("top_k")
        return InferenceResult(source_width=1, source_height=1, detections=[], inference_time=0.0)

    monkeypatch.setattr(manager, "recognize", fake_recognize)

    await manager.recognize_many(
        model_names=["yolo_test", "yolo_test2"],
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        conf_thresh=0.5,
        top_k={"yolo_test": 3},
    )

    assert captured == {"yolo_test": 3, "yolo_test2": 0}


async def test_detect_batch_scalar_top_k_applies_to_every_model(manager, monkeypatch):
    """The plain-int form is unchanged: every model in the stack gets it."""
    captured: dict = {}

    async def fake_recognize(model_name, image, conf_thresh, return_annotated_img=False, pin_id=None, **kwargs):
        captured[model_name] = kwargs.get("top_k")
        return InferenceResult(source_width=1, source_height=1, detections=[], inference_time=0.0)

    monkeypatch.setattr(manager, "recognize", fake_recognize)

    await manager.recognize_many(
        model_names=["yolo_test", "yolo_test2"],
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        conf_thresh=0.5,
        top_k=5,
    )

    assert captured == {"yolo_test": 5, "yolo_test2": 5}


async def test_detect_batch_route_is_not_shadowed_by_the_model_name_route(http_client):
    """/api/models/{model_name}/detect would happily match a model called
    "detect-batch"; the batch route is declared first so it cannot."""
    resp = await http_client.post(
        "/api/models/detect-batch",
        json={"confThresh": 0.5, "imageBase64": make_png_b64(), "modelNames": ["yolo_test"]},
    )
    assert resp.status_code == 200
    assert "results" in resp.json()
