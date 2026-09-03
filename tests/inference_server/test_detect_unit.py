"""Running a hierarchical model config as ONE model.

`detect-unit` is the only place the cascade runs, so these tests pin the behaviour every
caller now inherits: the SPA's inference panel, the watcher and the review preview all
hand over a config uuid and expect detections that are already as specific as the
hierarchy can make them.

The failures worth guarding are all silent ones — a second pass that does not run, or
runs in the wrong order, produces a *plausible* coarser label ("Cat" instead of "Willow")
and nothing anywhere errors.
"""

import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from apps.helpers.types import BoundingBoxMetadata, DetectionResult, InferenceResult
from apps.inference_server.manager import Manager
from apps.inference_server.web import WebApp
from tests.conftest import make_png

CFG = "cfg-hierarchy"
CAT = "label-cat"
WILLOW = "label-willow"
DOG = "label-dog"
ANIMAL = "label-animal"


def _png_b64() -> str:
    return base64.b64encode(make_png()).decode()


def _detection(label_uuid: str, confidence: float) -> DetectionResult:
    return DetectionResult(
        bbox=BoundingBoxMetadata(
            uuid="bbox-1",
            x=0.1,
            y=0.1,
            width=0.2,
            height=0.2,
            label_uuid=label_uuid,
            class_str="Cat",
        ),
        confidence=confidence,
    )


def _write(models_dir, name: str, sidecar: dict) -> None:
    (models_dir / f"{name}.onnx").write_bytes(b"fake onnx")
    (models_dir / f"{name}.json").write_text(json.dumps(sidecar))


def _top_sidecar() -> dict:
    return {
        "yolo": 8,
        "classes": ["cat", "dog"],
        "class_map": {},
        "model_kind": "object_detection",
        "model_config_uuid": CFG,
        "role": "top",
        "model_config_node_uuid": "node-root",
    }


def _sub_sidecar(parent: str, classes: list, node: str) -> dict:
    return {
        "classes": ["a", "b"],
        "class_map": {},
        "model_kind": "crop_classifier",
        "model_config_uuid": CFG,
        "role": "submodel",
        "model_config_node_uuid": node,
        "parent_label_uuid": parent,
        "label_uuids": classes,
    }


@pytest.fixture
def crops():
    """Per-model crop predictions the fake classifier will return, keyed by model name."""
    return {}


@pytest.fixture
def provider(crops):
    prov = MagicMock()
    prov.processImage = AsyncMock(
        return_value=InferenceResult(
            source_width=100,
            source_height=100,
            detections=[_detection(CAT, 0.8)],
            inference_time=0.01,
        )
    )
    return prov


@pytest.fixture
def make_client(tmp_path, provider, crops, monkeypatch):
    """Build a client over a models dir the test populates itself.

    A factory rather than a fixture with fixed contents: what makes a unit interesting
    is exactly *which* artifacts exist, and half these tests are about one being absent.
    """

    def _build(**sidecars):
        for name, sidecar in sidecars.items():
            _write(tmp_path, name, sidecar)

        def _load(path):
            name = path.split("/")[-1].rsplit(".", 1)[0]
            prov = MagicMock()
            prov.processImage = provider.processImage
            prov.processCrops = AsyncMock(return_value=crops.get(name, []))
            return prov

        monkeypatch.setattr("apps.inference_server.manager.load_yolo_onnx", _load)
        return tmp_path

    return _build


async def _post(models_dir, payload: dict) -> dict:
    manager = Manager(models_path=str(models_dir))
    await manager.list_models()
    web_app = WebApp(app_name="test", manager=manager)
    transport = ASGITransport(app=web_app.app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/api/models/detect-unit", json=payload)
    assert r.status_code == 200, r.text
    return r.json()


def _body(**kw) -> dict:
    return {"modelConfigUuid": CFG, "imageBase64": _png_b64(), "confThresh": 0.3, **kw}


async def test_units_listing_groups_artifacts_by_config(make_client):
    """What lets a picker show one entry per hierarchy. Two configs share a models
    directory, and each must list only its own parts — the sidecars sit side by side
    with nothing but the config uuid separating them."""
    other_top = dict(_top_sidecar(), model_config_uuid="cfg-other")
    models_dir = make_client(
        top_model=_top_sidecar(),
        cat_namer=_sub_sidecar(CAT, [WILLOW], "node-cat"),
        other_model=other_top,
    )

    manager = Manager(models_path=str(models_dir))
    await manager.list_models()
    web_app = WebApp(app_name="test", manager=manager)
    transport = ASGITransport(app=web_app.app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/models/units")
    body = r.json()

    by_uuid = {u["modelConfigUuid"]: u for u in body["units"]}
    assert by_uuid[CFG]["detectorNames"] == ["top_model"]
    assert by_uuid[CFG]["subModelNames"] == ["cat_namer"]
    assert by_uuid["cfg-other"]["detectorNames"] == ["other_model"]
    assert by_uuid["cfg-other"]["subModelNames"] == []


async def test_boundary_detection_is_refined_by_the_sub_model(make_client, crops):
    """The headline: one call in, a specific label out. The top model said 'cat' and the
    caller never asked for the sub-model by name."""
    crops["cat_namer"] = [[(WILLOW, 0.5)]]
    models_dir = make_client(
        top_model=_top_sidecar(),
        cat_namer=_sub_sidecar(CAT, [WILLOW], "node-cat"),
    )

    body = await _post(models_dir, _body())

    detections = body["results"]["top_model"]["detections"]
    assert detections[0]["bbox"]["labelUuid"] == WILLOW
    # Confidence is the PRODUCT of both passes: 0.8 that it is a cat, 0.5 that this cat
    # is Willow. Reporting the classifier's 0.5 alone would let a weak detection come back
    # looking strong, and every downstream confidence band reads this number.
    assert detections[0]["confidence"] == pytest.approx(0.4)
    assert body["subModelNames"] == ["cat_namer"]


async def test_detection_outside_the_boundary_is_left_alone(make_client, crops, provider):
    """A dog is not a cat: the sub-model must not be asked about it, and must not
    relabel it. Without this the cascade would rewrite every box on the frame."""
    provider.processImage = AsyncMock(
        return_value=InferenceResult(
            source_width=100, source_height=100, detections=[_detection(DOG, 0.9)], inference_time=0.01
        )
    )
    crops["cat_namer"] = [[(WILLOW, 0.99)]]
    models_dir = make_client(top_model=_top_sidecar(), cat_namer=_sub_sidecar(CAT, [WILLOW], "node-cat"))

    body = await _post(models_dir, _body())

    detections = body["results"]["top_model"]["detections"]
    assert detections[0]["bbox"]["labelUuid"] == DOG
    assert detections[0]["confidence"] == pytest.approx(0.9)


async def test_untrained_sub_model_degrades_visibly_not_silently(make_client):
    """The failure that started this feature: the top model trained, the sub-model never
    did, and detections stopped at 'Cat' with nothing anywhere to point at.

    The detections legitimately stay coarse — that IS the answer without a sub-model —
    so what has to hold is that the response says which parts ran, letting the caller
    (which holds the config) see that a part is missing.
    """
    models_dir = make_client(top_model=_top_sidecar())

    body = await _post(models_dir, _body())

    assert body["results"]["top_model"]["detections"][0]["bbox"]["labelUuid"] == CAT
    assert body["subModelNames"] == []
    assert [p["modelName"] for p in body["parts"]] == ["top_model"]


async def test_a_failing_sub_model_keeps_the_first_pass_detections(make_client, crops, monkeypatch):
    """A broken second pass costs its own refinement, not the boxes already found —
    and says so, rather than looking like a hierarchy that had nothing to add."""
    models_dir = make_client(top_model=_top_sidecar(), cat_namer=_sub_sidecar(CAT, [WILLOW], "node-cat"))

    async def _boom(*a, **kw):
        raise RuntimeError("onnx session died")

    monkeypatch.setattr(Manager, "classify_crops", _boom)

    body = await _post(models_dir, _body())

    assert body["results"]["top_model"]["detections"][0]["bbox"]["labelUuid"] == CAT
    assert any("cat_namer" in n for n in body["notes"])


async def test_a_sub_model_with_no_parent_label_is_reported_and_not_run(make_client):
    """Left in the detector list it would be sent to the detection endpoint, which
    answers with meaningless boxes instead of failing."""
    orphan = _sub_sidecar(CAT, [WILLOW], "node-cat")
    orphan.pop("parent_label_uuid")
    models_dir = make_client(top_model=_top_sidecar(), orphan=orphan)

    body = await _post(models_dir, _body())

    assert body["subModelNames"] == []
    assert body["detectorNames"] == ["top_model"]
    assert any("orphan" in n for n in body["notes"])


async def test_nested_unit_resolves_two_levels_in_one_call(make_client, crops, provider):
    """animal -> cat -> willow, with no label tree supplied.

    Each pass reads what the previous wrote, so this only works if they run
    outermost-first — ordering derived from the artifacts alone. Run inner-first and the
    cat pass finds nothing labelled Cat yet and the answer stops at 'Cat', which looks
    exactly like a correct low-confidence result.
    """
    provider.processImage = AsyncMock(
        return_value=InferenceResult(
            source_width=100, source_height=100, detections=[_detection(ANIMAL, 1.0)], inference_time=0.01
        )
    )
    crops["z_animal_namer"] = [[(CAT, 1.0)]]
    crops["a_cat_namer"] = [[(WILLOW, 1.0)]]
    models_dir = make_client(
        top_model=_top_sidecar(),
        # Both the model names AND the node uuids are chosen so the INNER model sorts
        # first, because that is the order resolve_unit hands them over in. Pass order
        # therefore has to come from the derived hierarchy: anything that preserves the
        # incoming order runs the cat pass before anything is labelled Cat, and the
        # answer stops at "Cat" — which reads as a correct, merely unspecific result.
        z_animal_namer=_sub_sidecar(ANIMAL, [CAT], "node-z-animal"),
        a_cat_namer=_sub_sidecar(CAT, [WILLOW], "node-a-cat"),
    )

    body = await _post(models_dir, _body())

    assert body["results"]["top_model"]["detections"][0]["bbox"]["labelUuid"] == WILLOW
