"""Tests for Manager.import_model_upload and the /api/models/import/upload,
/api/models/curated endpoints (Phase 5 — generic model import, cold-start).

Builds a tiny real ONNX fixture in-process via the `onnx` package's helper API
so `get_model_metadata` can genuinely parse `names`/`description` metadata —
no network access involved.
"""

import base64
import json

import onnx
import pytest
from onnx import TensorProto, helper

from apps.helpers.curatedModels import CURATED_BASE_MODELS
from apps.inference_server.manager import Manager


def _build_onnx_bytes(class_names: dict, description: str = "Ultralytics YOLOv8n model") -> bytes:
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph)
    model.metadata_props.append(onnx.StringStringEntryProto(key="description", value=description))
    model.metadata_props.append(onnx.StringStringEntryProto(key="names", value=str(class_names)))
    return model.SerializeToString()


_ONNX_BYTES = _build_onnx_bytes({0: "cat", 1: "dog"})


class TestImportModelUploadManager:
    async def test_writes_onnx_and_sidecar_with_parsed_classes(self, tmp_path):
        mgr = Manager(models_path=str(tmp_path))
        result = await mgr.import_model_upload(model_name="uploaded", onnx_bytes=_ONNX_BYTES)

        onnx_path = tmp_path / "uploaded.onnx"
        sidecar_path = tmp_path / "uploaded.json"
        assert onnx_path.exists()
        assert onnx_path.read_bytes() == _ONNX_BYTES
        assert sidecar_path.exists()

        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["classes"] == ["cat", "dog"]
        assert sidecar["device"] == "auto"
        assert result == {"model_name": "uploaded", "classes": ["cat", "dog"]}

    async def test_appears_in_list_models(self, tmp_path):
        mgr = Manager(models_path=str(tmp_path))
        await mgr.import_model_upload(model_name="uploaded2", onnx_bytes=_ONNX_BYTES)
        assert "uploaded2" in await mgr.list_models()

    async def test_overwrite_false_existing_raises(self, tmp_path):
        mgr = Manager(models_path=str(tmp_path))
        await mgr.import_model_upload(model_name="dup", onnx_bytes=_ONNX_BYTES)
        with pytest.raises(FileExistsError):
            await mgr.import_model_upload(model_name="dup", onnx_bytes=_ONNX_BYTES, overwrite=False)

    async def test_overwrite_true_replaces_and_drops_stale_classes(self, tmp_path):
        mgr = Manager(models_path=str(tmp_path))
        await mgr.import_model_upload(model_name="dup2", onnx_bytes=_ONNX_BYTES)

        # Manually taint the sidecar the way a class-label association would.
        sidecar_path = tmp_path / "dup2.json"
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["class_map"] = {"cat": "some-label-uuid"}
        sidecar_path.write_text(json.dumps(sidecar))

        new_onnx = _build_onnx_bytes({0: "horse", 1: "sheep", 2: "cow"})
        result = await mgr.import_model_upload(model_name="dup2", onnx_bytes=new_onnx, overwrite=True)

        new_sidecar = json.loads(sidecar_path.read_text())
        assert new_sidecar["classes"] == ["horse", "sheep", "cow"]
        # Stale class_map must not survive — sidecar was dropped and regenerated fresh,
        # so class_map is simply absent now (get_models_classes treats missing as {}).
        assert new_sidecar.get("class_map", {}) == {}
        assert result["classes"] == ["horse", "sheep", "cow"]
        assert (tmp_path / "dup2.onnx").read_bytes() == new_onnx

    async def test_invalid_model_name_rejected(self, tmp_path):
        mgr = Manager(models_path=str(tmp_path))
        for bad_name in ["../escape", "a/b", "a\\b", "..", ""]:
            with pytest.raises(ValueError):
                await mgr.import_model_upload(model_name=bad_name, onnx_bytes=_ONNX_BYTES)

    async def test_garbage_onnx_bytes_rejected_without_touching_disk(self, tmp_path):
        mgr = Manager(models_path=str(tmp_path))
        with pytest.raises(ValueError):
            await mgr.import_model_upload(model_name="garbage", onnx_bytes=b"not an onnx file at all")

        assert not (tmp_path / "garbage.onnx").exists()
        assert not (tmp_path / "garbage.json").exists()

    async def test_garbage_onnx_bytes_on_overwrite_does_not_clobber_existing(self, tmp_path):
        mgr = Manager(models_path=str(tmp_path))
        await mgr.import_model_upload(model_name="good", onnx_bytes=_ONNX_BYTES)

        with pytest.raises(ValueError):
            await mgr.import_model_upload(model_name="good", onnx_bytes=b"garbage", overwrite=True)

        # The previously-written good model + sidecar must survive the failed overwrite.
        assert (tmp_path / "good.onnx").read_bytes() == _ONNX_BYTES
        sidecar = json.loads((tmp_path / "good.json").read_text())
        assert sidecar["classes"] == ["cat", "dog"]


class TestImportModelUploadEndpoint:
    async def test_upload_endpoint_success(self, http_client):
        resp = await http_client.post(
            "/api/models/import/upload",
            json={
                "modelName": "http_uploaded",
                "onnxBase64": base64.b64encode(_ONNX_BYTES).decode(),
                "overwrite": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["modelName"] == "http_uploaded"
        assert set(data["classes"]) == {"cat", "dog"}

    async def test_upload_endpoint_invalid_base64_returns_failure_not_500(self, http_client):
        resp = await http_client.post(
            "/api/models/import/upload",
            json={"modelName": "bad_b64", "onnxBase64": "not-valid-base64!!", "overwrite": False},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "failure"

    async def test_upload_endpoint_valid_base64_garbage_onnx_returns_failure_not_500(self, http_client):
        resp = await http_client.post(
            "/api/models/import/upload",
            json={
                "modelName": "garbage_http",
                "onnxBase64": base64.b64encode(b"not an onnx file").decode(),
                "overwrite": False,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "failure"

    async def test_upload_endpoint_duplicate_without_overwrite_returns_failure(self, http_client):
        payload = {
            "modelName": "dupe_http",
            "onnxBase64": base64.b64encode(_ONNX_BYTES).decode(),
            "overwrite": False,
        }
        resp1 = await http_client.post("/api/models/import/upload", json=payload)
        assert resp1.json()["status"] == "success"

        resp2 = await http_client.post("/api/models/import/upload", json=payload)
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "failure"


class TestCuratedModelsEndpoint:
    async def test_lists_curated_models_without_internal_fields(self, http_client):
        resp = await http_client.get("/api/models/curated")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert len(data["models"]) == len(CURATED_BASE_MODELS)
        for m in data["models"]:
            assert set(m.keys()) == {"key", "displayName"}
