import json

import pytest

from apps.inference_server.manager import Manager


async def test_list_models_empty(tmp_path):
    mgr = Manager(models_path=str(tmp_path))
    assert await mgr.list_models() == []


async def test_list_models_api(http_client):
    resp = await http_client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "yolo_test" in data["models"]


async def test_get_classes(http_client):
    resp = await http_client.get("/api/models/yolo_test/classes")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert set(data["classes"].keys()) == {"cat", "dog"}


async def test_get_classes_unknown_model_returns_empty(http_client):
    resp = await http_client.get("/api/models/nonexistent/classes")
    assert resp.status_code == 200
    data = resp.json()
    assert data["classes"] == {}


async def test_bulk_classes(http_client):
    resp = await http_client.post(
        "/api/models/classes/bulk",
        json={"modelNames": ["yolo_test"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert set(data["models"]["yolo_test"].keys()) == {"cat", "dog"}


async def test_associate_label(http_client, models_dir):
    resp = await http_client.put(
        "/api/models/yolo_test/classes/cat/label",
        json={"labelUuid": "label-uuid-abc"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["labelSet"] is True

    meta = json.loads((models_dir / "yolo_test.json").read_text())
    assert meta["class_map"]["cat"] == "label-uuid-abc"


async def test_associate_label_clears_with_null(http_client, models_dir):
    await http_client.put(
        "/api/models/yolo_test/classes/cat/label",
        json={"labelUuid": "label-uuid-abc"},
    )
    resp = await http_client.put(
        "/api/models/yolo_test/classes/cat/label",
        json={"labelUuid": None},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    meta = json.loads((models_dir / "yolo_test.json").read_text())
    assert "cat" not in meta["class_map"]


async def test_clearing_one_dynamic_label_does_not_clear_others(http_client, models_dir):
    # Simulate a model trained in-app: no explicit class_map yet, but the
    # trainer wrote positional label_uuids that get_models_classes()
    # falls back to for pre-filling the mapping UI.
    meta_path = models_dir / "yolo_test.json"
    meta = json.loads(meta_path.read_text())
    meta["label_uuids"] = ["uuid-cat", "uuid-dog"]
    meta_path.write_text(json.dumps(meta))

    # Clearing "cat" (which was only ever positionally mapped, never in
    # class_map) should not wipe out "dog"'s positional mapping too.
    resp = await http_client.put(
        "/api/models/yolo_test/classes/cat/label",
        json={"labelUuid": None},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    resp = await http_client.get("/api/models/yolo_test/classes")
    data = resp.json()
    assert data["classes"]["cat"] is None
    assert data["classes"]["dog"] == "uuid-dog"


async def test_associate_label_unknown_class_returns_failure(http_client):
    resp = await http_client.put(
        "/api/models/yolo_test/classes/horse/label",
        json={"labelUuid": "label-uuid-xyz"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "failure"


async def test_associate_label_unknown_model_returns_failure(http_client):
    resp = await http_client.put(
        "/api/models/nonexistent/classes/cat/label",
        json={"labelUuid": "label-uuid-xyz"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "failure"


async def test_pin_model_returns_pin_id(http_client):
    resp = await http_client.post(
        "/api/models/yolo_test/pin",
        json={"modelName": "yolo_test", "duration": 60},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["pinId"] is not None


async def test_pin_unknown_model_returns_failure(http_client):
    resp = await http_client.post(
        "/api/models/nonexistent/pin",
        json={"modelName": "nonexistent", "duration": 60},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "failure"


async def test_get_model_meta_defaults_device_to_auto(http_client):
    resp = await http_client.get("/api/models/yolo_test/meta")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["device"] == "auto"


async def test_set_model_device(http_client, models_dir):
    resp = await http_client.post(
        "/api/models/yolo_test/device",
        json={"device": "cpu"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["deviceSet"] is True

    meta = json.loads((models_dir / "yolo_test.json").read_text())
    assert meta["device"] == "cpu"

    resp = await http_client.get("/api/models/yolo_test/meta")
    assert resp.json()["device"] == "cpu"


async def test_set_model_device_invalid_returns_failure(http_client):
    resp = await http_client.post(
        "/api/models/yolo_test/device",
        json={"device": "tpu"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "failure"


async def test_set_model_device_unknown_model_returns_failure(http_client):
    resp = await http_client.post(
        "/api/models/nonexistent/device",
        json={"device": "gpu"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "failure"
