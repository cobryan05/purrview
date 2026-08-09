import json
import os
import time

import pytest

import apps.inference_server.manager as manager_module
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


class TestGetModelTrainingImages:
    """The training page reads a previous run's image manifest through this
    endpoint, so both the manifest and the provenance that identifies which run
    produced it have to survive the round trip."""

    async def _write_sidecar(self, models_dir, extra: dict) -> None:
        sidecar = models_dir / "yolo_test.json"
        data = json.loads(sidecar.read_text())
        data.update(extra)
        sidecar.write_text(json.dumps(data))

    async def test_returns_recorded_manifest(self, http_client, models_dir):
        await self._write_sidecar(
            models_dir,
            {"train_image_uuids": ["a", "b", "c"], "val_image_uuids": ["d"]},
        )
        resp = await http_client.get("/api/models/yolo_test/training-images")
        data = resp.json()
        assert data["status"] == "success"
        # camelCase on the wire: the response inherits StatusResponse's to_camel
        # alias generator. A client reading train_image_uuids gets nothing, which
        # is exactly how the Training Data modal came to report "no UUIDs
        # recorded" for every model.
        assert data["trainImageUuids"] == ["a", "b", "c"]
        assert data["valImageUuids"] == ["d"]

    async def test_returns_run_provenance(self, http_client, models_dir):
        await self._write_sidecar(
            models_dir,
            {
                "train_image_uuids": ["a"],
                "trained_at": "2026-08-01T12:00:00+00:00",
                "model_config_node_uuid": "node-123",
            },
        )
        resp = await http_client.get("/api/models/yolo_test/training-images")
        data = resp.json()
        assert data["trainedAt"] == "2026-08-01T12:00:00+00:00"
        assert data["modelConfigNodeUuid"] == "node-123"

    async def test_sidecar_without_provenance_returns_nulls(self, http_client):
        """Models trained before those fields existed must still resolve, with the
        provenance simply absent rather than the request failing."""
        resp = await http_client.get("/api/models/yolo_test/training-images")
        data = resp.json()
        assert data["status"] == "success"
        assert data["trainedAt"] is None
        assert data["modelConfigNodeUuid"] is None

    async def test_unknown_model_returns_failure(self, http_client):
        resp = await http_client.get("/api/models/nonexistent/training-images")
        assert resp.json()["status"] == "failure"


class TestClassMapCache:
    """`recognize()` calls get_models_classes() for every image that produced a
    detection, so an uncached call meant a sidecar stat+read+json.parse per
    image on the event loop during a bulk run. The cache is keyed on the
    sidecar's mtime so it self-invalidates on edit.
    """

    async def test_repeat_calls_read_sidecar_once(self, manager, monkeypatch):
        calls = []
        real = manager_module.get_model_metadata

        def counting(model_path, *args, **kwargs):
            calls.append(model_path)
            return real(model_path, *args, **kwargs)

        monkeypatch.setattr(manager_module, "get_model_metadata", counting)

        first = await manager.get_models_classes(["yolo_test"])
        for _ in range(5):
            again = await manager.get_models_classes(["yolo_test"])
            assert again == first

        assert len(calls) == 1, f"sidecar re-read {len(calls)} times; cache is not holding"

    async def test_sidecar_edit_invalidates_cache(self, manager, models_dir):
        before = await manager.get_models_classes(["yolo_test"])
        assert before["yolo_test"]["cat"] is None

        meta_path = models_dir / "yolo_test.json"
        meta = json.loads(meta_path.read_text())
        meta["class_map"] = {"cat": "uuid-cat"}
        meta_path.write_text(json.dumps(meta))
        # Force a distinct mtime: the cache keys on it, and a same-nanosecond
        # rewrite would legitimately look unchanged.
        os.utime(meta_path, ns=(time.time_ns(), time.time_ns() + 1_000_000))

        after = await manager.get_models_classes(["yolo_test"])
        assert after["yolo_test"]["cat"] == "uuid-cat", "edited sidecar served from a stale cache"

    async def test_missing_sidecar_does_not_poison_cache(self, manager, models_dir):
        """With no sidecar, get_model_metadata derives metadata from the ONNX and
        writes one. Caching that as a permanent miss would pin the derived value
        forever, so no-sidecar results must simply not be cached."""
        (models_dir / "yolo_test.json").unlink()

        # Derivation fails on the fake ONNX bytes; the point is only that the
        # failure isn't cached as a hit for the next (now-valid) call.
        try:
            await manager.get_models_classes(["yolo_test"])
        except Exception:
            pass
        assert str(models_dir / "yolo_test.onnx") not in manager._class_map_cache

    async def test_label_association_is_visible_immediately(self, http_client, manager):
        """End-to-end guard on the invalidation path: the write goes through the
        manager's own setter, so a stale cache here would make the UI show the
        old mapping until restart."""
        await http_client.put(
            "/api/models/yolo_test/classes/cat/label",
            json={"labelUuid": "label-uuid-abc"},
        )
        classes = await manager.get_models_classes(["yolo_test"])
        assert classes["yolo_test"]["cat"] == "label-uuid-abc"

    async def test_sidecar_stat_does_not_block_event_loop(self, manager, monkeypatch):
        """get_models_classes() is awaited per-detection during a bulk inference
        run; a synchronous os.stat() there would stall every other in-flight
        request for as long as the stat takes. Assert the stat itself runs off
        the event loop: a concurrent short timer must fire on schedule (~0.05s)
        rather than only after the slow stat (~0.2s) finally lets the loop turn.
        """
        import asyncio
        import time as time_module

        real_stat = os.stat

        def slow_stat(path, *args, **kwargs):
            if str(path).endswith(".json"):
                time_module.sleep(0.2)
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(os, "stat", slow_stat)

        loop = asyncio.get_running_loop()
        start = loop.time()
        ticker_elapsed = None

        async def ticker():
            nonlocal ticker_elapsed
            await asyncio.sleep(0.05)
            ticker_elapsed = loop.time() - start

        ticker_task = asyncio.create_task(ticker())
        await manager.get_models_classes(["yolo_test"])
        await ticker_task

        assert ticker_elapsed is not None and ticker_elapsed < 0.15, (
            f"ticker (scheduled for ~0.05s) fired after {ticker_elapsed:.3f}s — the event loop "
            "was blocked by the sidecar stat() until it finished (~0.2s) instead of letting the "
            "timer run concurrently"
        )
