import json
import os
import time

import onnx
import onnx.helper as onnx_helper
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


def _write_onnx_with_static_shape(path, height: int, width: int) -> None:
    inp = onnx_helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, height, width])
    out = onnx_helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1, 84, 8400])
    node = onnx_helper.make_node("Identity", ["images"], ["out"])
    graph = onnx_helper.make_graph([node], "g", [inp], [out])
    onnx.save(onnx_helper.make_model(graph), str(path))


async def test_get_model_meta_input_shape_none_for_unparseable_onnx(http_client):
    """`models_dir`'s fixtures are placeholder bytes, not real ONNX — the meta
    endpoint must degrade to None rather than fail the whole response over a
    field that's a bonus, not the reason for the call."""
    resp = await http_client.get("/api/models/yolo_test/meta")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["inputShape"] is None


async def test_get_model_meta_input_shape_from_real_onnx_graph(models_dir, mock_provider, monkeypatch):
    _write_onnx_with_static_shape(models_dir / "yolo_test.onnx", 640, 480)
    monkeypatch.setattr("apps.inference_server.manager.load_yolo_onnx", lambda path: mock_provider)
    mgr = Manager(models_path=str(models_dir))
    await mgr.list_models()

    shape = await mgr.get_model_input_shape("yolo_test")

    assert shape == (640, 480)


async def test_get_model_input_shape_invalidates_on_retrain(models_dir, mock_provider, monkeypatch):
    """A retrain replaces the .onnx with a new architecture; a stale cached
    shape would silently mis-size everything downstream that trusts it."""
    _write_onnx_with_static_shape(models_dir / "yolo_test.onnx", 640, 640)
    monkeypatch.setattr("apps.inference_server.manager.load_yolo_onnx", lambda path: mock_provider)
    mgr = Manager(models_path=str(models_dir))
    await mgr.list_models()

    assert await mgr.get_model_input_shape("yolo_test") == (640, 640)

    time.sleep(0.01)  # ensure a distinct mtime_ns on filesystems with coarse resolution
    _write_onnx_with_static_shape(models_dir / "yolo_test.onnx", 1280, 1280)

    assert await mgr.get_model_input_shape("yolo_test") == (1280, 1280)


async def test_get_model_input_shape_unknown_model_returns_none(manager):
    assert await manager.get_model_input_shape("nonexistent") is None


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
        run, so file I/O it performs on the event loop stalls every other
        in-flight request for its duration. Both halves must stay off the loop
        thread: the sidecar stat, and the sidecar read behind
        get_model_metadata() that only a cache miss reaches.

        Asserts which thread each one ran on rather than how long anything took
        — a latency bound here would make machine load the input.
        """
        import threading

        loop_thread = threading.current_thread()
        stat_threads: set = set()
        read_threads: set = set()

        real_stat = os.stat

        def recording_stat(path, *args, **kwargs):
            if str(path).endswith(".json"):
                stat_threads.add(threading.current_thread())
            return real_stat(path, *args, **kwargs)

        real_get_model_metadata = manager_module.get_model_metadata

        def recording_get_model_metadata(*args, **kwargs):
            read_threads.add(threading.current_thread())
            return real_get_model_metadata(*args, **kwargs)

        monkeypatch.setattr(os, "stat", recording_stat)
        monkeypatch.setattr(manager_module, "get_model_metadata", recording_get_model_metadata)
        # The miss path is the one that reads the sidecar; a cache warmed by an
        # earlier call would skip it and prove nothing.
        manager._class_map_cache.clear()

        await manager.get_models_classes(["yolo_test"])

        assert stat_threads, "no sidecar stat happened — the test no longer covers its path"
        assert loop_thread not in stat_threads, "the sidecar stat ran on the event loop thread"
        assert read_threads, "the cache-miss sidecar read never happened — the cache was warm"
        assert loop_thread not in read_threads, "the sidecar read ran on the event loop thread"


class TestSuppressionGroupsPreviewAndApply:
    """/api/models/suppression-groups/{preview,apply} — the label editor's "N

    trained models affected" notice after an exclusive_variants edit. See
    apps/helpers/suppressionGroups.py for the grouping rule itself.
    """

    @staticmethod
    def _write_detector_sidecar(models_dir, name: str, label_uuids: list, existing_groups=None, thresholds=None):
        metadata = {
            "model_kind": "object_detection",
            "classes": [f"class-{i}" for i in range(len(label_uuids))],
            "label_uuids": label_uuids,
        }
        if existing_groups is not None:
            metadata["suppression_group_uuids"] = existing_groups
            metadata["suppression_group_iou_thresholds"] = thresholds
        (models_dir / f"{name}.onnx").write_bytes(b"fake onnx")
        (models_dir / f"{name}.json").write_text(json.dumps(metadata))

    @staticmethod
    def _labels_payload(parent, children, threshold=0.9):
        return [
            {"uuid": parent, "parentUuid": None, "exclusiveVariants": True, "exclusiveVariantsIouThreshold": threshold},
            *[
                {"uuid": c, "parentUuid": parent, "exclusiveVariants": False, "exclusiveVariantsIouThreshold": None}
                for c in children
            ],
        ]

    async def test_preview_reports_a_model_with_no_prior_grouping(self, http_client, models_dir):
        self._write_detector_sidecar(models_dir, "detector1", ["parent-a", "child-a", "child-b"])
        await http_client.get("/api/models")  # refresh Manager._avail_models

        resp = await http_client.post(
            "/api/models/suppression-groups/preview",
            json={"labels": self._labels_payload("parent-a", ["child-a", "child-b"])},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert [m["modelName"] for m in data["models"]] == ["detector1"]
        # All 3 classes count as changed: child-a/child-b newly join the group,
        # and parent-a's own threshold goes from unset to 0.9 — its suppression
        # key ("parent-a") doesn't move, but it now actually competes against the
        # two children it previously suppressed independently of.
        assert data["models"][0]["changedClasses"] == 3
        assert data["models"][0]["totalClasses"] == 3

    async def test_preview_omits_a_model_already_up_to_date(self, http_client, models_dir):
        self._write_detector_sidecar(
            models_dir,
            "detector1",
            ["parent-a", "child-a", "child-b"],
            existing_groups=["parent-a", "parent-a", "parent-a"],
            thresholds=[0.9, 0.9, 0.9],
        )
        await http_client.get("/api/models")

        resp = await http_client.post(
            "/api/models/suppression-groups/preview",
            json={"labels": self._labels_payload("parent-a", ["child-a", "child-b"])},
        )
        assert resp.json()["models"] == []

    async def test_preview_omits_unrelated_models(self, http_client, models_dir):
        """A model whose label_uuids never mention the changed label tree at all

        must not show up as "affected" just because it's an object_detection
        model with no suppression_group fields yet."""
        self._write_detector_sidecar(models_dir, "unrelated", ["some-other-label"])
        await http_client.get("/api/models")

        resp = await http_client.post(
            "/api/models/suppression-groups/preview",
            json={"labels": self._labels_payload("parent-a", ["child-a", "child-b"])},
        )
        assert resp.json()["models"] == []

    async def test_apply_writes_the_sidecar_and_is_picked_up_without_a_restart(self, http_client, models_dir):
        self._write_detector_sidecar(models_dir, "detector1", ["parent-a", "child-a", "child-b"])
        await http_client.get("/api/models")

        resp = await http_client.post(
            "/api/models/suppression-groups/apply",
            json={
                "labels": self._labels_payload("parent-a", ["child-a", "child-b"]),
                "modelNames": ["detector1"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["updated"] == ["detector1"]

        sidecar = json.loads((models_dir / "detector1.json").read_text())
        assert sidecar["suppression_group_uuids"] == ["parent-a", "parent-a", "parent-a"]
        assert sidecar["suppression_group_iou_thresholds"] == [0.9, 0.9, 0.9]

        # A second preview against the same tree now reports nothing left to do.
        resp2 = await http_client.post(
            "/api/models/suppression-groups/preview",
            json={"labels": self._labels_payload("parent-a", ["child-a", "child-b"])},
        )
        assert resp2.json()["models"] == []

    async def test_apply_only_touches_the_named_models(self, http_client, models_dir):
        self._write_detector_sidecar(models_dir, "detector1", ["parent-a", "child-a"])
        self._write_detector_sidecar(models_dir, "detector2", ["parent-a", "child-a"])
        await http_client.get("/api/models")

        resp = await http_client.post(
            "/api/models/suppression-groups/apply",
            json={"labels": self._labels_payload("parent-a", ["child-a"]), "modelNames": ["detector1"]},
        )
        assert resp.json()["updated"] == ["detector1"]
        assert "suppression_group_uuids" not in json.loads((models_dir / "detector2.json").read_text())
