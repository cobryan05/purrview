"""Tests for the model version endpoints and the archiving the install path does.

The behaviour under test is "a retrain is undoable": pushing an artifact archives
what it replaces, the archive is listable, and restoring one puts those weights
back while archiving the ones it displaced.
"""

import base64
import io
import json
import tarfile

import pytest

from apps.helpers.modelVersions import list_versions, snapshot_current
from apps.inference_server.manager import Manager


def _artifact_tar(files: dict) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


async def _push(http_client, files: dict, *, versions_keep=None):
    payload = {"artifactTarBase64": _artifact_tar(files)}
    if versions_keep is not None:
        payload["versionsKeep"] = versions_keep
    r = await http_client.post("/api/models/artifact/upload", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body


class TestArtifactPushArchives:
    async def test_pushing_a_retrain_archives_the_weights_it_replaces(self, http_client, models_dir):
        original = (models_dir / "yolo_test.onnx").read_bytes()

        await _push(http_client, {"yolo_test.onnx": b"retrained", "yolo_test.json": b"{}"})

        assert (models_dir / "yolo_test.onnx").read_bytes() == b"retrained"
        versions = list_versions(models_dir, "yolo_test")
        assert len(versions) == 1
        archived = models_dir / "_versions" / "yolo_test" / versions[0]["version_id"] / "yolo_test.onnx"
        assert archived.read_bytes() == original

    async def test_versions_keep_from_the_pusher_bounds_the_history(self, http_client, models_dir):
        """The training host sends the limit; a remote inference server can't read it."""
        for i in range(4):
            await _push(
                http_client,
                {"yolo_test.onnx": f"gen{i}".encode(), "yolo_test.json": b"{}"},
                versions_keep=1,
            )
        assert len(list_versions(models_dir, "yolo_test")) == 1

    async def test_a_model_the_push_does_not_mention_keeps_its_own_history(self, http_client, models_dir):
        snapshot_current(models_dir, "yolo_test2")
        await _push(http_client, {"yolo_test.onnx": b"new", "yolo_test.json": b"{}"}, versions_keep=1)
        assert len(list_versions(models_dir, "yolo_test2")) == 1


class TestVersionEndpoints:
    async def test_listing_a_model_with_no_history_is_success_not_error(self, http_client):
        r = await http_client.get("/api/models/yolo_test/versions")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["versions"] == []

    async def test_restore_swaps_the_live_weights_and_archives_the_displaced_ones(self, http_client, models_dir):
        (models_dir / "yolo_test.onnx").write_bytes(b"good")
        await _push(http_client, {"yolo_test.onnx": b"bad", "yolo_test.json": b"{}"})

        listed = (await http_client.get("/api/models/yolo_test/versions")).json()["versions"]
        assert len(listed) == 1
        version_id = listed[0]["versionId"]

        r = await http_client.post(f"/api/models/yolo_test/versions/{version_id}/restore")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "success", body
        assert body["restored"] == version_id
        assert (models_dir / "yolo_test.onnx").read_bytes() == b"good"

        # The bad weights are now themselves a restore target.
        after = (await http_client.get("/api/models/yolo_test/versions")).json()["versions"]
        assert body["archived"] in [v["versionId"] for v in after]
        archived_file = models_dir / "_versions" / "yolo_test" / body["archived"] / "yolo_test.onnx"
        assert archived_file.read_bytes() == b"bad"

    async def test_restoring_an_unknown_version_fails_without_touching_the_model(self, http_client, models_dir):
        before = (models_dir / "yolo_test.onnx").read_bytes()
        r = await http_client.post("/api/models/yolo_test/versions/20200101T000000Z/restore")
        assert r.status_code == 200
        assert r.json()["status"] == "failure"
        assert (models_dir / "yolo_test.onnx").read_bytes() == before

    async def test_restore_drops_the_loaded_copy_so_inference_stops_serving_old_weights(
        self, manager, models_dir, monkeypatch
    ):
        """A pinned model holds its weights in memory; a restore has to evict it.

        The reload check keys on mtime, and a restore moves that backwards as often
        as forwards, so leaving the pin means inference keeps answering from the
        version the user just rolled away from.
        """
        (models_dir / "yolo_test.onnx").write_bytes(b"old")
        version_id = snapshot_current(models_dir, "yolo_test")
        (models_dir / "yolo_test.onnx").write_bytes(b"new")

        manager._pinned_models["yolo_test"] = object()
        await manager.restore_model_version("yolo_test", version_id)
        assert "yolo_test" not in manager._pinned_models


class TestDeleteTakesHistoryWithIt:
    async def test_deleting_a_model_removes_its_versions_and_external_tensors(self, manager, models_dir):
        (models_dir / "yolo_test.onnx.data").write_bytes(b"tensors")
        snapshot_current(models_dir, "yolo_test")
        assert list_versions(models_dir, "yolo_test")

        await manager.delete_model("yolo_test")

        assert not (models_dir / "yolo_test.onnx").exists()
        # Both were left behind before: the external tensors as an orphaned blob,
        # and the history as disk nothing could reach.
        assert not (models_dir / "yolo_test.onnx.data").exists()
        assert list_versions(models_dir, "yolo_test") == []


class TestVersionPreviews:
    async def test_preview_is_runnable_by_name_but_not_offered_in_the_model_list(self, http_client, models_dir):
        """The whole point: a version has to resolve like a model to be scored,
        and must never be pickable as a watcher's model or a training target."""
        (models_dir / "yolo_test.onnx").write_bytes(b"old")
        await _push(http_client, {"yolo_test.onnx": b"new", "yolo_test.json": b"{}"})
        version_id = (await http_client.get("/api/models/yolo_test/versions")).json()["versions"][0]["versionId"]

        r = await http_client.post(f"/api/models/yolo_test/versions/{version_id}/preview")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "success", body
        preview = body["previewName"]

        # Resolvable — this is what lets a comparison run it.
        meta = await http_client.get(f"/api/models/{preview}/meta")
        assert meta.status_code == 200
        assert meta.json()["status"] == "success"

        # But absent from the list every picker is built from.
        listed = (await http_client.get("/api/models")).json()["models"]
        assert preview not in listed
        assert "yolo_test" in listed

    async def test_preview_holds_the_archived_weights_not_the_live_ones(self, http_client, models_dir):
        (models_dir / "yolo_test.onnx").write_bytes(b"old")
        await _push(http_client, {"yolo_test.onnx": b"new", "yolo_test.json": b"{}"})
        version_id = (await http_client.get("/api/models/yolo_test/versions")).json()["versions"][0]["versionId"]

        preview = (await http_client.post(f"/api/models/yolo_test/versions/{version_id}/preview")).json()["previewName"]

        assert (models_dir / f"{preview}.onnx").read_bytes() == b"old"
        assert (models_dir / "yolo_test.onnx").read_bytes() == b"new"

    async def test_deleting_a_preview_leaves_the_model_and_its_history_alone(self, http_client, models_dir):
        (models_dir / "yolo_test.onnx").write_bytes(b"old")
        await _push(http_client, {"yolo_test.onnx": b"new", "yolo_test.json": b"{}"})
        version_id = (await http_client.get("/api/models/yolo_test/versions")).json()["versions"][0]["versionId"]
        preview = (await http_client.post(f"/api/models/yolo_test/versions/{version_id}/preview")).json()["previewName"]

        r = await http_client.delete(f"/api/models/{preview}/preview")
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "success"

        assert not (models_dir / f"{preview}.onnx").exists()
        assert (models_dir / "yolo_test.onnx").exists()
        assert len(list_versions(models_dir, "yolo_test")) == 1

    async def test_the_preview_delete_route_refuses_a_real_model(self, http_client, models_dir):
        """A cleanup path that loses track of its name must not delete a model."""
        r = await http_client.delete("/api/models/yolo_test/preview")
        assert r.status_code == 200
        assert r.json()["status"] == "failure"
        assert (models_dir / "yolo_test.onnx").exists()

    async def test_materializing_an_unknown_version_fails_cleanly(self, http_client):
        r = await http_client.post("/api/models/yolo_test/versions/20200101T000000Z/preview")
        assert r.status_code == 200
        assert r.json()["status"] == "failure"
