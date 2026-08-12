from pathlib import Path

import pytest
from sqlalchemy import text

from apps.helpers.db.db_client import DbClient
from tests.conftest import make_png


@pytest.fixture
def image_file(files_root):
    """A real PNG file on disk that the manager can open."""
    img = files_root / "test.png"
    img.write_bytes(make_png())
    return img


@pytest.fixture
async def label_uuid(http_client):
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    return r.json()["label"]["uuid"]


async def test_list_images_empty_dir(http_client):
    r = await http_client.get("/api/images/list")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["files"] == []


async def test_list_images_finds_file(http_client, image_file):
    r = await http_client.get("/api/images/list", params={"pattern": "*.png"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    names = [f["name"] for f in body["files"]]
    assert "test.png" in names


async def test_get_image_metadata_not_found(http_client):
    r = await http_client.post("/api/images/metadata/get", json={"image_path": "nonexistent.png"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failure"


async def test_get_image_metadata(http_client, image_file):
    r = await http_client.post("/api/images/metadata/get", json={"image_path": "test.png"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success" or body["image"] is not None


async def test_update_and_retrieve_bboxes(http_client, image_file, label_uuid):
    bbox = {
        "uuid": "bbbbbbbb-0000-0000-0000-000000000001",
        "label_uuid": label_uuid,
        "x": 0.1,
        "y": 0.2,
        "width": 0.3,
        "height": 0.4,
        "tag_uuids": [],
    }
    update_r = await http_client.post(
        "/api/images/metadata/update",
        json={"image_path": "test.png", "boxes": [bbox]},
    )
    assert update_r.status_code == 200
    assert update_r.json()["status"] == "success" or update_r.json()["image"] is not None

    meta_r = await http_client.post("/api/images/metadata/get", json={"image_path": "test.png"})
    assert meta_r.status_code == 200
    image_data = meta_r.json()["image"]
    assert image_data is not None
    assert len(image_data["bboxes"]) == 1
    assert abs(image_data["bboxes"][0]["x"] - 0.1) < 1e-6


async def test_get_image_file(http_client, image_file):
    r = await http_client.get("/api/images/test.png/file")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["filename"] == "test.png"
    assert body["imageBase64"] is not None


async def test_get_image_file_not_found(http_client):
    r = await http_client.get("/api/images/missing.png/file")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failure"


async def test_delete_image_not_found(http_client):
    r = await http_client.delete("/api/images/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failure"
    assert "not found" in body["message"].lower()


async def test_delete_image_removes_file_and_record(http_client, image_file, files_root):
    # Create a sidecar so the image can be reconciled into the DB with a stable UUID.
    import json as _json
    import uuid as _uuid

    stable_uuid = str(_uuid.uuid4())
    sidecar = image_file.with_suffix(".json")
    sidecar.write_text(_json.dumps({"uuid": stable_uuid, "boxes": []}))

    r = await http_client.post(
        "/api/images/reconcile",
        json={"relPath": "test.png", "syncMode": "replace_db_with_json"},
    )
    assert r.status_code == 200

    # Confirm the record is in the DB.
    r = await http_client.post("/api/images/metadata/get", json={"imagePath": "test.png"})
    assert r.status_code == 200
    image_uuid = r.json()["image"]["uuid"]
    assert image_uuid == stable_uuid

    assert image_file.exists()
    assert sidecar.exists()

    # Delete via the API.
    r = await http_client.delete(f"/api/images/{image_uuid}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["imageUuid"] == image_uuid

    # File and sidecar should be gone from disk.
    assert not image_file.exists()
    assert not sidecar.exists()

    # DB record should be gone too.
    r2 = await http_client.post("/api/images/metadata/get", json={"imagePath": "test.png"})
    assert r2.status_code == 200
    # file no longer exists, so metadata returns failure
    assert r2.json()["status"] == "failure"


async def test_delete_image_permanently_aborts_when_sidecar_undeletable(http_client, db_client, image_file, files_root):
    """If the sidecar cannot be deleted, the DB record must NOT be deleted.

    Otherwise the DB record is gone but the sidecar survives on disk, and the
    next reconcile silently re-imports (resurrects) the "deleted" image. The
    sidecar must be removed before (and gate) the DB delete. The DB state is
    checked directly (metadata/get would auto-resync from the surviving sidecar
    and mask the bug).
    """
    import json as _json
    import os
    import uuid as _uuid

    stable_uuid = str(_uuid.uuid4())
    sidecar = image_file.with_suffix(".json")
    sidecar.write_text(_json.dumps({"uuid": stable_uuid, "boxes": []}))

    r = await http_client.post(
        "/api/images/reconcile",
        json={"relPath": "test.png", "syncMode": "replace_db_with_json"},
    )
    assert r.status_code == 200
    assert await db_client.get_image_by_uuid(stable_uuid) is not None

    # Make the sidecar undeletable by removing write permission on its directory.
    os.chmod(files_root, 0o500)
    try:
        await http_client.delete(f"/api/images/{stable_uuid}")
    finally:
        os.chmod(files_root, 0o700)

    assert sidecar.exists(), "sidecar should not have been removed"
    # Raw DB check (no resync): the record must survive a failed sidecar delete.
    assert await db_client.get_image_by_uuid(stable_uuid) is not None, "DB record must survive a failed sidecar delete"


async def test_delete_image_db_only_not_found(http_client):
    r = await http_client.delete("/api/images/00000000-0000-0000-0000-000000000000/db-only")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failure"
    assert "not found" in body["message"].lower()


async def test_delete_image_db_only_keeps_file(http_client, image_file, files_root):
    """DB-only delete removes the DB record and sidecar but leaves the image file on disk."""
    import json as _json
    import uuid as _uuid

    stable_uuid = str(_uuid.uuid4())
    sidecar = image_file.with_suffix(".json")
    sidecar.write_text(_json.dumps({"uuid": stable_uuid, "boxes": []}))

    r = await http_client.post(
        "/api/images/reconcile",
        json={"relPath": "test.png", "syncMode": "replace_db_with_json"},
    )
    assert r.status_code == 200

    # Confirm the record is in the DB with the expected UUID.
    r = await http_client.post("/api/images/metadata/get", json={"imagePath": "test.png"})
    assert r.status_code == 200
    image_uuid = r.json()["image"]["uuid"]
    assert image_uuid == stable_uuid

    assert image_file.exists()
    assert sidecar.exists()

    # Delete from DB only.
    r = await http_client.delete(f"/api/images/{image_uuid}/db-only")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["imageUuid"] == image_uuid

    # Image file must still exist; sidecar must be removed (prevents re-import on reconcile).
    assert image_file.exists(), "image file should NOT be deleted in db-only mode"
    assert not sidecar.exists(), "sidecar SHOULD be deleted so reconcile cannot re-import the record"

    # DB record must be gone: by-uuid lookup returns 404 without triggering re-sync.
    r2 = await http_client.get(f"/api/images/by-uuid/{image_uuid}/meta")
    assert r2.status_code == 404


async def test_bulk_delete_over_threshold_reclaims_freed_pages(tmp_path, files_root):
    """A large delete (e.g. an accidental mass-ingest cleanup) must not leave its
    freed pages sitting on the freelist forever — that's what silently balloons
    db.sqlite until next/prev and other paginated queries degrade. Uses a real
    file-backed DbClient (not the in-memory test fixture) since freelist_count
    is only meaningful on disk."""
    client = DbClient(db_dir=tmp_path, memory=False, files_root=files_root)
    await client.init_db()
    client._INCREMENTAL_VACUUM_THRESHOLD = 5  # keep the test fast

    images = [await client.add_image(Path(f"img-{i}.jpg")) for i in range(10)]
    deleted = await client.delete_images_bulk([img.uuid for img in images])
    assert deleted == 10

    async with client._engine.connect() as conn:
        result = await conn.execute(text("PRAGMA freelist_count"))
        assert result.scalar() == 0, "freed pages should have been reclaimed by incremental_vacuum"

    await client.close()
