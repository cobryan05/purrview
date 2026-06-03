"""Tests for move-aware image reconcile, prune, and orphan generated-tag cleanup.

Scenarios:
  - Scenario A: reconcile before prune — moved file reconciled correctly, prune skips it
  - Scenario B: prune before reconcile — deleted path pruned, then new path reconciled from sidecar
  - True deletion: file gone + no sidecar → prune deletes DB record
  - sync_mode variations: replace_db_with_json, replace_json_with_db, merge, keep_newer
  - Orphan TASK_GENERATED tag cleanup: unattached tag deleted, attached tag survives

All tests use the db_server conftest fixtures: db_client, files_root, http_client.
"""

import json
import shutil
import struct
import uuid as uuid_mod
import zlib
from pathlib import Path
from typing import Optional

import pytest

from apps.helpers.db.types import UNVERIFIED_TAG_UUID, TagKind

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_png() -> bytes:
    """Minimal 1×1 white PNG."""

    def chunk(tag, data):
        raw = tag + data
        return struct.pack(">I", len(data)) + raw + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        + chunk(b"IEND", b"")
    )


def _write_image_and_register(files_root: Path, rel_path: str, http_client=None) -> dict:
    """Write a PNG to files_root/rel_path and register it in the DB via the API.

    Returns {"abs_path": Path, "rel_path": str}.
    """
    abs_path = files_root / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(_make_png())
    return {"abs_path": abs_path, "rel_path": rel_path}


async def _register_image(http_client, rel_path: str, label_uuid: Optional[str] = None) -> dict:
    """POST to /api/images/metadata/update to create a DB record (no bboxes or one unverified)."""
    boxes = []
    if label_uuid:
        boxes.append(
            {
                "uuid": str(uuid_mod.uuid4()),
                "labelUuid": label_uuid,
                "x": 0.1,
                "y": 0.1,
                "width": 0.2,
                "height": 0.2,
                "tagUuids": [UNVERIFIED_TAG_UUID],
            }
        )
    r = await http_client.post(
        "/api/images/metadata/update",
        json={"imagePath": rel_path, "boxes": boxes},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _read_sidecar(files_root: Path, rel_path: str) -> dict:
    abs_path = files_root / rel_path
    sidecar = abs_path.with_suffix(".json")
    assert sidecar.exists(), f"Sidecar not found at {sidecar}"
    return json.loads(sidecar.read_text())


async def _get_image_uuid_from_sidecar(files_root: Path, rel_path: str) -> str:
    return _read_sidecar(files_root, rel_path)["uuid"]


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
async def label_uuid(http_client):
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#f00"})
    assert r.status_code == 200
    return r.json()["label"]["uuid"]


# ──────────────────────────────────────────────────────────────────────────────
# Image stubs endpoint
# ──────────────────────────────────────────────────────────────────────────────


async def test_list_image_stubs_empty(http_client):
    r = await http_client.get("/api/images/stubs")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["stubs"] == []


async def test_list_image_stubs_after_register(http_client, files_root, label_uuid):
    _write_image_and_register(files_root, "img.png")
    await _register_image(http_client, "img.png", label_uuid)

    r = await http_client.get("/api/images/stubs")
    assert r.status_code == 200
    stubs = r.json()["stubs"]
    assert len(stubs) == 1
    assert "uuid" in stubs[0]
    assert stubs[0]["filename"].endswith("img.png")


# ──────────────────────────────────────────────────────────────────────────────
# Move Scenario A: reconcile BEFORE prune
# ──────────────────────────────────────────────────────────────────────────────


async def test_move_scenario_a(http_client, db_client, files_root, label_uuid):
    """
    1. Write image + register → DB record at old_path.
    2. Move file to new_path on disk.
    3. Reconcile new_path → DB record's filename updated (move detected).
    4. Prune with known_disk_uuids = {uuid} → nothing deleted (uuid is known).
    """
    # 1. Setup.
    _write_image_and_register(files_root, "orig.png")
    await _register_image(http_client, "orig.png", label_uuid)
    img_uuid = await _get_image_uuid_from_sidecar(files_root, "orig.png")

    # 2. Move on disk (keep sidecar alongside new file).
    old_abs = files_root / "orig.png"
    new_abs = files_root / "moved.png"
    new_abs.write_bytes(old_abs.read_bytes())
    # Copy the sidecar too (reconcile reads from new sidecar location).
    old_sidecar = old_abs.with_suffix(".json")
    new_sidecar = new_abs.with_suffix(".json")
    new_sidecar.write_bytes(old_sidecar.read_bytes())
    old_abs.unlink()
    old_sidecar.unlink()

    # 3. Reconcile the new path.
    r = await http_client.post(
        "/api/images/reconcile",
        json={"relPath": "moved.png", "syncMode": "merge"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["action"] in ("moved", "synced")

    # DB record should now point to the new absolute path.
    updated = await db_client.get_image(img_uuid)
    assert updated is not None
    assert str(new_abs) == updated.filename

    # 4. Prune with uuid in known set → nothing deleted.
    r = await http_client.post(
        "/api/images/prune",
        json={"knownDiskUuids": [img_uuid]},
    )
    assert r.status_code == 200
    assert r.json()["deletedCount"] == 0

    # Record still exists.
    assert await db_client.get_image(img_uuid) is not None


# ──────────────────────────────────────────────────────────────────────────────
# Move Scenario B: prune BEFORE reconcile
# ──────────────────────────────────────────────────────────────────────────────


async def test_move_scenario_b(http_client, db_client, files_root, label_uuid):
    """
    1. Write image + register → DB record at old_path.
    2. Delete old file (without adding new file yet) → prune removes DB record.
    3. Add new file + new sidecar (with same uuid).
    4. Reconcile new_path from sidecar → record recreated with same uuid/bboxes.
    """
    # 1. Setup.
    _write_image_and_register(files_root, "before_prune.png")
    await _register_image(http_client, "before_prune.png", label_uuid)
    img_uuid = await _get_image_uuid_from_sidecar(files_root, "before_prune.png")

    # Save sidecar content before deleting.
    sidecar_content = (files_root / "before_prune.json").read_bytes()

    # 2. Delete old file → prune.
    (files_root / "before_prune.png").unlink()
    (files_root / "before_prune.json").unlink()

    r = await http_client.post(
        "/api/images/prune",
        json={"knownDiskUuids": []},
    )
    assert r.status_code == 200
    assert r.json()["deletedCount"] == 1

    # Record should be gone.
    assert await db_client.get_image(img_uuid) is None

    # 3. Write sidecar to new location (same uuid).
    new_abs = files_root / "after_prune.png"
    new_abs.write_bytes(_make_png())
    new_abs.with_suffix(".json").write_bytes(sidecar_content)

    # 4. Reconcile new path → recreates record from sidecar.
    r = await http_client.post(
        "/api/images/reconcile",
        json={"relPath": "after_prune.png", "syncMode": "replace_db_with_json"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    restored = await db_client.get_image(img_uuid)
    assert restored is not None


# ──────────────────────────────────────────────────────────────────────────────
# True deletion prune
# ──────────────────────────────────────────────────────────────────────────────


async def test_true_deletion_prune(http_client, db_client, files_root, label_uuid):
    """Image file gone and uuid not known → DB record deleted by prune."""
    _write_image_and_register(files_root, "deleted.png")
    await _register_image(http_client, "deleted.png", label_uuid)
    img_uuid = await _get_image_uuid_from_sidecar(files_root, "deleted.png")

    (files_root / "deleted.png").unlink()
    (files_root / "deleted.json").unlink()

    r = await http_client.post(
        "/api/images/prune",
        json={"knownDiskUuids": []},
    )
    assert r.status_code == 200
    assert r.json()["deletedCount"] == 1
    assert await db_client.get_image(img_uuid) is None


async def test_prune_skips_files_that_still_exist(http_client, db_client, files_root, label_uuid):
    """Existing image file is NOT pruned even if not in known_disk_uuids."""
    _write_image_and_register(files_root, "keep.png")
    await _register_image(http_client, "keep.png", label_uuid)
    img_uuid = await _get_image_uuid_from_sidecar(files_root, "keep.png")

    r = await http_client.post(
        "/api/images/prune",
        json={"knownDiskUuids": []},
    )
    assert r.status_code == 200
    assert r.json()["deletedCount"] == 0
    assert await db_client.get_image(img_uuid) is not None


# ──────────────────────────────────────────────────────────────────────────────
# sync_mode tests
# ──────────────────────────────────────────────────────────────────────────────


async def test_sync_mode_replace_db_with_json(http_client, db_client, files_root, label_uuid):
    """replace_db_with_json: sidecar wins over DB bboxes."""
    _write_image_and_register(files_root, "rdbj.png")
    # Register with a bbox.
    await _register_image(http_client, "rdbj.png", label_uuid)

    # Now manually corrupt the DB bboxes by removing them.
    img_uuid = await _get_image_uuid_from_sidecar(files_root, "rdbj.png")
    image = await db_client.get_image(img_uuid)
    # Write empty bboxes to DB directly.
    from apps.helpers.db.types import ImageRecordUpdate

    await db_client.write_image_metadata_to_db(img_uuid, ImageRecordUpdate(bboxes=[]))

    # Reconcile with replace_db_with_json → sidecar restores bbox.
    r = await http_client.post(
        "/api/images/reconcile",
        json={"relPath": "rdbj.png", "syncMode": "replace_db_with_json"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    bboxes_after = await db_client.get_bboxes_for_image(img_uuid)
    assert len(bboxes_after) == 1


async def test_sync_mode_replace_json_with_db(http_client, db_client, files_root, label_uuid):
    """replace_json_with_db: DB wins, sidecar is rewritten from DB."""
    _write_image_and_register(files_root, "rjdb.png")
    await _register_image(http_client, "rjdb.png", label_uuid)
    img_uuid = await _get_image_uuid_from_sidecar(files_root, "rjdb.png")

    # Overwrite the sidecar with empty boxes.
    sidecar = files_root / "rjdb.json"
    data = json.loads(sidecar.read_text())
    data["boxes"] = []
    sidecar.write_text(json.dumps(data))

    r = await http_client.post(
        "/api/images/reconcile",
        json={"relPath": "rjdb.png", "syncMode": "replace_json_with_db"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    # Sidecar should now reflect the DB bbox.
    updated_sidecar = json.loads(sidecar.read_text())
    assert len(updated_sidecar["boxes"]) == 1


async def test_sync_mode_merge(http_client, db_client, files_root, label_uuid):
    """merge: JSON bboxes not in DB are added; existing DB bboxes are kept."""
    _write_image_and_register(files_root, "merge.png")
    await _register_image(http_client, "merge.png", label_uuid)
    img_uuid = await _get_image_uuid_from_sidecar(files_root, "merge.png")

    # DB currently has 1 bbox. Add a second bbox only in the sidecar.
    sidecar = files_root / "merge.json"
    data = json.loads(sidecar.read_text())
    extra_bbox_uuid = str(uuid_mod.uuid4())
    data["boxes"].append(
        {
            "uuid": extra_bbox_uuid,
            "label_uuid": label_uuid,
            "label_name": "cat",
            "x": 0.5,
            "y": 0.5,
            "width": 0.1,
            "height": 0.1,
            "tag_uuids": [UNVERIFIED_TAG_UUID],
            "tag_names": ["Unverified"],
        }
    )
    sidecar.write_text(json.dumps(data))

    r = await http_client.post(
        "/api/images/reconcile",
        json={"relPath": "merge.png", "syncMode": "merge"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    bboxes_after = await db_client.get_bboxes_for_image(img_uuid)
    bbox_uuids = {b.uuid for b in bboxes_after}
    assert extra_bbox_uuid in bbox_uuids


async def test_sync_mode_keep_newer_db_wins(http_client, db_client, files_root, label_uuid):
    """keep_newer: when DB record is newer than sidecar, sidecar is rewritten from DB."""
    _write_image_and_register(files_root, "kn.png")
    await _register_image(http_client, "kn.png", label_uuid)
    img_uuid = await _get_image_uuid_from_sidecar(files_root, "kn.png")

    # Overwrite sidecar with empty boxes and set its mtime to the past.
    import os
    import time

    sidecar = files_root / "kn.json"
    data = json.loads(sidecar.read_text())
    data["boxes"] = []
    sidecar.write_text(json.dumps(data))
    past_time = time.time() - 3600  # 1 hour ago
    os.utime(sidecar, (past_time, past_time))

    r = await http_client.post(
        "/api/images/reconcile",
        json={"relPath": "kn.png", "syncMode": "keep_newer"},
    )
    assert r.status_code == 200

    # DB is newer → sidecar should be rewritten with the DB bbox.
    updated_sidecar = json.loads(sidecar.read_text())
    assert len(updated_sidecar["boxes"]) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Orphan TASK_GENERATED tag cleanup
# ──────────────────────────────────────────────────────────────────────────────


async def test_clear_orphan_generated_tags_removes_unattached(http_client, db_client):
    """A TASK_GENERATED tag with no bbox_tags rows is deleted."""
    tag = await db_client.add_tag("orphan-task-tag", kind=TagKind.TASK_GENERATED)

    r = await http_client.post("/api/tags/clear-orphan-generated")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["deletedCount"] == 1

    # Tag should be gone.
    remaining = await db_client.list_tags()
    uuids = {t.uuid for t in remaining}
    assert tag.uuid not in uuids


async def test_clear_orphan_generated_tags_keeps_attached(http_client, db_client, files_root, label_uuid):
    """A TASK_GENERATED tag attached to at least one bbox is NOT deleted."""
    _write_image_and_register(files_root, "kept.png")
    task_tag = await db_client.add_tag("kept-task-tag", kind=TagKind.TASK_GENERATED)

    # Register image with the task tag on its bbox.
    await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "kept.png",
            "boxes": [
                {
                    "uuid": str(uuid_mod.uuid4()),
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [UNVERIFIED_TAG_UUID, task_tag.uuid],
                }
            ],
        },
    )

    r = await http_client.post("/api/tags/clear-orphan-generated")
    assert r.status_code == 200
    assert r.json()["deletedCount"] == 0

    remaining = await db_client.list_tags()
    uuids = {t.uuid for t in remaining}
    assert task_tag.uuid in uuids


async def test_clear_orphan_generic_tags_untouched(http_client, db_client):
    """Generic (non-TASK_GENERATED) tags are never deleted by clear_orphan."""
    tag = await db_client.add_tag("generic-orphan", kind=TagKind.GENERIC)

    r = await http_client.post("/api/tags/clear-orphan-generated")
    assert r.status_code == 200
    assert r.json()["deletedCount"] == 0

    remaining = await db_client.list_tags()
    uuids = {t.uuid for t in remaining}
    assert tag.uuid in uuids


# ──────────────────────────────────────────────────────────────────────────────
# Reconcile with no sidecar
# ──────────────────────────────────────────────────────────────────────────────


async def test_reconcile_no_sidecar_skips(http_client, files_root):
    """reconcile_image skips files with no sidecar."""
    (files_root / "nosidecar.png").write_bytes(_make_png())
    r = await http_client.post(
        "/api/images/reconcile",
        json={"relPath": "nosidecar.png", "syncMode": "merge"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["action"] == "skipped"
