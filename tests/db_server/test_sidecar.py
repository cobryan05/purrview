"""Tests for the JSON sidecar mechanism.

Two concerns:
  1. Write side — DB operations correctly produce/update the .json sidecar
     alongside the image file, including label_name and tag_names for robustness.
  2. Read side — sync_image_from_json can restore a fresh DB from those sidecars,
     using UUID-first resolution and falling back to names when UUIDs differ.
"""

import json
import uuid as uuid_mod
from pathlib import Path

import pytest

from apps.db_server.manager import Manager
from apps.db_server.web import WebApp
from apps.helpers.db.db_client import DbClient
from apps.helpers.db.types import UNVERIFIED_TAG_UUID, BBoxUpdate, ImageRecordUpdate
from tests.conftest import make_png

# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures (the http_client / db_client / files_root fixtures already
# live in tests/db_server/conftest.py and are available automatically)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def image_file(files_root):
    img = files_root / "sidecar_test.png"
    img.write_bytes(make_png())
    return img


@pytest.fixture
async def label_uuid(http_client):
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    return r.json()["label"]["uuid"]


@pytest.fixture
async def custom_tag_uuid(http_client):
    r = await http_client.post("/api/tags/add", json={"name": "my_tag"})
    return r.json()["tag"]["uuid"]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Write side: sidecar is created / updated by bbox operations
# ──────────────────────────────────────────────────────────────────────────────


async def test_sidecar_created_on_metadata_update(http_client, image_file, label_uuid):
    """Calling the update endpoint writes a .json sidecar next to the image."""
    bid = str(uuid_mod.uuid4())
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "sidecar_test.png",
            "boxes": [
                {
                    "uuid": bid,
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.2,
                    "width": 0.3,
                    "height": 0.4,
                    "tagUuids": [UNVERIFIED_TAG_UUID],
                }
            ],
        },
    )
    assert r.status_code == 200

    sidecar = image_file.with_suffix(".json")
    assert sidecar.exists(), "sidecar file should exist after metadata update"

    data = json.loads(sidecar.read_text())
    assert "uuid" in data, "sidecar must include image uuid"
    assert len(data["boxes"]) == 1

    box = data["boxes"][0]
    assert box["uuid"] == bid
    assert box["label_uuid"] == label_uuid
    assert box["x"] == pytest.approx(0.1)
    assert box["y"] == pytest.approx(0.2)
    assert box["width"] == pytest.approx(0.3)
    assert box["height"] == pytest.approx(0.4)


async def test_sidecar_includes_label_and_tag_names(http_client, image_file, label_uuid, custom_tag_uuid):
    """Sidecar stores label_name and tag_names for rebuild robustness."""
    bid = str(uuid_mod.uuid4())
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "sidecar_test.png",
            "boxes": [
                {
                    "uuid": bid,
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [UNVERIFIED_TAG_UUID, custom_tag_uuid],
                }
            ],
        },
    )
    assert r.status_code == 200

    sidecar = image_file.with_suffix(".json")
    data = json.loads(sidecar.read_text())
    box = data["boxes"][0]

    assert box.get("label_name") == "cat", "sidecar must store the label name"
    tag_names = box.get("tag_names", [])
    assert "my_tag" in tag_names, "sidecar must store custom tag name"
    assert "Unverified" in tag_names, "sidecar must store system tag name"


async def test_sidecar_updated_on_bbox_patch(http_client, image_file, label_uuid):
    """PATCH /api/images/bbox/{uuid} keeps the sidecar in sync."""
    bid = str(uuid_mod.uuid4())
    await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "sidecar_test.png",
            "boxes": [
                {
                    "uuid": bid,
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [UNVERIFIED_TAG_UUID],
                }
            ],
        },
    )

    r2 = await http_client.post("/api/labels", json={"name": "dog", "color": "#0000FF"})
    other_uuid = r2.json()["label"]["uuid"]

    patch_r = await http_client.patch(
        f"/api/images/bbox/{bid}",
        json={"labelUuid": other_uuid, "tagUuids": [UNVERIFIED_TAG_UUID]},
    )
    assert patch_r.status_code == 204

    sidecar = image_file.with_suffix(".json")
    data = json.loads(sidecar.read_text())
    box = data["boxes"][0]
    assert box["label_uuid"] == other_uuid
    assert box.get("label_name") == "dog"


async def test_sidecar_updated_on_bbox_delete(http_client, image_file, label_uuid):
    """DELETE /api/images/bbox/{uuid} removes the bbox from the sidecar."""
    bid = str(uuid_mod.uuid4())
    await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "sidecar_test.png",
            "boxes": [
                {
                    "uuid": bid,
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [UNVERIFIED_TAG_UUID],
                }
            ],
        },
    )

    del_r = await http_client.delete(f"/api/images/bbox/{bid}")
    assert del_r.status_code == 204

    sidecar = image_file.with_suffix(".json")
    data = json.loads(sidecar.read_text())
    assert data["boxes"] == [], "deleted bbox must be removed from sidecar"


async def test_sidecar_reflects_multiple_updates(http_client, image_file, label_uuid):
    """Successive metadata updates overwrite the sidecar with the latest state."""
    bid1 = str(uuid_mod.uuid4())
    bid2 = str(uuid_mod.uuid4())

    # First write: one bbox
    await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "sidecar_test.png",
            "boxes": [
                {
                    "uuid": bid1,
                    "labelUuid": label_uuid,
                    "x": 0.0,
                    "y": 0.0,
                    "width": 0.1,
                    "height": 0.1,
                    "tagUuids": [],
                }
            ],
        },
    )

    # Second write: two bboxes
    await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "sidecar_test.png",
            "boxes": [
                {
                    "uuid": bid1,
                    "labelUuid": label_uuid,
                    "x": 0.0,
                    "y": 0.0,
                    "width": 0.1,
                    "height": 0.1,
                    "tagUuids": [],
                },
                {
                    "uuid": bid2,
                    "labelUuid": label_uuid,
                    "x": 0.5,
                    "y": 0.5,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [],
                },
            ],
        },
    )

    sidecar = image_file.with_suffix(".json")
    data = json.loads(sidecar.read_text())
    box_uuids = {b["uuid"] for b in data["boxes"]}
    assert box_uuids == {bid1, bid2}, "sidecar should reflect the latest two-bbox state"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Read side: sync_image_from_json restores a fresh DB from sidecar files
# ──────────────────────────────────────────────────────────────────────────────


async def _write_sidecar_directly(
    db_client: DbClient,
    image_path: Path,
    image_uuid: str,
    bbox_uuid: str,
    label_uuid: str,
    label_name: str,
    tag_uuid: str,
    tag_name: str,
) -> None:
    """Write a sidecar with fully populated name fields."""
    sidecar_path = image_path.with_suffix(".json")
    update = ImageRecordUpdate(
        bboxes=[
            BBoxUpdate(
                uuid=bbox_uuid,
                label_uuid=label_uuid,
                label_name=label_name,
                x=0.1,
                y=0.2,
                width=0.3,
                height=0.4,
                tag_uuids=[tag_uuid],
                tag_names=[tag_name],
            )
        ]
    )
    db_client.write_image_metadata_to_json(sidecar_path, image_uuid, str(image_path), update)


async def test_rebuild_from_sidecar_by_uuid(tmp_path):
    """
    When the DB is deleted but sidecars remain, sync_image_from_json rebuilds
    the image record using the stored UUIDs (the common case).
    """
    files_root = tmp_path / "files"
    files_root.mkdir()

    # ── Phase 1: build original DB and write sidecar ──────────────────────────
    db_dir1 = tmp_path / "db1"
    db_dir1.mkdir()
    client1 = DbClient(db_dir=db_dir1)
    await client1.init_db()

    label = await client1.add_label("cat", "#FF0000")
    image_path = files_root / "rebuild_test.png"
    image_path.write_bytes(make_png())

    image = await client1.add_image(image_path)
    bbox_uuid = str(uuid_mod.uuid4())
    await client1.write_image_metadata_to_db(
        image.uuid,
        ImageRecordUpdate(
            bboxes=[
                BBoxUpdate(
                    uuid=bbox_uuid,
                    label_uuid=label.uuid,
                    x=0.1,
                    y=0.2,
                    width=0.3,
                    height=0.4,
                    tag_uuids=[UNVERIFIED_TAG_UUID],
                )
            ]
        ),
    )
    await _write_sidecar_directly(
        client1,
        image_path,
        image.uuid,
        bbox_uuid,
        label.uuid,
        "cat",
        UNVERIFIED_TAG_UUID,
        "Unverified",
    )
    original_image_uuid = image.uuid
    original_label_uuid = label.uuid
    await client1.close()

    # ── Phase 2: fresh DB, same label UUID, rebuild from sidecar ─────────────
    db_dir2 = tmp_path / "db2"
    db_dir2.mkdir()
    client2 = DbClient(db_dir=db_dir2)
    await client2.init_db()
    # Recreate the label with the SAME uuid so UUID-based resolution works
    await client2.add_label("cat", "#FF0000", uuid=original_label_uuid)

    restored = await client2.sync_image_from_json(image_path)

    assert restored.uuid == original_image_uuid, "restored image UUID must match sidecar"
    bboxes = await client2.get_bboxes_for_image(restored.uuid)
    assert len(bboxes) == 1
    b = bboxes[0]
    assert b.uuid == bbox_uuid
    assert b.label_uuid == original_label_uuid
    # System tag should be linked
    tag_uuids = [t.uuid for t in (b.tags or [])]
    assert UNVERIFIED_TAG_UUID in tag_uuids

    await client2.close()


async def test_rebuild_from_sidecar_by_name_fallback(tmp_path):
    """
    When the DB is rebuilt with different UUIDs (e.g. after re-generation),
    sync_image_from_json falls back to label_name / tag_names for resolution.
    """
    files_root = tmp_path / "files"
    files_root.mkdir()

    # ── Phase 1: write sidecar with original UUIDs ────────────────────────────
    old_label_uuid = str(uuid_mod.uuid4())
    old_tag_uuid = str(uuid_mod.uuid4())
    image_uuid = str(uuid_mod.uuid4())
    bbox_uuid = str(uuid_mod.uuid4())

    image_path = files_root / "fallback_test.png"
    image_path.write_bytes(make_png())

    # Write the sidecar directly — no DB needed for phase 1
    sidecar_path = image_path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(
            {
                "uuid": image_uuid,
                "filename": str(image_path),
                "boxes": [
                    {
                        "uuid": bbox_uuid,
                        "label_uuid": old_label_uuid,
                        "label_name": "cat",
                        "x": 0.1,
                        "y": 0.2,
                        "width": 0.3,
                        "height": 0.4,
                        "tag_uuids": [old_tag_uuid],
                        "tag_names": ["my_tag"],
                    }
                ],
            }
        )
    )

    # ── Phase 2: fresh DB with NEW UUIDs but same names ───────────────────────
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    client = DbClient(db_dir=db_dir)
    await client.init_db()

    new_label = await client.add_label("cat", "#FF0000")  # new UUID
    new_tag = await client.add_tag("my_tag")  # new UUID

    assert new_label.uuid != old_label_uuid, "precondition: UUIDs differ"
    assert new_tag.uuid != old_tag_uuid, "precondition: UUIDs differ"

    restored = await client.sync_image_from_json(image_path)

    assert restored.uuid == image_uuid, "image UUID comes from the sidecar"
    bboxes = await client.get_bboxes_for_image(restored.uuid)
    assert len(bboxes) == 1
    b = bboxes[0]
    assert b.uuid == bbox_uuid
    # Label resolved by name to the new UUID
    assert b.label_uuid == new_label.uuid, "label should be resolved by name"
    tag_uuids = [t.uuid for t in (b.tags or [])]
    assert new_tag.uuid in tag_uuids, "tag should be resolved by name"

    await client.close()


async def test_rebuild_skips_bbox_with_unresolvable_label(tmp_path):
    """
    A bbox whose label UUID is not in the DB and whose label_name is missing
    (or also absent) is silently skipped — the image is still created.
    """
    files_root = tmp_path / "files"
    files_root.mkdir()

    image_uuid = str(uuid_mod.uuid4())
    bbox_uuid = str(uuid_mod.uuid4())
    image_path = files_root / "skip_test.png"
    image_path.write_bytes(make_png())

    sidecar_path = image_path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(
            {
                "uuid": image_uuid,
                "filename": str(image_path),
                "boxes": [
                    {
                        "uuid": bbox_uuid,
                        "label_uuid": "00000000-dead-beef-0000-000000000000",  # no such label
                        # no label_name provided — no fallback possible
                        "x": 0.1,
                        "y": 0.1,
                        "width": 0.2,
                        "height": 0.2,
                        "tag_uuids": [],
                        "tag_names": [],
                    }
                ],
            }
        )
    )

    db_dir = tmp_path / "db"
    db_dir.mkdir()
    client = DbClient(db_dir=db_dir)
    await client.init_db()

    restored = await client.sync_image_from_json(image_path)

    assert restored.uuid == image_uuid, "image should still be created"
    bboxes = await client.get_bboxes_for_image(restored.uuid)
    assert len(bboxes) == 0, "unresolvable bbox must be skipped"

    await client.close()


async def test_rebuild_preserves_existing_bboxes_on_merge(tmp_path):
    """
    sync_image_from_json in merge mode (overwrite=False) does not replace
    bboxes that are already in the DB — only new UUID bboxes are added.
    """
    files_root = tmp_path / "files"
    files_root.mkdir()

    db_dir = tmp_path / "db"
    db_dir.mkdir()
    client = DbClient(db_dir=db_dir)
    await client.init_db()

    label = await client.add_label("cat", "#FF0000")
    image_path = files_root / "merge_test.png"
    image_path.write_bytes(make_png())

    # Seed the DB with one bbox
    image = await client.add_image(image_path)
    existing_bid = str(uuid_mod.uuid4())
    await client.write_image_metadata_to_db(
        image.uuid,
        ImageRecordUpdate(
            bboxes=[
                BBoxUpdate(
                    uuid=existing_bid,
                    label_uuid=label.uuid,
                    x=0.0,
                    y=0.0,
                    width=0.1,
                    height=0.1,
                    tag_uuids=[],
                )
            ]
        ),
    )

    # Sidecar has the existing bbox plus a brand-new one
    new_bid = str(uuid_mod.uuid4())
    sidecar_path = image_path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(
            {
                "uuid": image.uuid,
                "filename": str(image_path),
                "boxes": [
                    {
                        "uuid": existing_bid,
                        "label_uuid": label.uuid,
                        "label_name": "cat",
                        "x": 0.0,
                        "y": 0.0,
                        "width": 0.1,
                        "height": 0.1,
                        "tag_uuids": [],
                        "tag_names": [],
                    },
                    {
                        "uuid": new_bid,
                        "label_uuid": label.uuid,
                        "label_name": "cat",
                        "x": 0.5,
                        "y": 0.5,
                        "width": 0.2,
                        "height": 0.2,
                        "tag_uuids": [],
                        "tag_names": [],
                    },
                ],
            }
        )
    )

    # merge (default overwrite=False)
    await client.sync_image_from_json(image_path, overwrite=False)

    bboxes = await client.get_bboxes_for_image(image.uuid)
    bbox_uuids = {b.uuid for b in bboxes}
    assert existing_bid in bbox_uuids, "existing bbox must be preserved"
    assert new_bid in bbox_uuids, "new bbox from sidecar must be added"
    assert len(bboxes) == 2

    await client.close()


async def test_rebuild_overwrites_bboxes_when_requested(tmp_path):
    """
    sync_image_from_json with overwrite=True replaces DB bboxes with the
    sidecar's version (the authoritative source after a DB wipe).
    """
    files_root = tmp_path / "files"
    files_root.mkdir()

    db_dir = tmp_path / "db"
    db_dir.mkdir()
    client = DbClient(db_dir=db_dir)
    await client.init_db()

    label = await client.add_label("cat", "#FF0000")
    image_path = files_root / "overwrite_test.png"
    image_path.write_bytes(make_png())

    # DB has two bboxes
    image = await client.add_image(image_path)
    stale_bid1 = str(uuid_mod.uuid4())
    stale_bid2 = str(uuid_mod.uuid4())
    await client.write_image_metadata_to_db(
        image.uuid,
        ImageRecordUpdate(
            bboxes=[
                BBoxUpdate(
                    uuid=stale_bid1,
                    label_uuid=label.uuid,
                    x=0.0,
                    y=0.0,
                    width=0.1,
                    height=0.1,
                    tag_uuids=[],
                ),
                BBoxUpdate(
                    uuid=stale_bid2,
                    label_uuid=label.uuid,
                    x=0.5,
                    y=0.5,
                    width=0.1,
                    height=0.1,
                    tag_uuids=[],
                ),
            ]
        ),
    )

    # Sidecar has only one bbox (the authoritative truth)
    canonical_bid = str(uuid_mod.uuid4())
    sidecar_path = image_path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(
            {
                "uuid": image.uuid,
                "filename": str(image_path),
                "boxes": [
                    {
                        "uuid": canonical_bid,
                        "label_uuid": label.uuid,
                        "label_name": "cat",
                        "x": 0.2,
                        "y": 0.3,
                        "width": 0.4,
                        "height": 0.1,
                        "tag_uuids": [],
                        "tag_names": [],
                    }
                ],
            }
        )
    )

    await client.sync_image_from_json(image_path, overwrite=True)

    bboxes = await client.get_bboxes_for_image(image.uuid)
    assert len(bboxes) == 1, "overwrite should leave only the sidecar's bbox"
    assert bboxes[0].uuid == canonical_bid


async def test_overwrite_when_sidecar_bbox_uuids_already_in_db(tmp_path):
    """Re-syncing a sidecar over a DB that already holds the same bbox UUIDs.

    Reassigning the relationship made SQLAlchemy UPDATE the existing rows rather
    than insert new ones, and that UPDATE spans `confidence` — a GENERATED
    column SQLite refuses to write, so the whole restore raised. Reconcile does
    exactly this whenever a sidecar is newer than the DB row.
    """
    files_root = tmp_path / "files"
    files_root.mkdir()
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    client = DbClient(db_dir=db_dir)
    await client.init_db()

    label = await client.add_label("cat", "#FF0000")
    image_path = files_root / "resync.png"
    image_path.write_bytes(make_png())

    image = await client.add_image(image_path)
    bid = str(uuid_mod.uuid4())
    await client.write_image_metadata_to_db(
        image.uuid,
        ImageRecordUpdate(
            bboxes=[BBoxUpdate(uuid=bid, label_uuid=label.uuid, x=0.0, y=0.0, width=0.1, height=0.1, tag_uuids=[])]
        ),
    )

    # Same bbox UUID as the DB, moved coordinates — the sidecar is authoritative.
    image_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "uuid": image.uuid,
                "filename": str(image_path),
                "boxes": [
                    {
                        "uuid": bid,
                        "label_uuid": label.uuid,
                        "label_name": "cat",
                        "x": 0.5,
                        "y": 0.5,
                        "width": 0.2,
                        "height": 0.2,
                        "tag_uuids": [],
                        "tag_names": [],
                    }
                ],
            }
        )
    )

    await client.sync_image_from_json(image_path, overwrite=True)

    bboxes = await client.get_bboxes_for_image(image.uuid)
    assert len(bboxes) == 1
    assert bboxes[0].uuid == bid
    assert bboxes[0].x == pytest.approx(0.5), "sidecar geometry should win"
    await client.close()


async def test_overwrite_from_corrupt_sidecar_does_not_wipe_bboxes(tmp_path):
    """A present-but-unparseable sidecar must NOT be treated as "empty" and
    overwrite the DB's bboxes with nothing. sync_image_from_json(overwrite=True)
    must refuse (raise SidecarParseError) and leave the DB untouched.
    """
    from apps.helpers.db.db_client import SidecarParseError

    files_root = tmp_path / "files"
    files_root.mkdir()
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    client = DbClient(db_dir=db_dir)
    await client.init_db()

    label = await client.add_label("cat", "#FF0000")
    image_path = files_root / "corrupt_test.png"
    image_path.write_bytes(make_png())
    image = await client.add_image(image_path)
    await client.write_image_metadata_to_db(
        image.uuid,
        ImageRecordUpdate(
            bboxes=[
                BBoxUpdate(
                    uuid=str(uuid_mod.uuid4()),
                    label_uuid=label.uuid,
                    x=0.0,
                    y=0.0,
                    width=0.1,
                    height=0.1,
                    tag_uuids=[],
                )
            ]
        ),
    )

    # Corrupt (truncated / invalid JSON) sidecar.
    sidecar_path = image_path.with_suffix(".json")
    sidecar_path.write_text('{ "uuid": "%s", "boxes": [ {invalid' % image.uuid)

    with pytest.raises(SidecarParseError):
        await client.sync_image_from_json(image_path, overwrite=True)

    bboxes = await client.get_bboxes_for_image(image.uuid)
    assert len(bboxes) == 1, "corrupt sidecar must not wipe existing annotations"

    await client.close()


def test_sidecar_write_failure_preserves_existing_file(tmp_path, monkeypatch):
    """write_image_metadata_to_json must be atomic (temp file + os.replace) so a
    failure mid-write leaves the previous valid sidecar intact rather than a
    truncated/torn file.
    """
    import os

    client = DbClient(db_dir=tmp_path, memory=True)

    json_path = tmp_path / "atomic.json"
    original = {"uuid": "original-uuid", "boxes": [{"uuid": "keep-me"}]}
    json_path.write_text(json.dumps(original))

    def boom(*_args, **_kwargs):
        raise OSError("simulated crash during replace")

    monkeypatch.setattr(os, "replace", boom)

    with pytest.raises(OSError):
        client.write_image_metadata_to_json(
            json_path,
            "new-uuid",
            str(tmp_path / "img.png"),
            ImageRecordUpdate(bboxes=[]),
        )

    # The original sidecar must be unchanged and still valid JSON.
    data = json.loads(json_path.read_text())
    assert data["uuid"] == "original-uuid"
    # No leftover temp file next to the sidecar.
    assert not list(tmp_path.glob("*.tmp"))


# ──────────────────────────────────────────────────────────────────────────────
# sync_image_from_json must recover `sourceUuid` from writer-authored sidecars
# (e.g. SaveSnapshotAction's watch-triggered captures) that never go through
# add_image/upload_images. Without this, an image is filed with source_uuid=NULL
# and silently drops out of every "filter by source" query.
# ──────────────────────────────────────────────────────────────────────────────


async def test_sync_from_json_picks_up_source_uuid_on_insert(tmp_path):
    client = DbClient(db_dir=tmp_path, memory=True)
    await client.init_db()
    source = await client.add_source("cam1", "rtsp", {})

    image_path = tmp_path / "captured.png"
    image_path.write_bytes(make_png())
    image_path.with_suffix(".json").write_text(json.dumps({"sourceUuid": source.uuid}))

    record = await client.sync_image_from_json(image_path)

    assert record.source_uuid == source.uuid
    await client.close()


async def test_sync_from_json_ignores_deleted_source(tmp_path):
    """A stale sourceUuid (source deleted since the sidecar was written) must not
    fail the whole sync — the image is filed unattributed instead."""
    client = DbClient(db_dir=tmp_path, memory=True)
    await client.init_db()

    image_path = tmp_path / "captured.png"
    image_path.write_bytes(make_png())
    image_path.with_suffix(".json").write_text(json.dumps({"sourceUuid": str(uuid_mod.uuid4())}))

    record = await client.sync_image_from_json(image_path)

    assert record.source_uuid is None
    await client.close()


async def test_sync_from_json_backfills_blank_source_but_never_overwrites(tmp_path):
    client = DbClient(db_dir=tmp_path, memory=True)
    await client.init_db()
    source_a = await client.add_source("cam1", "rtsp", {})
    source_b = await client.add_source("cam2", "rtsp", {})

    image_path = tmp_path / "captured.png"
    image_path.write_bytes(make_png())
    await client.add_image(image_path)  # no source yet

    image_path.with_suffix(".json").write_text(json.dumps({"sourceUuid": source_a.uuid}))
    record = await client.sync_image_from_json(image_path, overwrite=True)
    assert record.source_uuid == source_a.uuid

    # A later sync with a different source must not clobber an already-attributed image.
    image_path.with_suffix(".json").write_text(json.dumps({"sourceUuid": source_b.uuid}))
    record = await client.sync_image_from_json(image_path, overwrite=True)
    assert record.source_uuid == source_a.uuid

    await client.close()
