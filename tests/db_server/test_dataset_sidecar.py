"""Tests for the datasets.json sidecar mechanism.

Two concerns:
  1. Write side — create/update/delete operations write datasets.json with the
     correct structure.
  2. Read side — sync_datasets_from_json and init_db can restore a fresh DB
     from datasets.json, with UUID-first, name-fallback, and insert-new semantics.
"""

import json
import uuid as uuid_mod

import pytest

from apps.helpers.db.db_client import DATASETS_JSON, DbClient
from apps.helpers.db.types import DatasetCreate, DatasetFilterCriteria, DatasetUpdate

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


async def _create(client: DbClient, name: str, **kwargs) -> str:
    """Create a dataset and return its UUID."""
    ds = await client.create_dataset(DatasetCreate(name=name, **kwargs))
    return ds.uuid


async def _read_sidecar(db_dir) -> list:
    return json.loads((db_dir / DATASETS_JSON).read_text())


# ──────────────────────────────────────────────────────────────────────────────
# 1. Write side
# ──────────────────────────────────────────────────────────────────────────────


async def test_create_writes_sidecar(tmp_path):
    """Creating a dataset produces datasets.json with the correct fields."""
    client = DbClient(db_dir=tmp_path, memory=True)
    await client.init_db()

    uuid = await _create(client, "test-ds", description="hello")

    sidecar = tmp_path / DATASETS_JSON
    assert sidecar.exists(), "datasets.json must be written after create"

    rows = json.loads(sidecar.read_text())
    assert len(rows) == 1
    row = rows[0]
    assert row["uuid"] == uuid
    assert row["name"] == "test-ds"
    assert row["description"] == "hello"
    assert isinstance(row["filter_json"], dict)
    assert isinstance(row["label_uuids"], list)

    await client.close()


async def test_update_refreshes_sidecar(tmp_path):
    """Updating a dataset name/filter is reflected in datasets.json."""
    client = DbClient(db_dir=tmp_path, memory=True)
    await client.init_db()

    uuid = await _create(client, "before")

    await client.update_dataset(uuid, DatasetUpdate(name="after"))

    rows = await _read_sidecar(tmp_path)
    assert len(rows) == 1
    assert rows[0]["name"] == "after"

    await client.close()


async def test_delete_removes_from_sidecar(tmp_path):
    """Deleting a dataset removes it from datasets.json."""
    client = DbClient(db_dir=tmp_path, memory=True)
    await client.init_db()

    uuid1 = await _create(client, "keep")
    uuid2 = await _create(client, "delete-me")

    await client.delete_dataset(uuid2)

    rows = await _read_sidecar(tmp_path)
    uuids = {r["uuid"] for r in rows}
    assert uuid1 in uuids
    assert uuid2 not in uuids

    await client.close()


async def test_sidecar_includes_label_uuids(tmp_path):
    """datasets.json stores the label_uuids for the dataset."""
    client = DbClient(db_dir=tmp_path, memory=True)
    await client.init_db()

    label = await client.add_label("cat", "#FF0000")
    uuid = await _create(client, "with-labels", label_uuids=[label.uuid])

    rows = await _read_sidecar(tmp_path)
    assert rows[0]["label_uuids"] == [label.uuid]

    await client.close()


async def test_sidecar_stores_filter_json(tmp_path):
    """datasets.json preserves flat filter_json fields."""
    client = DbClient(db_dir=tmp_path, memory=True)
    await client.init_db()

    await _create(
        client,
        "filtered",
        filter_json=DatasetFilterCriteria(filename_search=".jpg"),
    )

    rows = await _read_sidecar(tmp_path)
    assert rows[0]["filter_json"]["filename_search"] == ".jpg"

    await client.close()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Read side
# ──────────────────────────────────────────────────────────────────────────────


async def test_sync_from_json_uuid_match(tmp_path):
    """sync_datasets_from_json restores a dataset when its UUID matches."""
    db_dir1 = tmp_path / "db1"
    db_dir1.mkdir()
    client1 = DbClient(db_dir=db_dir1, memory=True)
    await client1.init_db()
    uuid = await _create(client1, "restore-me")
    await client1.close()

    # Copy the sidecar to a fresh DB dir
    db_dir2 = tmp_path / "db2"
    db_dir2.mkdir()
    (db_dir2 / DATASETS_JSON).write_text((db_dir1 / DATASETS_JSON).read_text())

    client2 = DbClient(db_dir=db_dir2, memory=True)
    await client2.init_db()

    datasets = await client2.list_datasets()
    assert any(d.uuid == uuid and d.name == "restore-me" for d in datasets)

    await client2.close()


async def test_sync_from_json_name_fallback(tmp_path):
    """sync_datasets_from_json reuses an existing dataset when names match but UUIDs differ."""
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    client = DbClient(db_dir=db_dir, memory=True)
    await client.init_db()

    # Seed DB with a dataset under a known UUID
    existing = await client.create_dataset(DatasetCreate(name="shared-name"))
    existing_uuid = existing.uuid

    # Write a sidecar with the same name but a different UUID
    new_uuid = str(uuid_mod.uuid4())
    assert new_uuid != existing_uuid
    sidecar_data = [
        {
            "uuid": new_uuid,
            "name": "shared-name",
            "description": "from-json",
            "filter_json": {},
            "label_uuids": [],
        }
    ]
    (db_dir / DATASETS_JSON).write_text(json.dumps(sidecar_data))

    await client.sync_datasets_from_json()

    datasets = await client.list_datasets()
    # Should still be one dataset — the existing record was updated in-place
    named = [d for d in datasets if d.name == "shared-name"]
    assert len(named) == 1
    assert named[0].uuid == existing_uuid, "name-fallback must reuse the DB UUID"
    assert named[0].description == "from-json", "description should be updated"

    await client.close()


async def test_sync_from_json_inserts_new(tmp_path):
    """sync_datasets_from_json inserts a dataset when neither UUID nor name matches."""
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    client = DbClient(db_dir=db_dir, memory=True)
    await client.init_db()

    brand_new_uuid = str(uuid_mod.uuid4())
    sidecar_data = [
        {
            "uuid": brand_new_uuid,
            "name": "brand-new",
            "description": None,
            "filter_json": {},
            "label_uuids": [],
        }
    ]
    (db_dir / DATASETS_JSON).write_text(json.dumps(sidecar_data))

    await client.sync_datasets_from_json()

    datasets = await client.list_datasets()
    assert any(d.uuid == brand_new_uuid and d.name == "brand-new" for d in datasets)

    await client.close()


async def test_sync_from_json_restores_label_links(tmp_path):
    """sync_datasets_from_json recreates DatasetLabelLink rows."""
    db_dir1 = tmp_path / "db1"
    db_dir1.mkdir()
    client1 = DbClient(db_dir=db_dir1, memory=True)
    await client1.init_db()

    label = await client1.add_label("cat", "#FF0000")
    uuid = await _create(client1, "linked", label_uuids=[label.uuid])
    await client1.close()

    # Fresh DB — labels.json will restore the label, datasets.json will restore the link
    db_dir2 = tmp_path / "db2"
    db_dir2.mkdir()
    for fname in (DATASETS_JSON, "labels.json"):
        (db_dir2 / fname).write_text((db_dir1 / fname).read_text())

    client2 = DbClient(db_dir=db_dir2, memory=True)
    await client2.init_db()

    datasets = await client2.list_datasets()
    restored = next(d for d in datasets if d.uuid == uuid)
    assert label.uuid in restored.label_uuids

    await client2.close()


async def test_sync_from_json_skips_missing_label_uuids(tmp_path):
    """sync_datasets_from_json silently skips label_uuids that don't exist in DB."""
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    client = DbClient(db_dir=db_dir, memory=True)
    await client.init_db()

    ghost_uuid = str(uuid_mod.uuid4())
    ds_uuid = str(uuid_mod.uuid4())
    sidecar_data = [
        {
            "uuid": ds_uuid,
            "name": "ghost-labels",
            "description": None,
            "filter_json": {},
            "label_uuids": [ghost_uuid],
        }
    ]
    (db_dir / DATASETS_JSON).write_text(json.dumps(sidecar_data))

    # Must not raise; ghost label UUID should be silently dropped
    await client.sync_datasets_from_json()

    datasets = await client.list_datasets()
    restored = next(d for d in datasets if d.uuid == ds_uuid)
    assert restored.label_uuids == [], "non-existent label UUID must be skipped"

    await client.close()


async def test_init_db_auto_restores_from_sidecar(tmp_path):
    """init_db on a fresh DB calls sync_datasets_from_json when datasets table is empty."""
    db_dir1 = tmp_path / "db1"
    db_dir1.mkdir()
    client1 = DbClient(db_dir=db_dir1, memory=True)
    await client1.init_db()
    uuid = await _create(client1, "auto-restore", description="via init_db")
    await client1.close()

    db_dir2 = tmp_path / "db2"
    db_dir2.mkdir()
    (db_dir2 / DATASETS_JSON).write_text((db_dir1 / DATASETS_JSON).read_text())

    client2 = DbClient(db_dir=db_dir2, memory=True)
    # init_db should pick up datasets.json automatically
    await client2.init_db()

    datasets = await client2.list_datasets()
    assert any(d.uuid == uuid and d.description == "via init_db" for d in datasets)

    await client2.close()


async def test_no_sidecar_is_noop(tmp_path):
    """sync_datasets_from_json is a no-op when datasets.json doesn't exist."""
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    client = DbClient(db_dir=db_dir, memory=True)
    await client.init_db()

    # No sidecar file — should not raise
    await client.sync_datasets_from_json()

    datasets = await client.list_datasets()
    assert datasets == []

    await client.close()
