"""Tests for task-config JSON sidecar syncing.

Scenarios:
  - create config → task_configs.json written
  - update config → JSON updated
  - delete config (no instances) → JSON updated (config removed)
  - wipe task_configs table → sync_task_configs_from_json restores configs
  - marked_for_delete configs are excluded from the JSON export
"""

import json

import pytest

from apps.helpers.db.db_client import TASK_CONFIGS_JSON, DbClient
from apps.helpers.db.types import TaskConfig

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _read_json(db_client: DbClient) -> list[dict]:
    path = db_client._task_configs_json_path
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _uuids_in_json(db_client: DbClient) -> set[str]:
    return {c["uuid"] for c in _read_json(db_client)}


# ──────────────────────────────────────────────────────────────────────────────
# Create
# ──────────────────────────────────────────────────────────────────────────────


async def test_create_config_writes_json(db_client):
    config = await db_client.add_task_config("my-task", "FooTask", {"k": "v"})
    assert db_client._task_configs_json_path.exists()
    data = _read_json(db_client)
    assert any(c["uuid"] == config.uuid for c in data)


async def test_json_contains_name_and_typename(db_client):
    config = await db_client.add_task_config("named-task", "BarTask", {"x": 1})
    data = _read_json(db_client)
    entry = next(c for c in data if c["uuid"] == config.uuid)
    assert entry["name"] == "named-task"
    assert entry["typename"] == "BarTask"
    assert "params_json" in entry
    assert entry["params_json"]["x"] == 1


async def test_json_excludes_timestamps(db_client):
    config = await db_client.add_task_config("ts-task", "T", {})
    data = _read_json(db_client)
    entry = next(c for c in data if c["uuid"] == config.uuid)
    assert "created_at" not in entry
    assert "updated_at" not in entry


async def test_json_excludes_instances_key(db_client):
    config = await db_client.add_task_config("inst-task", "T", {})
    data = _read_json(db_client)
    entry = next(c for c in data if c["uuid"] == config.uuid)
    assert "instances" not in entry


# ──────────────────────────────────────────────────────────────────────────────
# Update
# ──────────────────────────────────────────────────────────────────────────────


async def test_update_config_updates_json(db_client):
    config = await db_client.add_task_config("orig", "T", {"a": 1})
    await db_client.update_task_config(config.uuid, name="renamed")
    data = _read_json(db_client)
    entry = next(c for c in data if c["uuid"] == config.uuid)
    assert entry["name"] == "renamed"


# ──────────────────────────────────────────────────────────────────────────────
# Delete
# ──────────────────────────────────────────────────────────────────────────────


async def test_delete_config_removes_from_json(db_client):
    config = await db_client.add_task_config("to-delete", "T", {})
    uuid = config.uuid
    assert uuid in _uuids_in_json(db_client)
    await db_client.delete_task_config(uuid)
    assert uuid not in _uuids_in_json(db_client)


async def test_multiple_configs_in_json(db_client):
    c1 = await db_client.add_task_config("c1", "T", {})
    c2 = await db_client.add_task_config("c2", "T", {})
    uuids = _uuids_in_json(db_client)
    assert c1.uuid in uuids
    assert c2.uuid in uuids


# ──────────────────────────────────────────────────────────────────────────────
# Restore from JSON (sync_task_configs_from_json)
# ──────────────────────────────────────────────────────────────────────────────


async def test_restore_after_db_wipe(db_client):
    """Create configs, wipe the table, restore from JSON, verify they're back."""
    c1 = await db_client.add_task_config("restore-1", "TypeA", {"n": 1})
    c2 = await db_client.add_task_config("restore-2", "TypeB", {"n": 2})

    # Wipe the task_configs table directly.
    from sqlmodel import delete as sql_delete

    from apps.helpers.db.types import TaskConfig

    async with db_client._async_session_maker() as session:
        await session.exec(sql_delete(TaskConfig))
        await session.commit()

    # Verify the table is empty.
    restored_empty = await db_client.get_task_configs()
    assert len(restored_empty) == 0

    # Restore from JSON.
    await db_client.sync_task_configs_from_json()

    restored = await db_client.get_task_configs()
    assert len(restored) == 2
    uuids = set(restored.keys())
    assert c1.uuid in uuids
    assert c2.uuid in uuids
    assert restored[c1.uuid].name == "restore-1"
    assert restored[c2.uuid].params_json == {"n": 2}


async def test_restore_uuid_match(db_client):
    """UUID-first resolution: existing record is updated in place, not duplicated."""
    config = await db_client.add_task_config("orig-name", "T", {"k": 1})

    # Manually write a JSON with the same UUID but a different name.
    json_data = [
        {
            "uuid": config.uuid,
            "name": "json-name",
            "typename": "T",
            "params_json": {"k": 99},
        }
    ]
    db_client._task_configs_json_path.write_text(json.dumps(json_data))

    await db_client.sync_task_configs_from_json()

    configs = await db_client.get_task_configs()
    assert len(configs) == 1
    assert configs[config.uuid].name == "json-name"
    assert configs[config.uuid].params_json == {"k": 99}


async def test_restore_name_fallback(db_client):
    """Name-fallback: JSON UUID differs but name matches existing DB record."""
    config = await db_client.add_task_config("same-name", "T", {})
    other_uuid = "00000000-0000-0000-0000-000000000099"

    json_data = [{"uuid": other_uuid, "name": "same-name", "typename": "T2", "params_json": {}}]
    db_client._task_configs_json_path.write_text(json.dumps(json_data))

    await db_client.sync_task_configs_from_json()

    configs = await db_client.get_task_configs()
    # Should reuse the DB UUID (no new record created).
    assert len(configs) == 1
    assert config.uuid in configs
    assert other_uuid not in configs


async def test_restore_new_insert(db_client):
    """No match: inserts a new record with the JSON UUID."""
    new_uuid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    json_data = [
        {
            "uuid": new_uuid,
            "name": "brand-new",
            "typename": "NewType",
            "params_json": {},
        }
    ]
    db_client._task_configs_json_path.write_text(json.dumps(json_data))

    await db_client.sync_task_configs_from_json()

    configs = await db_client.get_task_configs()
    assert new_uuid in configs
    assert configs[new_uuid].name == "brand-new"


async def test_missing_json_file_is_no_op(db_client):
    """sync_task_configs_from_json does nothing if the file doesn't exist."""
    if db_client._task_configs_json_path.exists():
        db_client._task_configs_json_path.unlink()
    await db_client.sync_task_configs_from_json()  # Should not raise.


# ──────────────────────────────────────────────────────────────────────────────
# Marked-for-delete exclusion
# ──────────────────────────────────────────────────────────────────────────────


async def test_marked_for_delete_excluded_from_json(db_client):
    """Configs marked_for_delete are not written to task_configs.json."""
    config = await db_client.add_task_config("mfd-task", "T", {})
    # Mark directly (bypassing the delete method's instance check).
    await db_client.update_task_config(config.uuid, marked_for_delete=True)
    # The sync should have fired after update; the config should be excluded.
    assert config.uuid not in _uuids_in_json(db_client)


# ──────────────────────────────────────────────────────────────────────────────
# Startup seed (init_db)
# ──────────────────────────────────────────────────────────────────────────────


async def test_startup_seeds_from_json(tmp_path):
    """init_db populates from task_configs.json when the table is empty."""
    seed_uuid = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    json_data = [
        {
            "uuid": seed_uuid,
            "name": "seeded-task",
            "typename": "T",
            "params_json": {"x": 42},
        }
    ]
    (tmp_path / TASK_CONFIGS_JSON).write_text(json.dumps(json_data))

    client = DbClient(db_dir=tmp_path, memory=True)
    await client.init_db()
    try:
        configs = await client.get_task_configs()
        assert seed_uuid in configs
        assert configs[seed_uuid].name == "seeded-task"
        assert configs[seed_uuid].params_json == {"x": 42}
    finally:
        await client.close()
