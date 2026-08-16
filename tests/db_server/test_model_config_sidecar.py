"""Tests for model-config JSON dump/restore (model_configs.json).

model_configs were the last user-authored table with no on-disk copy, so a
lost database took every per-label model assignment with it.
"""

import json

import pytest

from apps.helpers.db.db_client import DbClient
from apps.helpers.db.types import ModelConfigCreate, ModelConfigNodeWrite


def _read_json(db_client: DbClient) -> list[dict]:
    path = db_client._model_configs_json_path
    if not path.exists():
        return []
    return json.loads(path.read_text())


async def _make_config(db_client, name="cfg"):
    label = await db_client.add_label("cat", "#FF0000")
    config = await db_client.create_model_config(
        ModelConfigCreate(
            name=name,
            description="a config",
            nodes=[
                ModelConfigNodeWrite(
                    label_uuid=label.uuid,
                    model_type="detection",
                    architecture="yolov8",
                )
            ],
        )
    )
    return label, config


class TestModelConfigDump:
    async def test_create_writes_json(self, db_client):
        _label, config = await _make_config(db_client)
        data = _read_json(db_client)
        assert [c["uuid"] for c in data] == [config.uuid]
        assert data[0]["name"] == "cfg"
        assert data[0]["description"] == "a config"

    async def test_nodes_carry_label_name(self, db_client):
        """The name is the fallback key when a rebuild mints new label UUIDs."""
        label, _config = await _make_config(db_client)
        node = _read_json(db_client)[0]["nodes"][0]
        assert node["label_uuid"] == label.uuid
        assert node["label_name"] == "cat"
        assert node["model_type"] == "detection"
        assert node["architecture"] == "yolov8"

    async def test_nodes_carry_their_own_uuid(self, db_client):
        """training_configs.model_config_node_uuid is an FK to the node, and
        trained models record it in their metadata — mint a new one on restore
        and every one of those references dangles."""
        _label, config = await _make_config(db_client)
        node = _read_json(db_client)[0]["nodes"][0]
        assert node["uuid"] == config.nodes[0].uuid

    async def test_delete_removes_from_json(self, db_client):
        _label, config = await _make_config(db_client)
        assert await db_client.delete_model_config(config.uuid) is True
        assert _read_json(db_client) == []


class TestModelConfigRestore:
    async def test_restore_into_a_fresh_db(self, db_client, tmp_path, files_root):
        """A fresh DB pointed at the same db_dir recovers the configs."""
        label, config = await _make_config(db_client)
        dump = db_client._model_configs_json_path.read_text()

        fresh = DbClient(db_dir=tmp_path / "fresh", memory=True, files_root=files_root)
        await fresh.init_db()
        try:
            fresh._model_configs_json_path.parent.mkdir(parents=True, exist_ok=True)
            fresh._model_configs_json_path.write_text(dump)
            await fresh.add_label("cat", "#FF0000")
            await fresh.sync_model_configs_from_json()
            restored = await fresh.list_model_configs()
        finally:
            await fresh.close()

        assert [c.name for c in restored] == ["cfg"]
        assert [n.architecture for n in restored[0].nodes] == ["yolov8"]

    async def test_restore_resolves_label_by_name(self, db_client, tmp_path, files_root):
        """A node whose label_uuid no longer exists resolves via label_name."""
        _label, _config = await _make_config(db_client)
        dump = db_client._model_configs_json_path.read_text()

        fresh = DbClient(db_dir=tmp_path / "fresh2", memory=True, files_root=files_root)
        await fresh.init_db()
        try:
            fresh._model_configs_json_path.write_text(dump)
            new_label = await fresh.add_label("cat", "#FF0000")
            await fresh.sync_model_configs_from_json()
            restored = await fresh.list_model_configs()
        finally:
            await fresh.close()

        assert [n.label_uuid for n in restored[0].nodes] == [new_label.uuid]

    async def test_restore_preserves_node_uuids(self, db_client, tmp_path, files_root):
        """Node UUIDs round-trip, so training configs and trained models keep
        pointing at the node they were built against."""
        _label, config = await _make_config(db_client)
        node_uuid = config.nodes[0].uuid
        dump = db_client._model_configs_json_path.read_text()

        fresh = DbClient(db_dir=tmp_path / "fresh_nodes", memory=True, files_root=files_root)
        await fresh.init_db()
        try:
            fresh._model_configs_json_path.write_text(dump)
            await fresh.add_label("cat", "#FF0000")
            await fresh.sync_model_configs_from_json()
            restored = await fresh.list_model_configs()
        finally:
            await fresh.close()

        assert [n.uuid for n in restored[0].nodes] == [node_uuid]

    async def test_node_missing_required_key_is_dropped_not_fatal(self, db_client, tmp_path, files_root):
        """model_configs.json sits beside labels.json, which users do edit. A
        missing key must not raise out of init_db() and take db_server down."""
        _label, _config = await _make_config(db_client)
        data = json.loads(db_client._model_configs_json_path.read_text())
        del data[0]["nodes"][0]["model_type"]

        fresh = DbClient(db_dir=tmp_path / "fresh_partial", memory=True, files_root=files_root)
        await fresh.init_db()
        try:
            fresh._model_configs_json_path.write_text(json.dumps(data))
            await fresh.add_label("cat", "#FF0000")
            await fresh.sync_model_configs_from_json()
            restored = await fresh.list_model_configs()
        finally:
            await fresh.close()

        assert [c.name for c in restored] == ["cfg"]
        assert restored[0].nodes == []

    async def test_unresolvable_node_is_dropped_not_fatal(self, db_client, tmp_path, files_root):
        """Losing one node beats losing the whole config."""
        _label, _config = await _make_config(db_client)
        data = json.loads(db_client._model_configs_json_path.read_text())
        data[0]["nodes"][0]["label_name"] = "no-such-label"
        data[0]["nodes"][0]["label_uuid"] = "00000000-0000-0000-0000-0000000000ff"

        fresh = DbClient(db_dir=tmp_path / "fresh3", memory=True, files_root=files_root)
        await fresh.init_db()
        try:
            fresh._model_configs_json_path.write_text(json.dumps(data))
            await fresh.sync_model_configs_from_json()
            restored = await fresh.list_model_configs()
        finally:
            await fresh.close()

        assert [c.name for c in restored] == ["cfg"]
        assert restored[0].nodes == []
