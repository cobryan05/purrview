"""Tests for the tags.json sidecar restore path.

sync_tags_from_json() runs whenever the tags table is empty — i.e. the
rebuild/restore path the sidecars exist for. A tag that a migration removed
from the DB is still present in any tags.json dumped before that migration
ran, so the restore would otherwise resurrect it.
"""

import json

import pytest

from apps.helpers.db.db_client import TAGS_JSON, DbClient
from apps.helpers.db.types import RETIRED_TAG_UUIDS

FALSE_POSITIVE_TAG_UUID = "00000000-0000-0000-0000-000000000003"


def _write_tags_json(db_dir, rows: list) -> None:
    (db_dir / TAGS_JSON).write_text(json.dumps(rows, indent=2))


def test_false_positive_tag_is_retired():
    """The tag removed by migration c1e5a83f7d92 must be on the denylist."""
    assert FALSE_POSITIVE_TAG_UUID in RETIRED_TAG_UUIDS


async def test_retired_tag_not_restored_from_sidecar(tmp_path):
    """A stale tags.json must not resurrect a retired system tag."""
    _write_tags_json(
        tmp_path,
        [
            {
                "uuid": FALSE_POSITIVE_TAG_UUID,
                "name": "False Positive",
                "color": "#FF0000",
                "icon": "x",
                "protected": True,
                "kind": "system",
                "exclusive_group": "verification",
            }
        ],
    )

    client = DbClient(db_dir=tmp_path, memory=True)
    await client.init_db()

    tags = await client.list_tags()
    assert not [t for t in tags if t.uuid == FALSE_POSITIVE_TAG_UUID], "retired tag must not be restored"
    assert not [t for t in tags if t.exclusive_group == "verification" and t.name == "False Positive"]

    await client.close()


async def test_retired_tag_is_pruned_from_the_sidecar(tmp_path):
    """Once a tag write reruns the dump, the stale entry is gone for good."""
    _write_tags_json(
        tmp_path,
        [
            {
                "uuid": FALSE_POSITIVE_TAG_UUID,
                "name": "False Positive",
                "kind": "system",
                "exclusive_group": "verification",
            }
        ],
    )

    client = DbClient(db_dir=tmp_path, memory=True)
    await client.init_db()
    await client.add_tag("occluded", color="#00FF00")

    rows = json.loads((tmp_path / TAGS_JSON).read_text())
    assert not [r for r in rows if r["uuid"] == FALSE_POSITIVE_TAG_UUID]

    await client.close()


async def test_non_retired_tags_still_restore(tmp_path):
    """The denylist must not disturb ordinary sidecar restore."""
    _write_tags_json(
        tmp_path,
        [
            {"uuid": "11111111-1111-1111-1111-111111111111", "name": "blurry", "kind": "generic"},
            {"uuid": FALSE_POSITIVE_TAG_UUID, "name": "False Positive", "kind": "system"},
        ],
    )

    client = DbClient(db_dir=tmp_path, memory=True)
    await client.init_db()

    names = {t.name for t in await client.list_tags()}
    assert "blurry" in names
    assert "False Positive" not in names

    await client.close()
