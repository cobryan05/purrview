"""Whole-image label assessments (image_labels) in the JSON sidecar.

ABSENT rows — "I looked at this image and this label is not in it" — leave no
trace in `boxes`, so before this they existed only in SQLite. These tests cover
the round trip: assessments reach the sidecar, survive into a rebuilt DB, and
the backfill repairs sidecars written before the feature existed.
"""

import json
import uuid as uuid_mod

import pytest

from apps.db_server.manager import Manager
from apps.helpers.db.db_client import DbClient
from apps.helpers.db.types import UNVERIFIED_TAG_UUID, VERIFIED_TAG_UUID
from tests.conftest import make_png


@pytest.fixture
def image_file(files_root):
    img = files_root / "labels_sidecar.png"
    img.write_bytes(make_png())
    return img


@pytest.fixture
def sidecar(image_file):
    return image_file.with_suffix(".json")


@pytest.fixture
async def label_uuid(http_client):
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    return r.json()["label"]["uuid"]


async def _seed_sidecar(http_client, label_uuid, tag_uuid=UNVERIFIED_TAG_UUID):
    """Create a bbox so the image has a sidecar to attach assessments to."""
    bid = str(uuid_mod.uuid4())
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "labels_sidecar.png",
            "boxes": [
                {
                    "uuid": bid,
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [tag_uuid],
                }
            ],
        },
    )
    assert r.status_code == 200
    return bid


async def _set_status(http_client, label_uuid, status):
    return await http_client.post(
        "/api/images/labels/set",
        json={"imagePath": "labels_sidecar.png", "labelUuid": label_uuid, "status": status},
    )


class TestImageLabelSidecarWrite:
    async def test_absent_status_lands_in_sidecar(self, http_client, label_uuid, sidecar):
        """Marking a label absent records it in the sidecar, with its name."""
        await _seed_sidecar(http_client, label_uuid)
        r = await _set_status(http_client, label_uuid, "absent")
        assert r.status_code == 200

        data = json.loads(sidecar.read_text())
        assert data["image_labels"] == [{"label_uuid": label_uuid, "label_name": "cat", "status": "absent"}]

    async def test_uncertain_clears_the_sidecar_entry(self, http_client, label_uuid, sidecar):
        """Reverting to uncertain removes the row, and the sidecar follows."""
        await _seed_sidecar(http_client, label_uuid)
        await _set_status(http_client, label_uuid, "absent")
        await _set_status(http_client, label_uuid, "uncertain")

        data = json.loads(sidecar.read_text())
        assert data["image_labels"] == []

    async def test_verified_bbox_records_present(self, http_client, label_uuid, sidecar):
        """A verified bbox derives a PRESENT row, which reaches the sidecar too."""
        await _seed_sidecar(http_client, label_uuid, tag_uuid=VERIFIED_TAG_UUID)

        data = json.loads(sidecar.read_text())
        statuses = {row["label_uuid"]: row["status"] for row in data["image_labels"]}
        assert statuses == {label_uuid: "present"}

    async def test_bbox_edit_preserves_assessments(self, http_client, label_uuid, sidecar):
        """A later bbox write must not wipe the image_labels key."""
        await _seed_sidecar(http_client, label_uuid)
        await _set_status(http_client, label_uuid, "absent")

        # A second, unrelated label marked absent, then another bbox write.
        r = await http_client.post("/api/labels", json={"name": "dog", "color": "#00FF00"})
        dog_uuid = r.json()["label"]["uuid"]
        await _set_status(http_client, dog_uuid, "absent")
        await _seed_sidecar(http_client, label_uuid)

        data = json.loads(sidecar.read_text())
        absent = {row["label_uuid"] for row in data["image_labels"] if row["status"] == "absent"}
        assert dog_uuid in absent, "an unrelated bbox write must not drop assessments"

    async def test_bulk_set_absent_updates_sidecar(self, http_client, label_uuid, sidecar):
        """The bulk path writes the sidecar, not just the DB."""
        await _seed_sidecar(http_client, label_uuid)
        image_uuid = json.loads(sidecar.read_text())["uuid"]

        r = await http_client.post(
            "/api/images/bulk-apply-by-uuids",
            json={"imageUuids": [image_uuid], "action": "setAbsent", "labelUuids": [label_uuid]},
        )
        assert r.status_code == 200

        data = json.loads(sidecar.read_text())
        assert [row["status"] for row in data["image_labels"]] == ["absent"]


class TestImageLabelSidecarRestore:
    async def test_absent_survives_a_rebuild(self, http_client, files_root, image_file, label_uuid, sidecar, tmp_path):
        """A fresh DB restored from the sidecar recovers the ABSENT assessment."""
        await _seed_sidecar(http_client, label_uuid)
        await _set_status(http_client, label_uuid, "absent")

        fresh = DbClient(db_dir=tmp_path / "fresh", memory=True, files_root=files_root)
        await fresh.init_db()
        try:
            # The rebuilt DB needs the label to exist, exactly as a labels.json
            # restore would have provided it.
            await fresh.add_label("cat", "#FF0000")
            record = await fresh.sync_image_from_json(image_file)
            rows = await fresh.get_image_labels_for_sidecar(record.uuid)
        finally:
            await fresh.close()

        assert [(r["label_name"], r["status"]) for r in rows] == [("cat", "absent")]

    async def test_restore_falls_back_to_label_name(
        self, http_client, files_root, image_file, label_uuid, sidecar, tmp_path
    ):
        """A rebuild that minted a new label UUID still resolves by name."""
        await _seed_sidecar(http_client, label_uuid)
        await _set_status(http_client, label_uuid, "absent")

        fresh = DbClient(db_dir=tmp_path / "fresh2", memory=True, files_root=files_root)
        await fresh.init_db()
        try:
            new_label = await fresh.add_label("cat", "#FF0000")
            assert new_label.uuid != label_uuid, "fixture must exercise the fallback"
            record = await fresh.sync_image_from_json(image_file)
            rows = await fresh.get_image_labels_for_sidecar(record.uuid)
        finally:
            await fresh.close()

        assert [(r["label_uuid"], r["status"]) for r in rows] == [(new_label.uuid, "absent")]

    async def test_sidecar_without_the_key_leaves_db_rows_alone(
        self, http_client, files_root, image_file, label_uuid, sidecar, tmp_path, db_client
    ):
        """A pre-feature sidecar must not delete assessments on an overwrite sync."""
        await _seed_sidecar(http_client, label_uuid)
        await _set_status(http_client, label_uuid, "absent")
        image_uuid = json.loads(sidecar.read_text())["uuid"]

        # Simulate a sidecar written before image_labels existed.
        data = json.loads(sidecar.read_text())
        del data["image_labels"]
        sidecar.write_text(json.dumps(data))

        await db_client.sync_image_from_json(image_file, overwrite=True)
        rows = await db_client.get_image_labels_for_sidecar(image_uuid)
        assert [r["status"] for r in rows] == ["absent"]

    async def test_overwrite_sync_drops_cleared_assessments(
        self, http_client, image_file, label_uuid, sidecar, db_client
    ):
        """An empty image_labels list is a real instruction: clear the rows."""
        await _seed_sidecar(http_client, label_uuid)
        await _set_status(http_client, label_uuid, "absent")
        image_uuid = json.loads(sidecar.read_text())["uuid"]

        data = json.loads(sidecar.read_text())
        data["image_labels"] = []
        sidecar.write_text(json.dumps(data))

        await db_client.sync_image_from_json(image_file, overwrite=True)
        assert await db_client.get_image_labels_for_sidecar(image_uuid) == []


class TestImageLabelSidecarBackfill:
    async def test_backfill_repairs_a_pre_feature_sidecar(
        self, http_client, db_client, files_root, image_file, label_uuid, sidecar
    ):
        """The backfill writes assessments into a sidecar that lacks them."""
        await _seed_sidecar(http_client, label_uuid)
        await _set_status(http_client, label_uuid, "absent")

        data = json.loads(sidecar.read_text())
        del data["image_labels"]
        sidecar.write_text(json.dumps(data))

        manager = Manager(db_client=db_client, files_root=files_root)
        stats = await manager.backfill_image_label_sidecars()

        assert stats["written"] == 1
        restored = json.loads(sidecar.read_text())
        assert [r["status"] for r in restored["image_labels"]] == ["absent"]
        assert restored["boxes"], "backfill must not clobber the boxes it doesn't own"

    async def test_backfill_skips_images_with_no_sidecar(
        self, http_client, db_client, files_root, image_file, label_uuid, sidecar
    ):
        """No sidecar means no file to repair — never invent a boxless one."""
        await _seed_sidecar(http_client, label_uuid)
        await _set_status(http_client, label_uuid, "absent")
        sidecar.unlink()

        manager = Manager(db_client=db_client, files_root=files_root)
        stats = await manager.backfill_image_label_sidecars()

        assert stats == {"written": 0, "skipped_no_sidecar": 1, "skipped_no_path": 0}
        assert not sidecar.exists()
