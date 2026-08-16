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


class TestImageLabelSidecarCreation:
    """Images with no bounding boxes have no sidecar until an assessment needs one.

    Sidecars are otherwise only ever created by a bbox write, and "I checked,
    this label is not here" is precisely the annotation made on a box-free
    image — so refusing to create the file lost exactly the assessments the
    sidecars exist to survive.

    The file is created only when it would carry something: an image with no
    assessments and no boxes gets none, because "this image exists" is already
    said by the image.
    """

    @pytest.fixture
    async def boxless(self, db_client, files_root):
        img = files_root / "boxless.png"
        img.write_bytes(make_png())
        record = await db_client.add_image(img)
        return record.uuid, img, img.with_suffix(".json")

    async def test_absent_on_a_boxless_image_creates_the_sidecar(self, http_client, label_uuid, boxless):
        _image_uuid, _img, side = boxless
        assert not side.exists(), "fixture must start with no sidecar"

        r = await http_client.post(
            "/api/images/labels/set",
            json={"imagePath": "boxless.png", "labelUuid": label_uuid, "status": "absent"},
        )
        assert r.status_code == 200

        assert side.exists(), "an absent mark on a boxless image must reach disk"
        data = json.loads(side.read_text())
        assert [row["status"] for row in data["image_labels"]] == ["absent"]

    async def test_created_sidecar_is_reconcilable(self, http_client, label_uuid, boxless):
        """A sidecar with no uuid/boxes is skipped by the reconcile path, so
        creating one without them would trade silent loss for silent loss."""
        image_uuid, _img, side = boxless
        await http_client.post(
            "/api/images/labels/set",
            json={"imagePath": "boxless.png", "labelUuid": label_uuid, "status": "absent"},
        )

        data = json.loads(side.read_text())
        assert data["uuid"] == image_uuid
        assert data["boxes"] == []
        assert data["filename"]

    async def test_created_sidecar_survives_a_rebuild(self, http_client, db_client, files_root, label_uuid, boxless):
        _image_uuid, img, _side = boxless
        await http_client.post(
            "/api/images/labels/set",
            json={"imagePath": "boxless.png", "labelUuid": label_uuid, "status": "absent"},
        )

        fresh = DbClient(db_dir=files_root.parent / "freshbox", memory=True, files_root=files_root)
        await fresh.init_db()
        try:
            await fresh.add_label("cat", "#FF0000")
            record = await fresh.sync_image_from_json(img)
            rows = await fresh.get_image_labels_for_sidecar(record.uuid)
        finally:
            await fresh.close()

        assert [(r["label_name"], r["status"]) for r in rows] == [("cat", "absent")]

    async def test_bulk_set_absent_creates_the_sidecar(self, http_client, label_uuid, boxless):
        image_uuid, _img, side = boxless

        r = await http_client.post(
            "/api/images/bulk-apply-by-uuids",
            json={"imageUuids": [image_uuid], "action": "setAbsent", "labelUuids": [label_uuid]},
        )
        assert r.status_code == 200

        assert side.exists()
        data = json.loads(side.read_text())
        assert [row["status"] for row in data["image_labels"]] == ["absent"]
        assert data["uuid"] == image_uuid

    async def test_no_sidecar_when_there_is_nothing_to_record(self, http_client, label_uuid, boxless):
        """ "This image exists" is not information — the image file already says
        that. Without assessments or boxes there is nothing to preserve, so no
        file is written."""
        _image_uuid, _img, side = boxless

        # Reverting to uncertain deletes the row, leaving an image with no
        # assessments and no boxes.
        r = await http_client.post(
            "/api/images/labels/set",
            json={"imagePath": "boxless.png", "labelUuid": label_uuid, "status": "uncertain"},
        )
        assert r.status_code == 200
        assert not side.exists(), "an empty sidecar is litter, not durability"

    async def test_bulk_action_with_nothing_to_record_writes_no_file(self, http_client, label_uuid, boxless):
        image_uuid, _img, side = boxless

        r = await http_client.post(
            "/api/images/bulk-apply-by-uuids",
            json={"imageUuids": [image_uuid], "action": "setUncertain", "labelUuids": [label_uuid]},
        )
        assert r.status_code == 200
        assert not side.exists()

    async def test_clearing_the_last_assessment_still_updates_an_existing_sidecar(
        self, http_client, label_uuid, boxless
    ):
        """Once the file exists, an empty list is a real instruction ("the user
        cleared everything") and must be written — the skip rule is only about
        creating a file, never about refusing to update one."""
        _image_uuid, _img, side = boxless
        for status in ("absent", "uncertain"):
            r = await http_client.post(
                "/api/images/labels/set",
                json={"imagePath": "boxless.png", "labelUuid": label_uuid, "status": status},
            )
            assert r.status_code == 200

        assert side.exists()
        assert json.loads(side.read_text())["image_labels"] == []

    async def test_recreated_sidecar_keeps_the_images_boxes(self, http_client, label_uuid, image_file, sidecar):
        """A deleted sidecar is rebuilt from the DB, not stamped with boxes: []."""
        bbox_uuid = await _seed_sidecar(http_client, label_uuid)
        sidecar.unlink()

        r = await _set_status(http_client, label_uuid, "absent")
        assert r.status_code == 200

        data = json.loads(sidecar.read_text())
        assert [b["uuid"] for b in data["boxes"]] == [bbox_uuid], "must not erase boxes the DB still has"


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

    async def test_backfill_creates_a_missing_sidecar(
        self, http_client, db_client, files_root, image_file, label_uuid, sidecar
    ):
        """A missing sidecar is written, complete, rather than skipped — the
        images most likely to lack one are the boxless ones, which are exactly
        the ones whose assessments have nowhere else to live."""
        bbox_uuid = await _seed_sidecar(http_client, label_uuid)
        await _set_status(http_client, label_uuid, "absent")
        image_uuid = json.loads(sidecar.read_text())["uuid"]
        sidecar.unlink()

        manager = Manager(db_client=db_client, files_root=files_root)
        stats = await manager.backfill_image_label_sidecars()

        assert stats == {"written": 0, "created": 1, "skipped_no_data": 0, "skipped_no_path": 0}
        data = json.loads(sidecar.read_text())
        assert data["uuid"] == image_uuid
        assert [b["uuid"] for b in data["boxes"]] == [bbox_uuid]
        assert [r["status"] for r in data["image_labels"]] == ["absent"]
