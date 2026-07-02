"""Tests for the chunked readiness/bucket query helpers.

These helpers exist so unbounded dataset unions never exceed the SQLite
bind-variable cap ("too many SQL variables"): every IN (uuid, ...) list is
chunked and the per-chunk results merged. The tests force a tiny chunk size to
prove the merge logic is correct across chunk boundaries.
"""

import uuid as uuidlib

import pytest

import apps.helpers.db.db_client as db_client_module
from apps.helpers.db.db_client import _chunked
from apps.helpers.db.types import (
    VERIFIED_TAG_UUID,
    BBoxUpdate,
    ImageItemStatus,
    ImageRecordUpdate,
)


class TestChunked:
    def test_partitions_in_order(self):
        assert [list(c) for c in _chunked(list(range(7)), 3)] == [[0, 1, 2], [3, 4, 5], [6]]

    def test_exact_multiple(self):
        assert [list(c) for c in _chunked([1, 2, 3, 4], 2)] == [[1, 2], [3, 4]]

    def test_empty(self):
        assert list(_chunked([], 3)) == []

    def test_default_size_read_at_call_time(self, monkeypatch):
        monkeypatch.setattr(db_client_module, "_UUID_IN_CHUNK", 2)
        assert [list(c) for c in _chunked([1, 2, 3])] == [[1, 2], [3]]


@pytest.fixture
async def labelled_images(db_client):
    """10 images; even ones have label-A PRESENT, images 0-2 label-B ABSENT."""
    label_a = (await db_client.add_label(name="A", color="#111111")).uuid
    label_b = (await db_client.add_label(name="B", color="#222222")).uuid
    image_uuids = []
    for i in range(10):
        img = await db_client.add_image(filename=f"/imgs/helper_{i:02d}.jpg")
        image_uuids.append(img.uuid)
        if i % 2 == 0:
            await db_client.set_image_label_status(img.uuid, label_a, ImageItemStatus.PRESENT)
        if i < 3:
            await db_client.set_image_label_status(img.uuid, label_b, ImageItemStatus.ABSENT)
    return {"image_uuids": image_uuids, "label_a": label_a, "label_b": label_b}


class TestVerificationCountsChunked:
    async def test_counts_identical_across_chunk_sizes(self, db_client, labelled_images, monkeypatch):
        uuids = labelled_images["image_uuids"]
        labels = [labelled_images["label_a"], labelled_images["label_b"]]

        expected_verified, expected_present = await db_client._get_image_label_verification_counts(uuids, labels)

        monkeypatch.setattr(db_client_module, "_UUID_IN_CHUNK", 3)
        verified, present = await db_client._get_image_label_verification_counts(uuids, labels)

        assert verified == expected_verified
        assert present == expected_present
        # Spot-check: image 0 has A PRESENT + B ABSENT = 2 verified, 1 present.
        assert verified[uuids[0]] == 2
        assert present[uuids[0]] == 1
        # Image 1 has only B ABSENT.
        assert verified[uuids[1]] == 1
        assert uuids[1] not in present
        # Image 9 has nothing.
        assert uuids[9] not in verified


class TestGetImagesWithRelations:
    async def test_preserves_input_order_and_loads_relations(self, db_client):
        label = (await db_client.add_label(name="lbl", color="#333333")).uuid
        uuids = []
        for i in range(5):
            img = await db_client.add_image(filename=f"/imgs/hydrate_{i}.jpg")
            uuids.append(img.uuid)
            await db_client.write_image_metadata_to_db(
                img.uuid,
                ImageRecordUpdate(
                    bboxes=[
                        BBoxUpdate(
                            uuid=str(uuidlib.uuid4()),
                            label_uuid=label,
                            x=0.1,
                            y=0.1,
                            width=0.2,
                            height=0.2,
                            tag_uuids=[VERIFIED_TAG_UUID],
                        )
                    ]
                ),
            )

        requested = [uuids[3], uuids[0], uuids[4]]
        page = await db_client._get_images_with_relations(requested)
        assert [img.uuid for img in page] == requested
        # Relations are eagerly loaded (accessible outside a session).
        assert page[0].bboxes[0].label.name == "lbl"
        assert page[0].bboxes[0].tags[0].uuid == VERIFIED_TAG_UUID

    async def test_missing_uuids_dropped_empty_input_ok(self, db_client):
        assert await db_client._get_images_with_relations([]) == []
        img = await db_client.add_image(filename="/imgs/hydrate_solo.jpg")
        page = await db_client._get_images_with_relations([str(uuidlib.uuid4()), img.uuid])
        assert [i.uuid for i in page] == [img.uuid]
