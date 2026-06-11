"""Tests for image-level label state tracking.

PRESENT rows are derived automatically from verified bboxes.
ABSENT rows are set explicitly by the user and survive recomputation.
Unverified and false-positive bboxes produce no image_label rows.
"""

import uuid

import pytest

from apps.helpers.db.types import FALSE_POS_TAG_UUID, UNVERIFIED_TAG_UUID
from tests.conftest import make_png


@pytest.fixture
def image_file(files_root):
    img = files_root / "labels_test.png"
    img.write_bytes(make_png())
    return img


@pytest.fixture
async def label_uuid(http_client):
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    return r.json()["label"]["uuid"]


@pytest.fixture
async def label2_uuid(http_client):
    r = await http_client.post("/api/labels", json={"name": "dog", "color": "#0000FF"})
    return r.json()["label"]["uuid"]


async def _get_image_labels(http_client, image_path: str) -> dict:
    """Return {label_uuid: status} for the given image path."""
    r = await http_client.get(f"/api/images/{image_path}/labels")
    assert r.status_code == 200
    body = r.json()
    return {row["labelUuid"]: row["status"] for row in body.get("labels", [])}


async def _update_bboxes(http_client, image_path, boxes):
    r = await http_client.post(
        "/api/images/metadata/update",
        json={"imagePath": image_path, "boxes": boxes},
    )
    assert r.status_code == 200
    return r.json()


async def _set_label(http_client, image_path, label_uuid, status):
    r = await http_client.post(
        "/api/images/labels/set",
        json={"imagePath": image_path, "labelUuid": label_uuid, "status": status},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"


# ──────────────────────────────────────────────────────────────────────────────
# Derived PRESENT from verified bboxes
# ──────────────────────────────────────────────────────────────────────────────


async def test_confirmed_bbox_gives_present(http_client, image_file, label_uuid):
    """A bbox with no UNVERIFIED tag is 'confirmed' and produces a PRESENT row."""
    await _update_bboxes(
        http_client,
        "labels_test.png",
        [
            {
                "uuid": str(uuid.uuid4()),
                "labelUuid": label_uuid,
                "x": 0.1,
                "y": 0.1,
                "width": 0.2,
                "height": 0.2,
                "tagUuids": [],
            }
        ],
    )
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"


async def test_unverified_bbox_produces_no_row(http_client, image_file, label_uuid):
    await _update_bboxes(
        http_client,
        "labels_test.png",
        [
            {
                "uuid": str(uuid.uuid4()),
                "labelUuid": label_uuid,
                "x": 0.1,
                "y": 0.1,
                "width": 0.2,
                "height": 0.2,
                "tagUuids": [UNVERIFIED_TAG_UUID],
            }
        ],
    )
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert label_uuid not in labels


async def test_non_unverified_bbox_gives_present(http_client, image_file, label_uuid):
    """A bbox with a non-verification custom tag (but no UNVERIFIED) is PRESENT."""
    r = await http_client.post("/api/tags/add", json={"name": "custom-tag"})
    custom_uuid = r.json()["tag"]["uuid"]
    await _update_bboxes(
        http_client,
        "labels_test.png",
        [
            {
                "uuid": str(uuid.uuid4()),
                "labelUuid": label_uuid,
                "x": 0.1,
                "y": 0.1,
                "width": 0.1,
                "height": 0.1,
                "tagUuids": [custom_uuid],
            },
        ],
    )
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"


async def test_one_confirmed_bbox_gives_present_despite_unverified_sibling(http_client, image_file, label_uuid):
    """If any bbox for a label is confirmed, the label is PRESENT even if others are UNVERIFIED."""
    await _update_bboxes(
        http_client,
        "labels_test.png",
        [
            {
                "uuid": str(uuid.uuid4()),
                "labelUuid": label_uuid,
                "x": 0.1,
                "y": 0.1,
                "width": 0.1,
                "height": 0.1,
                "tagUuids": [UNVERIFIED_TAG_UUID],
            },
            {
                "uuid": str(uuid.uuid4()),
                "labelUuid": label_uuid,
                "x": 0.5,
                "y": 0.5,
                "width": 0.1,
                "height": 0.1,
                "tagUuids": [],
            },
        ],
    )
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"


async def test_stale_present_row_removed(http_client, image_file, label_uuid, label2_uuid):
    bbox1 = str(uuid.uuid4())
    bbox2 = str(uuid.uuid4())
    await _update_bboxes(
        http_client,
        "labels_test.png",
        [
            {"uuid": bbox1, "labelUuid": label_uuid, "x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1, "tagUuids": []},
            {"uuid": bbox2, "labelUuid": label2_uuid, "x": 0.5, "y": 0.5, "width": 0.1, "height": 0.1, "tagUuids": []},
        ],
    )
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"
    assert labels[label2_uuid] == "present"

    # Remove confirmed bbox for label2 → its PRESENT row should be cleaned up
    await _update_bboxes(
        http_client,
        "labels_test.png",
        [
            {"uuid": bbox1, "labelUuid": label_uuid, "x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1, "tagUuids": []},
        ],
    )
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"
    assert label2_uuid not in labels


async def test_bbox_patch_to_confirmed_gives_present(http_client, image_file, label_uuid):
    """Patching a bbox to remove UNVERIFIED (empty tags) creates a PRESENT row."""
    bbox_id = str(uuid.uuid4())
    await _update_bboxes(
        http_client,
        "labels_test.png",
        [
            {
                "uuid": bbox_id,
                "labelUuid": label_uuid,
                "x": 0.1,
                "y": 0.1,
                "width": 0.2,
                "height": 0.2,
                "tagUuids": [UNVERIFIED_TAG_UUID],
            }
        ],
    )
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert label_uuid not in labels  # unverified → no row

    patch_r = await http_client.patch(
        f"/api/images/bbox/{bbox_id}",
        json={"labelUuid": label_uuid, "tagUuids": []},
    )
    assert patch_r.status_code == 204

    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"


async def test_false_positive_bbox_produces_no_row(http_client, image_file, label_uuid):
    """A bbox tagged FALSE_POSITIVE must not generate a PRESENT image_label row."""
    await _update_bboxes(
        http_client,
        "labels_test.png",
        [
            {
                "uuid": str(uuid.uuid4()),
                "labelUuid": label_uuid,
                "x": 0.1,
                "y": 0.1,
                "width": 0.2,
                "height": 0.2,
                "tagUuids": [FALSE_POS_TAG_UUID],
            }
        ],
    )
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert label_uuid not in labels


async def test_one_confirmed_bbox_gives_present_despite_false_positive_sibling(http_client, image_file, label_uuid):
    """If any bbox for a label is confirmed, the label is PRESENT even if a sibling is FALSE_POSITIVE."""
    await _update_bboxes(
        http_client,
        "labels_test.png",
        [
            {
                "uuid": str(uuid.uuid4()),
                "labelUuid": label_uuid,
                "x": 0.1,
                "y": 0.1,
                "width": 0.1,
                "height": 0.1,
                "tagUuids": [FALSE_POS_TAG_UUID],
            },
            {
                "uuid": str(uuid.uuid4()),
                "labelUuid": label_uuid,
                "x": 0.5,
                "y": 0.5,
                "width": 0.1,
                "height": 0.1,
                "tagUuids": [],
            },
        ],
    )
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"


# ──────────────────────────────────────────────────────────────────────────────
# Explicit ABSENT
# ──────────────────────────────────────────────────────────────────────────────


async def test_set_absent_stored(http_client, image_file, label_uuid):
    await _update_bboxes(http_client, "labels_test.png", [])  # register image
    await _set_label(http_client, "labels_test.png", label_uuid, "absent")
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "absent"


async def test_explicit_absent_survives_recompute(http_client, image_file, label_uuid, label2_uuid):
    """ABSENT mark on label X must not be wiped when a bbox for label Y changes."""
    await _update_bboxes(http_client, "labels_test.png", [])  # register image
    await _set_label(http_client, "labels_test.png", label_uuid, "absent")

    # Trigger recompute via a bbox mutation on a different label
    await _update_bboxes(
        http_client,
        "labels_test.png",
        [
            {
                "uuid": str(uuid.uuid4()),
                "labelUuid": label2_uuid,
                "x": 0.1,
                "y": 0.1,
                "width": 0.1,
                "height": 0.1,
                "tagUuids": [],
            }
        ],
    )

    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "absent"  # survived recompute
    assert labels[label2_uuid] == "present"


async def test_set_uncertain_clears_absent(http_client, image_file, label_uuid):
    """Setting UNCERTAIN on an explicit ABSENT deletes the row."""
    await _update_bboxes(http_client, "labels_test.png", [])
    await _set_label(http_client, "labels_test.png", label_uuid, "absent")
    await _set_label(http_client, "labels_test.png", label_uuid, "uncertain")
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert label_uuid not in labels


async def test_confirmed_bbox_overrides_absent(http_client, image_file, label_uuid):
    """If a confirmed bbox is added for a label that was explicitly ABSENT, PRESENT wins."""
    await _update_bboxes(http_client, "labels_test.png", [])
    await _set_label(http_client, "labels_test.png", label_uuid, "absent")

    await _update_bboxes(
        http_client,
        "labels_test.png",
        [
            {
                "uuid": str(uuid.uuid4()),
                "labelUuid": label_uuid,
                "x": 0.1,
                "y": 0.1,
                "width": 0.2,
                "height": 0.2,
                "tagUuids": [],
            }
        ],
    )
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"


# ──────────────────────────────────────────────────────────────────────────────
# Error handling
# ──────────────────────────────────────────────────────────────────────────────


async def test_set_image_label_image_not_found(http_client, label_uuid):
    r = await http_client.post(
        "/api/images/labels/set",
        json={"imagePath": "ghost.png", "labelUuid": label_uuid, "status": "absent"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "failure"
