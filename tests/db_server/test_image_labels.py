"""Tests for image-level label state tracking.

PRESENT rows are derived automatically from verified bboxes.
ABSENT rows are set explicitly by the user and survive recomputation.
Unverified and false-positive bboxes produce no image_label rows.
"""

import uuid

import pytest

from apps.helpers.db.types import (
    FALSE_POS_TAG_UUID,
    UNVERIFIED_TAG_UUID,
    VERIFIED_TAG_UUID,
)
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
    """A bbox with the VERIFIED tag produces a PRESENT row."""
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
                "tagUuids": [VERIFIED_TAG_UUID],
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


async def test_verified_bbox_gives_present(http_client, image_file, label_uuid):
    """A bbox with VERIFIED tag is PRESENT, even when it also has a custom tag."""
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
                "tagUuids": [custom_uuid, VERIFIED_TAG_UUID],
            },
        ],
    )
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"


async def test_one_confirmed_bbox_gives_present_despite_unverified_sibling(http_client, image_file, label_uuid):
    """If any bbox for a label is VERIFIED, the label is PRESENT even if others are UNVERIFIED."""
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
                "tagUuids": [VERIFIED_TAG_UUID],
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
            {
                "uuid": bbox1,
                "labelUuid": label_uuid,
                "x": 0.1,
                "y": 0.1,
                "width": 0.1,
                "height": 0.1,
                "tagUuids": [VERIFIED_TAG_UUID],
            },
            {
                "uuid": bbox2,
                "labelUuid": label2_uuid,
                "x": 0.5,
                "y": 0.5,
                "width": 0.1,
                "height": 0.1,
                "tagUuids": [VERIFIED_TAG_UUID],
            },
        ],
    )
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"
    assert labels[label2_uuid] == "present"

    # Remove VERIFIED bbox for label2 → its PRESENT row should be cleaned up
    await _update_bboxes(
        http_client,
        "labels_test.png",
        [
            {
                "uuid": bbox1,
                "labelUuid": label_uuid,
                "x": 0.1,
                "y": 0.1,
                "width": 0.1,
                "height": 0.1,
                "tagUuids": [VERIFIED_TAG_UUID],
            },
        ],
    )
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"
    assert label2_uuid not in labels


async def test_bbox_patch_to_verified_gives_present(http_client, image_file, label_uuid):
    """Patching a bbox to add VERIFIED creates a PRESENT row."""
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
        json={"labelUuid": label_uuid, "tagUuids": [VERIFIED_TAG_UUID]},
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
    """If any bbox for a label is VERIFIED, the label is PRESENT even if a sibling is FALSE_POSITIVE."""
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
                "tagUuids": [VERIFIED_TAG_UUID],
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
                "tagUuids": [VERIFIED_TAG_UUID],
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
    """If a VERIFIED bbox is added for a label that was explicitly ABSENT, PRESENT wins."""
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
                "tagUuids": [VERIFIED_TAG_UUID],
            }
        ],
    )
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"


# ──────────────────────────────────────────────────────────────────────────────
# Manual PRESENT (new: can now be set directly)
# ──────────────────────────────────────────────────────────────────────────────


async def test_set_present_manually(http_client, image_file, label_uuid):
    """PRESENT can now be set directly without a verified bbox (for image classifiers)."""
    await _update_bboxes(http_client, "labels_test.png", [])  # register image
    await _set_label(http_client, "labels_test.png", label_uuid, "present")
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"


async def test_set_uncertain_clears_manual_present(http_client, image_file, label_uuid):
    """Setting UNCERTAIN on a manually-set PRESENT deletes the row."""
    await _update_bboxes(http_client, "labels_test.png", [])
    await _set_label(http_client, "labels_test.png", label_uuid, "present")
    await _set_label(http_client, "labels_test.png", label_uuid, "uncertain")
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert label_uuid not in labels


# ──────────────────────────────────────────────────────────────────────────────
# ABSENT contradiction check (blocked when verified bbox exists)
# ──────────────────────────────────────────────────────────────────────────────


async def test_absent_blocked_by_verified_bbox(http_client, image_file, label_uuid):
    """ABSENT is rejected when the image already has a verified bbox for that label."""
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
                "tagUuids": [VERIFIED_TAG_UUID],
            }
        ],
    )
    # Setting ABSENT should fail (contradicts verified bbox)
    r = await http_client.post(
        "/api/images/labels/set",
        json={
            "imagePath": "labels_test.png",
            "labelUuid": label_uuid,
            "status": "absent",
        },
    )
    assert r.status_code == 200
    assert r.json()["status"] == "failure"
    # Label should still be PRESENT (from the verified bbox)
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"


async def test_absent_allowed_without_verified_bbox(http_client, image_file, label_uuid):
    """ABSENT is accepted when there is no verified bbox for the label."""
    # Register image with no bboxes
    await _update_bboxes(http_client, "labels_test.png", [])
    r = await http_client.post(
        "/api/images/labels/set",
        json={
            "imagePath": "labels_test.png",
            "labelUuid": label_uuid,
            "status": "absent",
        },
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "absent"


async def test_absent_allowed_when_only_unverified_bbox_exists(http_client, image_file, label_uuid):
    """ABSENT is allowed when the only bbox for that label is unverified."""
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
    r = await http_client.post(
        "/api/images/labels/set",
        json={
            "imagePath": "labels_test.png",
            "labelUuid": label_uuid,
            "status": "absent",
        },
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "absent"


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


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/image-labels  (bulk query used by the training dataset loader)
# ──────────────────────────────────────────────────────────────────────────────


async def test_list_images_for_label_empty(http_client, label_uuid):
    r = await http_client.get(
        "/api/image-labels",
        params={"label_uuid": label_uuid, "status": ["present", "absent"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["images"] == []


async def test_list_images_for_label_returns_present_rows(http_client, image_file, label_uuid):
    """Images with verified bboxes appear as 'present' in the bulk query."""
    import uuid as _uuid

    from apps.helpers.db.types import VERIFIED_TAG_UUID

    await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "labels_test.png",
            "boxes": [
                {
                    "uuid": str(_uuid.uuid4()),
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [VERIFIED_TAG_UUID],
                }
            ],
        },
    )
    r = await http_client.get("/api/image-labels", params={"label_uuid": label_uuid, "status": ["present"]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["images"]) == 1
    assert body["images"][0]["status"] == "present"
    assert body["images"][0]["filename"].endswith("labels_test.png")


async def test_list_images_for_label_returns_absent_rows(http_client, image_file, label_uuid):
    """Explicitly-set absent labels appear in the bulk query."""
    # Register the image in the DB first (empty bbox update), then set the label.
    await http_client.post(
        "/api/images/metadata/update",
        json={"imagePath": "labels_test.png", "boxes": []},
    )
    r_set = await http_client.post(
        "/api/images/labels/set",
        json={
            "imagePath": "labels_test.png",
            "labelUuid": label_uuid,
            "status": "absent",
        },
    )
    assert r_set.json()["status"] == "success"

    r = await http_client.get("/api/image-labels", params={"label_uuid": label_uuid, "status": ["absent"]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["images"]) == 1
    assert body["images"][0]["status"] == "absent"
    assert "imageUuid" in body["images"][0]
    assert body["images"][0]["filename"].endswith("labels_test.png")


async def test_list_images_for_label_filters_by_status(http_client, image_file, label_uuid, label2_uuid):
    """Status filter is applied correctly — present query doesn't return absent rows."""
    # Register the image then set it to absent for label.
    await http_client.post(
        "/api/images/metadata/update",
        json={"imagePath": "labels_test.png", "boxes": []},
    )
    await http_client.post(
        "/api/images/labels/set",
        json={
            "imagePath": "labels_test.png",
            "labelUuid": label_uuid,
            "status": "absent",
        },
    )

    r_present = await http_client.get("/api/image-labels", params={"label_uuid": label_uuid, "status": ["present"]})
    assert r_present.json()["images"] == []

    r_absent = await http_client.get("/api/image-labels", params={"label_uuid": label_uuid, "status": ["absent"]})
    assert len(r_absent.json()["images"]) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Batch endpoint — POST /api/images/labels/batch
# ──────────────────────────────────────────────────────────────────────────────


async def _image_uuid_for(http_client, image_path: str) -> str:
    r = await http_client.post("/api/images/review", json={"show_all_bboxes": True, "limit": 100})
    for img in r.json()["images"]:
        if img["filename"].endswith(image_path):
            return img["uuid"]
    raise AssertionError(f"image {image_path} not found in review queue")


class TestBatchImageLabels:
    async def test_batch_returns_rows_per_image(self, http_client, files_root, label_uuid, label2_uuid):
        for name in ("batch_a.png", "batch_b.png", "batch_c.png"):
            (files_root / name).write_bytes(make_png())
            await _update_bboxes(http_client, name, [])
        await _set_label(http_client, "batch_a.png", label_uuid, "present")
        await _set_label(http_client, "batch_a.png", label2_uuid, "absent")
        await _set_label(http_client, "batch_b.png", label_uuid, "absent")
        # batch_c gets no rows.

        uuid_a = await _image_uuid_for(http_client, "batch_a.png")
        uuid_b = await _image_uuid_for(http_client, "batch_b.png")
        uuid_c = await _image_uuid_for(http_client, "batch_c.png")

        r = await http_client.post(
            "/api/images/labels/batch",
            json={"imageUuids": [uuid_a, uuid_b, uuid_c, str(uuid.uuid4())]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        by_image = body["imageLabels"]

        rows_a = {row["labelUuid"]: row["status"] for row in by_image[uuid_a]}
        assert rows_a == {label_uuid: "present", label2_uuid: "absent"}
        rows_b = {row["labelUuid"]: row["status"] for row in by_image[uuid_b]}
        assert rows_b == {label_uuid: "absent"}
        # No-row and unknown images are omitted entirely.
        assert uuid_c not in by_image
        assert len(by_image) == 2

    async def test_batch_empty_payload_returns_empty_map(self, http_client):
        r = await http_client.post("/api/images/labels/batch", json={"imageUuids": []})
        assert r.status_code == 200
        assert r.json()["imageLabels"] == {}
