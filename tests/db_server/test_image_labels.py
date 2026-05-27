"""Tests for image-level label derivation via _recompute_image_labels.

These tests exercise the full HTTP path but the real value is verifying that
after update_image_metadata / bbox PATCH / bbox DELETE, the image_labels table
reflects the correct PRESENT / ABSENT / UNCERTAIN state for each label.
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
    return {row["label_uuid"]: row["status"] for row in body.get("labels", [])}


async def _update_bboxes(http_client, image_path, boxes):
    """POST metadata/update and assert success."""
    r = await http_client.post(
        "/api/images/metadata/update",
        json={"imagePath": image_path, "boxes": boxes},
    )
    assert r.status_code == 200
    return r.json()


# ──────────────────────────────────────────────────────────────────────────────
# Basic derivation
# ──────────────────────────────────────────────────────────────────────────────

async def test_unverified_bbox_gives_uncertain(http_client, image_file, label_uuid):
    await _update_bboxes(http_client, "labels_test.png", [{
        "uuid": str(uuid.uuid4()),
        "labelUuid": label_uuid,
        "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2,
        "tagUuids": [UNVERIFIED_TAG_UUID],
    }])
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "uncertain"


async def test_verified_bbox_gives_present(http_client, image_file, label_uuid):
    await _update_bboxes(http_client, "labels_test.png", [{
        "uuid": str(uuid.uuid4()),
        "labelUuid": label_uuid,
        "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2,
        "tagUuids": [VERIFIED_TAG_UUID],
    }])
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"


async def test_all_false_pos_bboxes_gives_absent(http_client, image_file, label_uuid):
    await _update_bboxes(http_client, "labels_test.png", [
        {"uuid": str(uuid.uuid4()), "labelUuid": label_uuid,
         "x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1, "tagUuids": [FALSE_POS_TAG_UUID]},
        {"uuid": str(uuid.uuid4()), "labelUuid": label_uuid,
         "x": 0.5, "y": 0.5, "width": 0.1, "height": 0.1, "tagUuids": [FALSE_POS_TAG_UUID]},
    ])
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "absent"


async def test_any_verified_wins_over_false_pos(http_client, image_file, label_uuid):
    await _update_bboxes(http_client, "labels_test.png", [
        {"uuid": str(uuid.uuid4()), "labelUuid": label_uuid,
         "x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1, "tagUuids": [FALSE_POS_TAG_UUID]},
        {"uuid": str(uuid.uuid4()), "labelUuid": label_uuid,
         "x": 0.5, "y": 0.5, "width": 0.1, "height": 0.1, "tagUuids": [VERIFIED_TAG_UUID]},
    ])
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"


# ──────────────────────────────────────────────────────────────────────────────
# Multiple labels on same image are independent
# ──────────────────────────────────────────────────────────────────────────────

async def test_two_labels_independent(http_client, image_file, label_uuid, label2_uuid):
    await _update_bboxes(http_client, "labels_test.png", [
        {"uuid": str(uuid.uuid4()), "labelUuid": label_uuid,
         "x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1, "tagUuids": [VERIFIED_TAG_UUID]},
        {"uuid": str(uuid.uuid4()), "labelUuid": label2_uuid,
         "x": 0.5, "y": 0.5, "width": 0.1, "height": 0.1, "tagUuids": [FALSE_POS_TAG_UUID]},
    ])
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"
    assert labels[label2_uuid] == "absent"


# ──────────────────────────────────────────────────────────────────────────────
# Stale label rows are removed
# ──────────────────────────────────────────────────────────────────────────────

async def test_stale_label_removed_when_bbox_deleted(http_client, image_file, label_uuid, label2_uuid):
    bbox1 = str(uuid.uuid4())
    bbox2 = str(uuid.uuid4())
    # Set up image with two labels
    await _update_bboxes(http_client, "labels_test.png", [
        {"uuid": bbox1, "labelUuid": label_uuid,
         "x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1, "tagUuids": [UNVERIFIED_TAG_UUID]},
        {"uuid": bbox2, "labelUuid": label2_uuid,
         "x": 0.5, "y": 0.5, "width": 0.1, "height": 0.1, "tagUuids": [UNVERIFIED_TAG_UUID]},
    ])
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert label_uuid in labels
    assert label2_uuid in labels

    # Remove bbox for label2 only
    await _update_bboxes(http_client, "labels_test.png", [
        {"uuid": bbox1, "labelUuid": label_uuid,
         "x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1, "tagUuids": [UNVERIFIED_TAG_UUID]},
    ])
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert label_uuid in labels
    assert label2_uuid not in labels   # stale row removed


# ──────────────────────────────────────────────────────────────────────────────
# Update recomputes on PATCH (via bbox PATCH endpoint)
# ──────────────────────────────────────────────────────────────────────────────

async def test_bbox_patch_updates_image_label(http_client, image_file, label_uuid):
    bbox_id = str(uuid.uuid4())
    await _update_bboxes(http_client, "labels_test.png", [{
        "uuid": bbox_id,
        "labelUuid": label_uuid,
        "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2,
        "tagUuids": [UNVERIFIED_TAG_UUID],
    }])
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "uncertain"

    # Verify via PATCH
    patch_r = await http_client.patch(
        f"/api/images/bbox/{bbox_id}",
        json={"labelUuid": label_uuid, "tagUuids": [VERIFIED_TAG_UUID]},
    )
    assert patch_r.status_code == 204

    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"


# ──────────────────────────────────────────────────────────────────────────────
# set_image_label explicit override
# ──────────────────────────────────────────────────────────────────────────────

async def test_set_image_label_explicit(http_client, image_file, label_uuid):
    """The set_image_label endpoint allows overriding label state directly."""
    # First create the image record
    await _update_bboxes(http_client, "labels_test.png", [{
        "uuid": str(uuid.uuid4()),
        "labelUuid": label_uuid,
        "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2,
        "tagUuids": [UNVERIFIED_TAG_UUID],
    }])

    r = await http_client.post(
        "/api/images/labels/set",
        json={"imagePath": "labels_test.png", "labelUuid": label_uuid, "status": "present"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"


async def test_set_image_label_image_not_found(http_client, label_uuid):
    r = await http_client.post(
        "/api/images/labels/set",
        json={"imagePath": "ghost.png", "labelUuid": label_uuid, "status": "present"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "failure"
