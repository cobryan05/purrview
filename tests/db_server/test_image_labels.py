"""Tests for image-level label state tracking.

PRESENT rows are derived automatically from verified bboxes.
ABSENT rows are set explicitly by the user and survive recomputation.
Unverified and false-positive bboxes produce no image_label rows.
"""

import asyncio
import uuid

import pytest

from apps.db_server.manager import Manager
from apps.helpers.db.types import (
    UNVERIFIED_TAG_UUID,
    VERIFIED_TAG_UUID,
    ImageItemStatus,
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


@pytest.fixture
async def image_scope_label_uuid(http_client):
    r = await http_client.post("/api/labels", json={"name": "Corrupt", "color": "#00FF00", "scope": "image"})
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
# Image-scope labels: PRESENT is user-owned, never bbox-derived (data loss regression)
# ──────────────────────────────────────────────────────────────────────────────


async def test_image_scope_present_survives_recompute(http_client, image_file, image_scope_label_uuid):
    """Image-scope labels (e.g. 'Corrupt') can never have a bbox — the UI short-circuits
    the draw tool for them — so their PRESENT row must never be treated as stale by the
    bbox-derived sweep in _recompute_image_labels. Regression test for data loss where a
    manually-set image-scope PRESENT was silently wiped on the next image save."""
    await _update_bboxes(http_client, "labels_test.png", [])  # register image
    await _set_label(http_client, "labels_test.png", image_scope_label_uuid, "present")
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[image_scope_label_uuid] == "present"

    # Any recompute-triggering save (e.g. saving image metadata again with no bboxes)
    # must not wipe the image-scope PRESENT row.
    await _update_bboxes(http_client, "labels_test.png", [])

    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels.get(image_scope_label_uuid) == "present"


async def test_object_scope_manual_present_still_reaped_without_verified_bbox(http_client, image_file, label_uuid):
    """Deliberate asymmetry: unlike image-scope labels, an object-scope label's PRESENT
    state is owned by bbox sync, not the user. A manually-set PRESENT on an object-scope
    label with no backing verified bbox must still be reaped on recompute — you cannot
    durably mark e.g. 'Cat' present without a verified Cat bbox."""
    await _update_bboxes(http_client, "labels_test.png", [])  # register image
    await _set_label(http_client, "labels_test.png", label_uuid, "present")
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"

    # Trigger recompute via another bbox-mutating save; no verified bbox backs label_uuid.
    await _update_bboxes(http_client, "labels_test.png", [])

    labels = await _get_image_labels(http_client, "labels_test.png")
    assert label_uuid not in labels


async def test_stale_present_row_for_deleted_label_still_swept(http_client, files_root):
    """A PRESENT row whose label has since been deleted is an orphan and must still be
    swept by recompute — the image-scope exemption must not accidentally protect uuids
    that no longer resolve to any label at all."""
    img_path = files_root / "orphan_test.png"
    img_path.write_bytes(make_png())

    r = await http_client.post("/api/labels", json={"name": "to-delete", "color": "#123456"})
    label_uuid = r.json()["label"]["uuid"]

    await _update_bboxes(http_client, "orphan_test.png", [])  # register image
    await _set_label(http_client, "orphan_test.png", label_uuid, "present")
    labels = await _get_image_labels(http_client, "orphan_test.png")
    assert labels[label_uuid] == "present"

    del_r = await http_client.delete(f"/api/labels/{label_uuid}")
    assert del_r.status_code == 200

    # Trigger recompute via a bbox-mutating save on the (now label-less) image.
    await _update_bboxes(http_client, "orphan_test.png", [])

    labels = await _get_image_labels(http_client, "orphan_test.png")
    assert label_uuid not in labels


# ──────────────────────────────────────────────────────────────────────────────
# ABSENT contradiction check (blocked when verified bbox exists)
# ──────────────────────────────────────────────────────────────────────────────


async def test_absent_blocked_by_verified_bbox(http_client, image_file, label_uuid):
    """ABSENT is refused when the image already has a verified bbox for that
    label — but this is a deliberate business-rule skip, not a failure: the
    response keeps status=success and sets skipped=True so the client doesn't
    treat it as an error worth surfacing/retrying (see labelList.js flush)."""
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
    r = await http_client.post(
        "/api/images/labels/set",
        json={
            "imagePath": "labels_test.png",
            "labelUuid": label_uuid,
            "status": "absent",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["skipped"] is True
    assert body["message"]
    # Label should still be PRESENT (from the verified bbox) — nothing was overwritten.
    labels = await _get_image_labels(http_client, "labels_test.png")
    assert labels[label_uuid] == "present"


async def test_absent_not_blocked_reports_skipped_false(http_client, image_file, label_uuid):
    """The common, unblocked case must NOT carry skipped=True."""
    await _update_bboxes(http_client, "labels_test.png", [])
    r = await http_client.post(
        "/api/images/labels/set",
        json={"imagePath": "labels_test.png", "labelUuid": label_uuid, "status": "absent"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body.get("skipped", False) is False


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


# ──────────────────────────────────────────────────────────────────────────────
# Bulk apply by UUIDs — POST /api/images/bulk-apply-by-uuids (image-label actions)
# ──────────────────────────────────────────────────────────────────────────────


class TestBulkApplyImagesByUuidsLabelActions:
    """setPresent/setAbsent/setUncertain via the by-uuids bulk endpoint.

    setPresent regression: the manager's action dispatch used to end with a
    bare `else` that treated any unmatched action (including "setPresent",
    which the frontend's applyToAllCandidates DOES send) as "setAbsent" —
    silently storing ABSENT for a request that asked for PRESENT.
    """

    async def _make_image(self, http_client, files_root, name):
        (files_root / name).write_bytes(make_png())
        r = await _update_bboxes(http_client, name, [])
        return r["image"]["uuid"]

    async def test_bulk_apply_set_present(self, http_client, files_root, label_uuid):
        image_uuid = await self._make_image(http_client, files_root, "bulk_present.png")

        r = await http_client.post(
            "/api/images/bulk-apply-by-uuids",
            json={"imageUuids": [image_uuid], "action": "setPresent", "labelUuids": [label_uuid]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["modifiedCount"] == 1
        assert body["failCount"] == 0

        labels = await _get_image_labels(http_client, "bulk_present.png")
        assert labels[label_uuid] == "present"

    async def test_bulk_apply_set_absent(self, http_client, files_root, label_uuid):
        image_uuid = await self._make_image(http_client, files_root, "bulk_absent.png")

        r = await http_client.post(
            "/api/images/bulk-apply-by-uuids",
            json={"imageUuids": [image_uuid], "action": "setAbsent", "labelUuids": [label_uuid]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["modifiedCount"] == 1

        labels = await _get_image_labels(http_client, "bulk_absent.png")
        assert labels[label_uuid] == "absent"

    async def test_bulk_apply_set_uncertain(self, http_client, files_root, label_uuid):
        image_uuid = await self._make_image(http_client, files_root, "bulk_uncertain.png")
        await _set_label(http_client, "bulk_uncertain.png", label_uuid, "absent")

        r = await http_client.post(
            "/api/images/bulk-apply-by-uuids",
            json={"imageUuids": [image_uuid], "action": "setUncertain", "labelUuids": [label_uuid]},
        )
        assert r.status_code == 200
        assert r.json()["modifiedCount"] == 1

        labels = await _get_image_labels(http_client, "bulk_uncertain.png")
        assert label_uuid not in labels

    async def test_bulk_apply_set_absent_skipped_when_verified_bbox_present(self, http_client, files_root, label_uuid):
        """setAbsent must be reported as skipped (not failed, not modified) when
        the image already has a verified bbox for the label — mirrors the
        single-image set_image_label_status contradiction rule."""
        name = "bulk_absent_skip.png"
        (files_root / name).write_bytes(make_png())
        r = await _update_bboxes(
            http_client,
            name,
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
        image_uuid = r["image"]["uuid"]

        resp = await http_client.post(
            "/api/images/bulk-apply-by-uuids",
            json={"imageUuids": [image_uuid], "action": "setAbsent", "labelUuids": [label_uuid]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["modifiedCount"] == 0
        assert body["failCount"] == 0
        assert body["skippedCount"] == 1

        # Untouched: still present from the verified bbox.
        labels = await _get_image_labels(http_client, name)
        assert labels[label_uuid] == "present"

    async def test_bulk_apply_set_present_for_multiple_labels(self, http_client, files_root, label_uuid, label2_uuid):
        """labelUuids applies the action to every listed label in one pass, and
        still counts images (not image/label pairs)."""
        image_uuid = await self._make_image(http_client, files_root, "bulk_multi_present.png")

        r = await http_client.post(
            "/api/images/bulk-apply-by-uuids",
            json={
                "imageUuids": [image_uuid],
                "action": "setPresent",
                "labelUuids": [label_uuid, label2_uuid],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["modifiedCount"] == 1
        assert body["failCount"] == 0

        labels = await _get_image_labels(http_client, "bulk_multi_present.png")
        assert labels[label_uuid] == "present"
        assert labels[label2_uuid] == "present"

    async def test_bulk_apply_set_uncertain_clears_every_listed_label(
        self, http_client, files_root, label_uuid, label2_uuid
    ):
        name = "bulk_multi_uncertain.png"
        image_uuid = await self._make_image(http_client, files_root, name)
        await _set_label(http_client, name, label_uuid, "absent")
        await _set_label(http_client, name, label2_uuid, "absent")

        r = await http_client.post(
            "/api/images/bulk-apply-by-uuids",
            json={
                "imageUuids": [image_uuid],
                "action": "setUncertain",
                "labelUuids": [label_uuid, label2_uuid],
            },
        )
        assert r.status_code == 200
        assert r.json()["modifiedCount"] == 1

        labels = await _get_image_labels(http_client, name)
        assert label_uuid not in labels
        assert label2_uuid not in labels

    async def test_bulk_apply_multi_label_partial_skip_still_counts_as_modified(
        self, http_client, files_root, label_uuid, label2_uuid
    ):
        """setAbsent is skipped per-label when a verified bbox contradicts it,
        but the image is still "modified" if any other label was applied —
        an image is only "skipped" when every label it was asked about was."""
        name = "bulk_multi_partial_skip.png"
        (files_root / name).write_bytes(make_png())
        r = await _update_bboxes(
            http_client,
            name,
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
        image_uuid = r["image"]["uuid"]

        resp = await http_client.post(
            "/api/images/bulk-apply-by-uuids",
            json={
                "imageUuids": [image_uuid],
                "action": "setAbsent",
                "labelUuids": [label_uuid, label2_uuid],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["modifiedCount"] == 1
        assert body["skippedCount"] == 0
        assert body["failCount"] == 0

        labels = await _get_image_labels(http_client, name)
        # Contradicted label untouched, the other one marked absent.
        assert labels[label_uuid] == "present"
        assert labels[label2_uuid] == "absent"

    async def test_bulk_apply_multi_label_uses_one_session_per_image_not_per_label(
        self, http_client, files_root, label_uuid, label2_uuid, db_client, monkeypatch
    ):
        """setAbsent/setPresent used to open one DB session per (image, label) pair.
        With N labels applied to one image, that's N sessions where one suffices —
        assert the session count no longer scales with label count."""
        r = await http_client.post("/api/labels", json={"name": "bird", "color": "#00FF00"})
        label3_uuid = r.json()["label"]["uuid"]

        name = "bulk_multi_session_count.png"
        image_uuid = await self._make_image(http_client, files_root, name)

        real_session_maker = db_client._async_session_maker
        session_opens = []

        def counting_session_maker(*args, **kwargs):
            session_opens.append(1)
            return real_session_maker(*args, **kwargs)

        monkeypatch.setattr(db_client, "_async_session_maker", counting_session_maker)

        resp = await http_client.post(
            "/api/images/bulk-apply-by-uuids",
            json={
                "imageUuids": [image_uuid],
                "action": "setPresent",
                "labelUuids": [label_uuid, label2_uuid, label3_uuid],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["modifiedCount"] == 1

        assert len(session_opens) <= 2, (
            f"setPresent for 1 image x 3 labels opened {len(session_opens)} DB sessions; "
            "expected a constant number independent of label count"
        )

    async def test_bulk_apply_label_action_without_labels_is_a_no_op(self, http_client, files_root):
        """No labels selected must not report images as modified — the same
        guard the tag actions already have."""
        image_uuid = await self._make_image(http_client, files_root, "bulk_no_labels.png")

        r = await http_client.post(
            "/api/images/bulk-apply-by-uuids",
            json={"imageUuids": [image_uuid], "action": "setPresent", "labelUuids": []},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["modifiedCount"] == 0
        assert body["failCount"] == 0
        assert await _get_image_labels(http_client, "bulk_no_labels.png") == {}

    async def test_bulk_apply_unknown_action_raises(self, http_client, files_root, label_uuid):
        image_uuid = await self._make_image(http_client, files_root, "bulk_unknown_action.png")

        r = await http_client.post(
            "/api/images/bulk-apply-by-uuids",
            json={"imageUuids": [image_uuid], "action": "bogusAction", "labelUuids": [label_uuid]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["modifiedCount"] == 0
        assert body["failCount"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# Filter-based bulk apply — POST /api/images/bulk-apply
# ──────────────────────────────────────────────────────────────────────────────


class TestBulkApplyImagesByFilter:
    """The image-kind counterpart of POST /api/images/bbox/bulk-apply: paginates
    list_images_search server-side instead of requiring the caller to already
    know every uuid."""

    async def test_bulk_apply_by_filter_sets_absent_for_matching_images(self, http_client, files_root, label_uuid):
        matching = ["filter_match_a.png", "filter_match_b.png"]
        other = "filter_other.png"
        for name in (*matching, other):
            (files_root / name).write_bytes(make_png())
            await _update_bboxes(http_client, name, [])

        r = await http_client.post(
            "/api/images/bulk-apply",
            json={
                "filter": {"terms": [{"kind": "filename", "patterns": ["filter_match"]}]},
                "action": "setAbsent",
                "labelUuids": [label_uuid],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["modifiedCount"] == len(matching)
        assert body["failCount"] == 0

        for name in matching:
            labels = await _get_image_labels(http_client, name)
            assert labels[label_uuid] == "absent"
        other_labels = await _get_image_labels(http_client, other)
        assert label_uuid not in other_labels

    async def test_bulk_apply_by_filter_paginates_across_pages(self, http_client, files_root, label_uuid, monkeypatch):
        """Force a tiny page size so the keyset cursor loop actually runs
        multiple iterations, mirroring bulk_apply_bboxes's own pagination test."""
        monkeypatch.setattr(Manager, "BULK_APPLY_PAGE_SIZE", 2)
        names = [f"filter_page_{i}.png" for i in range(5)]
        for name in names:
            (files_root / name).write_bytes(make_png())
            await _update_bboxes(http_client, name, [])

        r = await http_client.post(
            "/api/images/bulk-apply",
            json={
                "filter": {"terms": [{"kind": "filename", "patterns": ["filter_page_"]}]},
                "action": "setAbsent",
                "labelUuids": [label_uuid],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["modifiedCount"] == len(names)

        for name in names:
            labels = await _get_image_labels(http_client, name)
            assert labels[label_uuid] == "absent"

    async def test_bulk_apply_by_filter_sets_multiple_labels(self, http_client, files_root, label_uuid, label2_uuid):
        """The filter-driven path takes the same label list as the by-uuids one,
        applying every label in a single pagination pass."""
        names = ["filter_multi_a.png", "filter_multi_b.png"]
        for name in names:
            (files_root / name).write_bytes(make_png())
            await _update_bboxes(http_client, name, [])

        r = await http_client.post(
            "/api/images/bulk-apply",
            json={
                "filter": {"terms": [{"kind": "filename", "patterns": ["filter_multi"]}]},
                "action": "setAbsent",
                "labelUuids": [label_uuid, label2_uuid],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["modifiedCount"] == len(names)
        assert body["failCount"] == 0

        for name in names:
            labels = await _get_image_labels(http_client, name)
            assert labels[label_uuid] == "absent"
            assert labels[label2_uuid] == "absent"

    async def test_bulk_apply_by_filter_reports_skipped(self, http_client, files_root, label_uuid):
        name = "filter_skip.png"
        (files_root / name).write_bytes(make_png())
        r = await _update_bboxes(
            http_client,
            name,
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

        resp = await http_client.post(
            "/api/images/bulk-apply",
            json={
                "filter": {"terms": [{"kind": "filename", "patterns": ["filter_skip"]}]},
                "action": "setAbsent",
                "labelUuids": [label_uuid],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["modifiedCount"] == 0
        assert body["skippedCount"] == 1

        labels = await _get_image_labels(http_client, name)
        assert labels[label_uuid] == "present"


# ──────────────────────────────────────────────────────────────────────────────
# Concurrent upsert — DbClient.set_image_label_status must not race
# ──────────────────────────────────────────────────────────────────────────────


class TestConcurrentSetImageLabelStatus:
    """A get-then-insert upsert lets two concurrent calls for the same
    (image_uuid, label_uuid) both miss the SELECT and both attempt an INSERT —
    the loser hits a PK violation. The fix must make the upsert atomic."""

    async def test_concurrent_calls_for_same_key_all_succeed(self, db_client, files_root):
        img = await db_client.add_image(files_root / "concurrent.png")
        label = await db_client.add_label(name="concurrent-label")

        await asyncio.gather(
            *[db_client.set_image_label_status(img.uuid, label.uuid, ImageItemStatus.PRESENT) for _ in range(10)]
        )

        rows = await db_client.get_image_labels(img.uuid)
        matching = [r for r in rows if r.label_uuid == label.uuid]
        assert len(matching) == 1
        assert matching[0].status == ImageItemStatus.PRESENT

    async def test_concurrent_calls_with_different_final_status_leave_one_consistent_row(self, db_client, files_root):
        """Mixed PRESENT/ABSENT calls racing on the same key must not crash; exactly
        one row must remain, holding whichever status won the race."""
        img = await db_client.add_image(files_root / "concurrent2.png")
        label = await db_client.add_label(name="concurrent-label-2")

        statuses = [ImageItemStatus.PRESENT, ImageItemStatus.ABSENT] * 5
        await asyncio.gather(*[db_client.set_image_label_status(img.uuid, label.uuid, s) for s in statuses])

        rows = await db_client.get_image_labels(img.uuid)
        matching = [r for r in rows if r.label_uuid == label.uuid]
        assert len(matching) == 1
        assert matching[0].status in (ImageItemStatus.PRESENT, ImageItemStatus.ABSENT)
