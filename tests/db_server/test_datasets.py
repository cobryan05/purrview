"""Tests for the dataset feature — CRUD, filter criteria, and pagination fixes.

Fix coverage:
  Fix 1 — filter_json validated at write time (invalid type → 422)
  Fix 2 — DatasetLabelStatusCondition + empty label_uuids → 0 images, not all
  Fix 3 — has_more reflects base-query page size, not post-bbox-age-filter count
  Fix 4 — after_sort_value query param accepted and forwarded
  Fix 5 — multiple FilenameConditions are AND-combined
"""
import uuid
from datetime import datetime, timezone, timedelta

import pytest
from sqlmodel import select

from apps.helpers.db.types import BBox, ImageRecord, UNVERIFIED_TAG_UUID
from tests.conftest import make_png


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _png(files_root, name: str) -> str:
    (files_root / name).write_bytes(make_png())
    return name


async def _register(http_client, name: str) -> str:
    """Register a PNG in the DB with no bboxes."""
    r = await http_client.post(
        "/api/images/metadata/update",
        json={"imagePath": name, "boxes": []},
    )
    assert r.status_code == 200, r.text
    return name


async def _add_bbox(
    http_client, image_path: str, label_uuid: str, *, tag_uuids: list | None = None
) -> str:
    bid = str(uuid.uuid4())
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": image_path,
            "boxes": [{
                "uuid": bid,
                "labelUuid": label_uuid,
                "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2,
                "tagUuids": tag_uuids or [],
            }],
        },
    )
    assert r.status_code == 200, r.text
    return bid


async def _create_dataset(
    http_client,
    name: str,
    *,
    conditions: list | None = None,
    label_uuids: list | None = None,
    description: str | None = None,
) -> dict:
    payload: dict = {
        "name": name,
        "filter_json": {"conditions": conditions or []},
        "label_uuids": label_uuids or [],
    }
    if description is not None:
        payload["description"] = description
    r = await http_client.post("/api/datasets", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["dataset"]


async def _dataset_images(http_client, dataset_uuid: str, **params) -> dict:
    r = await http_client.post(f"/api/datasets/{dataset_uuid}/images", params=params)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body


async def _backdate_created_at(db_client, filename: str, delta: timedelta) -> None:
    """Set created_at on an image to now() - delta."""
    ts = datetime.now(timezone.utc) - delta
    async with db_client._async_session_maker() as session:
        result = await session.exec(
            select(ImageRecord).where(ImageRecord.filename.like(f"%/{filename}"))
        )
        img = result.one()
        img.created_at = ts
        session.add(img)
        await session.commit()


async def _backdate_bbox(db_client, bbox_uuid: str, delta: timedelta) -> None:
    """Set updated_at on a bbox to now() - delta."""
    ts = datetime.now(timezone.utc) - delta
    async with db_client._async_session_maker() as session:
        result = await session.exec(select(BBox).where(BBox.uuid == bbox_uuid))
        bbox = result.one()
        bbox.updated_at = ts
        session.add(bbox)
        await session.commit()


async def _backdate(db_client, filename: str, delta: timedelta) -> None:
    """Set updated_at on an image to now() - delta directly in the DB.

    The DB stores absolute paths so we match by the trailing basename component.
    """
    ts = datetime.now(timezone.utc) - delta
    async with db_client._async_session_maker() as session:
        # filename is a bare name like "old.png"; DB stores the full path.
        result = await session.exec(
            select(ImageRecord).where(ImageRecord.filename.like(f"%/{filename}"))
        )
        img = result.one()
        img.updated_at = ts
        session.add(img)
        await session.commit()


# ─────────────────────────────────────────────────────────────────────────────
# CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def test_create_and_list_datasets(http_client):
    ds = await _create_dataset(http_client, "pets", description="All pets")
    assert ds["name"] == "pets"
    assert ds["description"] == "All pets"
    assert ds["uuid"]

    r = await http_client.get("/api/datasets")
    body = r.json()
    assert body["status"] == "success"
    names = [d["name"] for d in body["datasets"]]
    assert "pets" in names


async def test_create_dataset_with_label_uuids(http_client):
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    label_uuid = r.json()["label"]["uuid"]

    ds = await _create_dataset(http_client, "cats", label_uuids=[label_uuid])
    assert label_uuid in ds["label_uuids"]


async def test_update_dataset_replaces_labels(http_client):
    r1 = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    r2 = await http_client.post("/api/labels", json={"name": "dog", "color": "#0000FF"})
    lbl1 = r1.json()["label"]["uuid"]
    lbl2 = r2.json()["label"]["uuid"]

    ds = await _create_dataset(http_client, "animals", label_uuids=[lbl1])
    assert ds["label_uuids"] == [lbl1]

    r = await http_client.patch(
        f"/api/datasets/{ds['uuid']}", json={"label_uuids": [lbl2]}
    )
    assert r.json()["status"] == "success"
    assert r.json()["dataset"]["label_uuids"] == [lbl2]


async def test_delete_dataset(http_client):
    ds = await _create_dataset(http_client, "to-delete")
    r = await http_client.delete(f"/api/datasets/{ds['uuid']}")
    assert r.json()["status"] == "success"

    # Second delete → not found
    r2 = await http_client.delete(f"/api/datasets/{ds['uuid']}")
    assert r2.json()["status"] == "failure"


async def test_get_dataset_images_not_found(http_client):
    r = await http_client.post("/api/datasets/nonexistent-uuid/images")
    assert r.json()["status"] == "failure"


async def test_label_delete_removes_link(http_client):
    """Deleting a label removes its dataset_label_link; the dataset survives."""
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    label_uuid = r.json()["label"]["uuid"]
    ds = await _create_dataset(http_client, "linked", label_uuids=[label_uuid])
    assert label_uuid in ds["label_uuids"]

    await http_client.delete(f"/api/labels/{label_uuid}")

    r = await http_client.get("/api/datasets")
    datasets = r.json()["datasets"]
    found = next(d for d in datasets if d["uuid"] == ds["uuid"])
    assert label_uuid not in found["label_uuids"]


async def test_get_dataset_labels_endpoint(http_client):
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    label_uuid = r.json()["label"]["uuid"]
    ds = await _create_dataset(http_client, "label-test", label_uuids=[label_uuid])

    r = await http_client.get(f"/api/datasets/{ds['uuid']}/labels")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert any(lbl["uuid"] == label_uuid for lbl in body["labels"])


async def test_get_dataset_labels_not_found(http_client):
    r = await http_client.get("/api/datasets/nonexistent-uuid/labels")
    assert r.json()["status"] == "failure"


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1 — filter_json validated at write time
# ─────────────────────────────────────────────────────────────────────────────

async def test_filter_json_invalid_type_rejected_at_create(http_client):
    """POST with unknown condition type returns 422, not 200 (Fix 1)."""
    r = await http_client.post(
        "/api/datasets",
        json={
            "name": "bad",
            "filter_json": {"conditions": [{"type": "nonexistent_condition_type"}]},
        },
    )
    assert r.status_code == 422


async def test_filter_json_invalid_type_rejected_at_update(http_client):
    """PATCH with unknown condition type returns 422, not 200 (Fix 1)."""
    ds = await _create_dataset(http_client, "valid-first")
    r = await http_client.patch(
        f"/api/datasets/{ds['uuid']}",
        json={"filter_json": {"conditions": [{"type": "also_nonexistent"}]}},
    )
    assert r.status_code == 422


async def test_filter_json_valid_condition_accepted(http_client):
    """Valid condition types are accepted at create time."""
    ds = await _create_dataset(
        http_client, "valid-cond",
        conditions=[{"type": "filename", "pattern": "test"}],
    )
    conds = ds["filter_json"]["conditions"]
    assert len(conds) == 1
    assert conds[0]["type"] == "filename"


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2 — DatasetLabelStatusCondition + empty label_uuids → 0 images
# ─────────────────────────────────────────────────────────────────────────────

async def test_label_status_empty_label_uuids_returns_no_images(http_client, files_root):
    """DatasetLabelStatusCondition with no labels returns 0 images, not all (Fix 2)."""
    _png(files_root, "img_fix2.png")
    await _register(http_client, "img_fix2.png")

    ds = await _create_dataset(
        http_client, "no-labels-status",
        conditions=[{"type": "dataset_label_status", "check": "all_verified"}],
        label_uuids=[],  # intentionally empty
    )
    result = await _dataset_images(http_client, ds["uuid"])
    assert result["images"] == [], (
        "DatasetLabelStatusCondition with no labels should return 0 images, not all images"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fix 3 — has_more reflects base-query page, not post-filter count
# ─────────────────────────────────────────────────────────────────────────────

async def test_has_more_correct_with_bbox_age_filter(http_client, files_root, db_client):
    """has_more=True when base page is full, even if bbox-age filter reduces returned count (Fix 3)."""
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    label_uuid = r.json()["label"]["uuid"]

    for name in ("img_ha.png", "img_hb.png", "img_hc.png"):
        _png(files_root, name)

    # Only img_ha has a bbox (will pass bbox-age filter)
    await _add_bbox(http_client, "img_ha.png", label_uuid)
    await _register(http_client, "img_hb.png")
    await _register(http_client, "img_hc.png")

    # Backdate img_hb and img_hc so img_ha sorts first (most recent updated_at)
    await _backdate(db_client, "img_hb.png", timedelta(hours=2))
    await _backdate(db_client, "img_hc.png", timedelta(hours=3))

    ds = await _create_dataset(
        http_client, "bbox-age-has-more",
        conditions=[{"type": "bbox_age", "field": "updated_at", "within_days": 1}],
    )

    # limit=2: base query returns [img_ha, img_hb] (most recent first, full page)
    # bbox-age filter: only img_ha passes (img_hb has no bbox)
    # has_more must be True — base page was full, more images may exist
    result = await _dataset_images(http_client, ds["uuid"], limit=2)
    filenames = [img["filename"] for img in result["images"]]
    assert "img_ha.png" in filenames
    assert result["hasMore"] is True, (
        "has_more should be True when the base page was full, even if bbox-age "
        "filtering reduced the returned count below the limit"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fix 4 — after_sort_value query param forwarded through endpoint
# ─────────────────────────────────────────────────────────────────────────────

async def test_after_sort_value_accepted_in_endpoint(http_client, files_root):
    """after_sort_value is accepted and used for keyset pagination (Fix 4)."""
    _png(files_root, "img_pg1.png")
    _png(files_root, "img_pg2.png")
    await _register(http_client, "img_pg1.png")
    await _register(http_client, "img_pg2.png")

    ds = await _create_dataset(http_client, "pagination-ds")

    # Page 1
    page1 = await _dataset_images(http_client, ds["uuid"], limit=1, sort_by="updated_at")
    assert len(page1["images"]) == 1
    first = page1["images"][0]
    after_uuid = first["uuid"]
    after_sort_value = first["updatedAt"]  # camelCase per ImageRecordRead alias

    # Page 2 — endpoint must accept after_sort_value without error
    r = await http_client.post(
        f"/api/datasets/{ds['uuid']}/images",
        params={
            "limit": 1,
            "sort_by": "updated_at",
            "after_uuid": after_uuid,
            "after_sort_value": after_sort_value,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    # Second page should not repeat the first image
    second_names = [img["filename"] for img in body["images"]]
    assert first["filename"] not in second_names


# ─────────────────────────────────────────────────────────────────────────────
# Fix 5 — multiple FilenameConditions are AND-combined
# ─────────────────────────────────────────────────────────────────────────────

async def test_multiple_filename_conditions_and_logic(http_client, files_root):
    """Two FilenameConditions both must match — AND, not last-wins (Fix 5)."""
    for name in ("cats_and_dogs.png", "cats_only.png", "dogs_only.png"):
        _png(files_root, name)
        await _register(http_client, name)

    ds = await _create_dataset(
        http_client, "multi-filename",
        conditions=[
            {"type": "filename", "pattern": "cats"},
            {"type": "filename", "pattern": "dogs"},
        ],
    )
    result = await _dataset_images(http_client, ds["uuid"])
    names = [img["filename"] for img in result["images"]]
    assert "cats_and_dogs.png" in names, "Image matching both patterns should be included"
    assert "cats_only.png" not in names, "Image matching only first pattern should be excluded"
    assert "dogs_only.png" not in names, "Image matching only second pattern should be excluded"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset filter — ImageAgeCondition
# ─────────────────────────────────────────────────────────────────────────────

async def test_dataset_filter_image_age(http_client, files_root, db_client):
    """ImageAgeCondition(field=updated_at, within_days=7) filters correctly."""
    for name in ("old_img.png", "new_img.png"):
        _png(files_root, name)
        await _register(http_client, name)

    await _backdate(db_client, "old_img.png", timedelta(days=60))

    ds = await _create_dataset(
        http_client, "recent-images",
        conditions=[{"type": "image_age", "field": "updated_at", "within_days": 7}],
    )
    result = await _dataset_images(http_client, ds["uuid"])
    names = [img["filename"] for img in result["images"]]
    assert "new_img.png" in names
    assert "old_img.png" not in names


# ─────────────────────────────────────────────────────────────────────────────
# Dataset filter — TagCondition
# ─────────────────────────────────────────────────────────────────────────────

async def test_dataset_filter_tag_condition(http_client, files_root):
    """TagCondition includes/excludes images based on bbox tag presence."""
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    label_uuid = r.json()["label"]["uuid"]
    t = await http_client.post("/api/tags/add", json={"name": "occluded", "color": "#AAAAAA"})
    tag_uuid = t.json()["tag"]["uuid"]

    _png(files_root, "tagged_img.png")
    _png(files_root, "untagged_img.png")
    await _add_bbox(http_client, "tagged_img.png", label_uuid, tag_uuids=[tag_uuid])
    await _register(http_client, "untagged_img.png")

    # Positive: images that have the tag
    ds_pos = await _create_dataset(
        http_client, "has-tag",
        conditions=[{"type": "tag", "tag_uuid": tag_uuid, "exclude": False}],
    )
    result = await _dataset_images(http_client, ds_pos["uuid"])
    names = [img["filename"] for img in result["images"]]
    assert "tagged_img.png" in names
    assert "untagged_img.png" not in names

    # Negative: images that don't have the tag
    ds_neg = await _create_dataset(
        http_client, "no-tag",
        conditions=[{"type": "tag", "tag_uuid": tag_uuid, "exclude": True}],
    )
    result2 = await _dataset_images(http_client, ds_neg["uuid"])
    names2 = [img["filename"] for img in result2["images"]]
    assert "tagged_img.png" not in names2
    assert "untagged_img.png" in names2


# ─────────────────────────────────────────────────────────────────────────────
# Dataset filter — DatasetLabelStatusCondition (with labels)
# ─────────────────────────────────────────────────────────────────────────────

async def test_dataset_filter_label_status(http_client, files_root):
    """DatasetLabelStatusCondition returns only images with verified label rows."""
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    label_uuid = r.json()["label"]["uuid"]

    _png(files_root, "confirmed_img.png")
    _png(files_root, "unconfirmed_img.png")
    # confirmed: verified bbox (no UNVERIFIED tag) → image_label PRESENT row
    await _add_bbox(http_client, "confirmed_img.png", label_uuid, tag_uuids=[])
    # unconfirmed: UNVERIFIED bbox → no image_label row (status = UNCERTAIN)
    await _add_bbox(http_client, "unconfirmed_img.png", label_uuid, tag_uuids=[UNVERIFIED_TAG_UUID])

    ds = await _create_dataset(
        http_client, "verified-only",
        conditions=[{"type": "dataset_label_status", "check": "all_verified"}],
        label_uuids=[label_uuid],
    )
    result = await _dataset_images(http_client, ds["uuid"])
    names = [img["filename"] for img in result["images"]]
    assert "confirmed_img.png" in names
    assert "unconfirmed_img.png" not in names


# ─────────────────────────────────────────────────────────────────────────────
# ImageSearchParams unit-level tests (via review queue endpoint)
# ─────────────────────────────────────────────────────────────────────────────

async def test_updated_at_range_filter(http_client, files_root, db_client):
    """updated_date_from/updated_date_to filter images by updated_at timestamp."""
    for name in ("old_upd.png", "new_upd.png"):
        _png(files_root, name)
        await _register(http_client, name)

    await _backdate(db_client, "old_upd.png", timedelta(days=30))

    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    r = await http_client.post(
        "/api/images/review",
        json={"show_all_bboxes": True, "tag_uuids": [], "updated_date_from": cutoff},
    )
    assert r.status_code == 200
    names = [img["filename"] for img in r.json()["images"]]
    assert "new_upd.png" in names
    assert "old_upd.png" not in names


async def test_tag_presence_filter(http_client, files_root):
    """TagPresenceFilter requires/excludes images based on bbox-level tag presence."""
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    label_uuid = r.json()["label"]["uuid"]
    t = await http_client.post("/api/tags/add", json={"name": "rare", "color": "#BBBBBB"})
    tag_uuid = t.json()["tag"]["uuid"]

    _png(files_root, "with_tag.png")
    _png(files_root, "without_tag.png")
    await _add_bbox(http_client, "with_tag.png", label_uuid, tag_uuids=[tag_uuid])
    await _register(http_client, "without_tag.png")

    # Require presence
    r = await http_client.post(
        "/api/images/review",
        json={
            "show_all_bboxes": True,
            "tag_uuids": [],
            "tag_presence_filters": [{"tag_uuid": tag_uuid, "exclude": False}],
        },
    )
    names = [img["filename"] for img in r.json()["images"]]
    assert "with_tag.png" in names
    assert "without_tag.png" not in names

    # Require absence
    r2 = await http_client.post(
        "/api/images/review",
        json={
            "show_all_bboxes": True,
            "tag_uuids": [],
            "tag_presence_filters": [{"tag_uuid": tag_uuid, "exclude": True}],
        },
    )
    names2 = [img["filename"] for img in r2.json()["images"]]
    assert "with_tag.png" not in names2
    assert "without_tag.png" in names2


async def test_require_verified_filter(http_client, files_root):
    """ImageLabelFilter.require_verified=True matches images with PRESENT or ABSENT row."""
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    label_uuid = r.json()["label"]["uuid"]

    _png(files_root, "confirmed_rv.png")
    _png(files_root, "unconfirmed_rv.png")
    # confirmed: verified bbox → image_label PRESENT row
    await _add_bbox(http_client, "confirmed_rv.png", label_uuid, tag_uuids=[])
    # unconfirmed: UNVERIFIED bbox → no image_label row
    await _add_bbox(http_client, "unconfirmed_rv.png", label_uuid, tag_uuids=[UNVERIFIED_TAG_UUID])

    r = await http_client.post(
        "/api/images/review",
        json={
            "show_all_bboxes": True,
            "tag_uuids": [],
            "image_label_filters": [
                {"label_uuid": label_uuid, "status": "present", "require_verified": True}
            ],
        },
    )
    assert r.status_code == 200
    names = [img["filename"] for img in r.json()["images"]]
    assert "confirmed_rv.png" in names
    assert "unconfirmed_rv.png" not in names


# ─────────────────────────────────────────────────────────────────────────────
# CRUD — PATCH and uniqueness
# ─────────────────────────────────────────────────────────────────────────────

async def test_patch_name_and_description(http_client):
    """PATCH can update name and description without touching label_uuids."""
    ds = await _create_dataset(http_client, "original-name", description="original desc")

    r = await http_client.patch(
        f"/api/datasets/{ds['uuid']}",
        json={"name": "updated-name", "description": "updated desc"},
    )
    assert r.json()["status"] == "success"
    updated = r.json()["dataset"]
    assert updated["name"] == "updated-name"
    assert updated["description"] == "updated desc"
    assert updated["label_uuids"] == ds["label_uuids"]  # unchanged


async def test_patch_filter_json(http_client):
    """PATCH replaces filter_json while leaving other fields intact."""
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    label_uuid = r.json()["label"]["uuid"]
    ds = await _create_dataset(http_client, "filter-patch-ds", label_uuids=[label_uuid])

    new_filter = {"conditions": [{"type": "filename", "pattern": "cats"}]}
    r = await http_client.patch(
        f"/api/datasets/{ds['uuid']}",
        json={"filter_json": new_filter},
    )
    assert r.json()["status"] == "success"
    updated = r.json()["dataset"]
    assert updated["filter_json"] == new_filter
    assert label_uuid in updated["label_uuids"]  # label_uuids unchanged


async def test_duplicate_dataset_name_rejected(http_client):
    """Creating two datasets with the same name returns status=failure."""
    await _create_dataset(http_client, "unique-name")

    r = await http_client.post(
        "/api/datasets",
        json={"name": "unique-name", "filter_json": {"conditions": []}, "label_uuids": []},
    )
    # Handler catches the IntegrityError and returns HTTP 200 with status=failure
    assert r.json()["status"] == "failure"


async def test_delete_dataset_cascades_label_links(http_client):
    """Deleting a dataset removes its label links; the label itself survives."""
    r = await http_client.post("/api/labels", json={"name": "linked-cat", "color": "#FF0000"})
    label_uuid = r.json()["label"]["uuid"]

    ds = await _create_dataset(http_client, "cascade-test", label_uuids=[label_uuid])
    assert label_uuid in ds["label_uuids"]

    del_r = await http_client.delete(f"/api/datasets/{ds['uuid']}")
    assert del_r.json()["status"] == "success"

    # Dataset is gone
    r2 = await http_client.delete(f"/api/datasets/{ds['uuid']}")
    assert r2.json()["status"] == "failure"

    # The label itself is unaffected
    labels_r = await http_client.get("/api/labels")
    label_names = [lbl["name"] for lbl in labels_r.json()["labels"]]
    assert "linked-cat" in label_names


# ─────────────────────────────────────────────────────────────────────────────
# ImageAgeCondition — created_at field
# ─────────────────────────────────────────────────────────────────────────────

async def test_image_age_condition_created_at_within_days(http_client, files_root, db_client):
    """ImageAgeCondition with field=created_at and within_days filters on created_at."""
    for name in ("old_created.png", "new_created.png"):
        _png(files_root, name)
        await _register(http_client, name)

    await _backdate_created_at(db_client, "old_created.png", timedelta(days=60))

    ds = await _create_dataset(
        http_client, "recent-created",
        conditions=[{"type": "image_age", "field": "created_at", "within_days": 7}],
    )
    result = await _dataset_images(http_client, ds["uuid"])
    names = [img["filename"] for img in result["images"]]
    assert "new_created.png" in names
    assert "old_created.png" not in names


# ─────────────────────────────────────────────────────────────────────────────
# ImageAgeCondition — absolute after/before range
# ─────────────────────────────────────────────────────────────────────────────

async def test_image_age_condition_after_before_range(http_client, files_root, db_client):
    """ImageAgeCondition with after/before fields applies an absolute timestamp window."""
    for name in ("img_range_old.png", "img_range_mid.png", "img_range_new.png"):
        _png(files_root, name)
        await _register(http_client, name)

    now = datetime.now(timezone.utc)
    await _backdate(db_client, "img_range_old.png", timedelta(days=30))
    await _backdate(db_client, "img_range_mid.png", timedelta(days=10))
    # img_range_new.png remains at now

    after = (now - timedelta(days=20)).isoformat()
    before = (now - timedelta(days=5)).isoformat()

    ds = await _create_dataset(
        http_client, "range-ds",
        conditions=[{
            "type": "image_age",
            "field": "updated_at",
            "after": after,
            "before": before,
        }],
    )
    result = await _dataset_images(http_client, ds["uuid"])
    names = [img["filename"] for img in result["images"]]
    assert "img_range_mid.png" in names      # inside the window
    assert "img_range_old.png" not in names  # before lower bound
    assert "img_range_new.png" not in names  # after upper bound


# ─────────────────────────────────────────────────────────────────────────────
# BboxAgeCondition — actual bbox-timestamp age filtering
# ─────────────────────────────────────────────────────────────────────────────

async def test_bbox_age_condition_excludes_old_bboxes(http_client, files_root, db_client):
    """BboxAgeCondition filters out images whose bboxes are older than within_days."""
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    label_uuid = r.json()["label"]["uuid"]

    _png(files_root, "recent_bbox.png")
    _png(files_root, "old_bbox.png")
    recent_bid = await _add_bbox(http_client, "recent_bbox.png", label_uuid)
    old_bid = await _add_bbox(http_client, "old_bbox.png", label_uuid)

    await _backdate_bbox(db_client, old_bid, timedelta(days=60))

    ds = await _create_dataset(
        http_client, "recent-bboxes",
        conditions=[{"type": "bbox_age", "field": "updated_at", "within_days": 7}],
    )
    result = await _dataset_images(http_client, ds["uuid"])
    names = [img["filename"] for img in result["images"]]
    assert "recent_bbox.png" in names
    assert "old_bbox.png" not in names


# ─────────────────────────────────────────────────────────────────────────────
# FilenameCondition — basic single-pattern
# ─────────────────────────────────────────────────────────────────────────────

async def test_filename_condition_single_pattern(http_client, files_root):
    """A single FilenameCondition filters images by filename substring."""
    for name in ("orange_cat.png", "blue_dog.png"):
        _png(files_root, name)
        await _register(http_client, name)

    ds = await _create_dataset(
        http_client, "cats-only",
        conditions=[{"type": "filename", "pattern": "orange_cat"}],
    )
    result = await _dataset_images(http_client, ds["uuid"])
    names = [img["filename"] for img in result["images"]]
    assert "orange_cat.png" in names
    assert "blue_dog.png" not in names


# ─────────────────────────────────────────────────────────────────────────────
# DatasetLabelStatusCondition — exclude_uncertain variant
# ─────────────────────────────────────────────────────────────────────────────

async def test_dataset_label_status_exclude_uncertain(http_client, files_root):
    """exclude_uncertain excludes images with no verified label row (same as all_verified)."""
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    label_uuid = r.json()["label"]["uuid"]

    _png(files_root, "verified_eu.png")
    _png(files_root, "uncertain_eu.png")
    await _add_bbox(http_client, "verified_eu.png", label_uuid, tag_uuids=[])
    await _add_bbox(http_client, "uncertain_eu.png", label_uuid, tag_uuids=[UNVERIFIED_TAG_UUID])

    ds = await _create_dataset(
        http_client, "exclude-uncertain",
        conditions=[{"type": "dataset_label_status", "check": "exclude_uncertain"}],
        label_uuids=[label_uuid],
    )
    result = await _dataset_images(http_client, ds["uuid"])
    names = [img["filename"] for img in result["images"]]
    assert "verified_eu.png" in names
    assert "uncertain_eu.png" not in names


# ─────────────────────────────────────────────────────────────────────────────
# Empty conditions — no filter returns all images
# ─────────────────────────────────────────────────────────────────────────────

async def test_empty_conditions_returns_all_images(http_client, files_root):
    """A dataset with no conditions returns every registered image."""
    for name in ("img_ec1.png", "img_ec2.png", "img_ec3.png"):
        _png(files_root, name)
        await _register(http_client, name)

    ds = await _create_dataset(http_client, "no-conditions")
    result = await _dataset_images(http_client, ds["uuid"])
    names = [img["filename"] for img in result["images"]]
    assert "img_ec1.png" in names
    assert "img_ec2.png" in names
    assert "img_ec3.png" in names


# ─────────────────────────────────────────────────────────────────────────────
# ImageSearchParams — updated_at upper bound
# ─────────────────────────────────────────────────────────────────────────────

async def test_updated_at_upper_bound_filter(http_client, files_root, db_client):
    """updated_date_to excludes images updated after the cutoff timestamp."""
    for name in ("old_upper.png", "new_upper.png"):
        _png(files_root, name)
        await _register(http_client, name)

    await _backdate(db_client, "old_upper.png", timedelta(days=30))
    # new_upper.png has updated_at ≈ now

    cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    r = await http_client.post(
        "/api/images/review",
        json={"show_all_bboxes": True, "tag_uuids": [], "updated_date_to": cutoff},
    )
    assert r.status_code == 200
    names = [img["filename"] for img in r.json()["images"]]
    assert "old_upper.png" in names
    assert "new_upper.png" not in names


# ─────────────────────────────────────────────────────────────────────────────
# ImageSearchParams — created_at range (date_from / date_to)
# ─────────────────────────────────────────────────────────────────────────────

async def test_created_at_range_filter(http_client, files_root, db_client):
    """date_from / date_to filter images by the created_at timestamp."""
    for name in ("old_cr.png", "new_cr.png"):
        _png(files_root, name)
        await _register(http_client, name)

    await _backdate_created_at(db_client, "old_cr.png", timedelta(days=30))

    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    r = await http_client.post(
        "/api/images/review",
        json={"show_all_bboxes": True, "tag_uuids": [], "date_from": cutoff},
    )
    assert r.status_code == 200
    names = [img["filename"] for img in r.json()["images"]]
    assert "new_cr.png" in names
    assert "old_cr.png" not in names
