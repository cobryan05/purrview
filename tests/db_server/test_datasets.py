"""Tests for the dataset feature — CRUD, filter criteria, and pagination fixes.

This file was updated to use the new flat DatasetFilterCriteria format
(tagUuids, tagExcludeUuids, labelExcludeUuids, dateFrom, dateTo,
filenameSearch, verificationCheck) instead of the old conditions[] list.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pydantic
import pytest
from sqlmodel import select

from apps.helpers.db.types import (
    UNVERIFIED_TAG_UUID,
    VERIFIED_TAG_UUID,
    BBox,
    DatasetFilterCriteria,
    ImageRecord,
)
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


async def _add_bbox(http_client, image_path: str, label_uuid: str, *, tag_uuids: list | None = None) -> str:
    bid = str(uuid.uuid4())
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": image_path,
            "boxes": [
                {
                    "uuid": bid,
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": tag_uuids or [],
                }
            ],
        },
    )
    assert r.status_code == 200, r.text
    return bid


async def _create_dataset(
    http_client,
    name: str,
    *,
    filter_json: dict | None = None,
    label_uuids: list | None = None,
    description: str | None = None,
) -> dict:
    payload: dict = {
        "name": name,
        "filter_json": filter_json or {},
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
        result = await session.exec(select(ImageRecord).where(ImageRecord.filename.like(f"%/{filename}")))
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
    """Set updated_at on an image to now() - delta directly in the DB."""
    ts = datetime.now(timezone.utc) - delta
    async with db_client._async_session_maker() as session:
        result = await session.exec(select(ImageRecord).where(ImageRecord.filename.like(f"%/{filename}")))
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
    assert label_uuid in ds["labelUuids"]


async def test_update_dataset_replaces_labels(http_client):
    r1 = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    r2 = await http_client.post("/api/labels", json={"name": "dog", "color": "#0000FF"})
    lbl1 = r1.json()["label"]["uuid"]
    lbl2 = r2.json()["label"]["uuid"]

    ds = await _create_dataset(http_client, "animals", label_uuids=[lbl1])
    assert ds["labelUuids"] == [lbl1]

    r = await http_client.patch(f"/api/datasets/{ds['uuid']}", json={"label_uuids": [lbl2]})
    assert r.json()["status"] == "success"
    assert r.json()["dataset"]["labelUuids"] == [lbl2]


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
    assert label_uuid in ds["labelUuids"]

    await http_client.delete(f"/api/labels/{label_uuid}")

    r = await http_client.get("/api/datasets")
    datasets = r.json()["datasets"]
    found = next(d for d in datasets if d["uuid"] == ds["uuid"])
    assert label_uuid not in found["labelUuids"]


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
# filter_json validation
# ─────────────────────────────────────────────────────────────────────────────


async def test_filter_json_invalid_value_rejected_at_create(http_client):
    """POST with a field value that doesn't match the literal is rejected with 422."""
    r = await http_client.post(
        "/api/datasets",
        json={
            "name": "bad",
            "filter_json": {"label_match_mode": "invalid_mode"},
        },
    )
    assert r.status_code == 422


async def test_filter_json_invalid_value_rejected_at_update(http_client):
    """PATCH with an invalid field value returns 422."""
    ds = await _create_dataset(http_client, "valid-first")
    r = await http_client.patch(
        f"/api/datasets/{ds['uuid']}",
        json={"filter_json": {"label_match_mode": "also_invalid"}},
    )
    assert r.status_code == 422


async def test_filter_json_valid_flat_fields_accepted(http_client):
    """terms and the remaining flat (label_uuids-modifier) fields are accepted
    and round-trip correctly."""
    ds = await _create_dataset(
        http_client,
        "valid-flat",
        filter_json={"terms": [{"kind": "filename", "patterns": ["test"]}], "label_match_mode": "all"},
    )
    assert ds["filterJson"]["terms"] == [{"kind": "filename", "patterns": ["test"], "negate": False}]
    assert ds["filterJson"]["labelMatchMode"] == "all"


class TestDatasetFilterCriteriaExtraFields:
    """Unit-level coverage for the extra='ignore' -> 'forbid' fix (PR 1.4b).

    Before this fix, an unrecognised filter_json field would validate
    successfully, get silently dropped, and vanish from every subsequent GET —
    data loss by design. It must now be rejected outright.
    """

    def test_unknown_field_rejected(self):
        with pytest.raises(pydantic.ValidationError):
            DatasetFilterCriteria(**{"notARealField": "x"})

    def test_known_fields_still_accepted(self):
        # Sanity check the fix didn't also reject legitimate fields.
        criteria = DatasetFilterCriteria(
            terms=[{"kind": "filename", "patterns": ["x"]}],
            label_match_mode="all",
        )
        assert criteria.terms[0].patterns == ["x"]
        assert criteria.label_match_mode == "all"


async def test_filter_json_unknown_fields_rejected(http_client):
    """Unknown fields in filter_json are now rejected (extra='forbid'), not
    silently dropped — see TestDatasetFilterCriteriaExtraFields for the
    unit-level version of this."""
    r = await http_client.post(
        "/api/datasets",
        json={
            "name": "legacy-compat",
            "filter_json": {"conditions": [{"type": "filename", "pattern": "test"}]},
        },
    )
    assert r.status_code == 422


async def _corrupt_filter_json(db_client, dataset_uuid: str) -> None:
    """Write a filter_json no DatasetFilterCriteria can validate, bypassing the
    API's own validation — the shape a future schema change leaves behind."""
    from apps.helpers.db.types import Dataset

    async with db_client._async_session_maker() as session:
        row = (await session.exec(select(Dataset).where(Dataset.uuid == dataset_uuid))).one()
        row.filter_json = {"terms": [{"kind": "no_such_term_kind"}]}
        session.add(row)
        await session.commit()


async def test_unconvertible_row_does_not_take_down_the_listing(http_client, db_client):
    """One row too broken to validate must not 500 the whole listing.

    filter_json is validated on read, so a single bad row used to fail the
    entire GET /api/datasets response — and that listing feeds every dataset
    picker in the app, so one row took out training configs, model configs and
    the dataset checklist at once. PR 5.3b migrated the legacy shapes that
    caused this in practice; this is the defensive half that was left unbuilt
    (follow-up E), for the shapes a *future* schema change leaves behind.
    """
    good = await _create_dataset(http_client, "still-listable")
    bad = await _create_dataset(http_client, "broken-row")
    await _corrupt_filter_json(db_client, bad["uuid"])

    r = await http_client.get("/api/datasets")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body

    names = [d["name"] for d in body["datasets"]]
    assert "still-listable" in names, f"the healthy row must survive, got {names}"
    # Skipped, not silently coerced into something that would look editable and
    # then overwrite the user's real filter on the next save.
    assert "broken-row" not in names, f"the unconvertible row must be skipped, got {names}"
    assert good["uuid"] in [d["uuid"] for d in body["datasets"]]


# ─────────────────────────────────────────────────────────────────────────────
# verificationCheck — all_verified with label_uuids
# ─────────────────────────────────────────────────────────────────────────────


async def test_label_status_empty_label_uuids_returns_all_images(http_client, files_root):
    """verificationCheck with no label_uuids is a no-op — all images are returned."""
    _png(files_root, "img_fix2.png")
    await _register(http_client, "img_fix2.png")

    ds = await _create_dataset(
        http_client,
        "no-labels-status",
        filter_json={"verification_check": "all_verified"},
        label_uuids=[],  # intentionally empty — verification check requires labels to do anything
    )
    result = await _dataset_images(http_client, ds["uuid"])
    # With no label_uuids the verification check is ignored; all images pass
    names = [img["filename"] for img in result["images"]]
    assert "img_fix2.png" in names


async def test_dataset_filter_label_status_all_verified(http_client, files_root):
    """verificationCheck=all_verified returns only images with verified label rows."""
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    label_uuid = r.json()["label"]["uuid"]

    _png(files_root, "confirmed_img.png")
    _png(files_root, "unconfirmed_img.png")
    # confirmed: VERIFIED bbox → image_label PRESENT row
    await _add_bbox(http_client, "confirmed_img.png", label_uuid, tag_uuids=[VERIFIED_TAG_UUID])
    # unconfirmed: UNVERIFIED bbox → no image_label row (status = UNCERTAIN)
    await _add_bbox(http_client, "unconfirmed_img.png", label_uuid, tag_uuids=[UNVERIFIED_TAG_UUID])

    ds = await _create_dataset(
        http_client,
        "verified-only",
        filter_json={"verification_check": "all_verified"},
        label_uuids=[label_uuid],
    )
    result = await _dataset_images(http_client, ds["uuid"])
    names = [img["filename"] for img in result["images"]]
    assert "confirmed_img.png" in names
    assert "unconfirmed_img.png" not in names


async def test_dataset_label_status_has_unverified(http_client, files_root):
    """verificationCheck=has_unverified excludes fully-verified images."""
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    label_uuid = r.json()["label"]["uuid"]

    _png(files_root, "verified_eu.png")
    _png(files_root, "uncertain_eu.png")
    await _add_bbox(http_client, "verified_eu.png", label_uuid, tag_uuids=[VERIFIED_TAG_UUID])
    await _add_bbox(http_client, "uncertain_eu.png", label_uuid, tag_uuids=[UNVERIFIED_TAG_UUID])

    ds = await _create_dataset(
        http_client,
        "has-unverified",
        filter_json={"verification_check": "has_unverified"},
        label_uuids=[label_uuid],
    )
    result = await _dataset_images(http_client, ds["uuid"])
    names = [img["filename"] for img in result["images"]]
    assert "uncertain_eu.png" in names
    assert "verified_eu.png" not in names


# ─────────────────────────────────────────────────────────────────────────────
# after_sort_value — keyset pagination
# ─────────────────────────────────────────────────────────────────────────────


async def test_after_sort_value_accepted_in_endpoint(http_client, files_root):
    """after_sort_value is accepted and used for keyset pagination."""
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
# filenameSearch — single and compound patterns
# ─────────────────────────────────────────────────────────────────────────────


async def test_filename_search_single_pattern(http_client, files_root):
    """filenameSearch filters images by filename substring."""
    for name in ("orange_cat.png", "blue_dog.png"):
        _png(files_root, name)
        await _register(http_client, name)

    ds = await _create_dataset(
        http_client,
        "cats-only",
        filter_json={"terms": [{"kind": "filename", "patterns": ["orange_cat"]}]},
    )
    result = await _dataset_images(http_client, ds["uuid"])
    names = [img["filename"] for img in result["images"]]
    assert "orange_cat.png" in names
    assert "blue_dog.png" not in names


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
        json={"terms": [{"kind": "date", "field": "updated", "from": cutoff}]},
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
            "terms": [{"kind": "bboxTag", "tagUuids": [tag_uuid]}],
        },
    )
    names = [img["filename"] for img in r.json()["images"]]
    assert "with_tag.png" in names
    assert "without_tag.png" not in names

    # Require absence
    r2 = await http_client.post(
        "/api/images/review",
        json={
            "terms": [{"kind": "bboxTag", "tagUuids": [tag_uuid], "negate": True}],
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
    # confirmed: VERIFIED bbox → image_label PRESENT row
    await _add_bbox(http_client, "confirmed_rv.png", label_uuid, tag_uuids=[VERIFIED_TAG_UUID])
    # unconfirmed: UNVERIFIED bbox → no image_label row
    await _add_bbox(http_client, "unconfirmed_rv.png", label_uuid, tag_uuids=[UNVERIFIED_TAG_UUID])

    r = await http_client.post(
        "/api/images/review",
        json={
            "terms": [{"kind": "imageLabel", "labelUuid": label_uuid, "state": "any_reviewed"}],
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
    assert updated["labelUuids"] == ds["labelUuids"]  # unchanged


async def test_patch_filter_json(http_client):
    """PATCH replaces filter_json while leaving other fields intact."""
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    label_uuid = r.json()["label"]["uuid"]
    ds = await _create_dataset(http_client, "filter-patch-ds", label_uuids=[label_uuid])

    new_filter = {"terms": [{"kind": "filename", "patterns": ["cats"]}]}
    r = await http_client.patch(
        f"/api/datasets/{ds['uuid']}",
        json={"filter_json": new_filter},
    )
    assert r.json()["status"] == "success"
    updated = r.json()["dataset"]
    assert updated["filterJson"]["terms"] == [{"kind": "filename", "patterns": ["cats"], "negate": False}]
    assert label_uuid in updated["labelUuids"]  # labelUuids unchanged


async def test_duplicate_dataset_name_rejected(http_client):
    """Creating two datasets with the same name returns status=failure."""
    await _create_dataset(http_client, "unique-name")

    r = await http_client.post(
        "/api/datasets",
        json={"name": "unique-name", "filter_json": {}, "label_uuids": []},
    )
    # Handler catches the IntegrityError and returns HTTP 200 with status=failure
    assert r.json()["status"] == "failure"


async def test_delete_dataset_cascades_label_links(http_client):
    """Deleting a dataset removes its label links; the label itself survives."""
    r = await http_client.post("/api/labels", json={"name": "linked-cat", "color": "#FF0000"})
    label_uuid = r.json()["label"]["uuid"]

    ds = await _create_dataset(http_client, "cascade-test", label_uuids=[label_uuid])
    assert label_uuid in ds["labelUuids"]

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
# date_from / date_to filtering (flat fields on DatasetFilterCriteria)
# ─────────────────────────────────────────────────────────────────────────────


async def test_dataset_filter_date_from(http_client, files_root, db_client):
    """date_from filters images created before the cutoff (date_from maps to created_at)."""
    for name in ("old_img.png", "new_img.png"):
        _png(files_root, name)
        await _register(http_client, name)

    await _backdate_created_at(db_client, "old_img.png", timedelta(days=60))

    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    ds = await _create_dataset(
        http_client,
        "recent-images",
        filter_json={"terms": [{"kind": "date", "field": "created", "from": cutoff}]},
    )
    result = await _dataset_images(http_client, ds["uuid"])
    names = [img["filename"] for img in result["images"]]
    assert "new_img.png" in names
    assert "old_img.png" not in names


async def test_dataset_filter_date_to(http_client, files_root, db_client):
    """date_to excludes images created after the cutoff (date_to maps to created_at)."""
    for name in ("old_upper.png", "new_upper.png"):
        _png(files_root, name)
        await _register(http_client, name)

    await _backdate_created_at(db_client, "old_upper.png", timedelta(days=30))

    cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    ds = await _create_dataset(
        http_client,
        "older-images",
        filter_json={"terms": [{"kind": "date", "field": "created", "to": cutoff}]},
    )
    result = await _dataset_images(http_client, ds["uuid"])
    names = [img["filename"] for img in result["images"]]
    assert "old_upper.png" in names
    assert "new_upper.png" not in names


# ─────────────────────────────────────────────────────────────────────────────
# tagUuids / tagExcludeUuids filtering (flat fields)
# ─────────────────────────────────────────────────────────────────────────────


async def test_dataset_filter_tag_include(http_client, files_root):
    """tag_uuids includes only images whose bboxes carry any of those tags."""
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    label_uuid = r.json()["label"]["uuid"]
    t = await http_client.post("/api/tags/add", json={"name": "occluded", "color": "#AAAAAA"})
    tag_uuid = t.json()["tag"]["uuid"]

    _png(files_root, "tagged_img.png")
    _png(files_root, "untagged_img.png")
    await _add_bbox(http_client, "tagged_img.png", label_uuid, tag_uuids=[tag_uuid])
    await _register(http_client, "untagged_img.png")

    ds = await _create_dataset(
        http_client,
        "has-tag",
        filter_json={"terms": [{"kind": "bboxTag", "tagUuids": [tag_uuid]}]},
    )
    result = await _dataset_images(http_client, ds["uuid"])
    names = [img["filename"] for img in result["images"]]
    assert "tagged_img.png" in names
    assert "untagged_img.png" not in names


async def test_dataset_filter_tag_exclude(http_client, files_root):
    """tag_exclude_uuids excludes images whose bboxes carry any of those tags."""
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    label_uuid = r.json()["label"]["uuid"]
    t = await http_client.post("/api/tags/add", json={"name": "rare", "color": "#BBBBBB"})
    tag_uuid = t.json()["tag"]["uuid"]

    _png(files_root, "tagged_img2.png")
    _png(files_root, "untagged_img2.png")
    await _add_bbox(http_client, "tagged_img2.png", label_uuid, tag_uuids=[tag_uuid])
    await _register(http_client, "untagged_img2.png")

    ds = await _create_dataset(
        http_client,
        "no-tag",
        filter_json={"terms": [{"kind": "bboxTag", "tagUuids": [tag_uuid], "negate": True}]},
    )
    result = await _dataset_images(http_client, ds["uuid"])
    names = [img["filename"] for img in result["images"]]
    assert "tagged_img2.png" not in names
    assert "untagged_img2.png" in names


# ─────────────────────────────────────────────────────────────────────────────
# New feature: label_uuids image filter
# ─────────────────────────────────────────────────────────────────────────────


async def test_dataset_images_filtered_by_label_uuids(http_client, files_root):
    """A dataset with label_uuids restricts results to images with a bbox using one of those labels."""
    r = await http_client.post("/api/labels", json={"name": "cat-lu", "color": "#FF0000"})
    cat_uuid = r.json()["label"]["uuid"]
    r = await http_client.post("/api/labels", json={"name": "dog-lu", "color": "#0000FF"})
    dog_uuid = r.json()["label"]["uuid"]

    for name in ("lu_cat.png", "lu_dog.png", "lu_none.png"):
        _png(files_root, name)

    await _add_bbox(http_client, "lu_cat.png", cat_uuid)
    await _add_bbox(http_client, "lu_dog.png", dog_uuid)
    await _register(http_client, "lu_none.png")

    # Dataset linked only to "cat" — no filter_json
    ds = await _create_dataset(http_client, "label-uuid-filter", label_uuids=[cat_uuid])
    result = await _dataset_images(http_client, ds["uuid"])
    names = [img["filename"] for img in result["images"]]

    assert "lu_cat.png" in names, "image with a cat bbox must be included"
    assert "lu_dog.png" not in names, "image with only a dog bbox must be excluded"
    assert "lu_none.png" not in names, "image with no bboxes must be excluded"

    # Sanity: adding dog_uuid to the dataset exposes the dog image too
    r2 = await http_client.patch(
        f"/api/datasets/{ds['uuid']}",
        json={"label_uuids": [cat_uuid, dog_uuid]},
    )
    assert r2.json()["status"] == "success"
    result2 = await _dataset_images(http_client, ds["uuid"])
    names2 = [img["filename"] for img in result2["images"]]
    assert "lu_cat.png" in names2
    assert "lu_dog.png" in names2
    assert "lu_none.png" not in names2


# ─────────────────────────────────────────────────────────────────────────────
# Pagination sentinel
# ─────────────────────────────────────────────────────────────────────────────


async def test_has_more_correct_on_simple_path(http_client, files_root):
    """has_more uses the limit+1 sentinel correctly."""
    for name in ("img_sp1.png", "img_sp2.png", "img_sp3.png"):
        _png(files_root, name)
        await _register(http_client, name)

    ds = await _create_dataset(http_client, "simple-sentinel-ds")

    # Fetch 2 of 3 — has_more must be True
    page1 = await _dataset_images(http_client, ds["uuid"], limit=2)
    assert len(page1["images"]) == 2
    assert page1["hasMore"] is True, "has_more must be True when more results exist"

    # Fetch all 3 — has_more must be False
    page_all = await _dataset_images(http_client, ds["uuid"], limit=10)
    assert len(page_all["images"]) == 3
    assert page_all["hasMore"] is False, "has_more must be False when all results fit on one page"


# ─────────────────────────────────────────────────────────────────────────────
# /api/datasets/preview endpoint
# ─────────────────────────────────────────────────────────────────────────────


async def test_preview_dataset_images_camelcase_contract(http_client, files_root):
    """POST /api/datasets/preview accepts camelCase body, returns images+hasMore."""
    r = await http_client.post("/api/labels", json={"name": "cat-pv", "color": "#FF0000"})
    cat_uuid = r.json()["label"]["uuid"]

    for name in ("pv_cat.png", "pv_other.png"):
        _png(files_root, name)

    await _add_bbox(http_client, "pv_cat.png", cat_uuid)
    await _register(http_client, "pv_other.png")

    # camelCase body — flat filterJson with no conditions
    r = await http_client.post(
        "/api/datasets/preview",
        params={"limit": 50},
        json={"filterJson": {}, "labelUuids": [cat_uuid]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    names = [img["filename"] for img in body["images"]]
    assert "pv_cat.png" in names, "image with a matching bbox label must appear"
    assert "pv_other.png" not in names, "image without a matching bbox label must be excluded"
    assert body["hasMore"] is False


async def test_created_at_range_filter(http_client, files_root, db_client):
    """date_from / date_to filter images by the created_at timestamp."""
    for name in ("old_cr.png", "new_cr.png"):
        _png(files_root, name)
        await _register(http_client, name)

    await _backdate_created_at(db_client, "old_cr.png", timedelta(days=30))

    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    r = await http_client.post(
        "/api/images/review",
        json={"terms": [{"kind": "date", "field": "created", "from": cutoff}]},
    )
    assert r.status_code == 200
    names = [img["filename"] for img in r.json()["images"]]
    assert "new_cr.png" in names
    assert "old_cr.png" not in names


# ─────────────────────────────────────────────────────────────────────────────
# Empty filter — no filter returns all images
# ─────────────────────────────────────────────────────────────────────────────


async def test_empty_filter_returns_all_images(http_client, files_root):
    """A dataset with no filter_json returns every registered image."""
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
# image_label_conditions round-trip and filtering
# ─────────────────────────────────────────────────────────────────────────────


async def _set_image_label(http_client, image_path: str, label_uuid: str, status: str) -> None:
    r = await http_client.post(
        "/api/images/labels/set",
        json={"imagePath": image_path, "labelUuid": label_uuid, "status": status},
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "success"


async def _make_label(http_client, name: str) -> str:
    r = await http_client.post("/api/labels", json={"name": name, "color": "#AABBCC", "scope": "image"})
    assert r.status_code == 200, r.text
    return r.json()["label"]["uuid"]


async def test_image_label_conditions_round_trip(http_client, files_root):
    """An imageLabel term must survive a dataset save/load cycle unchanged.

    Replaces the pre-1.4b imageLabelConditions flat field, which is now
    expressed as a term (kind='imageLabel') in filter_json.terms.
    """
    label_uuid = await _make_label(http_client, "litter_box_rt")
    filter_json = {"terms": [{"kind": "imageLabel", "labelUuid": label_uuid, "state": "absent"}]}
    ds = await _create_dataset(http_client, "rt-test", filter_json=filter_json)

    saved = ds["filterJson"]
    assert "terms" in saved, "terms dropped from filterJson on save"
    assert saved["terms"] == [
        {"kind": "imageLabel", "labelUuid": label_uuid, "state": "absent", "rollUp": True, "negate": False}
    ]

    r = await http_client.get("/api/datasets")
    assert r.status_code == 200, r.text
    all_ds = {d["uuid"]: d for d in r.json()["datasets"]}
    fetched = all_ds[ds["uuid"]]["filterJson"]
    assert fetched.get("terms") == saved["terms"]


async def test_image_label_conditions_filter_absent(http_client, files_root):
    """Dataset with an imageLabel[absent] term returns only images where label is ABSENT."""
    label_uuid = await _make_label(http_client, "litter_box_flt")

    for name in ("ilc_absent.png", "ilc_present.png", "ilc_unset.png"):
        _png(files_root, name)
        await _register(http_client, name)

    await _set_image_label(http_client, "ilc_absent.png", label_uuid, "absent")
    await _set_image_label(http_client, "ilc_present.png", label_uuid, "present")
    # ilc_unset.png has no image_label row

    filter_json = {"terms": [{"kind": "imageLabel", "labelUuid": label_uuid, "state": "absent"}]}
    ds = await _create_dataset(http_client, "ilc-absent-ds", filter_json=filter_json)
    result = await _dataset_images(http_client, ds["uuid"])
    names = {img["filename"] for img in result["images"]}

    assert "ilc_absent.png" in names
    assert "ilc_present.png" not in names
    assert "ilc_unset.png" not in names


async def test_image_label_conditions_filter_present(http_client, files_root):
    """Dataset with an imageLabel[present] term returns only images where label is PRESENT."""
    label_uuid = await _make_label(http_client, "litter_box_prs")

    for name in ("ilcp_absent.png", "ilcp_present.png"):
        _png(files_root, name)
        await _register(http_client, name)

    await _set_image_label(http_client, "ilcp_absent.png", label_uuid, "absent")
    await _set_image_label(http_client, "ilcp_present.png", label_uuid, "present")

    filter_json = {"terms": [{"kind": "imageLabel", "labelUuid": label_uuid, "state": "present"}]}
    ds = await _create_dataset(http_client, "ilc-present-ds", filter_json=filter_json)
    result = await _dataset_images(http_client, ds["uuid"])
    names = {img["filename"] for img in result["images"]}

    assert "ilcp_present.png" in names
    assert "ilcp_absent.png" not in names


# ─────────────────────────────────────────────────────────────────────────────
# get_dataset_image_uuids pagination
# ─────────────────────────────────────────────────────────────────────────────


async def test_get_dataset_image_uuids_spans_multiple_pages(http_client, files_root, db_client, monkeypatch):
    """get_dataset_image_uuids must return all images when the result spans >1 page.

    Regression: the pagination loop incorrectly set after_sort_value to a UUID
    string (instead of leaving it None), causing pydantic to raise a validation
    error on the second page. The endpoint caught the error and silently returned
    an empty list, making the training dataset filter discard all labeled images.
    """
    import apps.helpers.db.db_client as db_client_module

    monkeypatch.setattr(db_client_module, "_READINESS_PAGE_SIZE", 2)

    names = [f"pag_{i}.png" for i in range(5)]
    for name in names:
        _png(files_root, name)
        await _register(http_client, name)

    ds = await _create_dataset(http_client, "all-images-paged")

    r = await http_client.get(
        "/api/datasets/image-uuids",
        params=[("dataset_uuid", ds["uuid"])],
    )
    body = r.json()
    assert body["status"] == "success", body.get("message")
    returned_uuids = body["imageUuids"]
    assert len(returned_uuids) == 5


class TestViewsAndPromotion:
    """Views and Datasets share the datasets table, separated by `promoted`.

    Decision register #10: "Dataset = promoted View, one record + flag."
    """

    async def _list(self, http_client, **params) -> list:
        r = await http_client.get("/api/datasets", params=params)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "success", body
        return body["datasets"]

    async def test_create_defaults_to_promoted(self, http_client):
        """POST /api/datasets creates a Dataset, not a View — existing callers
        of this endpoint all mean to create something training can consume."""
        ds = await _create_dataset(http_client, "defaults-promoted")
        assert ds["promoted"] is True

    async def test_views_are_excluded_from_the_default_listing(self, http_client):
        await _create_dataset(http_client, "a-real-dataset")
        r = await http_client.post(
            "/api/datasets",
            json={"name": "a-saved-view", "filter_json": {}, "label_uuids": [], "promoted": False},
        )
        assert r.status_code == 200, r.text

        names = {d["name"] for d in await self._list(http_client)}
        assert "a-real-dataset" in names
        assert "a-saved-view" not in names, (
            "A View leaked into the default dataset listing — the training-config "
            "and model-config pickers would offer it as trainable"
        )

    async def test_include_views_opts_the_view_back_in(self, http_client):
        await _create_dataset(http_client, "ds-for-optin")
        r = await http_client.post(
            "/api/datasets",
            json={"name": "view-for-optin", "filter_json": {}, "label_uuids": [], "promoted": False},
        )
        assert r.status_code == 200, r.text

        names = {d["name"] for d in await self._list(http_client, include_views="true")}
        assert {"ds-for-optin", "view-for-optin"} <= names

    async def test_promotion_round_trips(self, http_client):
        """Promotion is a flag flip on one record, and reverses."""
        r = await http_client.post(
            "/api/datasets",
            json={"name": "promote-me", "filter_json": {}, "label_uuids": [], "promoted": False},
        )
        uuid = r.json()["dataset"]["uuid"]
        assert "promote-me" not in {d["name"] for d in await self._list(http_client)}

        r = await http_client.patch(f"/api/datasets/{uuid}", json={"promoted": True})
        assert r.status_code == 200, r.text
        assert r.json()["dataset"]["promoted"] is True
        assert "promote-me" in {d["name"] for d in await self._list(http_client)}

        # ...and demote again
        r = await http_client.patch(f"/api/datasets/{uuid}", json={"promoted": False})
        assert r.status_code == 200, r.text
        assert r.json()["dataset"]["promoted"] is False
        assert "promote-me" not in {d["name"] for d in await self._list(http_client)}

    async def test_patching_other_fields_leaves_promoted_alone(self, http_client):
        """`promoted` is omitted from most PATCHes; exclude_unset must not reset it."""
        ds = await _create_dataset(http_client, "keep-promoted")
        r = await http_client.patch(f"/api/datasets/{ds['uuid']}", json={"description": "unrelated edit"})
        assert r.status_code == 200, r.text
        assert r.json()["dataset"]["promoted"] is True

    async def test_seeded_whole_database_dataset_is_promoted(self, http_client):
        """The seeded system dataset must be a Dataset, not a View.

        Regression guard: the table defaults to promoted=False, so the seed has
        to opt in explicitly. If it doesn't, "Whole Database" disappears from
        every training/model picker on any fresh database.
        """
        names = {d["name"] for d in await self._list(http_client)}
        assert "Whole Database" in names

    async def test_sidecar_roundtrip_keeps_datasets_promoted(self, http_client, db_client):
        """Restoring from datasets.json must not demote datasets to Views.

        The dump excludes Views, so everything in the sidecar is by definition a
        Dataset; a restore that dropped the flag would make them all invisible.
        """
        await _create_dataset(http_client, "sidecar-promoted")
        await db_client.sync_datasets_from_json()

        restored = {d.name: d for d in await db_client.list_datasets()}
        assert "sidecar-promoted" in restored
        assert restored["sidecar-promoted"].promoted is True


class TestDemotionGuard:
    """A Dataset a training config depends on must not be demotable to a View.

    `TrainingConfigDatasetLink` is ON DELETE RESTRICT and `delete_dataset`
    already refuses while links exist. Demotion had no such guard: the link row
    survives, but the row drops out of `GET /api/datasets`, so it silently
    vanishes from every training/model picker and the next save of that training
    config drops the link. Same silent-disappearance failure mode the rest of
    slice 5 guards against.
    """

    async def _link_training_config(self, http_client, dataset_uuid: str, name: str) -> str:
        r = await http_client.post(
            "/api/training-configs",
            json={"name": name, "datasetUuids": [dataset_uuid]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "success", body
        return body["config"]["uuid"]

    async def test_demoting_a_referenced_dataset_is_refused(self, http_client):
        ds = await _create_dataset(http_client, "referenced-dataset")
        await self._link_training_config(http_client, ds["uuid"], "tc-holding-a-reference")

        r = await http_client.patch(f"/api/datasets/{ds['uuid']}", json={"promoted": False})
        body = r.json()
        assert body["status"] == "failure", body
        assert "training config" in (body.get("message") or "").lower(), body

        # ...and it is still a Dataset.
        r = await http_client.get("/api/datasets")
        assert "referenced-dataset" in {d["name"] for d in r.json()["datasets"]}

    async def test_demoting_an_unreferenced_dataset_still_works(self, http_client):
        ds = await _create_dataset(http_client, "unreferenced-dataset")
        r = await http_client.patch(f"/api/datasets/{ds['uuid']}", json={"promoted": False})
        assert r.status_code == 200, r.text
        assert r.json()["dataset"]["promoted"] is False

    async def test_other_edits_to_a_referenced_dataset_still_work(self, http_client):
        """The guard keys off the promoted flip, not off having links at all."""
        ds = await _create_dataset(http_client, "referenced-but-editable")
        await self._link_training_config(http_client, ds["uuid"], "tc-editable-ref")

        r = await http_client.patch(f"/api/datasets/{ds['uuid']}", json={"description": "still fine"})
        assert r.status_code == 200, r.text
        assert r.json()["dataset"]["description"] == "still fine"
