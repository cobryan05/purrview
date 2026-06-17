"""Tests for whole-image tagging (image_tags table).

Distinct from bbox tags (bbox_tags table): these tags attach to the whole
image, independent of any bounding box, and are intended for whole-image
classification workflows (e.g. tagging corrupt/reject images for training
a reject filter).
"""

from pathlib import Path

import pytest

from tests.conftest import make_png


@pytest.fixture
def image_file(files_root):
    img = files_root / "image_tags_test.png"
    img.write_bytes(make_png())
    return img


@pytest.fixture
async def tag_uuid(http_client):
    r = await http_client.post("/api/tags/add", json={"name": "corrupt", "color": "#FF0000"})
    return r.json()["tag"]["uuid"]


@pytest.fixture
async def tag2_uuid(http_client):
    r = await http_client.post("/api/tags/add", json={"name": "blurry", "color": "#0000FF"})
    return r.json()["tag"]["uuid"]


async def _register_image(http_client, image_path):
    r = await http_client.post(
        "/api/images/metadata/update",
        json={"imagePath": image_path, "boxes": []},
    )
    assert r.status_code == 200
    return r.json()


async def _get_image_tags(http_client, image_path: str):
    r = await http_client.get(f"/api/images/{image_path}/tags")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    return [t["uuid"] for t in body.get("tags", [])]


async def _set_tag(http_client, image_path, tag_uuid, action):
    r = await http_client.post(
        "/api/images/tags/set",
        json={"imagePath": image_path, "tagUuid": tag_uuid, "action": action},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"


async def _set_tag_status(http_client, image_path, tag_uuid, status):
    r = await http_client.post(
        "/api/images/tags/set-status",
        json={"imagePath": image_path, "tagUuid": tag_uuid, "status": status},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"


async def _get_image_tag_statuses(http_client, image_path: str) -> dict:
    """Return {tag_uuid: status} for explicit (PRESENT/ABSENT) rows on the given image."""
    r = await http_client.get(f"/api/images/{image_path}/tags/statuses")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    return {row["tagUuid"]: row["status"] for row in body.get("tagStatuses", [])}


# ──────────────────────────────────────────────────────────────────────────────
# db_client level
# ──────────────────────────────────────────────────────────────────────────────


async def test_add_and_get_image_tag(db_client):
    image = await db_client.add_image(Path("a.png"))
    tag = await db_client.add_tag(name="reject")

    await db_client.add_image_tag(image.uuid, tag.uuid)
    assert await db_client.get_image_tag_uuids(image.uuid) == [tag.uuid]


async def test_add_image_tag_is_idempotent(db_client):
    image = await db_client.add_image(Path("a.png"))
    tag = await db_client.add_tag(name="reject")

    await db_client.add_image_tag(image.uuid, tag.uuid)
    await db_client.add_image_tag(image.uuid, tag.uuid)  # no-op, must not raise
    assert await db_client.get_image_tag_uuids(image.uuid) == [tag.uuid]


async def test_remove_image_tag(db_client):
    image = await db_client.add_image(Path("a.png"))
    tag = await db_client.add_tag(name="reject")

    await db_client.add_image_tag(image.uuid, tag.uuid)
    await db_client.remove_image_tag(image.uuid, tag.uuid)
    assert await db_client.get_image_tag_uuids(image.uuid) == []


async def test_remove_image_tag_missing_link_is_noop(db_client):
    image = await db_client.add_image(Path("a.png"))
    tag = await db_client.add_tag(name="reject")

    await db_client.remove_image_tag(image.uuid, tag.uuid)  # never added; must not raise
    assert await db_client.get_image_tag_uuids(image.uuid) == []


async def test_get_image_uuids_by_tag(db_client):
    image1 = await db_client.add_image(Path("a.png"))
    image2 = await db_client.add_image(Path("b.png"))
    tag = await db_client.add_tag(name="reject")

    await db_client.add_image_tag(image1.uuid, tag.uuid)
    assert await db_client.get_image_uuids_by_tag(tag.uuid) == {image1.uuid}
    assert image2.uuid not in await db_client.get_image_uuids_by_tag(tag.uuid)


# ──────────────────────────────────────────────────────────────────────────────
# API level — set/get
# ──────────────────────────────────────────────────────────────────────────────


async def test_set_tag_add_and_get(http_client, image_file, tag_uuid):
    await _register_image(http_client, "image_tags_test.png")
    await _set_tag(http_client, "image_tags_test.png", tag_uuid, "add")
    assert await _get_image_tags(http_client, "image_tags_test.png") == [tag_uuid]


async def test_set_tag_remove(http_client, image_file, tag_uuid):
    await _register_image(http_client, "image_tags_test.png")
    await _set_tag(http_client, "image_tags_test.png", tag_uuid, "add")
    await _set_tag(http_client, "image_tags_test.png", tag_uuid, "remove")
    assert await _get_image_tags(http_client, "image_tags_test.png") == []


async def test_set_tag_image_not_found(http_client, tag_uuid):
    r = await http_client.post(
        "/api/images/tags/set",
        json={"imagePath": "ghost.png", "tagUuid": tag_uuid, "action": "add"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "failure"


async def test_get_tags_image_not_found(http_client):
    r = await http_client.get("/api/images/ghost.png/tags")
    assert r.status_code == 200
    assert r.json()["status"] == "failure"


# ──────────────────────────────────────────────────────────────────────────────
# Bulk apply by UUIDs
# ──────────────────────────────────────────────────────────────────────────────


async def test_bulk_apply_add_tags(http_client, files_root, tag_uuid, tag2_uuid):
    paths = ["bulk_a.png", "bulk_b.png"]
    image_uuids = []
    for p in paths:
        (files_root / p).write_bytes(make_png())
        r = await _register_image(http_client, p)
        image_uuids.append(r["image"]["uuid"])

    r = await http_client.post(
        "/api/images/bulk-apply-by-uuids",
        json={"imageUuids": image_uuids, "action": "addTags", "tagUuids": [tag_uuid, tag2_uuid]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["modifiedCount"] == 2
    assert body["failCount"] == 0

    for p in paths:
        tags = await _get_image_tags(http_client, p)
        assert set(tags) == {tag_uuid, tag2_uuid}


async def test_bulk_apply_remove_tags(http_client, files_root, tag_uuid, tag2_uuid):
    p = "bulk_remove.png"
    (files_root / p).write_bytes(make_png())
    r = await _register_image(http_client, p)
    image_uuid = r["image"]["uuid"]
    await _set_tag(http_client, p, tag_uuid, "add")
    await _set_tag(http_client, p, tag2_uuid, "add")

    r = await http_client.post(
        "/api/images/bulk-apply-by-uuids",
        json={"imageUuids": [image_uuid], "action": "removeTags", "tagUuids": [tag_uuid]},
    )
    assert r.status_code == 200
    assert r.json()["modifiedCount"] == 1

    tags = await _get_image_tags(http_client, p)
    assert tags == [tag2_uuid]


# ──────────────────────────────────────────────────────────────────────────────
# Review queue filtering by whole-image tag
# ──────────────────────────────────────────────────────────────────────────────


async def test_review_queue_filters_by_image_tag(http_client, files_root, tag_uuid):
    tagged_path = "tagged.png"
    untagged_path = "untagged.png"
    for p in (tagged_path, untagged_path):
        (files_root / p).write_bytes(make_png())
        await _register_image(http_client, p)
    await _set_tag(http_client, tagged_path, tag_uuid, "add")

    r = await http_client.post(
        "/api/images/review",
        json={"showAllBboxes": True, "imageTagFilters": [{"tagUuid": tag_uuid, "exclude": False}]},
    )
    assert r.status_code == 200
    body = r.json()
    filenames = {img["filename"] for img in body["images"]}
    assert filenames == {tagged_path}


async def test_review_queue_excludes_by_image_tag(http_client, files_root, tag_uuid):
    tagged_path = "tagged2.png"
    untagged_path = "untagged2.png"
    for p in (tagged_path, untagged_path):
        (files_root / p).write_bytes(make_png())
        await _register_image(http_client, p)
    await _set_tag(http_client, tagged_path, tag_uuid, "add")

    r = await http_client.post(
        "/api/images/review",
        json={"showAllBboxes": True, "imageTagFilters": [{"tagUuid": tag_uuid, "exclude": True}]},
    )
    assert r.status_code == 200
    body = r.json()
    filenames = {img["filename"] for img in body["images"]}
    assert filenames == {untagged_path}


# ──────────────────────────────────────────────────────────────────────────────
# Tri-state verification (PRESENT / ABSENT / UNCERTAIN)
# ──────────────────────────────────────────────────────────────────────────────


async def test_set_tag_status_absent_and_get_statuses(http_client, image_file, tag_uuid):
    await _register_image(http_client, "image_tags_test.png")
    await _set_tag_status(http_client, "image_tags_test.png", tag_uuid, "absent")

    statuses = await _get_image_tag_statuses(http_client, "image_tags_test.png")
    assert statuses[tag_uuid] == "absent"
    # ABSENT must not appear in the PRESENT-only tags list.
    assert await _get_image_tags(http_client, "image_tags_test.png") == []


async def test_set_tag_status_present_via_status_endpoint(http_client, image_file, tag_uuid):
    await _register_image(http_client, "image_tags_test.png")
    await _set_tag_status(http_client, "image_tags_test.png", tag_uuid, "present")

    assert await _get_image_tags(http_client, "image_tags_test.png") == [tag_uuid]


async def test_set_tag_status_uncertain_clears_absent(http_client, image_file, tag_uuid):
    await _register_image(http_client, "image_tags_test.png")
    await _set_tag_status(http_client, "image_tags_test.png", tag_uuid, "absent")
    await _set_tag_status(http_client, "image_tags_test.png", tag_uuid, "uncertain")

    statuses = await _get_image_tag_statuses(http_client, "image_tags_test.png")
    assert tag_uuid not in statuses


async def test_set_tag_present_clears_existing_absent_mark(http_client, image_file, tag_uuid):
    """Applying a tag (add) after it was verified absent flips it back to PRESENT."""
    await _register_image(http_client, "image_tags_test.png")
    await _set_tag_status(http_client, "image_tags_test.png", tag_uuid, "absent")
    await _set_tag(http_client, "image_tags_test.png", tag_uuid, "add")

    assert await _get_image_tags(http_client, "image_tags_test.png") == [tag_uuid]
    statuses = await _get_image_tag_statuses(http_client, "image_tags_test.png")
    assert statuses[tag_uuid] == "present"


async def test_set_tag_status_image_not_found(http_client, tag_uuid):
    r = await http_client.post(
        "/api/images/tags/set-status",
        json={"imagePath": "ghost.png", "tagUuid": tag_uuid, "status": "absent"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "failure"


async def test_get_tag_statuses_image_not_found(http_client):
    r = await http_client.get("/api/images/ghost.png/tags/statuses")
    assert r.status_code == 200
    assert r.json()["status"] == "failure"


async def test_bulk_apply_set_tag_absent(http_client, files_root, tag_uuid):
    p = "bulk_tag_absent.png"
    (files_root / p).write_bytes(make_png())
    r = await _register_image(http_client, p)
    image_uuid = r["image"]["uuid"]

    r = await http_client.post(
        "/api/images/bulk-apply-by-uuids",
        json={"imageUuids": [image_uuid], "action": "setTagAbsent", "tagUuids": [tag_uuid]},
    )
    assert r.status_code == 200
    assert r.json()["modifiedCount"] == 1

    statuses = await _get_image_tag_statuses(http_client, p)
    assert statuses[tag_uuid] == "absent"


async def test_bulk_apply_set_tag_uncertain(http_client, files_root, tag_uuid):
    p = "bulk_tag_uncertain.png"
    (files_root / p).write_bytes(make_png())
    r = await _register_image(http_client, p)
    image_uuid = r["image"]["uuid"]
    await _set_tag_status(http_client, p, tag_uuid, "absent")

    r = await http_client.post(
        "/api/images/bulk-apply-by-uuids",
        json={"imageUuids": [image_uuid], "action": "setTagUncertain", "tagUuids": [tag_uuid]},
    )
    assert r.status_code == 200
    assert r.json()["modifiedCount"] == 1

    statuses = await _get_image_tag_statuses(http_client, p)
    assert tag_uuid not in statuses


async def test_review_queue_filters_by_verified_absent_tag(http_client, files_root, tag_uuid):
    absent_path = "absent.png"
    untouched_path = "untouched.png"
    for p in (absent_path, untouched_path):
        (files_root / p).write_bytes(make_png())
        await _register_image(http_client, p)
    await _set_tag_status(http_client, absent_path, tag_uuid, "absent")

    r = await http_client.post(
        "/api/images/review",
        json={"showAllBboxes": True, "imageTagFilters": [{"tagUuid": tag_uuid, "status": "absent", "exclude": False}]},
    )
    assert r.status_code == 200
    filenames = {img["filename"] for img in r.json()["images"]}
    assert filenames == {absent_path}


async def test_review_queue_require_verified_image_tag(http_client, files_root, tag_uuid):
    present_path = "verified_present.png"
    absent_path = "verified_absent.png"
    unchecked_path = "unchecked.png"
    for p in (present_path, absent_path, unchecked_path):
        (files_root / p).write_bytes(make_png())
        await _register_image(http_client, p)
    await _set_tag(http_client, present_path, tag_uuid, "add")
    await _set_tag_status(http_client, absent_path, tag_uuid, "absent")

    r = await http_client.post(
        "/api/images/review",
        json={
            "showAllBboxes": True,
            "imageTagFilters": [{"tagUuid": tag_uuid, "status": "present", "exclude": False, "requireVerified": True}],
        },
    )
    assert r.status_code == 200
    filenames = {img["filename"] for img in r.json()["images"]}
    assert filenames == {present_path, absent_path}


async def test_review_queue_filters_by_unset_image_tag(http_client, files_root, tag_uuid):
    """Images with no image_tags row for the tag (never assessed) match requireUnset=True."""
    unset_path = "unset_tag.png"
    present_path = "present_unset_test.png"
    absent_path = "absent_unset_test.png"
    for p in (unset_path, present_path, absent_path):
        (files_root / p).write_bytes(make_png())
        await _register_image(http_client, p)
    await _set_tag(http_client, present_path, tag_uuid, "add")
    await _set_tag_status(http_client, absent_path, tag_uuid, "absent")

    r = await http_client.post(
        "/api/images/review",
        json={"showAllBboxes": True, "imageTagFilters": [{"tagUuid": tag_uuid, "requireUnset": True}]},
    )
    assert r.status_code == 200
    filenames = {img["filename"] for img in r.json()["images"]}
    assert filenames == {unset_path}
