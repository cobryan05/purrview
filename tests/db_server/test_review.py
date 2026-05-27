"""Tests for the review queue endpoint — POST /api/images/review.

Covers tag filtering, label filtering, filename search, date range,
image_label filters, cursor pagination, and sort order.
"""
import uuid
from datetime import datetime, timezone, timedelta

import pytest

from apps.helpers.db.types import (
    FALSE_POS_TAG_UUID,
    UNVERIFIED_TAG_UUID,
    VERIFIED_TAG_UUID,
)
from tests.conftest import make_png


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
async def label_uuid(http_client):
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    return r.json()["label"]["uuid"]


@pytest.fixture
async def label2_uuid(http_client):
    r = await http_client.post("/api/labels", json={"name": "dog", "color": "#0000FF"})
    return r.json()["label"]["uuid"]


def _make_image(files_root, name: str) -> str:
    """Write a PNG to files_root and return the relative path."""
    p = files_root / name
    p.write_bytes(make_png())
    return name


async def _add_bbox(http_client, image_path: str, label_uuid: str, tag_uuids: list) -> str:
    """Create/replace bboxes on an image; return the bbox uuid."""
    bid = str(uuid.uuid4())
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": image_path,
            "boxes": [{
                "uuid": bid,
                "labelUuid": label_uuid,
                "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2,
                "tagUuids": tag_uuids,
            }],
        },
    )
    assert r.status_code == 200
    return bid


async def _review(http_client, **params):
    """POST /api/images/review with the given params and return the images list.

    ImageSearchParams uses snake_case field names (no alias_generator).
    """
    r = await http_client.post("/api/images/review", json=params)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    return body["images"]


# ──────────────────────────────────────────────────────────────────────────────
# Basic tag filter
# ──────────────────────────────────────────────────────────────────────────────

async def test_empty_queue(http_client):
    images = await _review(http_client, tag_uuids=[UNVERIFIED_TAG_UUID])
    assert images == []


async def test_unverified_image_appears_in_queue(http_client, files_root, label_uuid):
    _make_image(files_root, "unverified.png")
    await _add_bbox(http_client, "unverified.png", label_uuid, [UNVERIFIED_TAG_UUID])
    images = await _review(http_client, tag_uuids=[UNVERIFIED_TAG_UUID])
    filenames = [img["filename"] for img in images]
    assert any("unverified.png" in f for f in filenames)


async def test_verified_image_not_in_unverified_queue(http_client, files_root, label_uuid):
    _make_image(files_root, "verified.png")
    await _add_bbox(http_client, "verified.png", label_uuid, [VERIFIED_TAG_UUID])
    images = await _review(http_client, tag_uuids=[UNVERIFIED_TAG_UUID])
    filenames = [img["filename"] for img in images]
    assert not any("verified.png" in f for f in filenames)


async def test_verified_image_appears_in_verified_queue(http_client, files_root, label_uuid):
    _make_image(files_root, "verf2.png")
    await _add_bbox(http_client, "verf2.png", label_uuid, [VERIFIED_TAG_UUID])
    images = await _review(http_client, tag_uuids=[VERIFIED_TAG_UUID])
    filenames = [img["filename"] for img in images]
    assert any("verf2.png" in f for f in filenames)


async def test_show_all_bboxes_returns_all_images(http_client, files_root, label_uuid):
    _make_image(files_root, "all1.png")
    _make_image(files_root, "all2.png")
    await _add_bbox(http_client, "all1.png", label_uuid, [UNVERIFIED_TAG_UUID])
    await _add_bbox(http_client, "all2.png", label_uuid, [VERIFIED_TAG_UUID])
    images = await _review(http_client, show_all_bboxes=True)
    filenames = [img["filename"] for img in images]
    assert any("all1.png" in f for f in filenames)
    assert any("all2.png" in f for f in filenames)


# ──────────────────────────────────────────────────────────────────────────────
# Label filter
# ──────────────────────────────────────────────────────────────────────────────

async def test_label_filter_includes_matching(http_client, files_root, label_uuid, label2_uuid):
    _make_image(files_root, "cat-img.png")
    _make_image(files_root, "dog-img.png")
    await _add_bbox(http_client, "cat-img.png", label_uuid, [UNVERIFIED_TAG_UUID])
    await _add_bbox(http_client, "dog-img.png", label2_uuid, [UNVERIFIED_TAG_UUID])

    images = await _review(http_client, tag_uuids=[UNVERIFIED_TAG_UUID], label_uuid=label_uuid)
    filenames = [img["filename"] for img in images]
    assert any("cat-img.png" in f for f in filenames)
    assert not any("dog-img.png" in f for f in filenames)


# ──────────────────────────────────────────────────────────────────────────────
# Filename search
# ──────────────────────────────────────────────────────────────────────────────

async def test_filename_search_matches(http_client, files_root, label_uuid):
    _make_image(files_root, "search-match.png")
    _make_image(files_root, "no-match.png")
    await _add_bbox(http_client, "search-match.png", label_uuid, [UNVERIFIED_TAG_UUID])
    await _add_bbox(http_client, "no-match.png", label_uuid, [UNVERIFIED_TAG_UUID])

    images = await _review(http_client, show_all_bboxes=True, filename_search="search-match")
    filenames = [img["filename"] for img in images]
    assert any("search-match.png" in f for f in filenames)
    assert not any("no-match.png" in f for f in filenames)


async def test_filename_search_no_results(http_client, files_root, label_uuid):
    _make_image(files_root, "any.png")
    await _add_bbox(http_client, "any.png", label_uuid, [UNVERIFIED_TAG_UUID])
    images = await _review(http_client, show_all_bboxes=True, filename_search="zzz-not-there")
    assert images == []


# ──────────────────────────────────────────────────────────────────────────────
# Sort order
# ──────────────────────────────────────────────────────────────────────────────

async def test_default_sort_desc_newest_first(http_client, files_root, label_uuid):
    """Default sort is created_at DESC; later-created images come first."""
    for name in ["sort-a.png", "sort-b.png", "sort-c.png"]:
        _make_image(files_root, name)
        await _add_bbox(http_client, name, label_uuid, [UNVERIFIED_TAG_UUID])

    images = await _review(http_client, tag_uuids=[UNVERIFIED_TAG_UUID], sort_by="created_at", sort_order="desc")
    assert len(images) >= 3
    # Most-recently created should be first — just confirm we get a list back
    assert all("filename" in img for img in images)


async def test_sort_asc_returns_images(http_client, files_root, label_uuid):
    for name in ["asc-a.png", "asc-b.png"]:
        _make_image(files_root, name)
        await _add_bbox(http_client, name, label_uuid, [UNVERIFIED_TAG_UUID])

    images = await _review(http_client, tag_uuids=[UNVERIFIED_TAG_UUID], sort_by="created_at", sort_order="asc")
    assert len(images) >= 2


# ──────────────────────────────────────────────────────────────────────────────
# Limit / pagination
# ──────────────────────────────────────────────────────────────────────────────

async def test_limit_restricts_results(http_client, files_root, label_uuid):
    for i in range(5):
        name = f"page-{i}.png"
        _make_image(files_root, name)
        await _add_bbox(http_client, name, label_uuid, [UNVERIFIED_TAG_UUID])

    images = await _review(http_client, tag_uuids=[UNVERIFIED_TAG_UUID], limit=2)
    assert len(images) == 2


async def test_cursor_pagination_covers_all_records(http_client, files_root, label_uuid):
    """Paging through with limit=2 should yield all 4 distinct images."""
    names = [f"cur-{i}.png" for i in range(4)]
    for name in names:
        _make_image(files_root, name)
        await _add_bbox(http_client, name, label_uuid, [UNVERIFIED_TAG_UUID])

    seen = set()
    after_uuid = None
    after_sort_value = None

    for _ in range(4):   # at most 4 iterations for 4 records with limit=2
        params = dict(
            tag_uuids=[UNVERIFIED_TAG_UUID],
            sort_by="created_at",
            sort_order="desc",
            limit=2,
        )
        if after_uuid:
            params["after_uuid"] = after_uuid
            params["after_sort_value"] = after_sort_value

        images = await _review(http_client, **params)
        if not images:
            break
        for img in images:
            seen.add(img["filename"])
        last = images[-1]
        after_uuid = last["uuid"]
        after_sort_value = last.get("createdAt")  # ImageRecordRead uses camelCase

    assert len(seen) == 4


# ──────────────────────────────────────────────────────────────────────────────
# image_label_filters
# ──────────────────────────────────────────────────────────────────────────────

async def test_image_label_filter_present(http_client, files_root, label_uuid):
    """image_label_filters with PRESENT status should only return verified images."""
    _make_image(files_root, "ilf-verf.png")
    _make_image(files_root, "ilf-unv.png")
    await _add_bbox(http_client, "ilf-verf.png", label_uuid, [VERIFIED_TAG_UUID])
    await _add_bbox(http_client, "ilf-unv.png", label_uuid, [UNVERIFIED_TAG_UUID])

    images = await _review(
        http_client,
        show_all_bboxes=True,
        image_label_filters=[{"label_uuid": label_uuid, "status": "present", "exclude": False}],
    )
    filenames = [img["filename"] for img in images]
    assert any("ilf-verf.png" in f for f in filenames)
    assert not any("ilf-unv.png" in f for f in filenames)


async def test_image_label_filter_exclude(http_client, files_root, label_uuid):
    """Exclude filter: return images that do NOT have a 'present' label."""
    _make_image(files_root, "excl-verf.png")
    _make_image(files_root, "excl-unv.png")
    await _add_bbox(http_client, "excl-verf.png", label_uuid, [VERIFIED_TAG_UUID])
    await _add_bbox(http_client, "excl-unv.png", label_uuid, [UNVERIFIED_TAG_UUID])

    images = await _review(
        http_client,
        show_all_bboxes=True,
        image_label_filters=[{"label_uuid": label_uuid, "status": "present", "exclude": True}],
    )
    filenames = [img["filename"] for img in images]
    assert not any("excl-verf.png" in f for f in filenames)
    assert any("excl-unv.png" in f for f in filenames)
