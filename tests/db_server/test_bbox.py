"""Tests for single-bbox CRUD and thumbnail endpoints.

  PATCH  /api/images/bbox/{uuid}
  DELETE /api/images/bbox/{uuid}
  GET    /api/images/{path}/bbox/{uuid}/thumbnail
"""
import uuid
from pathlib import Path

import pytest

from apps.helpers.db.types import UNVERIFIED_TAG_UUID, VERIFIED_TAG_UUID
from tests.conftest import make_png


@pytest.fixture
def image_file(files_root):
    img = files_root / "bbox_test.png"
    img.write_bytes(make_png())
    return img


@pytest.fixture
async def label_uuid(http_client):
    r = await http_client.post("/api/labels", json={"name": "thing", "color": "#abcdef"})
    return r.json()["label"]["uuid"]


@pytest.fixture
async def setup_image_with_bbox(http_client, image_file, label_uuid):
    """Create one image with one bbox; return the bbox uuid."""
    bid = str(uuid.uuid4())
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "bbox_test.png",
            "boxes": [{
                "uuid": bid,
                "labelUuid": label_uuid,
                "x": 0.1, "y": 0.1, "width": 0.3, "height": 0.3,
                "tagUuids": [UNVERIFIED_TAG_UUID],
            }],
        },
    )
    assert r.status_code == 200
    return bid


# ──────────────────────────────────────────────────────────────────────────────
# PATCH /api/images/bbox/{uuid}
# ──────────────────────────────────────────────────────────────────────────────

async def test_patch_bbox_updates_label(http_client, setup_image_with_bbox, label_uuid):
    bid = setup_image_with_bbox

    # Create a second label to switch to
    r2 = await http_client.post("/api/labels", json={"name": "other", "color": "#000000"})
    other_uuid = r2.json()["label"]["uuid"]

    patch_r = await http_client.patch(
        f"/api/images/bbox/{bid}",
        json={"labelUuid": other_uuid, "tagUuids": [UNVERIFIED_TAG_UUID]},
    )
    assert patch_r.status_code == 204

    meta_r = await http_client.post(
        "/api/images/metadata/get", json={"imagePath": "bbox_test.png"}
    )
    bboxes = meta_r.json()["image"]["bboxes"]
    assert len(bboxes) == 1
    assert bboxes[0]["label_uuid"] == other_uuid


async def test_patch_bbox_updates_tags(http_client, setup_image_with_bbox, label_uuid):
    bid = setup_image_with_bbox

    patch_r = await http_client.patch(
        f"/api/images/bbox/{bid}",
        json={"labelUuid": label_uuid, "tagUuids": [VERIFIED_TAG_UUID]},
    )
    assert patch_r.status_code == 204

    meta_r = await http_client.post(
        "/api/images/metadata/get", json={"imagePath": "bbox_test.png"}
    )
    bboxes = meta_r.json()["image"]["bboxes"]
    tag_uuids = [t["uuid"] for t in bboxes[0]["tags"]]
    assert VERIFIED_TAG_UUID in tag_uuids
    assert UNVERIFIED_TAG_UUID not in tag_uuids


async def test_patch_nonexistent_bbox_returns_404(http_client):
    r = await http_client.patch(
        f"/api/images/bbox/{uuid.uuid4()}",
        json={"tagUuids": []},
    )
    assert r.status_code == 404


# ──────────────────────────────────────────────────────────────────────────────
# DELETE /api/images/bbox/{uuid}
# ──────────────────────────────────────────────────────────────────────────────

async def test_delete_bbox_removes_it(http_client, setup_image_with_bbox):
    bid = setup_image_with_bbox

    del_r = await http_client.delete(f"/api/images/bbox/{bid}")
    assert del_r.status_code == 204

    meta_r = await http_client.post(
        "/api/images/metadata/get", json={"imagePath": "bbox_test.png"}
    )
    bboxes = meta_r.json()["image"]["bboxes"]
    assert len(bboxes) == 0


async def test_delete_nonexistent_bbox_returns_404(http_client):
    r = await http_client.delete(f"/api/images/bbox/{uuid.uuid4()}")
    assert r.status_code == 404


async def test_delete_bbox_recomputes_image_labels(http_client, image_file, label_uuid):
    """After deleting the only verified bbox for a label, the PRESENT row is removed."""
    bid = str(uuid.uuid4())
    await http_client.post(
        "/api/images/metadata/update",
        json={"imagePath": "bbox_test.png", "boxes": [{
            "uuid": bid, "labelUuid": label_uuid,
            "x": 0.1, "y": 0.1, "width": 0.3, "height": 0.3,
            "tagUuids": [VERIFIED_TAG_UUID],
        }]},
    )

    labels_r = await http_client.get("/api/images/bbox_test.png/labels")
    assert label_uuid in {row["label_uuid"] for row in labels_r.json()["labels"]}

    await http_client.delete(f"/api/images/bbox/{bid}")

    labels_r = await http_client.get("/api/images/bbox_test.png/labels")
    assert label_uuid not in {row["label_uuid"] for row in labels_r.json()["labels"]}


# ──────────────────────────────────────────────────────────────────────────────
# Thumbnail
# ──────────────────────────────────────────────────────────────────────────────

async def test_thumbnail_returns_png_bytes(http_client, setup_image_with_bbox, monkeypatch, tmp_path):
    """Thumbnail endpoint returns PNG image data for a valid bbox."""
    # Point cache dir at a writable temp location
    monkeypatch.setattr("apps.db_server.manager.CACHE_DIR", tmp_path)

    bid = setup_image_with_bbox
    r = await http_client.get(f"/api/images/bbox_test.png/bbox/{bid}/thumbnail")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    # PNG magic bytes
    assert r.content[:4] == b"\x89PNG"


async def test_thumbnail_is_cached(http_client, setup_image_with_bbox, monkeypatch, tmp_path):
    """Second request for the same thumbnail is served from cache (file must exist)."""
    monkeypatch.setattr("apps.db_server.manager.CACHE_DIR", tmp_path)

    bid = setup_image_with_bbox
    url = f"/api/images/bbox_test.png/bbox/{bid}/thumbnail"

    r1 = await http_client.get(url)
    assert r1.status_code == 200

    # Cache file should exist
    cache_files = list((tmp_path / "thumb_cache").glob(f"{bid}_*.png"))
    assert len(cache_files) == 1

    # Second request returns identical bytes (from cache)
    r2 = await http_client.get(url)
    assert r2.status_code == 200
    assert r1.content == r2.content


async def test_thumbnail_nonexistent_bbox_returns_404(http_client, monkeypatch, tmp_path):
    monkeypatch.setattr("apps.db_server.manager.CACHE_DIR", tmp_path)
    r = await http_client.get(f"/api/images/bbox_test.png/bbox/{uuid.uuid4()}/thumbnail")
    assert r.status_code == 404


async def test_invalidate_thumbnails(http_client, setup_image_with_bbox, monkeypatch, tmp_path):
    """Invalidating thumbnails removes cached files."""
    monkeypatch.setattr("apps.db_server.manager.CACHE_DIR", tmp_path)

    bid = setup_image_with_bbox
    # Generate the thumbnail first
    await http_client.get(f"/api/images/bbox_test.png/bbox/{bid}/thumbnail")
    cache_files = list((tmp_path / "thumb_cache").glob(f"{bid}_*.png"))
    assert len(cache_files) == 1

    # Invalidate
    r = await http_client.post(
        "/api/images/bbox/thumbnails/invalidate",
        json={"bboxUuids": [bid]},
    )
    assert r.status_code == 200

    cache_files = list((tmp_path / "thumb_cache").glob(f"{bid}_*.png"))
    assert len(cache_files) == 0
