"""Tests for single-bbox CRUD and thumbnail endpoints.

PATCH  /api/images/bbox/{uuid}
DELETE /api/images/bbox/{uuid}
GET    /api/images/{path}/bbox/{uuid}/thumbnail
"""

import asyncio
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text as sa_text

from apps.db_server.manager import Manager
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
            "boxes": [
                {
                    "uuid": bid,
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.3,
                    "height": 0.3,
                    "tagUuids": [UNVERIFIED_TAG_UUID],
                }
            ],
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

    meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "bbox_test.png"})
    bboxes = meta_r.json()["image"]["bboxes"]
    assert len(bboxes) == 1
    assert bboxes[0]["labelUuid"] == other_uuid


async def test_patch_bbox_updates_tags(http_client, setup_image_with_bbox, label_uuid):
    """Patching to empty tags removes UNVERIFIED (bbox is now 'confirmed')."""
    bid = setup_image_with_bbox

    patch_r = await http_client.patch(
        f"/api/images/bbox/{bid}",
        json={"labelUuid": label_uuid, "tagUuids": []},
    )
    assert patch_r.status_code == 204

    meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "bbox_test.png"})
    bboxes = meta_r.json()["image"]["bboxes"]
    tag_uuids = [t["uuid"] for t in bboxes[0]["tags"]]
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

    meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "bbox_test.png"})
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
        json={
            "imagePath": "bbox_test.png",
            "boxes": [
                {
                    "uuid": bid,
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.3,
                    "height": 0.3,
                    "tagUuids": [VERIFIED_TAG_UUID],
                }
            ],
        },
    )

    labels_r = await http_client.get("/api/images/bbox_test.png/labels")
    assert label_uuid in {row["labelUuid"] for row in labels_r.json()["labels"]}

    await http_client.delete(f"/api/images/bbox/{bid}")

    labels_r = await http_client.get("/api/images/bbox_test.png/labels")
    assert label_uuid not in {row["labelUuid"] for row in labels_r.json()["labels"]}


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


# ──────────────────────────────────────────────────────────────────────────────
# Region thumbnail — the crop for a box that has no bbox row yet (a preview
# run's proposed *new* boxes), GET /api/images/{path}/region/thumbnail
# ──────────────────────────────────────────────────────────────────────────────

REGION_QS = "x=0.1&y=0.1&width_frac=0.3&height_frac=0.3"


async def test_region_thumbnail_returns_png_bytes(http_client, image_file, monkeypatch, tmp_path):
    monkeypatch.setattr("apps.db_server.manager.CACHE_DIR", tmp_path)
    r = await http_client.get(f"/api/images/bbox_test.png/region/thumbnail?{REGION_QS}")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    assert r.content[:4] == b"\x89PNG"


async def test_region_thumbnail_is_cached_per_region(http_client, image_file, monkeypatch, tmp_path):
    """The region is part of the cache key, so moving a proposed box misses the
    cache instead of serving the old crop."""
    monkeypatch.setattr("apps.db_server.manager.CACHE_DIR", tmp_path)

    url = f"/api/images/bbox_test.png/region/thumbnail?{REGION_QS}"
    r1 = await http_client.get(url)
    assert r1.status_code == 200
    cache_files = list((tmp_path / "thumb_cache").glob("rgn_*.png"))
    assert len(cache_files) == 1

    r2 = await http_client.get(url)
    assert r2.content == r1.content
    assert len(list((tmp_path / "thumb_cache").glob("rgn_*.png"))) == 1

    moved = await http_client.get(
        "/api/images/bbox_test.png/region/thumbnail?x=0.5&y=0.5&width_frac=0.3&height_frac=0.3"
    )
    assert moved.status_code == 200
    assert len(list((tmp_path / "thumb_cache").glob("rgn_*.png"))) == 2


async def test_region_thumbnail_is_cached_per_outline_color(http_client, image_file, monkeypatch, tmp_path):
    """The outline colour is baked into the PNG, so it has to be part of the key.

    Relabelling a proposed box re-points the <img> at the same region with a new
    colour; keyed on region alone, the second request serves the crop outlined
    in the previous label's colour — the chip says one class and the crop shows
    another.
    """
    monkeypatch.setattr("apps.db_server.manager.CACHE_DIR", tmp_path)

    red = await http_client.get(f"/api/images/bbox_test.png/region/thumbnail?{REGION_QS}&color=%23ff0000")
    assert red.status_code == 200

    blue = await http_client.get(f"/api/images/bbox_test.png/region/thumbnail?{REGION_QS}&color=%230000ff")
    assert blue.status_code == 200

    assert blue.content != red.content, "Re-colouring the outline must not serve the cached crop"
    assert len(list((tmp_path / "thumb_cache").glob("rgn_*.png"))) == 2


async def test_region_thumbnail_can_return_the_bare_region(http_client, image_file, monkeypatch, tmp_path):
    """`pad=0&outline=false` for a caller whose own frame already *is* the box.

    The watcher's match preview draws the crop inside a rectangle the user drags, so
    the default 20% context plus the stroked outline put a second, smaller box inside
    the first and made it unclear which edge the overlap was measured from.
    """
    monkeypatch.setattr("apps.db_server.manager.CACHE_DIR", tmp_path)

    padded = await http_client.get(f"/api/images/bbox_test.png/region/thumbnail?{REGION_QS}")
    bare = await http_client.get(f"/api/images/bbox_test.png/region/thumbnail?{REGION_QS}&pad=0&outline=false")
    assert bare.status_code == 200, bare.text
    assert bare.headers["content-type"] == "image/png"
    assert bare.content[:4] == b"\x89PNG"
    assert bare.content != padded.content


async def test_region_thumbnail_is_cached_per_crop_variant(http_client, image_file, monkeypatch, tmp_path):
    """Padding and the outline flag are baked into the PNG, so both must be keyed —
    otherwise a tight crop is served whatever the padded request left behind."""
    monkeypatch.setattr("apps.db_server.manager.CACHE_DIR", tmp_path)

    await http_client.get(f"/api/images/bbox_test.png/region/thumbnail?{REGION_QS}")
    assert len(list((tmp_path / "thumb_cache").glob("rgn_*.png"))) == 1

    bare = await http_client.get(f"/api/images/bbox_test.png/region/thumbnail?{REGION_QS}&pad=0&outline=false")
    assert bare.status_code == 200
    assert len(list((tmp_path / "thumb_cache").glob("rgn_*.png"))) == 2

    # And the tight crop is itself cached rather than regenerated per request.
    again = await http_client.get(f"/api/images/bbox_test.png/region/thumbnail?{REGION_QS}&pad=0&outline=false")
    assert again.content == bare.content
    assert len(list((tmp_path / "thumb_cache").glob("rgn_*.png"))) == 2


async def test_region_thumbnail_missing_image_returns_404(http_client, monkeypatch, tmp_path):
    monkeypatch.setattr("apps.db_server.manager.CACHE_DIR", tmp_path)
    r = await http_client.get(f"/api/images/no_such_image.png/region/thumbnail?{REGION_QS}")
    assert r.status_code == 404


async def test_region_thumbnail_zero_sized_region_returns_404(http_client, image_file, monkeypatch, tmp_path):
    """A degenerate region has nothing to crop — 404 beats a 1px image or a 500."""
    monkeypatch.setattr("apps.db_server.manager.CACHE_DIR", tmp_path)
    r = await http_client.get("/api/images/bbox_test.png/region/thumbnail?x=0.1&y=0.1&width_frac=0&height_frac=0.3")
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


# ──────────────────────────────────────────────────────────────────────────────
# Bulk apply bbox — relabel
# ──────────────────────────────────────────────────────────────────────────────


async def test_bulk_apply_bboxes_relabel(http_client, image_file, label_uuid):
    """POST /api/images/bbox/bulk-apply with action=relabel changes the label."""
    bid = str(uuid.uuid4())
    # Create a second label to relabel to
    r2 = await http_client.post("/api/labels", json={"name": "other_bulk", "color": "#123456"})
    other_uuid = r2.json()["label"]["uuid"]

    await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "bbox_test.png",
            "boxes": [
                {
                    "uuid": bid,
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [UNVERIFIED_TAG_UUID],
                }
            ],
        },
    )

    r = await http_client.post(
        "/api/images/bbox/bulk-apply",
        json={
            "filter": {"showAllBboxes": True, "limit": 50},
            "action": "relabel",
            "labelUuid": other_uuid,
            "tagUuids": [],
            "bboxTagFilter": [],
            "bboxLabelFilter": [],
            "showAllBboxes": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success", body
    assert body["modifiedCount"] == 1
    assert body["failCount"] == 0

    meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "bbox_test.png"})
    bboxes = meta_r.json()["image"]["bboxes"]
    assert len(bboxes) == 1
    assert bboxes[0]["labelUuid"] == other_uuid, f"Expected {other_uuid}, got {bboxes[0]['labelUuid']}"
    # Should have VERIFIED tag (relabeling strips UNVERIFIED)
    tag_uuids = [t["uuid"] for t in bboxes[0]["tags"]]
    assert VERIFIED_TAG_UUID in tag_uuids
    assert UNVERIFIED_TAG_UUID not in tag_uuids


async def _two_bboxes_one_verified(http_client, label_uuid):
    """One image carrying a VERIFIED bbox and an UNVERIFIED one. Returns
    (verified_uuid, unverified_uuid)."""
    verified_id = str(uuid.uuid4())
    unverified_id = str(uuid.uuid4())
    await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "bbox_test.png",
            "boxes": [
                {
                    "uuid": verified_id,
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [VERIFIED_TAG_UUID],
                },
                {
                    "uuid": unverified_id,
                    "labelUuid": label_uuid,
                    "x": 0.5,
                    "y": 0.5,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [UNVERIFIED_TAG_UUID],
                },
            ],
        },
    )
    return verified_id, unverified_id


class TestBulkApplyBboxVerificationScope:
    """bulk-apply must narrow to the same *bboxes* the review grid shows.

    The grid's bbox-verification tri-state is a per-bbox predicate
    (makeBboxFilterPredicate's verificationMatch), not merely an image-level
    "has a bbox in this state" match. Without a bbox-level equivalent here,
    filtering to "Unverified only" and clicking Apply to All also hit the
    verified bboxes sharing those images — silently modifying more than the
    grid showed. Follow-up PR A, UI_REDESIGN §32.5.
    """

    async def test_unverified_scope_skips_verified_bboxes(self, http_client, image_file, label_uuid):
        verified_id, unverified_id = await _two_bboxes_one_verified(http_client, label_uuid)

        r = await http_client.post(
            "/api/images/bbox/bulk-apply",
            json={
                "filter": {"showAllBboxes": True, "limit": 50},
                "action": "delete",
                "tagUuids": [],
                "bboxTagFilter": [],
                "bboxLabelFilter": [],
                "showAllBboxes": True,
                "bboxVerification": "unverified",
            },
        )
        assert r.status_code == 200
        assert r.json()["modifiedCount"] == 1, r.json()

        meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "bbox_test.png"})
        remaining = [b["uuid"] for b in meta_r.json()["image"]["bboxes"]]
        assert remaining == [verified_id], f"only the unverified bbox should be gone, got {remaining}"
        assert unverified_id not in remaining

    async def test_verified_scope_skips_unverified_bboxes(self, http_client, image_file, label_uuid):
        verified_id, unverified_id = await _two_bboxes_one_verified(http_client, label_uuid)

        r = await http_client.post(
            "/api/images/bbox/bulk-apply",
            json={
                "filter": {"showAllBboxes": True, "limit": 50},
                "action": "delete",
                "tagUuids": [],
                "bboxTagFilter": [],
                "bboxLabelFilter": [],
                "showAllBboxes": True,
                "bboxVerification": "verified",
            },
        )
        assert r.status_code == 200
        assert r.json()["modifiedCount"] == 1, r.json()

        meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "bbox_test.png"})
        remaining = [b["uuid"] for b in meta_r.json()["image"]["bboxes"]]
        assert remaining == [unverified_id], f"only the verified bbox should be gone, got {remaining}"

    async def test_omitted_scope_hits_every_bbox(self, http_client, image_file, label_uuid):
        """The narrowing is opt-in: no bboxVerification means today's behaviour."""
        await _two_bboxes_one_verified(http_client, label_uuid)

        r = await http_client.post(
            "/api/images/bbox/bulk-apply",
            json={
                "filter": {"showAllBboxes": True, "limit": 50},
                "action": "delete",
                "tagUuids": [],
                "bboxTagFilter": [],
                "bboxLabelFilter": [],
                "showAllBboxes": True,
            },
        )
        assert r.status_code == 200
        assert r.json()["modifiedCount"] == 2, r.json()

    async def test_verification_scope_ands_with_bbox_tag_filter(self, http_client, image_file, label_uuid):
        """bboxTagFilter is OR-semantics across its own list; the verification
        scope must AND with it, not join that OR. Merging the verification tag
        into bboxTagFilter would widen the scope instead of narrowing it."""
        r_tag = await http_client.post("/api/tags/add", json={"name": "review-me", "color": "#abcabc"})
        review_tag = r_tag.json()["tag"]["uuid"]

        tagged_verified = str(uuid.uuid4())
        tagged_unverified = str(uuid.uuid4())
        untagged_unverified = str(uuid.uuid4())
        await http_client.post(
            "/api/images/metadata/update",
            json={
                "imagePath": "bbox_test.png",
                "boxes": [
                    {
                        "uuid": tagged_verified,
                        "labelUuid": label_uuid,
                        "x": 0.1,
                        "y": 0.1,
                        "width": 0.1,
                        "height": 0.1,
                        "tagUuids": [VERIFIED_TAG_UUID, review_tag],
                    },
                    {
                        "uuid": tagged_unverified,
                        "labelUuid": label_uuid,
                        "x": 0.3,
                        "y": 0.3,
                        "width": 0.1,
                        "height": 0.1,
                        "tagUuids": [UNVERIFIED_TAG_UUID, review_tag],
                    },
                    {
                        "uuid": untagged_unverified,
                        "labelUuid": label_uuid,
                        "x": 0.6,
                        "y": 0.6,
                        "width": 0.1,
                        "height": 0.1,
                        "tagUuids": [UNVERIFIED_TAG_UUID],
                    },
                ],
            },
        )

        r = await http_client.post(
            "/api/images/bbox/bulk-apply",
            json={
                "filter": {"showAllBboxes": True, "limit": 50},
                "action": "delete",
                "tagUuids": [],
                "bboxTagFilter": [review_tag],
                "bboxLabelFilter": [],
                "showAllBboxes": False,
                "bboxVerification": "unverified",
            },
        )
        assert r.status_code == 200
        assert r.json()["modifiedCount"] == 1, r.json()

        meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "bbox_test.png"})
        remaining = sorted(b["uuid"] for b in meta_r.json()["image"]["bboxes"])
        assert remaining == sorted([tagged_verified, untagged_unverified])


async def _add_bbox_only(http_client, label_uuid) -> str:
    """One plain bbox on the shared test image; returns its uuid."""
    bid = str(uuid.uuid4())
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "bbox_test.png",
            "boxes": [
                {"uuid": bid, "labelUuid": label_uuid, "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2, "tagUuids": []}
            ],
        },
    )
    assert r.status_code == 200
    return bid


async def _two_bboxes_one_low_confidence(http_client, label_uuid):
    """One bbox at 20% confidence, one at 90%, on the same image.

    Confidences go on as strings because that is what the canvas save path
    writes (BoundingBoxMetadataModel.extra is Dict[str, str]); the SQL and the
    in-memory matcher both cast, so this exercises the harder of the two shapes.
    """
    low_id = str(uuid.uuid4())
    high_id = str(uuid.uuid4())
    await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "bbox_test.png",
            "boxes": [
                {
                    "uuid": low_id,
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [UNVERIFIED_TAG_UUID],
                    "extra": {"confidence": "0.2"},
                },
                {
                    "uuid": high_id,
                    "labelUuid": label_uuid,
                    "x": 0.5,
                    "y": 0.5,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [UNVERIFIED_TAG_UUID],
                    "extra": {"confidence": "0.9"},
                },
            ],
        },
    )
    return low_id, high_id


class TestBBoxConfidenceColumn:
    """`bboxes.confidence` is a GENERATED column over metadata_json, indexed so
    the confidence-band filter is a range seek instead of a full scan.

    The whole point of generating it rather than having each writer maintain a
    plain column is that it cannot fall out of sync — these tests pin that, plus
    the two things a naive `CAST(json_extract(...))` gets wrong.
    """

    async def _confidence_of(self, db_client, bbox_uuid):
        from sqlmodel import select

        from apps.helpers.db.types import BBox

        async with db_client._async_session_maker() as session:
            result = await session.exec(select(BBox.confidence).where(BBox.uuid == bbox_uuid))
            return result.one()

    async def test_populated_from_either_key_and_either_json_type(self, http_client, db_client, image_file, label_uuid):
        # String (canvas save path, extra is Dict[str, str]) under both keys.
        bid_conf = str(uuid.uuid4())
        bid_source = str(uuid.uuid4())
        bid_none = str(uuid.uuid4())
        await http_client.post(
            "/api/images/metadata/update",
            json={
                "imagePath": "bbox_test.png",
                "boxes": [
                    {
                        "uuid": bid_conf,
                        "labelUuid": label_uuid,
                        "x": 0.1,
                        "y": 0.1,
                        "width": 0.1,
                        "height": 0.1,
                        "extra": {"confidence": "0.25"},
                    },
                    {
                        "uuid": bid_source,
                        "labelUuid": label_uuid,
                        "x": 0.3,
                        "y": 0.1,
                        "width": 0.1,
                        "height": 0.1,
                        "extra": {"source_confidence": "0.75"},
                    },
                    {"uuid": bid_none, "labelUuid": label_uuid, "x": 0.5, "y": 0.1, "width": 0.1, "height": 0.1},
                ],
            },
        )
        assert await self._confidence_of(db_client, bid_conf) == pytest.approx(0.25)
        assert await self._confidence_of(db_client, bid_source) == pytest.approx(0.75)
        # No confidence at all stays NULL — it is not 0.0.
        assert await self._confidence_of(db_client, bid_none) is None

        # Float (task / preview-run path, metadata_json is Dict[str, Any]).
        r = await http_client.patch(f"/api/images/bbox/{bid_none}", json={"metadataJson": {"confidence": 0.5}})
        assert r.status_code == 204
        assert await self._confidence_of(db_client, bid_none) == pytest.approx(0.5)

    async def test_confidence_key_wins_over_source_confidence(self, http_client, db_client, image_file, label_uuid):
        """Same precedence as the Python and JS readers (newest key first)."""
        bid = await _add_bbox_only(http_client, label_uuid)
        await http_client.patch(
            f"/api/images/bbox/{bid}",
            json={"metadataJson": {"confidence": 0.9, "source_confidence": 0.1}},
        )
        assert await self._confidence_of(db_client, bid) == pytest.approx(0.9)

    async def test_tracks_metadata_edits_without_any_writer_setting_it(
        self, http_client, db_client, image_file, label_uuid
    ):
        """The sync property: nothing in the app ever writes this column."""
        bid = await _add_bbox_only(http_client, label_uuid)
        await http_client.patch(f"/api/images/bbox/{bid}", json={"metadataJson": {"confidence": 0.2}})
        assert await self._confidence_of(db_client, bid) == pytest.approx(0.2)

        # Re-write the metadata: the column must follow, not keep the old value.
        await http_client.patch(f"/api/images/bbox/{bid}", json={"metadataJson": {"confidence": 0.8}})
        assert await self._confidence_of(db_client, bid) == pytest.approx(0.8)

        # And clearing the confidence must take the column back to NULL, so the
        # box stops matching bands rather than lingering at its last value.
        await http_client.patch(f"/api/images/bbox/{bid}", json={"metadataJson": {"other": "x"}})
        assert await self._confidence_of(db_client, bid) is None

    async def test_junk_text_is_null_not_zero(self, http_client, db_client, image_file, label_uuid):
        """SQLite's CAST turns 'n/a' into 0.0, which would sweep junk into every
        low-confidence band — exactly the boxes the filter exists to surface."""
        bid = await _add_bbox_only(http_client, label_uuid)
        await http_client.patch(f"/api/images/bbox/{bid}", json={"metadataJson": {"confidence": "n/a"}})
        assert await self._confidence_of(db_client, bid) is None

        # A genuine zero, however, is a real confidence and must survive as 0.0.
        await http_client.patch(f"/api/images/bbox/{bid}", json={"metadataJson": {"confidence": "0.0"}})
        assert await self._confidence_of(db_client, bid) == 0.0

    async def test_band_query_uses_the_index(self, db_client):
        """Without the index this filter is a full scan of bboxes — the whole
        reason the column exists. Asserted on the query plan so a revert to a
        json_extract predicate fails here rather than silently getting slower."""
        async with db_client._async_session_maker() as session:
            result = await session.exec(
                sa_text(
                    "EXPLAIN QUERY PLAN SELECT image_uuid FROM bboxes" " WHERE confidence >= 0.4 AND confidence <= 0.6"
                )
            )
            plan = " ".join(str(cell) for row in result.all() for cell in row)
        assert "ix_bboxes_confidence_image_uuid" in plan, plan
        assert "SCAN bboxes" not in plan, plan


class TestVerifiedBBoxDropsConfidence:
    """A confidence is what a MODEL guessed; once a human verifies the box it is
    settled history, so a verified bbox stores none.

    Enforced in db_client at every bbox write, because the trigger is usually a
    TAG change rather than a metadata change — marking an existing inferred box
    Verified has to drop the confidence it was created with. `bboxes.confidence`
    is generated over those JSON keys, so dropping them is also what takes the
    box out of the confidence-band filter.
    """

    async def _metadata_of(self, db_client, bbox_uuid):
        from sqlmodel import select

        from apps.helpers.db.types import BBox

        async with db_client._async_session_maker() as session:
            result = await session.exec(select(BBox.metadata_json).where(BBox.uuid == bbox_uuid))
            return result.one()

    async def _confidence_of(self, db_client, bbox_uuid):
        from sqlmodel import select

        from apps.helpers.db.types import BBox

        async with db_client._async_session_maker() as session:
            result = await session.exec(select(BBox.confidence).where(BBox.uuid == bbox_uuid))
            return result.one()

    async def test_verifying_an_existing_bbox_drops_its_stored_confidence(
        self, http_client, db_client, image_file, label_uuid
    ):
        """The core case: nothing about the metadata changes in this request —
        only the tag — so a metadata-triggered rule alone would miss it."""
        bid = await _add_bbox_only(http_client, label_uuid)
        await http_client.patch(
            f"/api/images/bbox/{bid}",
            json={"tagUuids": [UNVERIFIED_TAG_UUID], "metadataJson": {"confidence": 0.42}},
        )
        assert await self._confidence_of(db_client, bid) == pytest.approx(0.42)

        r = await http_client.patch(f"/api/images/bbox/{bid}", json={"tagUuids": [VERIFIED_TAG_UUID]})
        assert r.status_code == 204

        assert await self._confidence_of(db_client, bid) is None
        assert "confidence" not in (await self._metadata_of(db_client, bid) or {})

    async def test_other_metadata_keys_survive(self, http_client, db_client, image_file, label_uuid):
        """Only the confidence keys go — the rest of the blob is unrelated state."""
        bid = await _add_bbox_only(http_client, label_uuid)
        await http_client.patch(
            f"/api/images/bbox/{bid}",
            json={
                "tagUuids": [UNVERIFIED_TAG_UUID],
                "metadataJson": {"confidence": 0.3, "source_confidence": 0.2, "run_model_added": True},
            },
        )
        await http_client.patch(f"/api/images/bbox/{bid}", json={"tagUuids": [VERIFIED_TAG_UUID]})

        metadata = await self._metadata_of(db_client, bid) or {}
        assert metadata == {"run_model_added": True}

    async def test_unverified_bbox_keeps_its_confidence(self, http_client, db_client, image_file, label_uuid):
        """Control: the rule must not blank every box it touches."""
        bid = await _add_bbox_only(http_client, label_uuid)
        await http_client.patch(
            f"/api/images/bbox/{bid}",
            json={"tagUuids": [UNVERIFIED_TAG_UUID], "metadataJson": {"confidence": 0.42}},
        )
        await http_client.patch(f"/api/images/bbox/{bid}", json={"tagUuids": [UNVERIFIED_TAG_UUID]})
        assert await self._confidence_of(db_client, bid) == pytest.approx(0.42)

    async def test_a_confidence_written_onto_an_already_verified_bbox_is_refused(
        self, http_client, db_client, image_file, label_uuid
    ):
        """Otherwise a later writer (a task re-scoring the box) puts one back."""
        bid = await _add_bbox_only(http_client, label_uuid)
        await http_client.patch(
            f"/api/images/bbox/{bid}",
            json={"tagUuids": [VERIFIED_TAG_UUID], "metadataJson": {"confidence": 0.9}},
        )
        assert await self._confidence_of(db_client, bid) is None

    async def test_canvas_save_of_a_verified_box_stores_no_confidence(
        self, http_client, db_client, image_file, label_uuid
    ):
        """The canvas save is a full DELETE+INSERT replace through a different
        db_client path than the single-bbox patch, so it needs its own cover."""
        bid = str(uuid.uuid4())
        r = await http_client.post(
            "/api/images/metadata/update",
            json={
                "imagePath": "bbox_test.png",
                "boxes": [
                    {
                        "uuid": bid,
                        "labelUuid": label_uuid,
                        "x": 0.1,
                        "y": 0.1,
                        "width": 0.2,
                        "height": 0.2,
                        "tagUuids": [VERIFIED_TAG_UUID],
                        "extra": {"confidence": "0.77"},
                    }
                ],
            },
        )
        assert r.status_code == 200
        assert await self._confidence_of(db_client, bid) is None

    async def test_bulk_relabel_drops_the_confidence_it_verifies_away(
        self, http_client, db_client, image_file, label_uuid
    ):
        """A bulk relabel promotes the box to VERIFIED — the confidence must go
        with it, on the bulk path as much as the single-bbox one."""
        bid = await _add_bbox_only(http_client, label_uuid)
        await http_client.patch(
            f"/api/images/bbox/{bid}",
            json={"tagUuids": [UNVERIFIED_TAG_UUID], "metadataJson": {"confidence": 0.31}},
        )
        r = await http_client.post("/api/labels", json={"name": "other-thing", "color": "#123456"})
        other_label = r.json()["label"]["uuid"]

        r = await http_client.post(
            "/api/images/bbox/bulk-apply-by-uuids",
            json={"bboxUuids": [bid], "action": "relabel", "labelUuid": other_label, "tagUuids": []},
        )
        assert r.status_code == 200
        assert r.json()["modifiedCount"] == 1

        assert await self._confidence_of(db_client, bid) is None

    async def test_a_verified_bbox_falls_out_of_the_confidence_band_filter(
        self, http_client, db_client, image_file, label_uuid
    ):
        """The user-visible consequence: verified boxes stop matching any band
        (they now match only `includeMissing`), which is what keeps the band
        filter pointed at the un-reviewed detections it exists to surface."""
        bid = await _add_bbox_only(http_client, label_uuid)
        await http_client.patch(
            f"/api/images/bbox/{bid}",
            json={"tagUuids": [UNVERIFIED_TAG_UUID], "metadataJson": {"confidence": 0.2}},
        )
        count_body = {
            "filter": {"showAllBboxes": True, "limit": 50},
            "bboxTagFilter": [],
            "bboxLabelFilter": [],
            "showAllBboxes": True,
            "bboxConfidence": {"max": 0.5},
        }
        r = await http_client.post("/api/images/bbox/count", json=count_body)
        assert r.json()["count"] == 1

        await http_client.patch(f"/api/images/bbox/{bid}", json={"tagUuids": [VERIFIED_TAG_UUID]})

        r = await http_client.post("/api/images/bbox/count", json=count_body)
        assert r.json()["count"] == 0


class TestBulkApplyConfidenceScope:
    """bulk-apply must narrow to the same *bboxes* the confidence-filtered grid
    shows. The bboxConfidence filter TERM only selects images that have an
    in-band bbox — on its own it would still let the action hit the out-of-band
    boxes sharing those images, exactly the failure mode bboxVerification's
    bbox-level field already exists to prevent.
    """

    async def test_band_skips_out_of_band_bboxes(self, http_client, image_file, label_uuid):
        low_id, high_id = await _two_bboxes_one_low_confidence(http_client, label_uuid)

        r = await http_client.post(
            "/api/images/bbox/bulk-apply",
            json={
                "filter": {"showAllBboxes": True, "limit": 50},
                "action": "delete",
                "tagUuids": [],
                "bboxTagFilter": [],
                "bboxLabelFilter": [],
                "showAllBboxes": True,
                "bboxConfidence": {"max": 0.5},
            },
        )
        assert r.status_code == 200
        assert r.json()["modifiedCount"] == 1, r.json()

        meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "bbox_test.png"})
        remaining = [b["uuid"] for b in meta_r.json()["image"]["bboxes"]]
        assert remaining == [high_id], f"only the low-confidence bbox should be gone, got {remaining}"
        assert low_id not in remaining

    async def test_unscored_bbox_needs_include_missing(self, http_client, image_file, label_uuid):
        """A hand-drawn box has no confidence; it is not 0 and must survive a
        low band unless the caller opts unscored boxes in."""
        scored_id = str(uuid.uuid4())
        unscored_id = str(uuid.uuid4())
        await http_client.post(
            "/api/images/metadata/update",
            json={
                "imagePath": "bbox_test.png",
                "boxes": [
                    {
                        "uuid": scored_id,
                        "labelUuid": label_uuid,
                        "x": 0.1,
                        "y": 0.1,
                        "width": 0.2,
                        "height": 0.2,
                        "tagUuids": [],
                        "extra": {"confidence": "0.2"},
                    },
                    {
                        "uuid": unscored_id,
                        "labelUuid": label_uuid,
                        "x": 0.5,
                        "y": 0.5,
                        "width": 0.2,
                        "height": 0.2,
                        "tagUuids": [],
                    },
                ],
            },
        )

        r = await http_client.post(
            "/api/images/bbox/bulk-apply",
            json={
                "filter": {"showAllBboxes": True, "limit": 50},
                "action": "delete",
                "tagUuids": [],
                "bboxTagFilter": [],
                "bboxLabelFilter": [],
                "showAllBboxes": True,
                "bboxConfidence": {"max": 0.5},
            },
        )
        assert r.status_code == 200
        assert r.json()["modifiedCount"] == 1, r.json()
        meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "bbox_test.png"})
        assert [b["uuid"] for b in meta_r.json()["image"]["bboxes"]] == [unscored_id]

        r2 = await http_client.post(
            "/api/images/bbox/bulk-apply",
            json={
                "filter": {"showAllBboxes": True, "limit": 50},
                "action": "delete",
                "tagUuids": [],
                "bboxTagFilter": [],
                "bboxLabelFilter": [],
                "showAllBboxes": True,
                "bboxConfidence": {"max": 0.5, "includeMissing": True},
            },
        )
        assert r2.status_code == 200
        assert r2.json()["modifiedCount"] == 1, r2.json()
        meta_r2 = await http_client.post("/api/images/metadata/get", json={"imagePath": "bbox_test.png"})
        assert meta_r2.json()["image"]["bboxes"] == []

    async def test_omitted_band_hits_every_bbox(self, http_client, image_file, label_uuid):
        """The narrowing is opt-in, like bboxVerification's."""
        await _two_bboxes_one_low_confidence(http_client, label_uuid)

        r = await http_client.post(
            "/api/images/bbox/bulk-apply",
            json={
                "filter": {"showAllBboxes": True, "limit": 50},
                "action": "delete",
                "tagUuids": [],
                "bboxTagFilter": [],
                "bboxLabelFilter": [],
                "showAllBboxes": True,
            },
        )
        assert r.status_code == 200
        assert r.json()["modifiedCount"] == 2, r.json()

    async def test_count_matches_what_bulk_apply_would_touch(self, http_client, image_file, label_uuid):
        """The count on the Apply-to-All button and the action itself take the
        same argument shape, so they must not disagree about the band."""
        await _two_bboxes_one_low_confidence(http_client, label_uuid)

        payload = {
            "filter": {"showAllBboxes": True, "limit": 50},
            "bboxTagFilter": [],
            "bboxLabelFilter": [],
            "showAllBboxes": True,
            "bboxConfidence": {"max": 0.5},
        }
        r_count = await http_client.post("/api/images/bbox/count", json=payload)
        assert r_count.status_code == 200, r_count.text
        assert r_count.json()["count"] == 1

        r = await http_client.post("/api/images/bbox/bulk-apply", json={**payload, "action": "delete", "tagUuids": []})
        assert r.json()["modifiedCount"] == 1


async def test_manual_relabel_clears_filter_chain_revert_stash(http_client, image_file, label_uuid):
    """A manual relabel of a filter-chain-relabeled bbox must drop the stashed
    relabel_prev_label_uuid, so a later revertRelabel can't silently restore the
    stale original label over the user's manual choice (it fails cleanly instead)."""
    bid = str(uuid.uuid4())
    r_prior = await http_client.post("/api/labels", json={"name": "orig_label", "color": "#654321"})
    prior_label_uuid = r_prior.json()["label"]["uuid"]
    r_manual = await http_client.post("/api/labels", json={"name": "manual_label", "color": "#00aa00"})
    manual_label_uuid = r_manual.json()["label"]["uuid"]

    # A bbox as left by ApplyFilterChainTask's relabel_bbox: current label + stashed prior.
    await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "bbox_test.png",
            "boxes": [
                {
                    "uuid": bid,
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [],
                    "extra": {"relabel_prev_label_uuid": prior_label_uuid},
                }
            ],
        },
    )

    # User manually relabels to a different label.
    r = await http_client.post(
        "/api/images/bbox/bulk-apply",
        json={
            "filter": {"showAllBboxes": True, "limit": 50},
            "action": "relabel",
            "labelUuid": manual_label_uuid,
            "tagUuids": [],
            "bboxTagFilter": [],
            "bboxLabelFilter": [],
            "showAllBboxes": True,
        },
    )
    assert r.status_code == 200
    assert r.json()["modifiedCount"] == 1

    meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "bbox_test.png"})
    bboxes = meta_r.json()["image"]["bboxes"]
    assert bboxes[0]["labelUuid"] == manual_label_uuid
    # Stash is gone, so a revert cannot resurrect the stale original label.
    assert "relabel_prev_label_uuid" not in (bboxes[0].get("extra") or {})

    r_revert = await http_client.post(
        "/api/images/bbox/bulk-apply",
        json={
            "filter": {"showAllBboxes": True, "limit": 50},
            "action": "revertRelabel",
            "tagUuids": [],
            "bboxTagFilter": [],
            "bboxLabelFilter": [],
            "showAllBboxes": True,
        },
    )
    assert r_revert.status_code == 200
    assert r_revert.json()["modifiedCount"] == 0  # nothing to revert; manual label preserved

    meta_r2 = await http_client.post("/api/images/metadata/get", json={"imagePath": "bbox_test.png"})
    assert meta_r2.json()["image"]["bboxes"][0]["labelUuid"] == manual_label_uuid


async def test_bulk_apply_bboxes_revert_relabel(http_client, image_file, label_uuid):
    """POST /api/images/bbox/bulk-apply with action=revertRelabel restores the
    prior label from metadata_json.relabel_prev_label_uuid and drops the tag(s)
    passed in tagUuids (the review tag), mirroring ApplyFilterChainTask's
    relabel_bbox action + Review tab's Revert bulk action."""
    bid = str(uuid.uuid4())
    review_tag_uuid = str(uuid.uuid4())
    await http_client.post(
        "/api/tags/add", json={"uuid": review_tag_uuid, "name": "auto-relabeled", "color": "#ff8800"}
    )
    r_prior = await http_client.post("/api/labels", json={"name": "prior_label", "color": "#654321"})
    prior_label_uuid = r_prior.json()["label"]["uuid"]

    await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "bbox_test.png",
            "boxes": [
                {
                    "uuid": bid,
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [review_tag_uuid],
                    "extra": {"relabel_prev_label_uuid": prior_label_uuid},
                }
            ],
        },
    )

    r = await http_client.post(
        "/api/images/bbox/bulk-apply",
        json={
            "filter": {"showAllBboxes": True, "limit": 50},
            "action": "revertRelabel",
            "tagUuids": [review_tag_uuid],
            "bboxTagFilter": [review_tag_uuid],
            "bboxLabelFilter": [],
            "showAllBboxes": False,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success", body
    assert body["modifiedCount"] == 1
    assert body["failCount"] == 0

    meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "bbox_test.png"})
    bboxes = meta_r.json()["image"]["bboxes"]
    assert len(bboxes) == 1
    assert bboxes[0]["labelUuid"] == prior_label_uuid
    tag_uuids = [t["uuid"] for t in bboxes[0]["tags"]]
    assert review_tag_uuid not in tag_uuids
    assert "relabel_prev_label_uuid" not in (bboxes[0].get("extra") or {})


async def test_bulk_apply_bboxes_revert_relabel_without_prior_label_fails(http_client, image_file, label_uuid):
    """A bbox with no relabel_prev_label_uuid in metadata_json can't be reverted;
    it's counted as a failure rather than silently no-op'd."""
    bid = str(uuid.uuid4())
    review_tag_uuid = str(uuid.uuid4())
    await http_client.post(
        "/api/tags/add", json={"uuid": review_tag_uuid, "name": "auto-relabeled2", "color": "#ff8800"}
    )

    await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "bbox_test.png",
            "boxes": [
                {
                    "uuid": bid,
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [review_tag_uuid],
                }
            ],
        },
    )

    r = await http_client.post(
        "/api/images/bbox/bulk-apply",
        json={
            "filter": {"showAllBboxes": True, "limit": 50},
            "action": "revertRelabel",
            "tagUuids": [review_tag_uuid],
            "bboxTagFilter": [review_tag_uuid],
            "bboxLabelFilter": [],
            "showAllBboxes": False,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success", body
    assert body["modifiedCount"] == 0
    assert body["failCount"] == 1


async def test_bulk_apply_bboxes_terminates_when_sorted_by_updated_at_asc(
    http_client, files_root, label_uuid, monkeypatch
):
    """Regression: bulk apply keyset-paginated by updated_at asc used to loop
    forever, because each action bumps updated_at past the cursor so processed
    rows re-match on the next page. It must terminate and touch each bbox once.
    """
    # Shrink the page size so a handful of images spans multiple pages.
    monkeypatch.setattr(Manager, "BULK_APPLY_PAGE_SIZE", 2)

    r2 = await http_client.post("/api/labels", json={"name": "relabel_target", "color": "#abcdef"})
    other_uuid = r2.json()["label"]["uuid"]

    n_images = 5
    for i in range(n_images):
        (files_root / f"loop_{i}.png").write_bytes(make_png())
        await http_client.post(
            "/api/images/metadata/update",
            json={
                "imagePath": f"loop_{i}.png",
                "boxes": [
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
            },
        )

    async def _do_bulk():
        return await http_client.post(
            "/api/images/bbox/bulk-apply",
            json={
                "filter": {"showAllBboxes": True, "limit": 50, "sortBy": "updated_at", "sortOrder": "asc"},
                "action": "relabel",
                "labelUuid": other_uuid,
                "tagUuids": [],
                "bboxTagFilter": [],
                "bboxLabelFilter": [],
                "showAllBboxes": True,
            },
        )

    # If the loop never terminates this raises TimeoutError instead of hanging CI.
    r = await asyncio.wait_for(_do_bulk(), timeout=15)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success", body
    # Each of the 5 bboxes must be modified exactly once — not re-processed.
    assert body["modifiedCount"] == n_images, body
    assert body["failCount"] == 0


async def test_bulk_apply_bboxes_by_uuids_relabel(http_client, image_file, label_uuid):
    """POST /api/images/bbox/bulk-apply-by-uuids with action=relabel changes the label."""
    bid = str(uuid.uuid4())
    r2 = await http_client.post("/api/labels", json={"name": "other_bulk2", "color": "#654321"})
    other_uuid = r2.json()["label"]["uuid"]

    await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "bbox_test.png",
            "boxes": [
                {
                    "uuid": bid,
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [UNVERIFIED_TAG_UUID],
                }
            ],
        },
    )

    r = await http_client.post(
        "/api/images/bbox/bulk-apply-by-uuids",
        json={"bboxUuids": [bid], "action": "relabel", "labelUuid": other_uuid, "tagUuids": []},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success", body
    assert body["modifiedCount"] == 1

    meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "bbox_test.png"})
    bboxes = meta_r.json()["image"]["bboxes"]
    assert bboxes[0]["labelUuid"] == other_uuid
    # Default (markVerified omitted) keeps the historical behaviour: a relabel is
    # a human decision, so the box is promoted to VERIFIED.
    tag_uuids = [t["uuid"] for t in bboxes[0]["tags"]]
    assert VERIFIED_TAG_UUID in tag_uuids
    assert UNVERIFIED_TAG_UUID not in tag_uuids


async def _seed_unverified_bbox(http_client, label_uuid, bid):
    await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "bbox_test.png",
            "boxes": [
                {
                    "uuid": bid,
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [UNVERIFIED_TAG_UUID],
                }
            ],
        },
    )


async def test_bulk_apply_by_uuids_relabel_mark_verified_false_keeps_unverified(http_client, image_file, label_uuid):
    """markVerified=False relabels without promoting the box to VERIFIED.

    "Run Model on Results" accepts a *model's* suggestion, not a human's, so the
    box must stay unverified (still reviewable) unless the user opts in.
    """
    bid = str(uuid.uuid4())
    r2 = await http_client.post("/api/labels", json={"name": "other_unverified", "color": "#654321"})
    other_uuid = r2.json()["label"]["uuid"]
    await _seed_unverified_bbox(http_client, label_uuid, bid)

    r = await http_client.post(
        "/api/images/bbox/bulk-apply-by-uuids",
        json={
            "bboxUuids": [bid],
            "action": "relabel",
            "labelUuid": other_uuid,
            "tagUuids": [],
            "markVerified": False,
        },
    )
    assert r.status_code == 200
    assert r.json()["modifiedCount"] == 1, r.json()

    meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "bbox_test.png"})
    bbox = meta_r.json()["image"]["bboxes"][0]
    # The relabel still landed…
    assert bbox["labelUuid"] == other_uuid
    # …but the verification state is untouched.
    tag_uuids = [t["uuid"] for t in bbox["tags"]]
    assert UNVERIFIED_TAG_UUID in tag_uuids
    assert VERIFIED_TAG_UUID not in tag_uuids


async def test_bulk_apply_by_uuids_relabel_mark_verified_true_promotes(http_client, image_file, label_uuid):
    """markVerified=True is the explicit opt-in and still stamps VERIFIED."""
    bid = str(uuid.uuid4())
    r2 = await http_client.post("/api/labels", json={"name": "other_verified", "color": "#654321"})
    other_uuid = r2.json()["label"]["uuid"]
    await _seed_unverified_bbox(http_client, label_uuid, bid)

    r = await http_client.post(
        "/api/images/bbox/bulk-apply-by-uuids",
        json={
            "bboxUuids": [bid],
            "action": "relabel",
            "labelUuid": other_uuid,
            "tagUuids": [],
            "markVerified": True,
        },
    )
    assert r.status_code == 200
    assert r.json()["modifiedCount"] == 1, r.json()

    meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "bbox_test.png"})
    bbox = meta_r.json()["image"]["bboxes"][0]
    tag_uuids = [t["uuid"] for t in bbox["tags"]]
    assert bbox["labelUuid"] == other_uuid
    assert VERIFIED_TAG_UUID in tag_uuids
    assert UNVERIFIED_TAG_UUID not in tag_uuids


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/images/bbox  (single-bbox insert — filter-chain add_bbox uses this)
# ──────────────────────────────────────────────────────────────────────────────


async def test_add_bbox_endpoint_inserts_new_box(http_client, setup_image_with_bbox, label_uuid):
    r = await http_client.post(
        "/api/images/bbox",
        json={
            "imagePath": "bbox_test.png",
            "labelUuid": label_uuid,
            "x": 0.5,
            "y": 0.5,
            "width": 0.2,
            "height": 0.2,
            "tagUuids": [UNVERIFIED_TAG_UUID],
            "metadataJson": {"filter_chain_added": True},
        },
    )
    assert r.status_code == 200
    new_uuid = r.json()["bboxUuid"]
    assert new_uuid

    meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "bbox_test.png"})
    bboxes = meta_r.json()["image"]["bboxes"]
    # Original bbox untouched + the newly inserted one.
    assert len(bboxes) == 2
    by_uuid = {b["uuid"]: b for b in bboxes}
    assert setup_image_with_bbox in by_uuid
    new = by_uuid[new_uuid]
    assert new["labelUuid"] == label_uuid
    assert new["x"] == 0.5 and new["width"] == 0.2
    assert UNVERIFIED_TAG_UUID in [t["uuid"] for t in new["tags"]]


async def test_add_bbox_endpoint_missing_image_returns_404(http_client, label_uuid):
    r = await http_client.post(
        "/api/images/bbox",
        json={
            "imagePath": "does_not_exist.png",
            "labelUuid": label_uuid,
            "x": 0.1,
            "y": 0.1,
            "width": 0.1,
            "height": 0.1,
        },
    )
    assert r.status_code == 404


async def test_add_bbox_accumulates_without_disturbing_siblings(http_client, setup_image_with_bbox, label_uuid):
    # A pure single-row insert: two successive adds must accumulate alongside the
    # original bbox (a full-image replace path would risk rewriting/dropping it).
    orig = setup_image_with_bbox
    for cx in (0.5, 0.8):
        r = await http_client.post(
            "/api/images/bbox",
            json={
                "imagePath": "bbox_test.png",
                "labelUuid": label_uuid,
                "x": cx,
                "y": cx,
                "width": 0.05,
                "height": 0.05,
            },
        )
        assert r.status_code == 200

    meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "bbox_test.png"})
    bboxes = meta_r.json()["image"]["bboxes"]
    uuids = {b["uuid"] for b in bboxes}
    assert len(bboxes) == 3
    assert orig in uuids  # original bbox never touched by the inserts


class TestCountBboxes:
    """POST /api/images/bbox/count — the number behind "Apply to All Filtered
    Bboxes". It must equal what a bulk-apply with the same filter modifies; the
    review queue's totalCount counts *images* and is not a substitute."""

    async def _make_three_bboxes(self, http_client, label_uuid, review_tag):
        tagged_verified = str(uuid.uuid4())
        tagged_unverified = str(uuid.uuid4())
        untagged_unverified = str(uuid.uuid4())
        await http_client.post(
            "/api/images/metadata/update",
            json={
                "imagePath": "bbox_test.png",
                "boxes": [
                    {
                        "uuid": tagged_verified,
                        "labelUuid": label_uuid,
                        "x": 0.1,
                        "y": 0.1,
                        "width": 0.1,
                        "height": 0.1,
                        "tagUuids": [VERIFIED_TAG_UUID, review_tag],
                    },
                    {
                        "uuid": tagged_unverified,
                        "labelUuid": label_uuid,
                        "x": 0.3,
                        "y": 0.3,
                        "width": 0.1,
                        "height": 0.1,
                        "tagUuids": [UNVERIFIED_TAG_UUID, review_tag],
                    },
                    {
                        "uuid": untagged_unverified,
                        "labelUuid": label_uuid,
                        "x": 0.6,
                        "y": 0.6,
                        "width": 0.1,
                        "height": 0.1,
                        "tagUuids": [UNVERIFIED_TAG_UUID],
                    },
                ],
            },
        )
        return tagged_verified, tagged_unverified, untagged_unverified

    async def test_counts_every_bbox_when_unfiltered(self, http_client, image_file, label_uuid):
        r_tag = await http_client.post("/api/tags/add", json={"name": "review-me", "color": "#abcabc"})
        await self._make_three_bboxes(http_client, label_uuid, r_tag.json()["tag"]["uuid"])

        r = await http_client.post(
            "/api/images/bbox/count",
            json={
                "filter": {"showAllBboxes": True, "limit": 50},
                "bboxTagFilter": [],
                "bboxLabelFilter": [],
                "showAllBboxes": True,
            },
        )
        assert r.status_code == 200
        # Three bboxes on ONE image — an image count would report 1 here.
        assert r.json()["count"] == 3, r.json()

    async def test_bbox_subset_filters_narrow_the_count(self, http_client, image_file, label_uuid):
        r_tag = await http_client.post("/api/tags/add", json={"name": "review-me", "color": "#abcabc"})
        review_tag = r_tag.json()["tag"]["uuid"]
        await self._make_three_bboxes(http_client, label_uuid, review_tag)

        body = {
            "filter": {"showAllBboxes": True, "limit": 50},
            "bboxTagFilter": [review_tag],
            "bboxLabelFilter": [],
            "showAllBboxes": False,
            "bboxVerification": "unverified",
        }
        r = await http_client.post("/api/images/bbox/count", json=body)
        assert r.status_code == 200
        # Only the tagged AND unverified bbox: verification ANDs with the tag
        # filter rather than joining its OR list.
        assert r.json()["count"] == 1, r.json()

    async def test_count_matches_what_bulk_apply_modifies(self, http_client, image_file, label_uuid):
        """Anti-drift: the count and the action share an argument shape, so the
        number shown must equal the number changed for the same filter."""
        r_tag = await http_client.post("/api/tags/add", json={"name": "review-me", "color": "#abcabc"})
        review_tag = r_tag.json()["tag"]["uuid"]
        await self._make_three_bboxes(http_client, label_uuid, review_tag)

        scope = {
            "filter": {"showAllBboxes": True, "limit": 50},
            "bboxTagFilter": [review_tag],
            "bboxLabelFilter": [],
            "showAllBboxes": False,
            "bboxVerification": "unverified",
        }
        r_count = await http_client.post("/api/images/bbox/count", json=scope)
        r_apply = await http_client.post(
            "/api/images/bbox/bulk-apply",
            json={**scope, "action": "delete", "tagUuids": []},
        )
        assert r_count.status_code == 200 and r_apply.status_code == 200
        assert r_count.json()["count"] == r_apply.json()["modifiedCount"] == 1

    async def test_label_filter_excludes_other_labels(self, http_client, image_file, label_uuid):
        r_tag = await http_client.post("/api/tags/add", json={"name": "review-me", "color": "#abcabc"})
        await self._make_three_bboxes(http_client, label_uuid, r_tag.json()["tag"]["uuid"])
        r_other = await http_client.post("/api/labels", json={"name": "count_other", "color": "#123456"})
        other_label = r_other.json()["label"]["uuid"]

        r = await http_client.post(
            "/api/images/bbox/count",
            json={
                "filter": {"showAllBboxes": True, "limit": 50},
                "bboxTagFilter": [],
                "bboxLabelFilter": [other_label],
                "showAllBboxes": True,
            },
        )
        assert r.status_code == 200
        assert r.json()["count"] == 0, r.json()


class TestCanvasSaveMetadataPreservation:
    """`POST /api/images/metadata/update` replaces the whole bbox list, and the
    canvas save posts only geometry/label/tags — no `extra`.

    That used to coerce to `{}` and wipe metadata_json on every bbox of the
    image, taking source_model / source_confidence / run_model_added with it,
    and relabel_prev_label_uuid, which "revert filter-chain relabel" reads back.
    """

    @staticmethod
    async def _save(http_client, image_path, bbox_uuid, label_uuid, extra=...):
        box = {
            "uuid": bbox_uuid,
            "labelUuid": label_uuid,
            "x": 0.1,
            "y": 0.1,
            "width": 0.2,
            "height": 0.2,
            "tagUuids": [],
        }
        if extra is not ...:
            box["extra"] = extra
        r = await http_client.post(
            "/api/images/metadata/update",
            json={"imagePath": image_path, "boxes": [box]},
        )
        assert r.status_code == 200, r.text
        return r

    @staticmethod
    async def _metadata(http_client, image_path):
        r = await http_client.post("/api/images/metadata/get", json={"imagePath": image_path})
        return r.json()["image"]["bboxes"][0]["metadataJson"]

    async def test_omitted_extra_preserves_stored_metadata(self, http_client, files_root):
        (files_root / "meta_keep.png").write_bytes(make_png())
        label_uuid = (await http_client.post("/api/labels", json={"name": "Cat", "color": "#f00"})).json()["label"][
            "uuid"
        ]
        r = await http_client.post(
            "/api/images/bbox",
            json={
                "imagePath": "meta_keep.png",
                "labelUuid": label_uuid,
                "x": 0.1,
                "y": 0.1,
                "width": 0.2,
                "height": 0.2,
                "tagUuids": [],
                "metadataJson": {"source_model": "det-a", "source_confidence": 0.82, "run_model_added": True},
            },
        )
        bbox_uuid = r.json()["bboxUuid"]

        await self._save(http_client, "meta_keep.png", bbox_uuid, label_uuid)

        metadata = await self._metadata(http_client, "meta_keep.png")
        assert metadata["source_confidence"] == pytest.approx(0.82)
        assert metadata["source_model"] == "det-a"
        assert metadata["run_model_added"] is True

    async def test_explicit_empty_extra_still_clears(self, http_client, files_root):
        """`{}` and "omitted" must stay distinguishable, or nothing can ever
        clear a bbox's metadata again."""
        (files_root / "meta_clear.png").write_bytes(make_png())
        label_uuid = (await http_client.post("/api/labels", json={"name": "Dog", "color": "#0f0"})).json()["label"][
            "uuid"
        ]
        r = await http_client.post(
            "/api/images/bbox",
            json={
                "imagePath": "meta_clear.png",
                "labelUuid": label_uuid,
                "x": 0.1,
                "y": 0.1,
                "width": 0.2,
                "height": 0.2,
                "tagUuids": [],
                "metadataJson": {"source_model": "det-a"},
            },
        )
        bbox_uuid = r.json()["bboxUuid"]

        await self._save(http_client, "meta_clear.png", bbox_uuid, label_uuid, extra={})

        assert await self._metadata(http_client, "meta_clear.png") == {}

    async def test_supplied_extra_persists_and_round_trips(self, http_client, files_root):
        """The canvas save path for a freshly-inferred box: it sends
        source_confidence as a **string**, because `extra` is Dict[str, str]."""
        (files_root / "meta_new.png").write_bytes(make_png())
        label_uuid = (await http_client.post("/api/labels", json={"name": "Fox", "color": "#00f"})).json()["label"][
            "uuid"
        ]
        r = await http_client.post(
            "/api/images/bbox",
            json={
                "imagePath": "meta_new.png",
                "labelUuid": label_uuid,
                "x": 0.1,
                "y": 0.1,
                "width": 0.2,
                "height": 0.2,
                "tagUuids": [],
            },
        )
        bbox_uuid = r.json()["bboxUuid"]

        await self._save(http_client, "meta_new.png", bbox_uuid, label_uuid, extra={"source_confidence": "0.82"})

        assert (await self._metadata(http_client, "meta_new.png"))["source_confidence"] == "0.82"


class TestGeneratedClientBboxMetadataRoundTrip:
    """The generated db_client is how background tasks write bboxes, and its
    field name for metadata (`bbox_meta`) differs from the server model's
    (`extra`). Asserting on the model object alone can't catch a mismatch here —
    pydantic ignores unknown kwargs — so this drives the real endpoint.
    """

    async def test_task_shaped_payload_persists_metadata(self, http_client, files_root):
        import json

        from db_client.models.bounding_box_metadata_model import BoundingBoxMetadataModel
        from db_client.models.update_metadata_payload import UpdateMetadataPayload

        (files_root / "collect.png").write_bytes(make_png())
        label_uuid = (await http_client.post("/api/labels", json={"name": "Collected", "color": "#f00"})).json()[
            "label"
        ]["uuid"]

        # Built exactly as CollectTrainingImagesTask builds it.
        payload = UpdateMetadataPayload(
            image_path="collect.png",
            boxes=[
                BoundingBoxMetadataModel(
                    bbox_uuid="bbox-collect-1",
                    x=0.1,
                    y=0.1,
                    width=0.2,
                    height=0.2,
                    label_uuid=label_uuid,
                    tag_uuids=[UNVERIFIED_TAG_UUID],
                    bbox_meta={"confidence": str(0.6231)},
                )
            ],
        )

        r = await http_client.post("/api/images/metadata/update", json=json.loads(payload.to_json()))
        assert r.status_code == 200, r.text

        r = await http_client.post("/api/images/metadata/get", json={"imagePath": "collect.png"})
        assert r.json()["image"]["bboxes"][0]["metadataJson"] == {"confidence": "0.6231"}


# ──────────────────────────────────────────────────────────────────────────────
# Crop framing — the outline must not scale with the source box
# ──────────────────────────────────────────────────────────────────────────────

THUMB_WIDTH = 300


@pytest.fixture
def framing_image(files_root):
    """A plain 1000x1000 frame — big enough that a 2% box is only 20 source px."""
    from PIL import Image as PILImage

    path = files_root / "framing.png"
    PILImage.new("RGB", (1000, 1000), (30, 30, 30)).save(path)
    return path


async def _make_bboxes(http_client, label, boxes):
    """Create every box in ONE metadata update — the endpoint replaces the
    image's box list wholesale, so a second call would delete the first box."""
    uuids = [str(uuid.uuid4()) for _ in boxes]
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "framing.png",
            "boxes": [
                {
                    "uuid": bid,
                    "labelUuid": label,
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "tagUuids": [UNVERIFIED_TAG_UUID],
                }
                for bid, (x, y, w, h) in zip(uuids, boxes)
            ],
        },
    )
    assert r.status_code == 200, r.text
    return uuids


def _outline_runs(png_bytes):
    """Lengths of the runs of outline-coloured pixels across the crop's middle row.

    The label is pure red on a grey frame, so a run is one vertical edge of the
    drawn rectangle. Two runs = the left and right edges of the box.
    """
    import io as _io

    from PIL import Image as PILImage

    im = PILImage.open(_io.BytesIO(png_bytes)).convert("RGB")
    row = [im.getpixel((x, im.height // 2)) for x in range(im.width)]
    runs = []
    current = 0
    for r, g, b in row:
        if r > 150 and g < 100 and b < 100:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs, im.size


async def test_crop_outline_width_is_independent_of_box_size(http_client, framing_image, monkeypatch, tmp_path):
    """A 20px box and a 600px box get the same outline weight in the thumbnail.

    Regression: the outline was stroked at a fixed 3 *source* pixels before the
    resize, so it came out ~36px thick on a tiny box's upscaled crop (eating the
    box until it read as a small box floating in context) and under a pixel on a
    large box's downscaled one (where LANCZOS then blurred it away).
    """
    monkeypatch.setattr("apps.db_server.manager.CACHE_DIR", tmp_path)
    r = await http_client.post("/api/labels", json={"name": "framing", "color": "#ff0000"})
    label = r.json()["label"]["uuid"]

    tiny, large = await _make_bboxes(http_client, label, [(0.4, 0.4, 0.02, 0.02), (0.2, 0.2, 0.6, 0.6)])

    tiny_runs, tiny_size = _outline_runs(
        (await http_client.get(f"/api/images/framing.png/bbox/{tiny}/thumbnail?width={THUMB_WIDTH}")).content
    )
    large_runs, large_size = _outline_runs(
        (await http_client.get(f"/api/images/framing.png/bbox/{large}/thumbnail?width={THUMB_WIDTH}")).content
    )

    assert tiny_size == large_size == (THUMB_WIDTH, THUMB_WIDTH)
    assert len(tiny_runs) == 2, f"expected two vertical edges, got {tiny_runs}"
    assert len(large_runs) == 2, f"expected two vertical edges, got {large_runs}"
    for run in tiny_runs + large_runs:
        assert 2 <= run <= 5, f"outline should be ~3px at width={THUMB_WIDTH}, got {tiny_runs} / {large_runs}"


async def test_crop_fills_the_same_share_of_the_thumbnail_at_any_box_size(
    http_client, framing_image, monkeypatch, tmp_path
):
    """The box's *interior* spans ~1/1.4 of the crop (20% context each side) either way.

    This is the user-visible half of the same regression: with a 32px outline
    stroked around a tiny box, only ~50% of the card was image, so a small box
    read as "zoomed to half the card with context around it" next to a large one.
    """
    monkeypatch.setattr("apps.db_server.manager.CACHE_DIR", tmp_path)
    r = await http_client.post("/api/labels", json={"name": "framing2", "color": "#ff0000"})
    label = r.json()["label"]["uuid"]

    tiny, large = await _make_bboxes(http_client, label, [(0.4, 0.4, 0.02, 0.02), (0.2, 0.2, 0.6, 0.6)])

    spans = {}
    for name, bid in (("tiny", tiny), ("large", large)):
        png = (await http_client.get(f"/api/images/framing.png/bbox/{bid}/thumbnail?width={THUMB_WIDTH}")).content
        import io as _io

        from PIL import Image as PILImage

        im = PILImage.open(_io.BytesIO(png)).convert("RGB")
        row = [im.getpixel((x, im.height // 2)) for x in range(im.width)]
        reds = [i for i, (rr, gg, bb) in enumerate(row) if rr > 150 and gg < 100 and bb < 100]
        # Inner edges of the two strokes: what the user actually sees of the box.
        first_gap = next(i for i in range(len(reds) - 1) if reds[i + 1] != reds[i] + 1)
        spans[name] = (reds[-1] - reds[first_gap]) / im.width

    assert abs(spans["tiny"] - spans["large"]) < 0.05, f"box fills different shares of the card: {spans}"
    for name, span in spans.items():
        assert span > 0.6, f"{name} box only fills {span:.0%} of the crop"
