"""Tests for single-bbox CRUD and thumbnail endpoints.

PATCH  /api/images/bbox/{uuid}
DELETE /api/images/bbox/{uuid}
GET    /api/images/{path}/bbox/{uuid}/thumbnail
"""

import asyncio
import uuid
from pathlib import Path

import pytest

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
