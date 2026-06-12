"""Integration tests for the UUID-based similarity search endpoints and batch helpers.

Covers:
  - db_client.get_image_uuids_for_bboxes
  - db_client.get_images_by_uuids
  - manager.find_similar_images_by_uuid  (excludes query image from results)
  - manager.find_similar_bboxes_by_uuid  (excludes query bbox from results)
  - POST /api/images/find-similar-by-image-uuid
  - POST /api/images/bbox/find-similar-by-bbox-uuid
  - POST /api/images/batch-by-uuids
  - GET  /api/images/by-uuid/{image_uuid}/meta   (returns relative path)
  - GET  /api/images/by-uuid/{image_uuid}/thumbnail
  - GET  /api/images/bboxes/{bbox_uuid}/thumbnail
"""

import io
import uuid as uuid_mod

import pytest
from PIL import Image

from apps.helpers.db.types import UNVERIFIED_TAG_UUID


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_png(r: int = 100, g: int = 150, b: int = 200, size: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (r, g, b)).save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def image_file(files_root):
    p = files_root / "similar_test.png"
    p.write_bytes(_make_png())
    return p


@pytest.fixture
async def label_uuid(http_client):
    r = await http_client.post("/api/labels", json={"name": "dog", "color": "#00FF00"})
    assert r.status_code == 200
    return r.json()["label"]["uuid"]


@pytest.fixture
async def written_image(http_client, image_file, label_uuid):
    """Write one image with one bbox; return (image_uuid, bbox_uuid)."""
    bid = str(uuid_mod.uuid4())
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "similar_test.png",
            "boxes": [
                {
                    "uuid": bid,
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.6,
                    "height": 0.6,
                    "tagUuids": [UNVERIFIED_TAG_UUID],
                }
            ],
        },
    )
    assert r.status_code == 200
    meta = await http_client.post(
        "/api/images/metadata/get", json={"imagePath": "similar_test.png"}
    )
    image_uuid = meta.json()["image"]["uuid"]
    return image_uuid, bid


@pytest.fixture
async def index_built(http_client, written_image):
    """Rebuild the hash index so UUID-based search endpoints can find matches."""
    r = await http_client.post("/api/images/phash/rebuild-index")
    assert r.json()["status"] == "success"
    return written_image


# ─────────────────────────────────────────────────────────────────────────────
# DbClient batch helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestDbClientBatchHelpers:
    async def test_get_image_uuids_for_bboxes_returns_mapping(
        self, db_client, written_image
    ):
        image_uuid, bbox_uuid = written_image
        result = await db_client.get_image_uuids_for_bboxes([bbox_uuid])
        assert isinstance(result, dict)
        assert result.get(bbox_uuid) == image_uuid

    async def test_get_image_uuids_for_bboxes_empty_input(self, db_client):
        result = await db_client.get_image_uuids_for_bboxes([])
        assert result == {}

    async def test_get_image_uuids_for_bboxes_unknown_uuid_absent(self, db_client):
        result = await db_client.get_image_uuids_for_bboxes([str(uuid_mod.uuid4())])
        assert result == {}

    async def test_get_images_by_uuids_returns_records(self, db_client, written_image):
        image_uuid, _ = written_image
        records = await db_client.get_images_by_uuids([image_uuid])
        assert len(records) == 1
        assert records[0].uuid == image_uuid

    async def test_get_images_by_uuids_eager_loads_bboxes(self, db_client, written_image):
        image_uuid, bbox_uuid = written_image
        records = await db_client.get_images_by_uuids([image_uuid])
        assert len(records[0].bboxes) >= 1
        assert any(b.uuid == bbox_uuid for b in records[0].bboxes)

    async def test_get_images_by_uuids_empty_input(self, db_client):
        records = await db_client.get_images_by_uuids([])
        assert records == []

    async def test_get_images_by_uuids_unknown_uuid_absent(self, db_client):
        records = await db_client.get_images_by_uuids([str(uuid_mod.uuid4())])
        assert records == []


# ─────────────────────────────────────────────────────────────────────────────
# find-similar-by-image-uuid
# ─────────────────────────────────────────────────────────────────────────────


class TestFindSimilarByImageUuid:
    async def test_includes_source_image_in_results(self, http_client, index_built):
        """Query image must appear in results with distance 0 (exact self-match)."""
        image_uuid, _ = index_built
        r = await http_client.post(
            "/api/images/find-similar-by-image-uuid",
            json={"imageUuid": image_uuid, "threshold": 64},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        matches_by_uuid = {m["imageUuid"]: m for m in body["matches"]}
        assert image_uuid in matches_by_uuid
        assert matches_by_uuid[image_uuid]["distance"] == 0

    async def test_unknown_uuid_returns_empty(self, http_client):
        r = await http_client.post(
            "/api/images/find-similar-by-image-uuid",
            json={"imageUuid": str(uuid_mod.uuid4()), "threshold": 10},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["matches"] == []

    async def test_response_shape(self, http_client, index_built):
        image_uuid, _ = index_built
        r = await http_client.post(
            "/api/images/find-similar-by-image-uuid",
            json={"imageUuid": image_uuid, "threshold": 64},
        )
        assert r.status_code == 200
        for m in r.json()["matches"]:
            assert "imageUuid" in m
            assert "distance" in m


# ─────────────────────────────────────────────────────────────────────────────
# find-similar-by-bbox-uuid
# ─────────────────────────────────────────────────────────────────────────────


class TestFindSimilarByBboxUuid:
    async def test_includes_source_bbox_in_results(self, http_client, index_built):
        """Query bbox must appear in results with distance 0 (exact self-match)."""
        _, bbox_uuid = index_built
        r = await http_client.post(
            "/api/images/bbox/find-similar-by-bbox-uuid",
            json={"bboxUuid": bbox_uuid, "threshold": 64},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        matches_by_uuid = {m["bboxUuid"]: m for m in body["matches"]}
        assert bbox_uuid in matches_by_uuid
        assert matches_by_uuid[bbox_uuid]["distance"] == 0

    async def test_unknown_uuid_returns_empty(self, http_client):
        r = await http_client.post(
            "/api/images/bbox/find-similar-by-bbox-uuid",
            json={"bboxUuid": str(uuid_mod.uuid4()), "threshold": 10},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["matches"] == []

    async def test_response_shape(self, http_client, index_built):
        _, bbox_uuid = index_built
        r = await http_client.post(
            "/api/images/bbox/find-similar-by-bbox-uuid",
            json={"bboxUuid": bbox_uuid, "threshold": 64},
        )
        assert r.status_code == 200
        for m in r.json()["matches"]:
            assert "bboxUuid" in m
            assert "distance" in m


# ─────────────────────────────────────────────────────────────────────────────
# batch-by-uuids
# ─────────────────────────────────────────────────────────────────────────────


class TestBatchByUuids:
    async def test_returns_requested_images(self, http_client, written_image):
        image_uuid, _ = written_image
        r = await http_client.post(
            "/api/images/batch-by-uuids",
            json={"imageUuids": [image_uuid]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert len(body["images"]) == 1
        assert body["images"][0]["uuid"] == image_uuid

    async def test_empty_list_returns_empty(self, http_client):
        r = await http_client.post(
            "/api/images/batch-by-uuids",
            json={"imageUuids": []},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["images"] == []

    async def test_unknown_uuid_absent_from_results(self, http_client):
        r = await http_client.post(
            "/api/images/batch-by-uuids",
            json={"imageUuids": [str(uuid_mod.uuid4())]},
        )
        assert r.status_code == 200
        assert r.json()["images"] == []


# ─────────────────────────────────────────────────────────────────────────────
# /api/images/by-uuid/{image_uuid}/meta
# ─────────────────────────────────────────────────────────────────────────────


class TestImageMetaByUuid:
    async def test_returns_relative_path(self, http_client, written_image):
        """Filename in response must be a relative path, not an absolute container path."""
        image_uuid, _ = written_image
        r = await http_client.get(f"/api/images/by-uuid/{image_uuid}/meta")
        assert r.status_code == 200
        body = r.json()
        assert body["uuid"] == image_uuid
        # Filename must be relative, not starting with '/'
        assert not body["filename"].startswith("/"), (
            f"Expected relative path, got absolute: {body['filename']!r}"
        )

    async def test_unknown_uuid_returns_404(self, http_client):
        r = await http_client.get(f"/api/images/by-uuid/{uuid_mod.uuid4()}/meta")
        assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# /api/images/by-uuid/{image_uuid}/thumbnail
# ─────────────────────────────────────────────────────────────────────────────


class TestImageThumbnailByUuid:
    async def test_returns_png(self, http_client, written_image):
        image_uuid, _ = written_image
        r = await http_client.get(f"/api/images/by-uuid/{image_uuid}/thumbnail")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert len(r.content) > 0

    async def test_unknown_uuid_returns_404(self, http_client):
        r = await http_client.get(f"/api/images/by-uuid/{uuid_mod.uuid4()}/thumbnail")
        assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# /api/images/bboxes/{bbox_uuid}/thumbnail
# ─────────────────────────────────────────────────────────────────────────────


class TestBboxThumbnailByUuid:
    async def test_returns_png(self, http_client, written_image):
        _, bbox_uuid = written_image
        r = await http_client.get(f"/api/images/bboxes/{bbox_uuid}/thumbnail")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert len(r.content) > 0

    async def test_unknown_uuid_returns_404(self, http_client):
        r = await http_client.get(f"/api/images/bboxes/{uuid_mod.uuid4()}/thumbnail")
        assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# filter_bboxes_by_tags (db_client)
# ─────────────────────────────────────────────────────────────────────────────


class TestFilterBboxesByTags:
    """Unit tests for DbClient.filter_bboxes_by_tags — any/all/exclude modes."""

    @pytest.fixture
    async def two_tags(self, http_client):
        """Create two tags; return (tag_a_uuid, tag_b_uuid)."""
        ra = await http_client.post("/api/tags/add", json={"name": "tag-a", "color": "#FF0000"})
        rb = await http_client.post("/api/tags/add", json={"name": "tag-b", "color": "#00FF00"})
        return ra.json()["tag"]["uuid"], rb.json()["tag"]["uuid"]

    @pytest.fixture
    async def two_bboxes(self, http_client, image_file, label_uuid, two_tags):
        """bbox1 has tag-a only; bbox2 has tag-a and tag-b."""
        tag_a, tag_b = two_tags
        bid1, bid2 = str(uuid_mod.uuid4()), str(uuid_mod.uuid4())
        r = await http_client.post(
            "/api/images/metadata/update",
            json={
                "imagePath": "similar_test.png",
                "boxes": [
                    {
                        "uuid": bid1,
                        "labelUuid": label_uuid,
                        "x": 0.1, "y": 0.1, "width": 0.3, "height": 0.3,
                        "tagUuids": [tag_a],
                    },
                    {
                        "uuid": bid2,
                        "labelUuid": label_uuid,
                        "x": 0.5, "y": 0.5, "width": 0.3, "height": 0.3,
                        "tagUuids": [tag_a, tag_b],
                    },
                ],
            },
        )
        assert r.status_code == 200
        return bid1, bid2, tag_a, tag_b

    async def test_empty_input_returns_empty(self, db_client):
        result = await db_client.filter_bboxes_by_tags([], [], [])
        assert result == set()

    async def test_no_filters_returns_all(self, db_client, two_bboxes):
        bid1, bid2, _, _ = two_bboxes
        result = await db_client.filter_bboxes_by_tags([bid1, bid2], [], [])
        assert result == {bid1, bid2}

    async def test_any_mode_selects_bboxes_with_any_required_tag(self, db_client, two_bboxes):
        """'any' with tag-b: only bbox2 carries it."""
        bid1, bid2, _, tag_b = two_bboxes
        result = await db_client.filter_bboxes_by_tags([bid1, bid2], [tag_b], [], "any")
        assert result == {bid2}

    async def test_any_mode_shared_tag_includes_both(self, db_client, two_bboxes):
        """'any' with tag-a: both bboxes carry it."""
        bid1, bid2, tag_a, _ = two_bboxes
        result = await db_client.filter_bboxes_by_tags([bid1, bid2], [tag_a], [], "any")
        assert result == {bid1, bid2}

    async def test_all_mode_requires_every_tag(self, db_client, two_bboxes):
        """'all' requiring tag-a AND tag-b: only bbox2 has both."""
        bid1, bid2, tag_a, tag_b = two_bboxes
        result = await db_client.filter_bboxes_by_tags([bid1, bid2], [tag_a, tag_b], [], "all")
        assert result == {bid2}

    async def test_all_mode_single_tag_matches_all_carrying_it(self, db_client, two_bboxes):
        """'all' requiring only tag-a: both have it."""
        bid1, bid2, tag_a, _ = two_bboxes
        result = await db_client.filter_bboxes_by_tags([bid1, bid2], [tag_a], [], "all")
        assert result == {bid1, bid2}

    async def test_exclude_removes_bboxes_with_excluded_tag(self, db_client, two_bboxes):
        """Excluding tag-b removes bbox2."""
        bid1, bid2, _, tag_b = two_bboxes
        result = await db_client.filter_bboxes_by_tags([bid1, bid2], [], [tag_b])
        assert result == {bid1}

    async def test_include_any_and_exclude_combined(self, db_client, two_bboxes):
        """'any' tag-a, exclude tag-b → bbox1 only (bbox2 has tag-b)."""
        bid1, bid2, tag_a, tag_b = two_bboxes
        result = await db_client.filter_bboxes_by_tags([bid1, bid2], [tag_a], [tag_b], "any")
        assert result == {bid1}

    async def test_require_missing_tag_returns_empty(self, db_client, two_bboxes):
        """Requiring a tag that no bbox has returns empty set."""
        bid1, bid2, _, _ = two_bboxes
        result = await db_client.filter_bboxes_by_tags(
            [bid1, bid2], [str(uuid_mod.uuid4())], [], "any"
        )
        assert result == set()

    async def test_exclude_all_returns_empty(self, db_client, two_bboxes):
        """Excluding both tags that bboxes carry returns empty set."""
        bid1, bid2, tag_a, tag_b = two_bboxes
        result = await db_client.filter_bboxes_by_tags([bid1, bid2], [], [tag_a, tag_b])
        assert result == set()


# ─────────────────────────────────────────────────────────────────────────────
# Tag and label filtering in find-similar-by-bbox-uuid
# ─────────────────────────────────────────────────────────────────────────────


class TestSimilarBboxesFiltering:
    """API-level tag and label_uuids filtering for bbox similarity search."""

    @pytest.fixture
    async def tagged_index(self, http_client, image_file, label_uuid):
        """
        Two near-identical bboxes: one carries a custom tag, one does not.
        Hash index is rebuilt so similarity search works.
        Returns (bbox_tagged_uuid, bbox_untagged_uuid, tag_uuid).
        """
        tag_r = await http_client.post(
            "/api/tags/add", json={"name": "verified", "color": "#ABCDEF"}
        )
        tag_uuid = tag_r.json()["tag"]["uuid"]
        bid_tagged, bid_bare = str(uuid_mod.uuid4()), str(uuid_mod.uuid4())
        r = await http_client.post(
            "/api/images/metadata/update",
            json={
                "imagePath": "similar_test.png",
                "boxes": [
                    {
                        "uuid": bid_tagged,
                        "labelUuid": label_uuid,
                        "x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5,
                        "tagUuids": [tag_uuid],
                    },
                    {
                        "uuid": bid_bare,
                        "labelUuid": label_uuid,
                        "x": 0.15, "y": 0.15, "width": 0.5, "height": 0.5,
                        "tagUuids": [],
                    },
                ],
            },
        )
        assert r.status_code == 200
        rb = await http_client.post("/api/images/phash/rebuild-index")
        assert rb.json()["status"] == "success"
        return bid_tagged, bid_bare, tag_uuid

    async def test_tag_filter_includes_only_matching_bboxes(self, http_client, tagged_index):
        """tagUuids=[tag] keeps only bboxes that carry that tag."""
        bid_tagged, bid_bare, tag_uuid = tagged_index
        r = await http_client.post(
            "/api/images/bbox/find-similar-by-bbox-uuid",
            json={"bboxUuid": bid_tagged, "threshold": 64, "tagUuids": [tag_uuid]},
        )
        assert r.status_code == 200
        uuids = {m["bboxUuid"] for m in r.json()["matches"]}
        assert bid_tagged in uuids
        assert bid_bare not in uuids

    async def test_exclude_tag_removes_tagged_bboxes(self, http_client, tagged_index):
        """excludeTagUuids=[tag] removes bboxes that carry the tag."""
        bid_tagged, bid_bare, tag_uuid = tagged_index
        r = await http_client.post(
            "/api/images/bbox/find-similar-by-bbox-uuid",
            json={"bboxUuid": bid_bare, "threshold": 64, "excludeTagUuids": [tag_uuid]},
        )
        assert r.status_code == 200
        uuids = {m["bboxUuid"] for m in r.json()["matches"]}
        assert bid_bare in uuids
        assert bid_tagged not in uuids

    async def test_no_tag_filter_returns_all_similar(self, http_client, tagged_index):
        """Without tag filters both bboxes appear in results."""
        bid_tagged, bid_bare, _ = tagged_index
        r = await http_client.post(
            "/api/images/bbox/find-similar-by-bbox-uuid",
            json={"bboxUuid": bid_tagged, "threshold": 64},
        )
        assert r.status_code == 200
        uuids = {m["bboxUuid"] for m in r.json()["matches"]}
        assert bid_tagged in uuids
        assert bid_bare in uuids

    async def test_label_uuids_restricts_to_matching_label(self, http_client, image_file):
        """labelUuids filters to only those label trees in the BK-tree search."""
        # Create two labels, one bbox each, index both.
        la = (await http_client.post("/api/labels", json={"name": "cat", "color": "#111111"})).json()["label"]["uuid"]
        lb = (await http_client.post("/api/labels", json={"name": "bird", "color": "#222222"})).json()["label"]["uuid"]
        bid_a, bid_b = str(uuid_mod.uuid4()), str(uuid_mod.uuid4())
        await http_client.post(
            "/api/images/metadata/update",
            json={
                "imagePath": "similar_test.png",
                "boxes": [
                    {"uuid": bid_a, "labelUuid": la, "x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5, "tagUuids": []},
                    {"uuid": bid_b, "labelUuid": lb, "x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5, "tagUuids": []},
                ],
            },
        )
        await http_client.post("/api/images/phash/rebuild-index")

        r = await http_client.post(
            "/api/images/bbox/find-similar-by-bbox-uuid",
            json={"bboxUuid": bid_a, "threshold": 64, "labelUuids": [la]},
        )
        assert r.status_code == 200
        uuids = {m["bboxUuid"] for m in r.json()["matches"]}
        assert bid_a in uuids
        assert bid_b not in uuids
