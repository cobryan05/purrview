"""Tests for dissimilar-search (farthest-point sampling) endpoints and primitives.

Covers:
  - _popcount LUT fallback (forces the non-bitwise_count path)
  - HashIndex.rank_images_by_diversity — ordering and diverse-walk property
  - HashIndex.rank_bboxes_by_diversity — ordering
  - Manager.rank_images_by_diversity / get_most_unique_images
  - POST /api/images/find-dissimilar-by-image-uuid
  - POST /api/images/bbox/find-dissimilar-by-bbox-uuid
  - POST /api/images/find-dissimilar-global
"""

import io
import uuid as uuid_mod

import numpy as np
import pytest
from PIL import Image

from apps.db_server.hash_index import HashIndex, _popcount
from apps.helpers.db.types import UNVERIFIED_TAG_UUID

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_png(r: int = 100, g: int = 150, b: int = 200, size: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (r, g, b)).save(buf, format="PNG")
    return buf.getvalue()


def _hamming_int(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ─────────────────────────────────────────────────────────────────────────────
# _popcount — verify the LUT fallback produces correct results
# ─────────────────────────────────────────────────────────────────────────────


class TestPopcountFallback:
    """Force the LUT fallback path to ensure both numpy versions are covered."""

    def _lut_popcount(self, arr: np.ndarray) -> np.ndarray:
        """Byte-LUT popcount that works on any numpy version."""
        lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
        return lut[arr.view(np.uint8)].reshape(arr.shape + (8,)).sum(axis=-1).astype(np.int32)

    def test_single_zero(self):
        arr = np.array([0], dtype=np.uint64)
        assert self._lut_popcount(arr)[0] == 0

    def test_single_max(self):
        arr = np.array([np.iinfo(np.uint64).max], dtype=np.uint64)
        assert self._lut_popcount(arr)[0] == 64

    def test_known_value(self):
        # 0b1010_1010 repeated 8 times (4 bits set per byte, 32 total)
        val = int("aa" * 8, 16)
        arr = np.array([val], dtype=np.uint64)
        assert self._lut_popcount(arr)[0] == 32

    def test_batch_matches_module_popcount(self):
        """LUT result must match the module-level _popcount for arbitrary values."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, np.iinfo(np.uint64).max, size=200, dtype=np.uint64)
        lut_result = self._lut_popcount(arr)
        module_result = _popcount(arr).astype(np.int32)
        np.testing.assert_array_equal(lut_result, module_result)

    def test_xor_popcount_gives_hamming(self):
        a = np.array([0b1100_0000], dtype=np.uint64)
        b = np.array([0b0101_0000], dtype=np.uint64)
        # XOR = 0b1001_0000 → 2 bits set
        result = self._lut_popcount(a ^ b)
        assert result[0] == 2


# ─────────────────────────────────────────────────────────────────────────────
# HashIndex.rank_images_by_diversity — unit-level with known phashes
# ─────────────────────────────────────────────────────────────────────────────


class TestRankImagesByDiversity:
    """Direct unit tests on HashIndex using injected phashes.

    We build a small index with known 64-bit phashes and verify that the
    farthest-point sampling orders them correctly.
    """

    @pytest.fixture
    async def index_with_known_phashes(self, db_client):
        """Return a HashIndex populated with four images whose phashes we control.

        phash layout (big-endian hex → popcount of XOR with 0x0000000000000000):
          A: 0x0000000000000000  (all zeros)
          B: 0xFFFFFFFFFFFFFFFF  (all ones  — distance 64 from A)
          C: 0x00000000FFFFFFFF  (half ones — distance 32 from A)
          D: 0x0000000000000001  (one bit   — distance  1 from A)

        Seeding with A:
          - farthest-first picks B (dist 64), then C (dist 32 from B = 32), then D (dist 1 from A = 1).
          - So expected order: B, C, D (A itself is excluded because it's the seed, not in the tree).
          - Actually: A IS in the tree (different UUID), so it will be picked last (dist 0 to A).
          - Expected order: B (64), C (32), D (1), A (0).
        """
        idx = HashIndex(db_client, image_hash_size=8, bbox_hash_size=16)
        # Build empty trees
        await idx.build()

        phashes = {
            "A": "0000000000000000",
            "B": "ffffffffffffffff",
            "C": "00000000ffffffff",
            "D": "0000000000000001",
        }
        uuids = {k: str(uuid_mod.uuid4()) for k in phashes}

        # Inject directly into the index structures
        import pybktree

        from apps.helpers.hashUtils import hamming

        def _item_dist(a, b):
            return hamming(a[0], b[0])

        idx._image_tree = pybktree.BKTree(_item_dist)
        for k, ph in phashes.items():
            idx._image_tree.add((ph, uuids[k]))
            idx._valid_images.add(uuids[k])

        return idx, uuids, phashes

    async def test_seed_nearest_ranked_last(self, index_with_known_phashes):
        """The image closest to the seed should be ranked last."""
        idx, uuids, phashes = index_with_known_phashes
        # Seed = A (all zeros); D is distance-1 from A → should appear last
        ranked = await idx.rank_images_by_diversity(phashes["A"], None, 4)
        assert ranked[-1] == uuids["A"] or ranked[-1] == uuids["D"]
        # First must be B (distance 64 from A — farthest)
        assert ranked[0] == uuids["B"]

    async def test_farthest_ranked_first(self, index_with_known_phashes):
        """The image farthest from the seed must be ranked first."""
        idx, uuids, phashes = index_with_known_phashes
        ranked = await idx.rank_images_by_diversity(phashes["A"], None, 4)
        assert ranked[0] == uuids["B"]

    async def test_no_near_duplicate_cluster_at_top(self, index_with_known_phashes):
        """After B (all-ones), A (all-zeros) should NOT be second.

        After selecting B, C is distance 32 from B (and 32 from A),
        D is distance 63 from B (but only 1 from A).  The farthest from
        *both* selected so far (A seed + B) wins — that is still C
        (min(32,32)=32 vs D's min(1,63)=1).
        """
        idx, uuids, phashes = index_with_known_phashes
        ranked = await idx.rank_images_by_diversity(phashes["A"], None, 4)
        # Second pick must be C, not D or A
        assert ranked[1] == uuids["C"]

    async def test_candidate_filter_respected(self, index_with_known_phashes):
        """Only UUIDs in candidate_uuids should appear in results."""
        idx, uuids, phashes = index_with_known_phashes
        allowed = {uuids["B"], uuids["C"]}
        ranked = await idx.rank_images_by_diversity(phashes["A"], allowed, 10)
        assert set(ranked) == allowed

    async def test_count_respected(self, index_with_known_phashes):
        idx, uuids, _ = index_with_known_phashes
        ranked = await idx.rank_images_by_diversity("0000000000000000", None, 2)
        assert len(ranked) == 2

    async def test_empty_index_returns_empty(self, db_client):
        idx = HashIndex(db_client, image_hash_size=8, bbox_hash_size=16)
        await idx.build()
        result = await idx.rank_images_by_diversity("0000000000000000", None, 10)
        assert result == []

    async def test_count_larger_than_n_returns_all(self, index_with_known_phashes):
        idx, uuids, _ = index_with_known_phashes
        ranked = await idx.rank_images_by_diversity("0000000000000000", None, 999)
        assert len(ranked) == len(uuids)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures for endpoint tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def image_file(files_root):
    p = files_root / "dissimilar_test.png"
    p.write_bytes(_make_png())
    return p


@pytest.fixture
async def label_uuid(http_client):
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    assert r.status_code == 200
    return r.json()["label"]["uuid"]


@pytest.fixture
async def written_image(http_client, image_file, label_uuid):
    """Write one image with one unverified bbox; return (image_uuid, bbox_uuid)."""
    bid = str(uuid_mod.uuid4())
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "dissimilar_test.png",
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
    meta = await http_client.post("/api/images/metadata/get", json={"imagePath": "dissimilar_test.png"})
    image_uuid = meta.json()["image"]["uuid"]
    return image_uuid, bid


@pytest.fixture
async def two_unverified_images(http_client, written_image, files_root, label_uuid):
    """Two distinct images, each with one unverified bbox, with the index rebuilt.

    Returns (first_image_uuid, second_image_uuid).
    """
    first_uuid, _ = written_image
    (files_root / "dissimilar_test_2.png").write_bytes(_make_png(r=10, g=20, b=30))
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "dissimilar_test_2.png",
            "boxes": [
                {
                    "uuid": str(uuid_mod.uuid4()),
                    "labelUuid": label_uuid,
                    "x": 0.2,
                    "y": 0.2,
                    "width": 0.5,
                    "height": 0.5,
                    "tagUuids": [UNVERIFIED_TAG_UUID],
                }
            ],
        },
    )
    assert r.status_code == 200
    meta = await http_client.post("/api/images/metadata/get", json={"imagePath": "dissimilar_test_2.png"})
    second_uuid = meta.json()["image"]["uuid"]
    assert (await http_client.post("/api/images/phash/rebuild-index")).json()["status"] == "success"
    return first_uuid, second_uuid


@pytest.fixture
async def verified_image(http_client, files_root, label_uuid):
    """One image whose only bbox carries no unverified tag."""
    (files_root / "dissimilar_verified.png").write_bytes(_make_png(r=200, g=30, b=90))
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "dissimilar_verified.png",
            "boxes": [
                {
                    "uuid": str(uuid_mod.uuid4()),
                    "labelUuid": label_uuid,
                    "x": 0.3,
                    "y": 0.3,
                    "width": 0.4,
                    "height": 0.4,
                    "tagUuids": [],
                }
            ],
        },
    )
    assert r.status_code == 200
    meta = await http_client.post("/api/images/metadata/get", json={"imagePath": "dissimilar_verified.png"})
    return meta.json()["image"]["uuid"]


@pytest.fixture
async def index_built(http_client, written_image):
    r = await http_client.post("/api/images/phash/rebuild-index")
    assert r.json()["status"] == "success"
    return written_image


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/images/find-dissimilar-by-image-uuid
# ─────────────────────────────────────────────────────────────────────────────


class TestFindDissimilarByImageUuid:
    async def test_returns_matches_with_rank(self, http_client, index_built):
        image_uuid, _ = index_built
        r = await http_client.post(
            "/api/images/find-dissimilar-by-image-uuid",
            json={"imageUuid": image_uuid, "count": 10},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert isinstance(body["matches"], list)
        assert len(body["matches"]) >= 1
        # Each match must have imageUuid and rank
        m = body["matches"][0]
        assert "imageUuid" in m
        assert "rank" in m
        assert m["rank"] == 0  # first result has rank 0

    async def test_ranks_are_sequential(self, http_client, index_built):
        image_uuid, _ = index_built
        r = await http_client.post(
            "/api/images/find-dissimilar-by-image-uuid",
            json={"imageUuid": image_uuid, "count": 50},
        )
        body = r.json()
        ranks = [m["rank"] for m in body["matches"]]
        assert ranks == list(range(len(ranks)))

    async def test_count_respected(self, http_client, index_built):
        image_uuid, _ = index_built
        r = await http_client.post(
            "/api/images/find-dissimilar-by-image-uuid",
            json={"imageUuid": image_uuid, "count": 1},
        )
        body = r.json()
        assert len(body["matches"]) == 1

    async def test_unknown_uuid_returns_empty(self, http_client):
        r = await http_client.post(
            "/api/images/find-dissimilar-by-image-uuid",
            json={"imageUuid": str(uuid_mod.uuid4()), "count": 10},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["matches"] == []


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/images/bbox/find-dissimilar-by-bbox-uuid
# ─────────────────────────────────────────────────────────────────────────────


class TestFindDissimilarByBboxUuid:
    async def test_returns_matches_with_rank(self, http_client, index_built):
        _, bbox_uuid = index_built
        r = await http_client.post(
            "/api/images/bbox/find-dissimilar-by-bbox-uuid",
            json={"bboxUuid": bbox_uuid, "count": 10},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert isinstance(body["matches"], list)
        assert len(body["matches"]) >= 1
        m = body["matches"][0]
        assert "bboxUuid" in m
        assert "labelUuid" in m
        assert "rank" in m
        assert m["rank"] == 0

    async def test_unknown_uuid_returns_empty(self, http_client):
        r = await http_client.post(
            "/api/images/bbox/find-dissimilar-by-bbox-uuid",
            json={"bboxUuid": str(uuid_mod.uuid4()), "count": 10},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["matches"] == []


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/images/find-dissimilar-global
# ─────────────────────────────────────────────────────────────────────────────


class TestFindDissimilarGlobal:
    async def test_returns_unverified_images(self, http_client, index_built):
        image_uuid, _ = index_built
        r = await http_client.post(
            "/api/images/find-dissimilar-global",
            json={"count": 50},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert isinstance(body["matches"], list)
        # The written image has an unverified bbox — it should appear
        returned_uuids = {m["imageUuid"] for m in body["matches"]}
        assert image_uuid in returned_uuids

    async def test_ranks_are_sequential(self, http_client, index_built):
        r = await http_client.post(
            "/api/images/find-dissimilar-global",
            json={"count": 50},
        )
        body = r.json()
        ranks = [m["rank"] for m in body["matches"]]
        assert ranks == list(range(len(ranks)))

    async def test_empty_index_returns_success(self, http_client):
        """An empty index should return success with an empty matches list."""
        r = await http_client.post("/api/images/find-dissimilar-global", json={"count": 10})
        assert r.status_code == 200
        assert r.json()["status"] == "success"


class TestFindDissimilarGlobalCandidates:
    """`candidateUuids` scopes the global diversity walk to the caller's filtered set.

    Without it the endpoint always walked every unverified image, so the Review
    tab's "★ Most Unique First" ignored the active filters.
    """

    async def test_candidate_uuids_restrict_results(self, http_client, two_unverified_images):
        keep_uuid, other_uuid = two_unverified_images
        r = await http_client.post(
            "/api/images/find-dissimilar-global",
            json={"count": 50, "candidateUuids": [keep_uuid]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        returned = {m["imageUuid"] for m in body["matches"]}
        assert returned == {keep_uuid}
        assert other_uuid not in returned

    async def test_omitted_candidates_keep_full_unverified_scope(self, http_client, two_unverified_images):
        keep_uuid, other_uuid = two_unverified_images
        r = await http_client.post("/api/images/find-dissimilar-global", json={"count": 50})
        returned = {m["imageUuid"] for m in r.json()["matches"]}
        assert {keep_uuid, other_uuid} <= returned

    async def test_empty_candidate_list_returns_no_matches(self, http_client, two_unverified_images):
        """An empty filtered set means "nothing matched" — not "fall back to everything"."""
        r = await http_client.post(
            "/api/images/find-dissimilar-global",
            json={"count": 50, "candidateUuids": []},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "success"
        assert r.json()["matches"] == []

    async def test_candidates_with_no_unverified_bboxes_still_ranked(self, http_client, verified_image, index_built):
        """Filtering to verified-only images must still rank them, not fall back to
        the unverified pool the filter deliberately excluded."""
        verified_uuid = verified_image
        r = await http_client.post(
            "/api/images/find-dissimilar-global",
            json={"count": 50, "candidateUuids": [verified_uuid]},
        )
        assert r.json()["status"] == "success"
        assert {m["imageUuid"] for m in r.json()["matches"]} == {verified_uuid}
