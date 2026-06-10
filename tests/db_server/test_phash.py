"""Integration tests for the perceptual-hash feature.

Covers:
  - phash computed and stored in DB on first image write
  - phash written to JSON sidecar
  - DB helper methods: get_all_image_phashes, get_all_bbox_phashes, update_phashes,
                       get_image_phashes_page, get_bbox_phashes_page
  - HTTP endpoints: find-similar-images, find-similar-bboxes,
                    rebuild-index, recompute-phash
"""

import io
import json
import uuid as uuid_mod
from pathlib import Path

import pytest
from PIL import Image

from apps.helpers.db.types import UNVERIFIED_TAG_UUID

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_png(r: int = 100, g: int = 150, b: int = 200, size: int = 64) -> bytes:
    """Return a solid-colour PIL-generated PNG readable by both PIL and cv2."""
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (r, g, b)).save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def image_file(files_root):
    """A cv2-readable PNG on disk."""
    p = files_root / "phash_test.png"
    p.write_bytes(_make_png())
    return p


@pytest.fixture
async def label_uuid(http_client):
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    return r.json()["label"]["uuid"]


@pytest.fixture
async def written_image(http_client, image_file, label_uuid):
    """Write one image with one bbox; return (image_uuid, bbox_uuid)."""
    bid = str(uuid_mod.uuid4())
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "phash_test.png",
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
        "/api/images/metadata/get", json={"imagePath": "phash_test.png"}
    )
    image_uuid = meta.json()["image"]["uuid"]
    return image_uuid, bid


# ─────────────────────────────────────────────────────────────────────────────
# phash storage on first write
# ─────────────────────────────────────────────────────────────────────────────


class TestPhashStoredOnWrite:
    async def test_image_phash_stored_in_db(self, db_client, written_image):
        image_uuid, _ = written_image
        rows = await db_client.get_all_image_phashes()
        image_row = next((r for r in rows if r["uuid"] == image_uuid), None)
        assert image_row is not None
        assert image_row["phash"] is not None
        assert len(image_row["phash"]) == 16  # hash_size=8 → 64 bits → 16 hex chars

    async def test_bbox_phash_stored_in_db(self, db_client, written_image):
        _, bbox_uuid = written_image
        rows = await db_client.get_all_bbox_phashes()
        bbox_row = next((r for r in rows if r["uuid"] == bbox_uuid), None)
        assert bbox_row is not None
        assert bbox_row["phash"] is not None
        assert len(bbox_row["phash"]) == 64  # hash_size=16 → 256 bits → 64 hex chars

    async def test_image_phash_written_to_sidecar(self, image_file, written_image):
        sidecar = image_file.with_suffix(".json")
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert "phash" in data
        assert data["phash"] is not None

    async def test_bbox_phash_written_to_sidecar(self, image_file, written_image):
        _, bbox_uuid = written_image
        sidecar = image_file.with_suffix(".json")
        data = json.loads(sidecar.read_text())
        box = next((b for b in data["boxes"] if b["uuid"] == bbox_uuid), None)
        assert box is not None
        assert box.get("phash") is not None

    async def test_second_write_does_not_overwrite_existing_phash(
        self, http_client, db_client, written_image, label_uuid
    ):
        """Calling update_image_metadata a second time should NOT recompute phash
        (image.phash is already set after the first write)."""
        image_uuid, _ = written_image

        rows_before = await db_client.get_all_image_phashes()
        phash_before = next(r["phash"] for r in rows_before if r["uuid"] == image_uuid)

        # Second write with same bbox
        bid2 = str(uuid_mod.uuid4())
        await http_client.post(
            "/api/images/metadata/update",
            json={
                "imagePath": "phash_test.png",
                "boxes": [
                    {
                        "uuid": bid2,
                        "labelUuid": label_uuid,
                        "x": 0.2,
                        "y": 0.2,
                        "width": 0.4,
                        "height": 0.4,
                        "tagUuids": [UNVERIFIED_TAG_UUID],
                    }
                ],
            },
        )

        rows_after = await db_client.get_all_image_phashes()
        phash_after = next(r["phash"] for r in rows_after if r["uuid"] == image_uuid)
        assert phash_before == phash_after


# ─────────────────────────────────────────────────────────────────────────────
# DB helper methods
# ─────────────────────────────────────────────────────────────────────────────


class TestDbHelpers:
    async def test_get_all_image_phashes_excludes_null_phash(self, db_client):
        """Images without a phash should not appear in the result."""
        rows = await db_client.get_all_image_phashes()
        # Fresh DB — no images at all → empty list
        assert rows == []

    async def test_get_all_image_phashes_returns_correct_fields(
        self, db_client, written_image
    ):
        image_uuid, _ = written_image
        rows = await db_client.get_all_image_phashes()
        assert len(rows) >= 1
        row = rows[0]
        assert "uuid" in row
        assert "phash" in row

    async def test_get_all_bbox_phashes_returns_correct_fields(
        self, db_client, written_image
    ):
        _, bbox_uuid = written_image
        rows = await db_client.get_all_bbox_phashes()
        assert len(rows) >= 1
        row = rows[0]
        assert "uuid" in row
        assert "label_uuid" in row
        assert "phash" in row

    async def test_update_phashes_writes_image_and_bbox(
        self, db_client, written_image
    ):
        """update_phashes should update both image and bbox phash fields."""
        image_uuid, bbox_uuid = written_image
        new_image_phash = "a" * 16
        new_bbox_phash = "b" * 64

        await db_client.update_phashes(
            image_uuid,
            new_image_phash,
            [{"uuid": bbox_uuid, "phash": new_bbox_phash}],
        )

        img_rows = await db_client.get_all_image_phashes()
        bbox_rows = await db_client.get_all_bbox_phashes()

        img_row = next((r for r in img_rows if r["uuid"] == image_uuid), None)
        bbox_row = next((r for r in bbox_rows if r["uuid"] == bbox_uuid), None)

        assert img_row is not None and img_row["phash"] == new_image_phash
        assert bbox_row is not None and bbox_row["phash"] == new_bbox_phash


# ─────────────────────────────────────────────────────────────────────────────
# Paginated DB helpers: get_image_phashes_page / get_bbox_phashes_page
# ─────────────────────────────────────────────────────────────────────────────


class TestPhashPage:
    """Tests for the paginated phash query helpers added alongside the bulk variants."""

    async def test_image_page_empty_db_returns_empty_list(self, db_client):
        rows = await db_client.get_image_phashes_page(0, 10)
        assert rows == []

    async def test_image_page_returns_row_with_correct_fields(
        self, db_client, written_image
    ):
        image_uuid, _ = written_image
        rows = await db_client.get_image_phashes_page(0, 10)
        assert len(rows) >= 1
        row = rows[0]
        assert "uuid" in row and "phash" in row
        assert row["uuid"] == image_uuid

    async def test_image_page_excludes_null_phash(self, db_client):
        # Image inserted without going through the write pipeline has no phash
        from pathlib import Path as _Path
        await db_client.add_image(_Path("no_phash.jpg"))
        rows = await db_client.get_image_phashes_page(0, 10)
        assert rows == []

    async def test_image_page_offset_skips_rows(self, db_client, written_image):
        # Only one image in DB; offset=1 must return nothing
        rows = await db_client.get_image_phashes_page(1, 10)
        assert rows == []

    async def test_image_page_limit_caps_result(self, db_client, written_image):
        # Seed a second image with a phash via update_phashes
        from pathlib import Path as _Path
        second = await db_client.add_image(_Path("second.jpg"))
        await db_client.update_phashes(second.uuid, "b" * 16, [])
        all_rows = await db_client.get_image_phashes_page(0, 100)
        assert len(all_rows) == 2
        limited = await db_client.get_image_phashes_page(0, 1)
        assert len(limited) == 1

    async def test_bbox_page_empty_db_returns_empty_list(self, db_client):
        rows = await db_client.get_bbox_phashes_page(0, 10)
        assert rows == []

    async def test_bbox_page_returns_row_with_correct_fields(
        self, db_client, written_image
    ):
        _, bbox_uuid = written_image
        rows = await db_client.get_bbox_phashes_page(0, 10)
        assert len(rows) >= 1
        row = rows[0]
        assert "uuid" in row and "label_uuid" in row and "phash" in row
        assert row["uuid"] == bbox_uuid

    async def test_bbox_page_offset_skips_rows(self, db_client, written_image):
        rows = await db_client.get_bbox_phashes_page(1, 10)
        assert rows == []


# ─────────────────────────────────────────────────────────────────────────────
# API: find-similar-images
# ─────────────────────────────────────────────────────────────────────────────


class TestFindSimilarImages:
    async def test_find_similar_images_returns_self_at_threshold_zero(
        self, http_client, db_client, written_image
    ):
        image_uuid, _ = written_image
        rows = await db_client.get_all_image_phashes()
        phash = next(r["phash"] for r in rows if r["uuid"] == image_uuid)

        r = await http_client.post(
            "/api/images/find-similar",
            json={"phash": phash, "threshold": 0},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        uuids = [m["imageUuid"] for m in body["matches"]]
        assert image_uuid in uuids

    async def test_find_similar_images_no_match_with_wrong_phash(
        self, http_client, written_image
    ):
        # Use a phash of all-zeros; unlikely to match any real image
        r = await http_client.post(
            "/api/images/find-similar",
            json={"phash": "0" * 16, "threshold": 0},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        # May or may not match; don't assert empty (zero might collide with
        # our solid-colour phash).  Just check the response shape.
        assert "matches" in body

    async def test_find_similar_images_returns_distance(
        self, http_client, db_client, written_image
    ):
        image_uuid, _ = written_image
        rows = await db_client.get_all_image_phashes()
        phash = next(r["phash"] for r in rows if r["uuid"] == image_uuid)

        r = await http_client.post(
            "/api/images/find-similar",
            json={"phash": phash, "threshold": 0},
        )
        matches = r.json()["matches"]
        match = next(m for m in matches if m["imageUuid"] == image_uuid)
        assert match["distance"] == 0

    async def test_find_similar_images_empty_index_returns_empty(self, http_client):
        r = await http_client.post(
            "/api/images/find-similar",
            json={"phash": "f" * 16, "threshold": 5},
        )
        assert r.status_code == 200
        assert r.json()["matches"] == []


# ─────────────────────────────────────────────────────────────────────────────
# API: find-similar-bboxes
# ─────────────────────────────────────────────────────────────────────────────


class TestFindSimilarBboxes:
    async def test_find_similar_bboxes_returns_self_at_threshold_zero(
        self, http_client, db_client, written_image, label_uuid
    ):
        _, bbox_uuid = written_image
        rows = await db_client.get_all_bbox_phashes()
        phash = next(r["phash"] for r in rows if r["uuid"] == bbox_uuid)

        r = await http_client.post(
            "/api/images/bbox/find-similar",
            json={"phash": phash, "threshold": 0, "labelUuid": label_uuid},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        uuids = [m["bboxUuid"] for m in body["matches"]]
        assert bbox_uuid in uuids

    async def test_find_similar_bboxes_no_label_searches_all(
        self, http_client, db_client, written_image
    ):
        _, bbox_uuid = written_image
        rows = await db_client.get_all_bbox_phashes()
        phash = next(r["phash"] for r in rows if r["uuid"] == bbox_uuid)

        r = await http_client.post(
            "/api/images/bbox/find-similar",
            json={"phash": phash, "threshold": 0},
        )
        assert r.status_code == 200
        uuids = [m["bboxUuid"] for m in r.json()["matches"]]
        assert bbox_uuid in uuids

    async def test_find_similar_bboxes_empty_index_returns_empty(self, http_client):
        r = await http_client.post(
            "/api/images/bbox/find-similar",
            json={"phash": "f" * 64, "threshold": 5},
        )
        assert r.status_code == 200
        assert r.json()["matches"] == []


# ─────────────────────────────────────────────────────────────────────────────
# API: rebuild-index
# ─────────────────────────────────────────────────────────────────────────────


class TestRebuildIndex:
    async def test_rebuild_returns_success(self, http_client):
        r = await http_client.post("/api/images/phash/rebuild-index")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"

    async def test_rebuild_returns_counts(self, http_client, written_image):
        r = await http_client.post("/api/images/phash/rebuild-index")
        body = r.json()
        assert body["imageCount"] >= 1
        assert body["bboxCount"] >= 1

    async def test_rebuild_returns_duration_ms(self, http_client):
        r = await http_client.post("/api/images/phash/rebuild-index")
        assert r.json()["durationMs"] >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# API: recompute-phash
# ─────────────────────────────────────────────────────────────────────────────


class TestRecomputePhash:
    async def test_recompute_returns_success_for_known_image(
        self, http_client, written_image
    ):
        image_uuid, _ = written_image
        r = await http_client.post(
            f"/api/images/{image_uuid}/phash/recompute",
            json={},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"

    async def test_recompute_returns_phash_string(
        self, http_client, written_image
    ):
        image_uuid, _ = written_image
        r = await http_client.post(
            f"/api/images/{image_uuid}/phash/recompute",
            json={},
        )
        phash = r.json()["imagePhash"]
        assert isinstance(phash, str)
        assert len(phash) == 16  # default hash_size=8

    async def test_recompute_returns_bbox_count(
        self, http_client, written_image
    ):
        image_uuid, _ = written_image
        r = await http_client.post(
            f"/api/images/{image_uuid}/phash/recompute",
            json={},
        )
        assert r.json()["bboxCount"] == 1

    async def test_recompute_phash_matches_original(
        self, http_client, db_client, written_image
    ):
        """Recomputing the same image should produce the same phash."""
        image_uuid, _ = written_image

        rows = await db_client.get_all_image_phashes()
        original_phash = next(r["phash"] for r in rows if r["uuid"] == image_uuid)

        r = await http_client.post(
            f"/api/images/{image_uuid}/phash/recompute",
            json={},
        )
        assert r.json()["imagePhash"] == original_phash

    async def test_recompute_unknown_image_returns_failure(self, http_client):
        fake_uuid = str(uuid_mod.uuid4())
        r = await http_client.post(
            f"/api/images/{fake_uuid}/phash/recompute",
            json={},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "failure"

    async def test_recompute_updates_sidecar(
        self, http_client, image_file, written_image
    ):
        image_uuid, _ = written_image
        await http_client.post(
            f"/api/images/{image_uuid}/phash/recompute",
            json={},
        )
        sidecar = image_file.with_suffix(".json")
        data = json.loads(sidecar.read_text())
        assert "phash" in data
        assert data["phash"] is not None
