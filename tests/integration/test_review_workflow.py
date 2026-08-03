"""Integration test: end-to-end review workflow through db_server.

This test exercises the full loop that the UI performs:
  1. Create label and tags (system tags already seeded by init_db)
  2. Add image + bboxes tagged UNVERIFIED
  3. Query review queue → image appears
  4. PATCH bbox to remove UNVERIFIED (confirm)
  5. Query review queue (UNVERIFIED filter) → image gone
  6. Check image_labels → label shows PRESENT

Nothing is mocked.  One real SQLite DB, one real Manager, one ASGI client.
"""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from apps.db_server.manager import Manager
from apps.db_server.web import WebApp
from apps.helpers.db.db_client import DbClient
from apps.helpers.db.types import (
    UNVERIFIED_TAG_UUID,
    VERIFIED_TAG_UUID,
    ImageItemStatus,
)
from tests.conftest import make_png
from tests.helpers.legacy_filter_params import legacy_params_to_body

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
async def db_client(tmp_path):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    client = DbClient(db_dir=db_dir, memory=True)
    await client.init_db()
    yield client
    await client.close()


@pytest.fixture
def files_root(tmp_path):
    d = tmp_path / "files"
    d.mkdir()
    return d


@pytest.fixture
async def client(db_client, files_root):
    manager = Manager(db_client=db_client, files_root=files_root)
    manager._db_ready = True  # db_client.init_db() was already called directly above
    web_app = WebApp(app_name="integration-test", manager=manager)
    transport = ASGITransport(app=web_app.app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


async def _create_label(client, name="cat", color="#FF0000"):
    r = await client.post("/api/labels", json={"name": name, "color": color})
    assert r.status_code == 200
    return r.json()["label"]["uuid"]


async def _put_image(files_root, client, filename="img.png", label_uuid=None, tag_uuids=None):
    """Write a PNG to files_root and register it with one bbox."""
    tag_uuids = [UNVERIFIED_TAG_UUID] if tag_uuids is None else tag_uuids
    p = files_root / filename
    p.write_bytes(make_png())
    bid = str(uuid.uuid4())
    r = await client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": filename,
            "boxes": [
                {
                    "uuid": bid,
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": tag_uuids,
                }
            ],
        },
    )
    assert r.status_code == 200
    return bid


async def _review(client, tag_uuids=None, show_all=False, **extra):
    # ImageSearchParams accepts both snake_case (field names) and camelCase (aliases).
    # Legacy flat-filter kwargs are translated to `terms` — see
    # legacy_filter_params.py's docstring for why.
    params = {"show_all_bboxes": show_all}
    if tag_uuids is not None:
        params["tag_uuids"] = tag_uuids
    params.update(extra)
    r = await client.post("/api/images/review", json=legacy_params_to_body(params))
    assert r.status_code == 200
    return r.json()["images"]


async def _image_labels(client, path):
    r = await client.get(f"/api/images/{path}/labels")
    assert r.status_code == 200
    return {row["labelUuid"]: row["status"] for row in r.json()["labels"]}


# ──────────────────────────────────────────────────────────────────────────────
# The full review workflow
# ──────────────────────────────────────────────────────────────────────────────


async def test_full_verify_workflow(client, files_root):
    """Core review loop: unverified → appears in queue → verify → gone from queue."""
    label = await _create_label(client)
    bid = await _put_image(files_root, client, "review.png", label, [UNVERIFIED_TAG_UUID])

    # Image appears in UNVERIFIED queue
    images = await _review(client, tag_uuids=[UNVERIFIED_TAG_UUID])
    filenames = [img["filename"] for img in images]
    assert any("review.png" in f for f in filenames)

    r = await client.patch(
        f"/api/images/bbox/{bid}",
        json={"labelUuid": label, "tagUuids": [VERIFIED_TAG_UUID]},
    )
    assert r.status_code == 204

    # No longer in UNVERIFIED queue
    images = await _review(client, tag_uuids=[UNVERIFIED_TAG_UUID])
    filenames = [img["filename"] for img in images]
    assert not any("review.png" in f for f in filenames)

    # Image label now PRESENT
    labels = await _image_labels(client, "review.png")
    assert labels[label] == ImageItemStatus.PRESENT


async def test_confirm_workflow_creates_present_label(client, files_root):
    """Confirming a bbox (removing UNVERIFIED) removes it from the UNVERIFIED queue
    and produces a PRESENT image_label row."""
    label = await _create_label(client, "dog")
    bid = await _put_image(files_root, client, "confirm.png", label, [UNVERIFIED_TAG_UUID])

    # In queue initially
    images = await _review(client, tag_uuids=[UNVERIFIED_TAG_UUID])
    assert any("confirm.png" in img["filename"] for img in images)

    r = await client.patch(
        f"/api/images/bbox/{bid}",
        json={"labelUuid": label, "tagUuids": [VERIFIED_TAG_UUID]},
    )
    assert r.status_code == 204

    images = await _review(client, tag_uuids=[UNVERIFIED_TAG_UUID])
    assert not any("confirm.png" in img["filename"] for img in images)

    labels = await _image_labels(client, "confirm.png")
    assert labels[label] == ImageItemStatus.PRESENT


async def test_delete_bbox_removes_from_queue(client, files_root):
    """Deleting the only bbox removes the image from the UNVERIFIED queue."""
    label = await _create_label(client, "bird")
    bid = await _put_image(files_root, client, "del.png", label, [UNVERIFIED_TAG_UUID])

    images = await _review(client, tag_uuids=[UNVERIFIED_TAG_UUID])
    assert any("del.png" in img["filename"] for img in images)

    # Delete bbox
    r = await client.delete(f"/api/images/bbox/{bid}")
    assert r.status_code == 204

    # No longer in queue
    images = await _review(client, tag_uuids=[UNVERIFIED_TAG_UUID])
    assert not any("del.png" in img["filename"] for img in images)


async def test_multiple_images_independent_verification(client, files_root):
    """Verifying one image doesn't affect the others in the queue."""
    label = await _create_label(client, "thing")
    bid_a = await _put_image(files_root, client, "a.png", label, [UNVERIFIED_TAG_UUID])
    bid_b = await _put_image(files_root, client, "b.png", label, [UNVERIFIED_TAG_UUID])
    await _put_image(files_root, client, "c.png", label, [UNVERIFIED_TAG_UUID])

    # All three in queue
    images = await _review(client, tag_uuids=[UNVERIFIED_TAG_UUID])
    filenames = {img["filename"] for img in images}
    assert sum(1 for f in filenames if "a.png" in f or "b.png" in f or "c.png" in f) == 3

    for bid in [bid_a, bid_b]:
        await client.patch(
            f"/api/images/bbox/{bid}",
            json={"labelUuid": label, "tagUuids": [VERIFIED_TAG_UUID]},
        )

    images = await _review(client, tag_uuids=[UNVERIFIED_TAG_UUID])
    filenames = [img["filename"] for img in images]
    assert not any("a.png" in f for f in filenames)
    assert not any("b.png" in f for f in filenames)
    assert any("c.png" in f for f in filenames)


async def test_review_queue_label_filter(client, files_root):
    """label_uuid filter returns only images with that label."""
    cat_label = await _create_label(client, "cat2")
    dog_label = await _create_label(client, "dog2")
    await _put_image(files_root, client, "cat.png", cat_label, [UNVERIFIED_TAG_UUID])
    await _put_image(files_root, client, "dog.png", dog_label, [UNVERIFIED_TAG_UUID])

    images = await _review(client, tag_uuids=[UNVERIFIED_TAG_UUID], label_uuid=cat_label)
    filenames = [img["filename"] for img in images]
    assert any("cat.png" in f for f in filenames)
    assert not any("dog.png" in f for f in filenames)


async def test_image_label_filter_excludes_confirmed(client, files_root):
    """image_label_filters with exclude=True keeps only non-confirmed images."""
    label = await _create_label(client, "fish")
    bid_unv = await _put_image(files_root, client, "unv.png", label, [UNVERIFIED_TAG_UUID])
    bid_ver = await _put_image(files_root, client, "ver.png", label, [VERIFIED_TAG_UUID])

    images = await _review(
        client,
        show_all=True,
        image_label_filters=[{"label_uuid": label, "status": "present", "exclude": True}],
    )
    filenames = [img["filename"] for img in images]
    assert any("unv.png" in f for f in filenames)
    assert not any("ver.png" in f for f in filenames)
