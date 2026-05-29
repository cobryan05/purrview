"""Extended tag tests — update endpoint and exclusive-group behaviour."""
import uuid

import pytest

from apps.helpers.db.types import UNVERIFIED_TAG_UUID
from tests.conftest import make_png


@pytest.fixture
async def custom_tag(http_client):
    """Create a generic tag and return its UUID."""
    r = await http_client.post("/api/tags/add", json={"name": "my-tag", "color": "#aabbcc"})
    assert r.status_code == 200
    return r.json()["tag"]["uuid"]


# ──────────────────────────────────────────────────────────────────────────────
# Tag update
# ──────────────────────────────────────────────────────────────────────────────

async def test_update_tag_name(http_client, custom_tag):
    r = await http_client.post(
        "/api/tags/update",
        json={"tagUuid": custom_tag, "data": {"name": "renamed-tag"}},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    # Verify the name changed
    list_r = await http_client.get("/api/tags/list")
    tags = {t["uuid"]: t for t in list_r.json()["tags"]}
    assert tags[custom_tag]["name"] == "renamed-tag"


async def test_update_tag_color(http_client, custom_tag):
    r = await http_client.post(
        "/api/tags/update",
        json={"tagUuid": custom_tag, "data": {"color": "#ff0000"}},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    list_r = await http_client.get("/api/tags/list")
    tags = {t["uuid"]: t for t in list_r.json()["tags"]}
    assert tags[custom_tag]["color"] == "#ff0000"


async def test_update_tag_no_changes_is_ok(http_client, custom_tag):
    r = await http_client.post(
        "/api/tags/update",
        json={"tagUuid": custom_tag, "data": {}},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"


# ──────────────────────────────────────────────────────────────────────────────
# Exclusive-group enforcement via bbox PATCH
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def image_file(files_root):
    img = files_root / "excl.png"
    img.write_bytes(make_png())
    return img


@pytest.fixture
async def label_uuid(http_client):
    r = await http_client.post("/api/labels", json={"name": "thing", "color": "#123456"})
    return r.json()["label"]["uuid"]


@pytest.fixture
async def bbox_uuid(http_client, image_file, label_uuid):
    """Create an image with one UNVERIFIED bbox; return the bbox uuid."""
    bbox_id = str(uuid.uuid4())
    await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "excl.png",
            "boxes": [{
                "uuid": bbox_id,
                "labelUuid": label_uuid,
                "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2,
                "tagUuids": [UNVERIFIED_TAG_UUID],
            }],
        },
    )
    return bbox_id


async def test_exclusive_group_custom_tag_ousts_unverified(http_client, bbox_uuid):
    """A custom tag in the 'verification' exclusive group replaces UNVERIFIED."""
    r = await http_client.post(
        "/api/tags/add",
        json={"name": "accepted", "exclusive_group": "verification"},
    )
    accepted_uuid = r.json()["tag"]["uuid"]

    patch_r = await http_client.patch(
        f"/api/images/bbox/{bbox_uuid}",
        json={"tagUuids": [accepted_uuid]},
    )
    assert patch_r.status_code == 204

    meta_r = await http_client.post(
        "/api/images/metadata/get", json={"imagePath": "excl.png"}
    )
    bboxes = meta_r.json()["image"]["bboxes"]
    assert len(bboxes) == 1
    tag_uuids = [t["uuid"] for t in bboxes[0]["tags"]]
    assert accepted_uuid in tag_uuids
    assert UNVERIFIED_TAG_UUID not in tag_uuids


async def test_exclusive_group_last_tag_wins(http_client, bbox_uuid):
    """When multiple exclusive-group tags are sent, only the last survives."""
    r1 = await http_client.post(
        "/api/tags/add",
        json={"name": "ver-tag-a", "exclusive_group": "verification"},
    )
    r2 = await http_client.post(
        "/api/tags/add",
        json={"name": "ver-tag-b", "exclusive_group": "verification"},
    )
    tag_a = r1.json()["tag"]["uuid"]
    tag_b = r2.json()["tag"]["uuid"]

    patch_r = await http_client.patch(
        f"/api/images/bbox/{bbox_uuid}",
        json={"tagUuids": [tag_a, tag_b]},
    )
    assert patch_r.status_code == 204

    meta_r = await http_client.post(
        "/api/images/metadata/get", json={"imagePath": "excl.png"}
    )
    bboxes = meta_r.json()["image"]["bboxes"]
    tag_uuids = [t["uuid"] for t in bboxes[0]["tags"]]
    assert tag_b in tag_uuids
    assert tag_a not in tag_uuids
    assert UNVERIFIED_TAG_UUID not in tag_uuids
