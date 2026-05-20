import pytest

from tests.conftest import make_png


@pytest.fixture
def image_file(files_root):
    """A real PNG file on disk that the manager can open."""
    img = files_root / "test.png"
    img.write_bytes(make_png())
    return img


@pytest.fixture
async def label_uuid(http_client):
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    return r.json()["label"]["uuid"]


async def test_list_images_empty_dir(http_client):
    r = await http_client.get("/api/images/list")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["files"] == []


async def test_list_images_finds_file(http_client, image_file):
    r = await http_client.get("/api/images/list", params={"pattern": "*.png"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    names = [f["name"] for f in body["files"]]
    assert "test.png" in names


async def test_get_image_metadata_not_found(http_client):
    r = await http_client.post("/api/images/metadata/get", json={"image_path": "nonexistent.png"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failure"


async def test_get_image_metadata(http_client, image_file):
    r = await http_client.post("/api/images/metadata/get", json={"image_path": "test.png"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success" or body["image"] is not None


async def test_update_and_retrieve_bboxes(http_client, image_file, label_uuid):
    bbox = {
        "uuid": "bbbbbbbb-0000-0000-0000-000000000001",
        "label_uuid": label_uuid,
        "x": 0.1,
        "y": 0.2,
        "width": 0.3,
        "height": 0.4,
        "tag_uuids": [],
    }
    update_r = await http_client.post(
        "/api/images/metadata/update",
        json={"image_path": "test.png", "boxes": [bbox]},
    )
    assert update_r.status_code == 200
    assert update_r.json()["status"] == "success" or update_r.json()["image"] is not None

    meta_r = await http_client.post("/api/images/metadata/get", json={"image_path": "test.png"})
    assert meta_r.status_code == 200
    image_data = meta_r.json()["image"]
    assert image_data is not None
    assert len(image_data["bboxes"]) == 1
    assert abs(image_data["bboxes"][0]["x"] - 0.1) < 1e-6


async def test_get_image_file(http_client, image_file):
    r = await http_client.get("/api/images/test.png/file")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["filename"] == "test.png"
    assert body["imageBase64"] is not None


async def test_get_image_file_not_found(http_client):
    r = await http_client.get("/api/images/missing.png/file")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failure"
