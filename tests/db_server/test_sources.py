"""Tests for source management endpoints — db_server /api/sources/."""

import pytest

# Import providers so they register themselves
import apps.helpers.imageProviders  # noqa: F401
from apps.helpers.imageProviders.Registry import image_provider_registry


@pytest.fixture
def provider_name():
    """Return the name of a valid registered image provider for tests."""
    # UrlImageProvider is the simplest; it's always registered
    assert "UrlImageProvider" in image_provider_registry, "UrlImageProvider must be registered"
    return "UrlImageProvider"


@pytest.fixture
def provider_params():
    return {"url": "http://example.com/image.jpg"}


# ──────────────────────────────────────────────────────────────────────────────
# Image provider discovery
# ──────────────────────────────────────────────────────────────────────────────


async def test_list_image_providers(http_client):
    r = await http_client.get("/api/sources/image-providers/list")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert isinstance(body["imageProviders"], list)
    assert len(body["imageProviders"]) > 0


async def test_get_image_provider_schema(http_client, provider_name):
    r = await http_client.post(
        "/api/sources/image-providers/schema",
        json={"imageProvider": provider_name},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["imageProviderSchema"] is not None
    # UrlImageProvider schema must include a "url" parameter
    assert "url" in body["imageProviderSchema"]


async def test_get_image_provider_schema_unknown(http_client):
    r = await http_client.post(
        "/api/sources/image-providers/schema",
        json={"imageProvider": "NonExistentProvider"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "failure"


# ──────────────────────────────────────────────────────────────────────────────
# Source CRUD
# ──────────────────────────────────────────────────────────────────────────────


async def test_list_sources_empty(http_client):
    r = await http_client.get("/api/sources/get")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["sources"] == {}


async def test_create_source(http_client, provider_name, provider_params):
    r = await http_client.post(
        "/api/sources/create",
        json={
            "sourceName": "my-cam",
            "imageProvider": provider_name,
            "providerParams": provider_params,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    source = body["source"]
    assert source["name"] == "my-cam"
    assert source["typename"] == provider_name
    assert "uuid" in source


async def test_create_source_unknown_provider(http_client):
    r = await http_client.post(
        "/api/sources/create",
        json={
            "sourceName": "bad-cam",
            "imageProvider": "DoesNotExist",
            "providerParams": {},
        },
    )
    assert r.status_code == 200
    assert r.json()["status"] == "failure"


async def test_list_sources_after_create(http_client, provider_name, provider_params):
    await http_client.post(
        "/api/sources/create",
        json={"sourceName": "cam-a", "imageProvider": provider_name, "providerParams": provider_params},
    )
    r = await http_client.get("/api/sources/get")
    assert r.status_code == 200
    sources = r.json()["sources"]
    names = [s["name"] for s in sources.values()]
    assert "cam-a" in names


async def test_delete_source(http_client, provider_name, provider_params):
    create_r = await http_client.post(
        "/api/sources/create",
        json={"sourceName": "cam-delete", "imageProvider": provider_name, "providerParams": provider_params},
    )
    source_uuid = create_r.json()["source"]["uuid"]

    del_r = await http_client.post(
        "/api/sources/delete",
        json={"sourceUuids": [source_uuid]},
    )
    assert del_r.status_code == 200
    assert del_r.json()["status"] == "success"

    # Verify it's gone
    list_r = await http_client.get("/api/sources/get")
    sources = list_r.json()["sources"]
    assert source_uuid not in sources


async def test_update_source(http_client, provider_name, provider_params):
    create_r = await http_client.post(
        "/api/sources/create",
        json={"sourceName": "cam-orig", "imageProvider": provider_name, "providerParams": provider_params},
    )
    source_uuid = create_r.json()["source"]["uuid"]

    update_r = await http_client.post(
        "/api/sources/update",
        json={
            "sourceUuid": source_uuid,
            "sourceName": "cam-renamed",
            "imageProvider": provider_name,
            "providerParams": {"url": "http://other.com/img.jpg"},
        },
    )
    assert update_r.status_code == 200
    body = update_r.json()
    assert body["status"] == "success"
    assert body["source"]["name"] == "cam-renamed"
    assert body["source"]["params"]["url"] == "http://other.com/img.jpg"


async def test_create_duplicate_source_name_fails(http_client, provider_name, provider_params):
    payload = {"sourceName": "dup-cam", "imageProvider": provider_name, "providerParams": provider_params}
    await http_client.post("/api/sources/create", json=payload)
    r = await http_client.post("/api/sources/create", json=payload)
    assert r.json()["status"] == "failure"
