import pytest


async def test_server_config_returns_hosts(http_client):
    resp = await http_client.get("/server-config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["config"]["inference_server"] == "http://inference:8001"
    assert data["config"]["tasks_server"] == "http://tasks:8002"
    assert data["config"]["db_server"] == "http://db:8003"


async def test_home_page_returns_html(http_client):
    resp = await http_client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


async def test_proxy_images_route_forwards(http_client):
    resp = await http_client.get("/api/images/list")
    assert resp.status_code == 200


async def test_proxy_labels_route_forwards(http_client):
    resp = await http_client.get("/api/labels/")
    assert resp.status_code == 200


async def test_proxy_models_route_forwards(http_client):
    resp = await http_client.get("/api/models/")
    assert resp.status_code == 200


async def test_proxy_sources_route_forwards(http_client):
    resp = await http_client.get("/api/sources/")
    assert resp.status_code == 200


async def test_proxy_tags_route_forwards(http_client):
    resp = await http_client.get("/api/tags/")
    assert resp.status_code == 200


async def test_proxy_tasks_route_forwards(http_client):
    resp = await http_client.get("/api/tasks/types/list")
    assert resp.status_code == 200


async def test_proxy_called_with_correct_path(http_client, monkeypatch):
    """Verify the proxy handle_request receives the path segment after the route prefix."""
    import apps.image_tagging_server.web as web_module
    from unittest.mock import AsyncMock
    from fastapi.responses import JSONResponse

    captured = {}
    async def capture(path, request):
        captured["path"] = path
        return JSONResponse(content={"status": "success"})

    monkeypatch.setattr(web_module.IMAGE_PROXY, "handle_request", capture)
    await http_client.get("/api/images/some/nested/path")
    assert captured["path"] == "some/nested/path"
