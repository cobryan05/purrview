import pytest


async def test_server_config_returns_hosts(http_client):
    resp = await http_client.get("/server-config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["config"]["inference_server"] == "http://inference:8001"
    assert data["config"]["tasks_server"] == "http://tasks:8002"
    assert data["config"]["db_server"] == "http://db:8003"
    assert data["config"]["training_server"] == "http://training:8005"


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
    from unittest.mock import AsyncMock

    from fastapi.responses import JSONResponse

    import apps.image_tagging_server.web as web_module

    captured = {}

    async def capture(path, request):
        captured["path"] = path
        return JSONResponse(content={"status": "success"})

    monkeypatch.setattr(web_module.IMAGE_PROXY, "handle_request", capture)
    await http_client.get("/api/images/some/nested/path")
    assert captured["path"] == "some/nested/path"


class TestStaticAssetRevalidation:
    """The SPA is unbundled, unversioned ES modules fetched by bare URL.

    Without an explicit cache-control the browser applies heuristic freshness
    per file, so one page load can mix modules from two revisions of the tree —
    and an import of an export that the cached copy predates is a module-level
    SyntaxError that takes the whole app down, not just the feature that owns
    the module.
    """

    async def test_app_static_asks_the_browser_to_revalidate(self, http_client):
        r = await http_client.get("/app-static/js/ui/tasks.js")

        assert r.status_code == 200
        assert r.headers.get("cache-control") == "no-cache"

    async def test_the_etag_survives_so_revalidation_stays_cheap(self, http_client):
        """no-cache must not mean re-download: the 304 is what makes it free."""
        r = await http_client.get("/app-static/js/ui/tasks.js")

        assert r.headers.get("etag"), "no etag to revalidate against"

        cached = await http_client.get(
            "/app-static/js/ui/tasks.js",
            headers={"if-none-match": r.headers["etag"]},
        )
        assert cached.status_code == 304


class TestTheWatchEventsProxyForwardsWrites:
    """Every method the Events page uses has to be listed on the proxy route.

    The SPA talks only to port 7999, so a method missing from the `api_route`
    declaration is a feature that cannot work at all: db_server answers the request
    correctly, and the browser never gets to make it — FastAPI rejects it here with a
    405 that reads, in the UI, as "failed to delete event".

    Nothing else catches this. The Playwright tests route `/api/watch-events` to a mock
    inside the browser, which is *above* this layer, and the db_server tests call the
    endpoint directly, which is below it. This is the only seam where the two lists of
    methods are compared.
    """

    @pytest.fixture
    def proxied(self, monkeypatch):
        """Record what actually reaches the proxy, so a 405 is distinguishable from a
        request that was forwarded and happened to fail upstream."""
        from fastapi.responses import JSONResponse

        import apps.image_tagging_server.web as web_module

        seen = []

        async def capture(path, request):
            seen.append((request.method, path))
            return JSONResponse(content={"status": "success"})

        monkeypatch.setattr(web_module.WATCH_EVENTS_PROXY, "handle_request", capture)
        return seen

    async def test_deleting_one_event_reaches_the_proxy(self, http_client, proxied):
        r = await http_client.delete("/api/watch-events/some-event-uuid")

        assert r.status_code == 200, r.text
        assert proxied == [("DELETE", "some-event-uuid")]

    async def test_deleting_a_selection_reaches_the_proxy(self, http_client, proxied):
        r = await http_client.post("/api/watch-events/delete", json={"uuids": ["a", "b"]})

        assert r.status_code == 200, r.text
        assert proxied == [("POST", "delete")]

    async def test_reading_events_still_works(self, http_client, proxied):
        r = await http_client.get("/api/watch-events/facets")

        assert r.status_code == 200, r.text
        assert proxied == [("GET", "facets")]
