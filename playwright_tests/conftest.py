"""
Playwright test fixtures.

Run with:
    pytest tests/playwright/ -p no:xdist
(xdist is incompatible with session-scoped browser fixtures)
"""

import socket
import threading
import time
from typing import Generator
from unittest.mock import MagicMock

import pytest
import uvicorn
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_label_mock():
    async def handle(path: str, request):
        return JSONResponse(content={"status": "success", "labels": []})

    return handle


def _make_tag_mock():
    async def handle(path: str, request):
        # tags.js fetches /api/tags/list and expects {"tags": [...]}
        return JSONResponse(content={"tags": []})

    return handle


def _make_models_mock():
    async def handle(path: str, request):
        return JSONResponse(content={"status": "success", "models": {}})

    return handle


def _make_sources_mock():
    async def handle(path: str, request):
        return JSONResponse(content={"status": "success", "sources": []})

    return handle


def _make_tasks_mock():
    async def handle(path: str, request):
        if path == "types/list":
            return JSONResponse(content={"status": "success", "task_types": []})
        if path == "types/status-schema":
            # Phase 3 status_schema() endpoint. Empty by default — callers
            # that need a specific schema override this route with page.route().
            return JSONResponse(content={"status": "success", "schemas": {}})
        if path == "instances/status-data":
            # Phase 3 per-instance status-data endpoint. Empty by default —
            # callers that need specific status-data override with page.route().
            return JSONResponse(content={"status": "success", "statusData": {}})
        if path == "instances/command":
            return JSONResponse(content={"status": "success", "message": "ok"})
        if path.startswith("configs/"):
            return JSONResponse(content={"status": "success", "task_configs": []})
        if path.startswith("instances/"):
            return JSONResponse(content={"status": "success", "tasks": []})
        return JSONResponse(content={"status": "success"})

    return handle


def _make_datasets_mock():
    async def handle(path: str, request):
        return JSONResponse(content={"status": "success", "datasets": []})

    return handle


def _make_model_configs_mock():
    async def handle(path: str, request):
        return JSONResponse(content={"status": "success", "model_configs": []})

    return handle


def _make_model_groups_mock():
    async def handle(path: str, request):
        return JSONResponse(content={"status": "success", "groups": []})

    return handle


def _make_filter_chains_mock():
    async def handle(path: str, request):
        return JSONResponse(content={"status": "success", "chains": []})

    return handle


def _make_preview_runs_mock():
    async def handle(path: str, request):
        # The review toolbar's "Saved Previews" count fetches this on mount.
        # Tests that need actual runs override the route with page.route().
        return JSONResponse(content={"status": "success", "runs": []})

    return handle


def _make_settings_mock():
    async def handle(path: str, request):
        return JSONResponse(content={"status": "success", "settings": {}})

    return handle


def _make_training_configs_mock():
    async def handle(path: str, request):
        return JSONResponse(content={"status": "success", "configs": []})

    return handle


def _make_images_mock():
    async def handle(path: str, request):
        return JSONResponse(content={"status": "success", "images": [], "total": 0})

    return handle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args: dict) -> dict:
    """
    Disable GPU/hardware compositing for the headless browser.

    Without this, page.goto() deadlocks indefinitely in some sandboxed/
    containerized environments: all page resources finish loading (verified
    via request/response tracing) and the JS main thread throws no errors,
    but the browser's 'load' event never fires and the renderer stops
    responding even to trivial CDP calls. The hang goes away when the
    GPU process is disabled and rendering is forced onto SwiftShader
    (software) rather than left to negotiate hardware compositing, which
    points to a stuck handshake with the sandboxed GPU/compositor process
    rather than anything in this app's JS.
    """
    return {
        **browser_type_launch_args,
        "args": [
            *browser_type_launch_args.get("args", []),
            "--disable-gpu",
            "--disable-software-rasterizer",
            "--use-gl=swiftshader",
        ],
    }


@pytest.fixture(scope="session")
def live_server() -> Generator[str, None, None]:
    """
    Spin up the image_tagging_server FastAPI app on a real port with all
    downstream proxies mocked out.  Returns the base URL, e.g.
    'http://127.0.0.1:54321'.
    """
    import apps.image_tagging_server.web as web_module
    from apps.image_tagging_server.manager import Manager
    from apps.image_tagging_server.web import WebApp

    # --- mock downstream API clients ---
    inf_api = MagicMock()
    inf_api.configuration.host = "http://inference:8001"
    tasks_api = MagicMock()
    tasks_api.configuration.host = "http://tasks:8002"
    db_api = MagicMock()
    db_api.configuration.host = "http://db:8003"

    manager = Manager(
        inference_api_client=inf_api,
        tasks_api_client=tasks_api,
        db_api_client=db_api,
        default_training_url="http://training:8005",
    )

    # --- patch all module-level proxy objects ---
    proxies_and_mocks = [
        ("IMAGE_PROXY", _make_images_mock()),
        ("LABEL_PROXY", _make_label_mock()),
        ("SOURCES_PROXY", _make_sources_mock()),
        ("TAG_PROXY", _make_tag_mock()),
        ("TASKS_PROXY", _make_tasks_mock()),
        ("DATASETS_PROXY", _make_datasets_mock()),
        ("MODEL_CONFIGS_PROXY", _make_model_configs_mock()),
        ("MODEL_GROUPS_PROXY", _make_model_groups_mock()),
        ("TRAINING_CONFIGS_PROXY", _make_training_configs_mock()),
        ("SETTINGS_PROXY", _make_settings_mock()),
        # Missed when filter chains landed: unmocked, this proxy makes a REAL
        # connection to the db service, so the two "no console errors" tests
        # passed only while a live stack happened to be running and failed with
        # httpx.ConnectError otherwise.
        ("FILTER_CHAINS_PROXY", _make_filter_chains_mock()),
        ("PREVIEW_RUNS_PROXY", _make_preview_runs_mock()),
    ]
    originals = []
    for attr, mock_fn in proxies_and_mocks:
        proxy_obj = getattr(web_module, attr)
        originals.append((proxy_obj, proxy_obj.handle_request))
        proxy_obj.handle_request = mock_fn

    # Prevent start() from rebuilding proxies (it reads from DB which isn't
    # reachable in tests, but still replaces the proxy instances we patched).
    async def _noop_start():
        pass

    manager.start = _noop_start

    # --- patch manager's dynamic proxies (inference + training) ---
    originals.append((manager.inference_proxy, manager.inference_proxy.handle_request))
    manager.inference_proxy.handle_request = _make_models_mock()
    originals.append((manager.training_proxy, manager.training_proxy.handle_request))
    manager.training_proxy.handle_request = _make_tasks_mock()

    # --- build and start the ASGI app ---
    web_app = WebApp(app_name="test", manager=manager)
    asgi_app = web_app.app()

    port = _free_port()
    config = uvicorn.Config(app=asgi_app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # wait until the server is accepting connections
    while not server.started:
        time.sleep(0.05)

    base = f"http://127.0.0.1:{port}"
    yield base

    # --- teardown ---
    server.should_exit = True
    thread.join(timeout=5)

    for proxy_obj, original_fn in originals:
        proxy_obj.handle_request = original_fn


@pytest.fixture(scope="session")
def base_url(live_server: str) -> str:
    """Override pytest-playwright's base_url to point at our test server."""
    return live_server


def _block_sse(page) -> None:
    """
    Intercept the two SSE endpoints so they return empty closed responses.

    Without this, EventSource connections to /api/notices/stream and
    /api/tasks/stream are kept open forever, preventing
    wait_for_load_state("networkidle") from ever resolving.
    """
    for pattern in ("**/api/notices/stream", "**/api/tasks/stream"):
        page.route(
            pattern,
            lambda route: route.fulfill(status=200, content_type="text/event-stream", body=""),
        )


@pytest.fixture()
def page(context, base_url):  # noqa: F811
    """
    Override pytest-playwright's page fixture to block SSE endpoints.
    All tests that receive a `page: Page` argument get this version.
    """
    p = context.new_page()
    _block_sse(p)
    # -n auto runs many workers per host; under contention (many chromium
    # processes competing for CPU — see e.g. concurrent agent sessions) a
    # multi-await chain like the coverage-popup's stats fetch can occasionally
    # miss Playwright's 30s default, which is otherwise plenty for a mocked
    # network call. Widen the ceiling rather than the per-test wait so every
    # action/wait_for_selector on this page benefits, not just one call site.
    p.set_default_timeout(60000)
    yield p
    p.close()


@pytest.fixture(scope="module")
def shared_page(browser, base_url):
    """
    A single pre-navigated page shared by every test in a module.

    Pure-logic tests that import an ES module and exercise a function don't
    need a fresh SPA reload each time — they build/tear down their own DOM.
    Reusing one page removes a full Konva/Pico/module re-init per test, which
    is the dominant cost of the Playwright suite. Tests using this fixture
    must clean up any DOM nodes they create, or rely on the autouse
    _canvas_dom_cleanup fixture below which removes fixed-position test
    containers after each test.
    """
    page = browser.new_page()
    _block_sse(page)
    page.goto(base_url)
    yield page
    page.close()


@pytest.fixture(autouse=True)
def _canvas_dom_cleanup(request):
    """
    After each test that uses shared_page, remove any fixed-position div
    containers and context-menu elements that canvas/Konva tests create but
    don't tear down themselves.  This lets those tests share a single page
    (shared_page) instead of calling page.goto() on every test, eliminating
    the dominant startup cost of the Playwright suite.

    No-op for tests that don't use shared_page (the funcargs check guards it).
    """
    yield
    sp = request.node.funcargs.get("shared_page")
    if sp is not None:
        sp.evaluate(
            "document.querySelectorAll('body > div[style*=\"position:fixed\"]').forEach(d => d.remove());"
            "document.querySelectorAll('.ctx-menu').forEach(m => m.remove());"
        )
