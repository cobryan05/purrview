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


def _make_settings_mock():
    async def handle(path: str, request):
        return JSONResponse(content={"status": "success", "settings": {}})
    return handle


def _make_images_mock():
    async def handle(path: str, request):
        return JSONResponse(content={"status": "success", "images": [], "total": 0})
    return handle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
    )

    # --- patch all module-level proxy objects ---
    proxies_and_mocks = [
        ("IMAGE_PROXY", _make_images_mock()),
        ("LABEL_PROXY", _make_label_mock()),
        ("MODELS_PROXY", _make_models_mock()),
        ("SOURCES_PROXY", _make_sources_mock()),
        ("TAG_PROXY", _make_tag_mock()),
        ("TASKS_PROXY", _make_tasks_mock()),
        ("DATASETS_PROXY", _make_datasets_mock()),
        ("MODEL_CONFIGS_PROXY", _make_model_configs_mock()),
        ("SETTINGS_PROXY", _make_settings_mock()),
    ]
    originals = []
    for attr, mock_fn in proxies_and_mocks:
        proxy_obj = getattr(web_module, attr)
        originals.append((proxy_obj, proxy_obj.handle_request))
        proxy_obj.handle_request = mock_fn

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
