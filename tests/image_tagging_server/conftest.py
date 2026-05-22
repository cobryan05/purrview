from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from apps.image_tagging_server.manager import Manager
from apps.image_tagging_server.web import WebApp


@pytest.fixture
def manager():
    inf_api = MagicMock()
    inf_api.configuration.host = "http://inference:8001"
    tasks_api = MagicMock()
    tasks_api.configuration.host = "http://tasks:8002"
    db_api = MagicMock()
    db_api.configuration.host = "http://db:8003"

    return Manager(
        inference_api_client=inf_api,
        tasks_api_client=tasks_api,
        db_api_client=db_api,
    )


@pytest.fixture
async def http_client(manager, monkeypatch):
    # Patch all proxy handle_request methods so no real HTTP calls are made
    proxy_response = JSONResponse(content={"status": "success"})
    proxy_mock = AsyncMock(return_value=proxy_response)

    import apps.image_tagging_server.web as web_module
    for proxy_attr in ("IMAGE_PROXY", "LABEL_PROXY", "MODELS_PROXY", "SOURCES_PROXY", "TAG_PROXY", "TASKS_PROXY"):
        monkeypatch.setattr(getattr(web_module, proxy_attr), "handle_request", proxy_mock)

    web_app = WebApp(app_name="test", manager=manager)
    transport = ASGITransport(app=web_app.app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
