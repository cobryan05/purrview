from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from apps.relay_buffer_server.manager import Manager
from apps.relay_buffer_server.web import WebApp


@pytest.fixture
def manager(monkeypatch):
    api_client = MagicMock()
    api_client.configuration.host = "http://localhost:9997"

    mgr = Manager(media_mtx_api_client=api_client)

    # Patch all methods that make real network calls to mediamtx
    monkeypatch.setattr(mgr, "refresh_streams", AsyncMock())
    monkeypatch.setattr(mgr, "create_rtsp_relay_stream", AsyncMock())
    monkeypatch.setattr(mgr, "create_delay_stream", AsyncMock())
    monkeypatch.setattr(mgr, "destroy_stream", AsyncMock())
    monkeypatch.setattr(mgr, "get_config", AsyncMock(return_value="rtspAddress: :8554\n"))

    return mgr


@pytest.fixture
async def http_client(manager):
    web_app = WebApp(app_name="test", manager=manager)
    transport = ASGITransport(app=web_app.app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
