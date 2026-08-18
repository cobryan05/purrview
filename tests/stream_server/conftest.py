from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from apps.stream_server.manager import Manager
from apps.stream_server.web import WebApp


@pytest.fixture
def manager(monkeypatch, tmp_path):
    api_client = MagicMock()
    api_client.configuration.host = "http://localhost:9997"

    mgr = Manager(media_mtx_api_client=api_client)
    # Keep lease persistence out of /data, which is a real bind mount in the container.
    mgr._lease_file = tmp_path / "leases.json"

    # Patch all methods that make real network calls to mediamtx
    monkeypatch.setattr(mgr, "refresh_streams", AsyncMock())
    monkeypatch.setattr(mgr, "destroy_stream", AsyncMock())
    monkeypatch.setattr(mgr, "get_config", AsyncMock(return_value="rtspAddress: :8554\n"))

    return mgr


@pytest.fixture
async def http_client(manager):
    web_app = WebApp(app_name="test", manager=manager)
    transport = ASGITransport(app=web_app.app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
