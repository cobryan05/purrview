import pytest
from httpx import ASGITransport, AsyncClient

from apps.training_server.manager import Manager
from apps.training_server.web import WebApp


@pytest.fixture
def manager():
    return Manager()


@pytest.fixture
async def http_client(manager):
    web_app = WebApp(app_name="test", manager=manager)
    transport = ASGITransport(app=web_app.app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
