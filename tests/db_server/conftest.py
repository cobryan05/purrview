import pytest
from httpx import ASGITransport, AsyncClient

from apps.db_server.manager import Manager
from apps.db_server.web import WebApp
from apps.helpers.db.db_client import DbClient


@pytest.fixture
async def db_client(tmp_path):
    client = DbClient(db_dir=tmp_path)
    await client.init_db()
    yield client
    await client.close()


@pytest.fixture
def files_root(tmp_path):
    d = tmp_path / "files"
    d.mkdir()
    return d


@pytest.fixture
async def http_client(db_client, files_root):
    manager = Manager(db_client=db_client, files_root=files_root)
    web_app = WebApp(app_name="test", manager=manager)
    transport = ASGITransport(app=web_app.app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
