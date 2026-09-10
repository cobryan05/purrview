import pytest
from httpx import ASGITransport, AsyncClient

from apps.db_server.manager import Manager
from apps.db_server.web import WebApp
from apps.helpers.db.db_client import DbClient


@pytest.fixture
def files_root(tmp_path):
    d = tmp_path / "files"
    d.mkdir()
    return d


@pytest.fixture
async def db_client(tmp_path, files_root):
    # `files_root` must match the Manager's below: the filename filter matches
    # the FILES_ROOT-relative path (_relative_filename_expr), so a client
    # pointed at a different root would fall back to matching the absolute
    # path and quietly diverge from what the API returns.
    client = DbClient(db_dir=tmp_path, memory=True, files_root=files_root)
    await client.init_db()
    yield client
    await client.close()


@pytest.fixture
def manager(db_client, files_root):
    """The same Manager `http_client` serves, for tests that need to reach past
    the HTTP surface (patching an internal, asserting cache state, …)."""
    mgr = Manager(db_client=db_client, files_root=files_root)
    mgr._db_ready = True  # db_client.init_db() was already called directly above
    return mgr


@pytest.fixture
async def http_client(manager, files_root):
    web_app = WebApp(app_name="test", manager=manager)
    transport = ASGITransport(app=web_app.app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
