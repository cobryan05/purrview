import pytest
from httpx import ASGITransport, AsyncClient

from apps.training_server.manager import Manager
from apps.training_server.web import WebApp


@pytest.fixture
def tmp_training_dirs(tmp_path):
    """Return (cache_dir, artifacts_dir) as str paths under pytest's tmp_path."""
    cache = tmp_path / "cache"
    cache.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    return str(cache), str(artifacts)


@pytest.fixture
def manager(tmp_training_dirs):
    cache_dir, artifacts_dir = tmp_training_dirs
    return Manager(cache_dir=cache_dir, artifacts_dir=artifacts_dir)


@pytest.fixture
async def http_client(manager):
    web_app = WebApp(app_name="test", manager=manager)
    transport = ASGITransport(app=web_app.app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
