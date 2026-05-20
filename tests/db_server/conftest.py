import struct
import zlib

import pytest
from httpx import ASGITransport, AsyncClient

from apps.db_server.manager import Manager
from apps.db_server.web import WebApp
from apps.helpers.db.db_client import DbClient


def make_png() -> bytes:
    """Return a minimal valid 1x1 white PNG."""
    def chunk(tag, data):
        raw = tag + data
        return struct.pack(">I", len(data)) + raw + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        + chunk(b"IEND", b"")
    )


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
