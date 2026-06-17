"""Shared fixtures for training-container tests.

These tests require PyTorch, torchvision, albumentations, timm, and
onnxruntime-gpu to be installed.  Run inside the training container only.
"""

import asyncio
import os
import tempfile

import pytest
from httpx import ASGITransport, AsyncClient

from apps.training_server.manager import Manager
from apps.training_server.web import WebApp


@pytest.fixture(scope="session")
def db_base_url() -> str:
    """URL of the db_server (default: localhost:8003, override with DB_SERVER_URL env var)."""
    host = os.environ.get("DB_SERVER_HOST", "localhost")
    port = os.environ.get("DB_SERVER_PORT", "8003")
    return f"http://{host}:{port}"


@pytest.fixture
def manager():
    return Manager()


@pytest.fixture
async def http_client(manager):
    web_app = WebApp(app_name="test", manager=manager)
    transport = ASGITransport(app=web_app.app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def tmp_cache_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d
