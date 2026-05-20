import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from apps.helpers.types import InferenceResult
from apps.inference_server.manager import Manager
from apps.inference_server.web import WebApp
from tests.conftest import make_png

MODEL_METADATA = {
    "yolo": 8,
    "classes": ["cat", "dog"],
    "class_map": {},
}


def make_png_b64() -> str:
    return base64.b64encode(make_png()).decode()


@pytest.fixture
def models_dir(tmp_path):
    (tmp_path / "yolo_test.onnx").write_bytes(b"fake onnx")
    (tmp_path / "yolo_test.json").write_text(json.dumps(MODEL_METADATA))
    return tmp_path


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.processImage = AsyncMock(
        return_value=InferenceResult(
            source_width=1,
            source_height=1,
            detections=[],
            inference_time=0.01,
        )
    )
    return provider


@pytest.fixture
async def manager(models_dir, mock_provider, monkeypatch):
    monkeypatch.setattr(
        "apps.inference_server.manager.load_yolo_onnx",
        lambda path: mock_provider,
    )
    mgr = Manager(models_path=str(models_dir))
    await mgr.list_models()
    return mgr


@pytest.fixture
async def http_client(manager):
    web_app = WebApp(app_name="test", manager=manager)
    transport = ASGITransport(app=web_app.app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
