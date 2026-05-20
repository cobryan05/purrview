import uuid
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from apps.helpers.consts import TaskStatus
from apps.helpers.types import TaskConfigMetadata, TaskInstanceMetadata
from apps.tasks_server.manager import Manager
from apps.tasks_server.tasks.Registry import register_task, task_registry
from apps.tasks_server.tasks.Task import Task
from apps.tasks_server.web import WebApp

# Trigger discovery of real tasks (registers PreviewSourceTask, etc.)
import apps.tasks_server.tasks  # noqa: F401

# Register a minimal task used only in tests
if "TestTask" not in task_registry:

    @register_task("TestTask")
    class TestTask(Task):
        async def _init(self, params: dict, resume_data: Optional[dict]) -> None:
            pass

        async def _run(self) -> dict[str, Any]:
            return {}

        async def _deinit(self) -> None:
            pass


CONFIG_UUID = str(uuid.uuid4())
TASK_UUID = str(uuid.uuid4())


@pytest.fixture
def legacy_db():
    mock = AsyncMock()
    mock.add_task_config.return_value = TaskConfigMetadata(
        uuid=CONFIG_UUID, typename="TestTask", params={}, name="test-config"
    )
    mock.get_task_configs.return_value = {
        CONFIG_UUID: TaskConfigMetadata(
            uuid=CONFIG_UUID, typename="TestTask", params={}, name="test-config"
        )
    }
    mock.insert_new_active_task.return_value = TaskInstanceMetadata(
        uuid=TASK_UUID, config_uuid=CONFIG_UUID, typename="TestTask", status=TaskStatus.PENDING
    )
    mock.get_tasks.return_value = []
    mock.delete_active_tasks.return_value = None
    mock.set_task_status.return_value = None
    mock.set_task_result.return_value = None
    mock.set_task_resume_data.return_value = None
    mock.get_task_results.return_value = {}
    return mock


@pytest.fixture
def mock_api_clients():
    db_api = MagicMock()
    db_api.configuration.host = "http://db:8000"
    inf_api = MagicMock()
    inf_api.configuration.host = "http://inference:8001"
    return db_api, inf_api


@pytest.fixture
def manager(legacy_db, mock_api_clients):
    db_api, inf_api = mock_api_clients
    return Manager(
        db_api_client=db_api,
        legacy_db_client=legacy_db,
        inference_api_client=inf_api,
    )


@pytest.fixture
async def http_client(manager):
    web_app = WebApp(app_name="test", manager=manager)
    transport = ASGITransport(app=web_app.app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
