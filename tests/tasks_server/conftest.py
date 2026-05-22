import uuid
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

import db_client.models as db_models
from apps.helpers.consts import TaskStatus
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
def tasks_api_mock():
    mock = MagicMock()
    mock_config = db_models.TaskConfigRead(
        uuid=CONFIG_UUID,
        name="test-config",
        typename="TestTask",
        params_json={},
        marked_for_delete=False,
        description=None,
    )
    mock_instance = db_models.TaskInstanceRead(
        uuid=TASK_UUID,
        config_uuid=CONFIG_UUID,
        status=TaskStatus.PENDING,
        resume_data_json=None,
        result_json=None,
        error_message=None,
    )
    mock.create_task_config.return_value = MagicMock(config=mock_config)
    mock.list_task_configs.return_value = MagicMock(configs={CONFIG_UUID: mock_config})
    mock.update_task_config.return_value = MagicMock()
    mock.delete_task_configs.return_value = MagicMock()
    mock.create_task_instance.return_value = MagicMock(instance=mock_instance)
    mock.list_task_instances.return_value = MagicMock(instances=[])
    mock.delete_task_instances.return_value = MagicMock()
    mock.set_task_instance_status.return_value = MagicMock()
    mock.set_task_instance_result.return_value = MagicMock()
    mock.set_task_instance_resume_data.return_value = MagicMock()
    mock.get_task_results.return_value = MagicMock(results={})
    return mock


@pytest.fixture
def manager(tasks_api_mock):
    db_api = MagicMock()
    db_api.configuration.host = "http://db:8000"
    inf_api = MagicMock()
    inf_api.configuration.host = "http://inference:8001"
    with patch("apps.tasks_server.manager.TasksApi", return_value=tasks_api_mock):
        yield Manager(db_api_client=db_api, inference_api_client=inf_api)


@pytest.fixture
async def http_client(manager):
    web_app = WebApp(app_name="test", manager=manager)
    transport = ASGITransport(app=web_app.app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
