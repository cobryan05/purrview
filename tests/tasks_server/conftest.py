import asyncio
import uuid
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import db_client.models as db_models
import pytest
from httpx import ASGITransport, AsyncClient

# Trigger discovery of real tasks (registers PreviewSourceTask, etc.)
import apps.tasks_server.tasks  # noqa: F401
from apps.helpers.consts import TaskStatus
from apps.tasks_server.manager import Manager
from apps.tasks_server.tasks.Registry import register_task, task_registry
from apps.tasks_server.tasks.Task import Task
from apps.tasks_server.web import WebApp

# Register a minimal task used only in tests
if "TestTask" not in task_registry:

    @register_task("TestTask")
    class TestTask(Task):
        async def _init(self, params: dict, resume_data: Optional[dict]) -> None:
            self._duration = float(params.get("duration", 0.0))
            self._status_msg = "Initialized"

        async def _run(self) -> dict[str, Any]:
            if self._duration <= 0:
                return {}

            cancel_task = asyncio.create_task(self._cancel_flag.wait())
            sleep_task = asyncio.create_task(asyncio.sleep(self._duration))

            try:
                done, pending = await asyncio.wait({cancel_task, sleep_task}, return_when=asyncio.FIRST_COMPLETED)

                if sleep_task in done:
                    return {"ran_for": self._duration}
                else:
                    self._status_msg = "Canceled"
                    return {"canceled": True}
            finally:
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass

        async def _deinit(self) -> None:
            pass


# Register a task that supports cooperative pause via _check_pause()
if "PausableTestTask" not in task_registry:

    @register_task("PausableTestTask")
    class PausableTestTask(Task):
        async def _init(self, params: dict, resume_data: Optional[dict]) -> None:
            self._iterations = int(params.get("iterations", 20))

        async def _run(self) -> dict[str, Any]:
            count = 0
            for _ in range(self._iterations):
                if self._cancel_flag.is_set():
                    break
                await self._check_pause()
                await asyncio.sleep(0.01)
                count += 1
            return {"count": count}

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
        params={"duration": 60.0},
        marked_for_delete=False,
        description=None,
    )
    mock_instance = db_models.TaskInstanceRead(
        uuid=TASK_UUID,
        config_uuid=CONFIG_UUID,
        status=TaskStatus.PENDING,
        resume_data=None,
        result=None,
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
    mock.set_task_instance_timestamps.return_value = MagicMock()
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
