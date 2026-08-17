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
            # Bound before the try, and to both children rather than to whatever
            # asyncio.wait returned: if this task is itself cancelled, `wait` raises
            # before the assignment ever happens, and the cleanup below then died on
            # an UnboundLocalError — turning a CancelledError into an ordinary
            # exception and leaking both children. Cancelling an already-finished
            # task is a no-op, so covering both is free.
            pending = {cancel_task, sleep_task}

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


# Register a task that reports indeterminate (None) progress while running, to
# exercise the progress: Optional[float] wire round-trip (TaskInfoRead.progress).
if "IndeterminateProgressTask" not in task_registry:

    @register_task("IndeterminateProgressTask")
    class IndeterminateProgressTask(Task):
        async def _init(self, params: dict, resume_data: Optional[dict]) -> None:
            pass

        async def _run(self) -> dict[str, Any]:
            self._update_progress(None)
            await self._cancel_flag.wait()
            return {"canceled": True}

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


# Register a task that overrides status_schema()/get_status_data()/handle_command()
# to exercise the Phase 3 task-status-schema framework end to end.
if "StatusSchemaTask" not in task_registry:

    @register_task("StatusSchemaTask")
    class StatusSchemaTask(Task):
        async def _init(self, params: dict, resume_data: Optional[dict]) -> None:
            self._capture_now_requested = False
            self._counter = 0

        async def _run(self) -> dict[str, Any]:
            await self._cancel_flag.wait()
            return {"canceled": True}

        async def _deinit(self) -> None:
            pass

        @classmethod
        def status_schema(cls) -> list[dict[str, Any]]:
            return [
                {"id": "progress", "type": "progress_bar", "source": "summary"},
                {"id": "counter", "type": "stat_grid"},
                {"id": "mystery_widget", "type": "some_future_widget_type"},
            ]

        def get_status_data(self) -> dict[str, Any]:
            return {"counter": self._counter}

        def get_status_actions(self) -> list[dict]:
            return [
                {
                    "type": "task_command",
                    "label": "Capture now",
                    "command": "capture_now",
                    "params": {},
                }
            ]

        async def handle_command(self, command: str, params: dict[str, Any]) -> None:
            if command == "capture_now":
                self._capture_now_requested = True
                self._counter += 1
                return
            raise NotImplementedError(f"Unknown command: {command}")

    STATUS_SCHEMA_TASK_TYPENAME = "StatusSchemaTask"


CONFIG_UUID = str(uuid.uuid4())
TASK_UUID = str(uuid.uuid4())
STATUS_SCHEMA_CONFIG_UUID = str(uuid.uuid4())
STATUS_SCHEMA_TASK_UUID = str(uuid.uuid4())


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
    status_schema_config = db_models.TaskConfigRead(
        uuid=STATUS_SCHEMA_CONFIG_UUID,
        name="status-schema-config",
        typename="StatusSchemaTask",
        params={},
        marked_for_delete=False,
        description=None,
    )
    status_schema_instance = db_models.TaskInstanceRead(
        uuid=STATUS_SCHEMA_TASK_UUID,
        config_uuid=STATUS_SCHEMA_CONFIG_UUID,
        status=TaskStatus.PENDING,
        resume_data=None,
        result=None,
        error_message=None,
    )
    mock.create_task_config.return_value = MagicMock(config=mock_config)
    mock.list_task_configs.return_value = MagicMock(
        configs={CONFIG_UUID: mock_config, STATUS_SCHEMA_CONFIG_UUID: status_schema_config}
    )
    mock.update_task_config.return_value = MagicMock()
    mock.delete_task_configs.return_value = MagicMock()

    def _create_task_instance(payload):
        if payload.config_uuid == STATUS_SCHEMA_CONFIG_UUID:
            return MagicMock(instance=status_schema_instance)
        return MagicMock(instance=mock_instance)

    mock.create_task_instance.side_effect = _create_task_instance
    mock.list_task_instances.return_value = MagicMock(instances=[])
    mock.delete_task_instances.return_value = MagicMock()
    mock.set_task_instance_status.return_value = MagicMock()
    mock.set_task_instance_result.return_value = MagicMock()
    mock.set_task_instance_resume_data.return_value = MagicMock()
    mock.set_task_instance_timestamps.return_value = MagicMock()
    mock.set_task_instance_heartbeat.return_value = MagicMock()
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
