"""Guards for stale/unknown typenames hitting the task/provider registries.

A config or source saved under an old typename (e.g. after a rename or a
removed plugin) used to raise a bare KeyError deep in registry lookup —
surfaced to the API as the cryptic message "'OldTypename'". Each lookup must
instead produce a clear error naming the typename.
"""

from unittest.mock import MagicMock

import db_client.models as db_models
import pytest

import apps.tasks_server.tasks  # noqa: F401  — registers all task types
from apps.tasks_server.tasks.Registry import task_registry
from apps.tasks_server.tasks.utils.sourceHelper import SourceHelper
from tests.tasks_server.conftest import CONFIG_UUID


async def test_start_task_with_stale_typename_returns_clear_failure(http_client, tasks_api_mock):
    tasks_api_mock.list_task_configs.return_value = MagicMock(
        configs={
            CONFIG_UUID: db_models.TaskConfigRead(
                uuid=CONFIG_UUID,
                name="stale-config",
                typename="RemovedTaskType",
                params={},
                marked_for_delete=False,
                description=None,
            )
        }
    )
    resp = await http_client.post("/api/tasks/instances/start", json={"configUuid": CONFIG_UUID})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failure"
    assert "RemovedTaskType" in (body.get("message") or ""), f"unhelpful error message: {body.get('message')!r}"
    assert "Unknown task type" in body["message"]


async def test_preview_task_with_unknown_provider_errors_cleanly():
    """An unknown provider typename must raise a clear ValueError from _run —
    Task._execute turns it into {"status": "error", "message": ...} — instead
    of a bare KeyError ("'RemovedProvider'")."""
    task = task_registry["PreviewSourceTask"](
        task_config_uuid="cfg",
        params={"source_config": {"provider": "RemovedProvider", "provider_params": {}}},
        manager=MagicMock(),
    )
    await task._init(params=task._params, resume_data=None)
    with pytest.raises(ValueError, match="RemovedProvider"):
        await task._run()


def test_source_helper_unknown_provider_typename_raises_value_error():
    helper = SourceHelper(MagicMock())
    helper._sources_cache = {
        "src-1": MagicMock(typename="RemovedProvider", params={}),
    }
    with pytest.raises(ValueError, match="RemovedProvider"):
        helper.get_image_provider("src-1")
