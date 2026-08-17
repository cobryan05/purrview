"""A task type decides whether its configs default to persistent.

"Persistent" means the instance auto-resumes after a server restart. It used to
be decided by whichever UI launched the task — `createTaskConfig` defaulted it
to false in JS — so a watcher started from the Runs launcher stopped silently
on every service restart, and nothing errored: the cameras were simply no
longer watched until somebody noticed.

The decision belongs to the task type, so these tests pin that an omitted
`persistent` resolves from `DEFAULT_PERSISTENT` and that an explicit value from
a caller who does care still wins.
"""

import pytest

from apps.tasks_server.tasks.Registry import task_registry
from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask
from apps.tasks_server.tasks.Task import Task


class TestDefaultPersistentDeclaration:
    def test_the_base_default_is_not_persistent(self):
        # Batch work must not silently resume hours later.
        assert Task.DEFAULT_PERSISTENT is False

    def test_a_watcher_defaults_to_persistent(self):
        assert StreamWatcherTask.DEFAULT_PERSISTENT is True

    def test_the_watcher_is_the_only_persistent_default(self):
        """A tripwire, not a rule: making another task persistent by default is
        a real decision, and it should not happen by copy-paste."""
        persistent = {name for name, cls in task_registry.items() if cls.DEFAULT_PERSISTENT}

        assert persistent == {"StreamWatcherTask"}

    def test_a_persistent_default_requires_checkpointing(self):
        """Auto-resume needs resume_data — a task that never checkpoints is
        marked ERROR on an interrupted restart, so a persistent default on one
        would promise something it cannot deliver."""
        for name, cls in task_registry.items():
            if not cls.DEFAULT_PERSISTENT:
                continue
            assert cls.CHECKPOINT_INTERVAL > 0, f"{name} defaults persistent but never checkpoints"


def _created_params(tasks_api_mock) -> dict:
    """The params of the single create_task_config call the manager made."""
    assert tasks_api_mock.create_task_config.call_count == 1
    payload = tasks_api_mock.create_task_config.call_args[0][0]
    return payload.params


class TestCreateResolvesTheDefault:
    async def test_omitted_persistent_uses_the_task_default(self, manager, tasks_api_mock):
        await manager.create_new_task_config(
            name="watch it",
            typename="StreamWatcherTask",
            params={"watch_profile_uuid": "wp-1"},
        )

        assert _created_params(tasks_api_mock)["_persistent"] is True

    async def test_omitted_persistent_on_batch_work_stays_false(self, manager, tasks_api_mock):
        await manager.create_new_task_config(name="one off", typename="TestTask", params={})

        assert _created_params(tasks_api_mock)["_persistent"] is False

    async def test_an_explicit_false_beats_the_task_default(self, manager, tasks_api_mock):
        # A preview/one-shot runner says false and must be obeyed, even for a
        # task type that would otherwise default persistent.
        await manager.create_new_task_config(
            name="watch once",
            typename="StreamWatcherTask",
            params={"watch_profile_uuid": "wp-1"},
            persistent=False,
        )

        assert _created_params(tasks_api_mock)["_persistent"] is False

    async def test_an_explicit_true_is_kept(self, manager, tasks_api_mock):
        await manager.create_new_task_config(
            name="keep it",
            typename="TestTask",
            params={},
            persistent=True,
        )

        assert _created_params(tasks_api_mock)["_persistent"] is True

    async def test_unknown_task_type_still_rejected(self, manager):
        with pytest.raises(ValueError):
            await manager.create_new_task_config(name="bad", typename="NoSuchTask", params={})


class TestCreateEndpointResolvesTheDefault:
    async def test_omitting_persistent_over_http_uses_the_task_default(self, http_client, tasks_api_mock):
        resp = await http_client.post(
            "/api/tasks/configs/create",
            json={"name": "watch it", "typename": "StreamWatcherTask", "params": {}},
        )

        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        assert _created_params(tasks_api_mock)["_persistent"] is True

    async def test_sending_false_over_http_is_obeyed(self, http_client, tasks_api_mock):
        resp = await http_client.post(
            "/api/tasks/configs/create",
            json={
                "name": "watch once",
                "typename": "StreamWatcherTask",
                "params": {},
                "persistent": False,
            },
        )

        assert resp.status_code == 200
        assert _created_params(tasks_api_mock)["_persistent"] is False
