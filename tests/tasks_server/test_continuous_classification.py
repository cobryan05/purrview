"""Tasks declare whether they run until stopped, and the API reports it.

A training runs a fixed number of epochs and finishes; a stream watcher is
`while not self._cancel_flag.is_set()` and never does. The frontend has to tell
them apart to list them separately and to keep its "still working" badge
meaningful — a permanently armed watcher folded into that count pins it at >=1
forever.

Nothing observable distinguishes the two, which is why `Task.CONTINUOUS` is
declared rather than inferred:

  - indeterminate progress does not: CollectTrainingImagesTask reports None the
    whole way and still completes
  - PERSISTENT does not either: that is about auto-restart, and the watcher is
    continuous *without* being persistent
"""

from dataclasses import dataclass
from typing import Optional

import db_client.models as db_models

import apps.tasks_server.tasks  # noqa: F401  — registers all task types
from apps.tasks_server.tasks.Registry import task_registry
from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask
from apps.tasks_server.tasks.Task import Task
from apps.tasks_server.types import TaskInfo, TaskInfoRead, _is_continuous


class TestDeclaration:
    def test_tasks_run_to_completion_by_default(self):
        """The default has to be the safe one: a new task class that forgets to
        declare itself is counted as a job, which is merely wrong rather than
        silently disabling the badge."""
        assert Task.CONTINUOUS is False

    def test_the_stream_watcher_declares_itself_continuous(self):
        assert StreamWatcherTask.CONTINUOUS is True

    def test_every_registered_task_declares_a_bool(self):
        """Catches a subclass that shadows CONTINUOUS with something truthy but
        not a bool, which would serialise as a non-bool through the API."""
        for name, cls in task_registry.items():
            assert isinstance(cls.CONTINUOUS, bool), f"{name}.CONTINUOUS is {cls.CONTINUOUS!r}"

    def test_only_the_watcher_is_continuous_today(self):
        """Not a rule, a snapshot: if this fails because a new continuous task
        landed, add it here. It exists so the flag cannot be switched on for
        something by accident."""
        continuous = {name for name, cls in task_registry.items() if cls.CONTINUOUS}
        assert continuous == {"StreamWatcherTask"}


class TestLookupByTypename:
    def test_known_typenames_resolve(self):
        assert _is_continuous("StreamWatcherTask") is True
        assert _is_continuous("TrainModelTask") is False

    def test_unknown_typename_is_not_continuous(self):
        """A config can outlive the class that served it (a rename, an
        unregistered plugin). A stale row is better filed as a finished job than
        as something claiming to still be running."""
        assert _is_continuous("RemovedTaskType") is False


@dataclass
class _Cfg:
    """Minimal stand-in for a task config row — only typename is read here."""

    typename: str
    uuid: str = "cfg-1"
    name: str = "a task"
    params: Optional[dict] = None
    marked_for_delete: bool = False
    description: Optional[str] = None


def _task_info(typename: str) -> TaskInfo:
    return TaskInfo(
        config=db_models.TaskConfigRead(
            uuid="cfg-1",
            name=f"run {typename}",
            typename=typename,
            params={},
            marked_for_delete=False,
            description=None,
        ),
        instance=None,
        task=None,
    )


class TestApiPayload:
    def test_continuous_is_reported_for_a_watcher(self):
        assert TaskInfoRead.from_task_info(_task_info("StreamWatcherTask")).continuous is True

    def test_continuous_is_false_for_a_training(self):
        assert TaskInfoRead.from_task_info(_task_info("TrainModelTask")).continuous is False

    def test_reported_without_a_loaded_task_object(self):
        """`TaskInfo.task` is None for any instance not currently loaded — a
        stopped or historical watcher. Reading the flag off the live object
        would misfile exactly those, so it is resolved from the registry.
        """
        info = _task_info("StreamWatcherTask")
        assert info.task is None
        assert TaskInfoRead.from_task_info(info).continuous is True

    def test_serialises_camel_case_like_its_siblings(self):
        """ApiModel aliases to camelCase; the frontend reads `continuous`, which
        is single-word and so must survive the alias generator unchanged."""
        dumped = TaskInfoRead.from_task_info(_task_info("StreamWatcherTask")).model_dump(by_alias=True)
        assert dumped["continuous"] is True
