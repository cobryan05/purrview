from dataclasses import dataclass
from typing import Optional

import db_client.models as db_models
from pydantic import Field

from apps.helpers.db.types import ApiModel, TaskConfigRead, TaskInstanceRead
from apps.tasks_server.tasks.Task import Task


def _is_continuous(typename: str) -> bool:
    """Whether a task type runs until stopped, looked up by name.

    Unknown typenames report False: a config can outlive the task class that
    served it (an unregistered plugin, a rename), and a stale row is better
    filed as a finished job than as something claiming to still be running.
    """
    # Imported here rather than at module scope: Registry imports Task, and Task
    # is already imported above, so a top-level import would close the cycle.
    from apps.tasks_server.tasks.Registry import task_registry

    task_cls = task_registry.get(typename)
    return bool(getattr(task_cls, "CONTINUOUS", False))


@dataclass
class TaskInfo:
    config: db_models.TaskConfigRead
    instance: Optional[db_models.TaskInstanceRead] = None
    task: Optional[Task] = None


class TaskInfoRead(ApiModel):
    config: TaskConfigRead
    instance: Optional[TaskInstanceRead] = None
    # None means indeterminate progress (frontend renders a striped bar); the field
    # default stays 0.0 so tasks that never call this out explicitly keep their
    # existing determinate 0% rendering.
    progress: Optional[float] = 0.0
    status_message: str = ""
    status_actions: list[dict] = Field(default_factory=list)
    log_lines: list[str] = Field(default_factory=list)
    is_paused_in_place: bool = False
    # Runs until stopped rather than to completion (see Task.CONTINUOUS). Read
    # from the registry by typename, not from `task_info.task`, which is None
    # for any instance not currently loaded -- a stopped watcher must still
    # report what kind of thing it is.
    continuous: bool = False

    @classmethod
    def from_task_info(cls, task_info: TaskInfo) -> "TaskInfoRead":
        return cls(
            continuous=_is_continuous(task_info.config.typename),
            config=TaskConfigRead.model_validate(task_info.config),
            instance=(TaskInstanceRead.model_validate(task_info.instance) if task_info.instance else None),
            progress=(task_info.task.get_progress() if task_info.task else 0.0),
            status_message=(task_info.task.get_status_message() if task_info.task else ""),
            status_actions=(task_info.task.get_status_actions() if task_info.task else []),
            log_lines=task_info.task.get_log_lines() if task_info.task else [],
            is_paused_in_place=(task_info.task.is_paused_in_place() if task_info.task else False),
        )
