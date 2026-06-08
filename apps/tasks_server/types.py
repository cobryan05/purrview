from dataclasses import dataclass
from typing import Optional

import db_client.models as db_models
from pydantic import Field

from apps.helpers.db.types import ApiModel, TaskConfigRead, TaskInstanceRead
from apps.tasks_server.tasks.Task import Task


@dataclass
class TaskInfo:
    config: db_models.TaskConfigRead
    instance: Optional[db_models.TaskInstanceRead] = None
    task: Optional[Task] = None


class TaskInfoRead(ApiModel):
    config: TaskConfigRead
    instance: Optional[TaskInstanceRead] = None
    progress: float = 0.0
    status_message: str = ""
    status_actions: list[dict] = Field(default_factory=list)
    log_lines: list[str] = Field(default_factory=list)
    is_paused_in_place: bool = False

    @classmethod
    def from_task_info(cls, task_info: TaskInfo) -> "TaskInfoRead":
        return cls(
            config=TaskConfigRead.model_validate(task_info.config),
            instance=TaskInstanceRead.model_validate(task_info.instance) if task_info.instance else None,
            progress=task_info.task.get_progress() if task_info.task else 0.0,
            status_message=task_info.task.get_status_message() if task_info.task else "",
            status_actions=task_info.task.get_status_actions() if task_info.task else [],
            log_lines=task_info.task.get_log_lines() if task_info.task else [],
            is_paused_in_place=task_info.task.is_paused_in_place() if task_info.task else False,
        )
