from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

import db_client.models as db_models
from apps.helpers.db.types import TaskConfigRead, TaskInstanceRead
from apps.tasks_server.tasks.Task import Task


@dataclass
class TaskInfo:
    config: db_models.TaskConfigRead
    instance: Optional[db_models.TaskInstanceRead] = None
    task: Optional[Task] = None


class TaskInfoRead(BaseModel):
    config: TaskConfigRead
    instance: Optional[TaskInstanceRead] = None
    progress: float = 0.0
    status_message: str = ""
    log_lines: list[str] = Field(default_factory=list)

    @classmethod
    def from_task_info(cls, task_info: TaskInfo) -> "TaskInfoRead":
        return cls(
            config=TaskConfigRead.model_validate(task_info.config),
            instance=TaskInstanceRead.model_validate(task_info.instance) if task_info.instance else None,
            progress=task_info.task.get_progress() if task_info.task else 0.0,
            status_message=task_info.task.get_status_message() if task_info.task else "",
            log_lines=task_info.task.get_log_lines() if task_info.task else [],
        )
