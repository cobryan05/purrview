from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel

from apps.helpers.db.types import TaskConfig, TaskConfigRead, TaskInstance, TaskInstanceRead
from apps.tasks_server.tasks.Task import Task


@dataclass
class TaskInfo:
    config: TaskConfig
    instance: Optional[TaskInstance] = None
    task: Optional[Task] = None


class TaskInfoRead(BaseModel):
    config: TaskConfigRead
    instance: Optional[TaskInstanceRead] = None

    @classmethod
    def from_task_info(cls, task_info: TaskInfo) -> "TaskInfoRead":
        return cls(
            config=TaskConfigRead.model_validate(task_info.config),
            instance=TaskInstanceRead.model_validate(task_info.instance) if task_info.instance else None,
        )
