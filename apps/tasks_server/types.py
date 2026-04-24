from dataclasses import dataclass
from typing import Optional

from apps.helpers.db.types import (
    TaskInstanceMetadata,
    TaskInstanceMetadataModel,
    DataclassMapper,
    TaskConfigMetadata,
    TaskConfigMetadataModel,
)
from apps.tasks_server.tasks.Task import Task


@dataclass
class TaskInfo:
    config_metadata: Optional[TaskConfigMetadata] = None
    task_metadata: Optional[TaskInstanceMetadata] = None
    task: Optional[Task] = None


class TaskInfoModel(DataclassMapper):
    config_metadata: TaskConfigMetadataModel
    task_metadata: Optional[TaskInstanceMetadataModel] = None
