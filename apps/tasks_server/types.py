from dataclasses import dataclass
from typing import Optional

from apps.helpers.types import (
    DataclassMapper,
    TaskConfigMetadata,
    TaskConfigMetadataModel,
    TaskInstanceMetadata,
    TaskInstanceMetadataModel,
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
