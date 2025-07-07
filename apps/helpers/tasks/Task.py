from abc import ABC, abstractmethod
from typing import Any

class Task(ABC):
    def __init__(self, task_id: str, params: dict, resume_data: dict = None):
        self.task_id = task_id
        self.params = params
        self.resume_data = resume_data or {}

    @abstractmethod
    async def run(self) -> dict:
        """Execute the task. Should periodically update progress and save resume data."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the unique task type name (used in DB)."""
        pass

    def serialize_resume_data(self) -> dict:
        return self.resume_data

    def update_progress(self, percent: float):
        # Store progress in DB, signal UI updates (e.g. via WebSocket or polling)
        pass
