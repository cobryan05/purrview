from abc import ABC, abstractmethod
from typing import Any

class Task(ABC):
    def __init__(self, id: int, params: dict, resume_data: dict = None):
        self.id = id
        self.params = params
        self.resume_data = resume_data or {}

    @abstractmethod
    async def run(self) -> dict:
        """Execute the task. Should periodically update progress and save resume data."""
        pass

    @abstractmethod
    def typename(self) -> str:
        """Return the unique task type name (used in DB)."""
        pass

    def serialize_resume_data(self) -> dict:
        return self.resume_data

    def update_progress(self, percent: float):
        # Store progress in DB, signal UI updates (e.g. via WebSocket or polling)
        pass
