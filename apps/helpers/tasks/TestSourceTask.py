import asyncio
import os
from typing import Any, Optional

from .Registry import register_task
from .Task import Task


@register_task()
class TestSourceTask(Task):
    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        """Run any initialization logic for the task."""
        self._directory = self._params.get("directory", ".")
        self._files_processed: int = 0

    async def _run(self) -> dict[str, Any]:
        """Run the main logic of the task."""
        if not os.path.exists(self._directory) or not os.path.isdir(self._directory):
            raise ValueError(f"Directory does not exist: {self._directory}")

        files = await asyncio.to_thread(os.listdir, self._directory)
        total = len(files)
        status = "success"
        result = []

        for i, filename in enumerate(files):
            self._files_processed += 1
            if self._cancel_flag.is_set():
                status = "canceled"
                break

            await asyncio.sleep(1)  # simulate some work
            result.append(filename)

            # simulate progress
            progress = (i + 1) / total * 100
            self._update_progress(progress)

            if i % 10 == 0 or self._data_req_flag.is_set():
                self._update_resume_data()

        self._update_resume_data()
        return {"directory": self._directory, "file_count": total, "files": result, "status": status}

    async def _deinit(self) -> None:
        pass

    def _update_resume_data(self) -> None:
        self._resume_data = {"files_processed": self._files_processed}
        self._data_ready_flag.set()

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        """
        Return a schema describing the parameters for this Task.
        Each key is a parameter name, value is a dict with:
            - type: str
            - required: bool
            - default: Any (optional)
            - help: str (optional)
            - options: list (optional, for enums)
            - schema: dict (optional, for nested objects)
        """
        return {
            "source_uuid": {"type": "string", "required": True, "help": "Source UUID to test"},
            "delete_source": {"type": "boolean", "required": True, "help": "Delete source when done"},
        }
