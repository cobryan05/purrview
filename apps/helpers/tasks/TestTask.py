import asyncio
import os
from typing import Any

from .Registry import register_task
from .Task import Task


@register_task()
class ListFilesTask(Task):
    async def _init(self) -> None:
        """Run any initialization logic for the task."""
        self._directory = self._params.get("directory", ".")

    async def _run(self) -> dict[str, Any]:
        """Run the main logic of the task."""
        if not os.path.exists(self._directory) or not os.path.isdir(self._directory):
            raise ValueError(f"Directory does not exist: {self._directory}")

        files = os.listdir(self._directory)
        total = len(files)

        result = []
        for i, filename in enumerate(files):
            await asyncio.sleep(1)  # simulate some work
            result.append(filename)

            # simulate progress
            progress = (i + 1) / total * 100
            self._update_progress(progress)

        return {
            "directory": self._directory,
            "file_count": total,
            "files": result,
        }

    async def _deinit(self) -> None:
        pass