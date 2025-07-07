import os
import asyncio
from typing import Dict
from .Task import Task  # assuming your base class is in base.py
from .Registry import register_task

@register_task(name="list-files")
class ListFilesTask(Task):
    def name(self) -> str:
        return "ListFiles"

    async def run(self) -> Dict:
        directory = self.params.get("directory", ".")
        if not os.path.exists(directory) or not os.path.isdir(directory):
            raise ValueError(f"Directory does not exist: {directory}")

        files = os.listdir(directory)
        total = len(files)

        result = []
        for i, filename in enumerate(files):
            await asyncio.sleep(0.1)  # simulate some work
            result.append(filename)

            # simulate progress
            progress = (i + 1) / total * 100
            self.update_progress(progress)

            # optionally save resume data
            self.resume_data["last_index"] = i

        return {
            "directory": directory,
            "file_count": total,
            "files": result,
        }
