import importlib
import pathlib

task_dir = pathlib.Path(__file__).parent

# Discover all Python files ending in _task.py
for path in task_dir.glob("*Task.py"):
    if path.name == "__init__.py":
        continue

    module_name = f"{__name__}.{path.stem}"
    importlib.import_module(module_name)
