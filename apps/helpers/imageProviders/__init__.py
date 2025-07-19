import importlib
import pathlib

# Get the directory this file is in
image_provider_dir = pathlib.Path(__file__).parent

# Discover all Python files ending in Provider.py
for path in image_provider_dir.glob("*Provider.py"):
    if path.name == "__init__.py":
        continue

    module_name = f"{__name__}.{path.stem}"
    importlib.import_module(module_name)
