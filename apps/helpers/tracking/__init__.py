"""Tracker plugin package.

Importing any submodule (in practice `Registry`) runs this file first, so the registry
is populated before the first `create_tracker()` call. Without it the package had no
`__init__.py` at all and nothing ever imported `iouTracker`, so every StreamWatcherTask
run died with "Unknown tracker 'IouTracker' (known: none registered)" — the test suite
missed it only because the tracker tests import the concrete module by name.

Same discovery shape as `imageProviders/__init__.py`: a new tracker registers by being
named `*Tracker.py`, with no import list to keep in sync.
"""

import importlib
import pathlib

tracker_dir = pathlib.Path(__file__).parent

for path in tracker_dir.glob("*Tracker.py"):
    importlib.import_module(f"{__name__}.{path.stem}")
