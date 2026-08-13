"""Watch action plugin package.

Mirrors `tracking/__init__.py` and `imageProviders/__init__.py`. Before this file
existed, `record_clip` registered only by luck — StreamWatcherTask imports
`recordClipAction` for its `PlaybackClipBackend` type — while `publish_mqtt` and
`save_snapshot`, both offered by the Watchers UI, resolved to
"Unknown action kind (known: record_clip)" the moment a rule fired.

A new action registers by being named `*Action.py`.
"""

import importlib
import pathlib

action_dir = pathlib.Path(__file__).parent

for path in action_dir.glob("*Action.py"):
    importlib.import_module(f"{__name__}.{path.stem}")
