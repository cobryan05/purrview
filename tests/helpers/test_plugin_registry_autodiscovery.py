"""Every plugin registry must populate itself from a bare package import.

`test_tracking.py` imports `apps.helpers.tracking.iouTracker` by name, which registers
IouTracker as a side effect — so the tracker registry looked fully populated under test
while being empty in production, where nothing imports that module. StreamWatcherTask
only ever imports `Registry.create_tracker`, and got:

    Unknown tracker 'IouTracker' (known: none registered)

The same hole existed in `apps.helpers.actions`: StreamWatcherTask imports
`recordClipAction` for its `PlaybackClipBackend` type, so `record_clip` registered by
accident, while `publish_mqtt` and `save_snapshot` did not.

These tests therefore import *only* the Registry module — the production path — and are
deliberately spelled that way. Importing the concrete plugin module here would restore
the exact blind spot they exist to close.
"""

import importlib
import pkgutil

import pytest


class TestTrackerRegistryAutodiscovery:
    def test_registry_import_alone_registers_iou_tracker(self):
        from apps.helpers.tracking.Registry import tracker_registry

        assert "IouTracker" in tracker_registry, (
            "importing the tracking Registry must be enough to populate it; "
            f"got {sorted(tracker_registry) or 'nothing'}"
        )

    def test_create_tracker_works_off_the_default_typename(self):
        """The DB default (`WatchProfile.tracker_typename`) must be constructible."""
        from apps.helpers.tracking.Registry import create_tracker

        tracker = create_tracker("IouTracker")
        assert tracker is not None

    def test_every_tracker_module_is_discovered(self):
        """A newly added tracker registers without editing any import list."""
        import apps.helpers.tracking as tracking
        from apps.helpers.tracking.Registry import tracker_registry

        modules = {name for _, name, _ in pkgutil.iter_modules(tracking.__path__) if name.endswith("Tracker")}
        assert modules, "no *Tracker.py modules found — has the naming convention changed?"
        for name in modules:
            importlib.import_module(f"apps.helpers.tracking.{name}")
        assert tracker_registry, "discovery imported no trackers"


class TestActionRegistryAutodiscovery:
    @pytest.mark.parametrize("kind", ["publish_mqtt", "record_clip", "save_snapshot"])
    def test_registry_import_alone_registers_every_action(self, kind):
        from apps.helpers.actions.Registry import action_registry

        assert kind in action_registry, (
            f"action {kind!r} is offered by the Watchers UI but is not registered; "
            f"got {sorted(action_registry) or 'nothing'}"
        )

    def test_ui_offered_action_kinds_all_resolve(self):
        """ACTION_KINDS in ui/watchers.js is the contract this registry has to meet."""
        from apps.helpers.actions.Registry import action_registry

        assert {"publish_mqtt", "record_clip", "save_snapshot"} <= set(action_registry)
