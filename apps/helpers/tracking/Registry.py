"""Registry of available Trackers.

Mirrors `imageProviders/Registry.py` deliberately: the established plugin shape is
what makes a tracker's `params_schema()` drive the UI with no UI change, which is the
whole payoff for adding ByteTrack later as an additive entry.
"""

from typing import Dict, Type

from .tracker import Tracker

tracker_registry: Dict[str, Type[Tracker]] = {}


def register_tracker(name: str = None):
    def wrapper(cls) -> Type[Tracker]:
        tracker_registry[name or cls.__name__] = cls
        return cls

    return wrapper


def create_tracker(typename: str, **params) -> Tracker:
    """Instantiate a registered tracker by name.

    Raises rather than falling back to a default: silently substituting a different
    tracker would change detection behaviour with no signal to the user.
    """
    cls = tracker_registry.get(typename)
    if cls is None:
        known = ", ".join(sorted(tracker_registry)) or "none registered"
        raise KeyError(f"Unknown tracker {typename!r} (known: {known})")
    return cls(**params)
