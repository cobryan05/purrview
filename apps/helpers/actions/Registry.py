"""Registry of available event Actions.

Same shape as the imageProviders and tracking registries, so `params_schema()` drives
the UI form automatically.
"""

from typing import Dict, Type

from .action import Action

action_registry: Dict[str, Type[Action]] = {}


def register_action(name: str = None):
    def wrapper(cls) -> Type[Action]:
        action_registry[name or cls.__name__] = cls
        return cls

    return wrapper


def create_action(kind: str, **params) -> Action:
    """Instantiate a registered action by its registry key."""
    cls = action_registry.get(kind)
    if cls is None:
        known = ", ".join(sorted(action_registry)) or "none registered"
        raise KeyError(f"Unknown action kind {kind!r} (known: {known})")
    return cls(**params)
