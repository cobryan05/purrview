"""Trainer registry for the training server.

Register a new trainer with:

    @register_trainer("my_trainer_type")
    class MyTrainer(Trainer):
        ...
"""

from typing import Any

from .base import Trainer  # noqa: F401 — re-export for convenience

TRAINER_REGISTRY: dict[str, Trainer] = {}


def register_trainer(name: str):
    """Class decorator that instantiates and registers a Trainer under *name*."""

    def decorator(cls: type) -> type:
        TRAINER_REGISTRY[name] = cls()
        return cls

    return decorator
