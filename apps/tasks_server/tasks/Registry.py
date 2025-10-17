"""Registry of available tasks"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Type

if TYPE_CHECKING:
    from .Task import Task

task_registry: dict[str, Type[Task]] = {}


def register_task(name: Optional[str] = None):
    def wrapper(cls) -> Task:
        task_registry[name or cls.__name__] = cls
        return cls

    return wrapper
