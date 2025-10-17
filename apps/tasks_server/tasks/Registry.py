"""Registry of available tasks"""
from .Task import Task
from typing import Type

task_registry: dict[str, Type[Task]] = {}
def register_task(name: str = None):
    def wrapper(cls) -> Task:
        task_registry[name or cls.__name__] = cls
        return cls
    return wrapper
