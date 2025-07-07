"""Registry of available tasks"""


task_registry = {}
def register_task(name: str = None):
    def wrapper(cls):
        task_registry[name or cls.__name__] = cls
        return cls
    return wrapper
