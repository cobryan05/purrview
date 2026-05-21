"""Registry of available ImageProviders"""
from typing import Type

from .imageProvider import ImageProvider

image_provider_registry: dict[str, Type[ImageProvider]] = {}
def register_image_provider(name: str = None):
    def wrapper(cls) -> ImageProvider:
        image_provider_registry[name or cls.__name__] = cls
        return cls
    return wrapper
