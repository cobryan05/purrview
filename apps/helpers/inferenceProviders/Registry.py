"""Registry of available InferenceProviders, keyed by model_family."""

from typing import Type

from .inferenceProvider import InferenceProvider

inference_provider_registry: dict[str, Type[InferenceProvider]] = {}


def register_inference_provider(name: str):
    def wrapper(cls) -> Type[InferenceProvider]:
        inference_provider_registry[name] = cls
        return cls

    return wrapper
