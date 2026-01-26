"""Task utilities for dealing with sources"""

import db_client
from db_client.api.sources_api import SourcesApi
from db_client.models.source_metadata import SourceMetadata

from apps.helpers.consts import JsonValues
from apps.helpers.imageProviders.imageProvider import ImageProvider
from apps.helpers.imageProviders.Registry import image_provider_registry


class SourceHelper:
    """Helper class for working with image sources and their providers."""

    def __init__(self, db_api_client: db_client.ApiClient):
        self._db_api_client = db_api_client
        self._sources_api = SourcesApi(db_api_client)
        self._sources_cache: dict[str, SourceMetadata] | None = None

    def _refresh_sources_cache(self) -> None:
        """Fetch sources from the API and update the cache."""
        api_response = self._sources_api.get_sources()
        if api_response.status != JsonValues.SUCCESS:
            raise Exception("Failed to retrieve source list")
        self._sources_cache = api_response.sources or {}

    def get_image_provider(self, source_uuid: str, refresh_cache: bool = False) -> ImageProvider:
        """
        Gets an image provider implementation from source UUID.

        Args:
            source_uuid: The UUID of the source to fetch.
            refresh_cache: If True, force re-fetching the sources list.
        """
        if refresh_cache or self._sources_cache is None:
            self._refresh_sources_cache()

        source_metadata = self._sources_cache.get(source_uuid)
        if not source_metadata:
            raise Exception(f"Source {source_uuid} not found")

        image_provider: ImageProvider = image_provider_registry[source_metadata.typename](**source_metadata.params)
        return image_provider
