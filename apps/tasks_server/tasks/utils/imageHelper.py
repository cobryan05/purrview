"""Task utilities for dealing with images"""

import re
from typing import Optional

import db_client
from db_client.api.images_api import ImagesApi
from db_client.models.get_image_metadata_payload import GetImageMetadataPayload
from db_client.models.get_image_metadata_response import GetImageMetadataResponse
from db_client.models.update_metadata_payload import UpdateMetadataPayload
from db_client.models.update_metadata_response import UpdateMetadataResponse
from db_client.models.image_metadata import ImageMetadata

from apps.helpers.consts import JsonValues


class ImageHelper:
    """Helper class for working with image sources and their providers."""

    def __init__(self, db_api_client: db_client.ApiClient):
        self._db_api_client = db_api_client
        self._images_api = ImagesApi(db_api_client)

    def get_image_metadata(self, image_path: str) -> Optional[ImageMetadata]:
        payload = GetImageMetadataPayload(image_path=image_path)
        response: GetImageMetadataResponse = self._images_api.get_image_metadata(payload)
        if response.status != JsonValues.SUCCESS:
            return None
        return response.metadata

    def update_image_metadata(self, image_path: str, metadata: ImageMetadata) -> bool:
        payload = UpdateMetadataPayload(image_path=image_path, boxes=metadata.boxes)
        response: UpdateMetadataResponse = self._images_api.update_image_metadata(payload)
        return response.status == JsonValues.SUCCESS

    @staticmethod
    def sanitize_filename(name: str, replacement: str = "_") -> str:
        return re.sub(r'[<>:"/\\|?*\x00-\x1f]', replacement, name)
