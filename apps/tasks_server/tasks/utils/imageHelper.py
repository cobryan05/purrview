"""Task utilities for dealing with images"""

import re
from typing import Optional

import db_client
from db_client.api.images_api import ImagesApi
from db_client.models.get_image_metadata_payload import GetImageMetadataPayload
from db_client.models.get_image_metadata_response import GetImageMetadataResponse
from db_client.models.update_metadata_payload import UpdateMetadataPayload
from db_client.models.update_metadata_response import UpdateMetadataResponse

from apps.helpers.consts import JsonValues

class ImageMetadata:
    # TODO: Remove these!
    pass


class ImageHelper:
    """Helper class for working with image sources and their providers."""

    def __init__(self, db_api_client: db_client.ApiClient):
        self._db_api_client = db_api_client
        self._images_api = ImagesApi(db_api_client)

    def get_image_metadata(self, image_path: str) -> Optional[ImageMetadata]:
        payload = GetImageMetadataPayload(image_path=image_path)
        response: GetImageMetadataResponse = self._images_api.get_image_metadata(payload)
        if response.metadata is None or response.status != JsonValues.SUCCESS:
            return None

        return ImageMetadata(**response.metadata.model_dump())

    def update_image_metadata(self, image_path: str, metadata: ImageMetadata) -> bool:
        input_boxes: list[db_client.BoundingBoxMetadataModel] = []
        for box in metadata.boxes:
            input_boxes.append(
                db_client.BoundingBoxMetadataModel(
                    uuid=box.uuid,
                    x=box.x,
                    y=box.y,
                    width=box.width,
                    height=box.height,
                    label_uuid=box.label_uuid
                )
            )
        payload = UpdateMetadataPayload(image_path=image_path, boxes=input_boxes)
        response: UpdateMetadataResponse = self._images_api.update_image_metadata(payload)
        return response.status == JsonValues.SUCCESS

    @staticmethod
    def sanitize_filename(name: str, replacement: str = "_") -> str:
        return re.sub(r'[<>:"/\.\\|?*\x00-\x1f]', replacement, name)
