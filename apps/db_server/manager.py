"""Manages the database"""

import asyncio
import logging
import sys
from typing import Any, Dict, List, Optional

from apps.helpers.db.db_client import DbClient, LabelData, LabelMetadata, SourceMetadata
from apps.helpers.imageProviders.Registry import image_provider_registry

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class Manager:
    """Manages interactions with the database"""

    POLLING_INTERVAL: float = 5.0  # Interval in seconds for periodic tasks

    def __init__(self, db_client: DbClient):
        """Initialize the Manager with a database client"""
        self._db_client = db_client
        self._task: Optional[asyncio.Task] = None  # Background task for periodic operations
        self._config = {}

    def start(self):
        """Start the periodic worker task, should be called from the event loop to run on"""
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.get_running_loop().create_task(self._worker_task())

    async def _init(self):
        """Initialization that should run on event loop"""
        pass

    async def _worker_task(self):
        """Periodic worker task that runs at regular intervals"""
        try:
            await self._init()
            while True:
                await asyncio.sleep(Manager.POLLING_INTERVAL)
        except asyncio.CancelledError:
            print("Periodic task was cancelled.")
        finally:
            print("Periodic task cleanup.")

    ################################################################################
    # LABELS API
    ################################################################################

    async def create_new_label(self, name: str, color: str, parent_uuid: Optional[str] = None) -> LabelMetadata:
        """
        Adds a new label to the database

        Returns:
            LabelMetaData: Label metadata added to database
        """
        ret = await self._db_client.add_label(name, color, parent_uuid=parent_uuid)
        # TODO CJO: await self._db_client.export_labels_from_db_to_json(str(self._labels_json))
        return ret

    async def delete_label(self, label_uuid: str) -> None:
        """
        Adds a new label to the database

        Returns:
            LabelMetaData: Label metadata added to database
        """
        # Check if the label has a parent
        child_uuids = await self._db_client.get_label_children(label_uuid)
        if child_uuids:
            raise ValueError(f"Cannot delete label '{label_uuid}' because it has child labels.")

        await self._db_client.delete_label(label_uuid)
        # TODO CJO: await self._db_client.export_labels_from_db_to_json(str(self._labels_json))

    async def update_label(self, label_uuid: str, name: Optional[str] = None, color: Optional[str] = None) -> None:
        """
        Update a label's name and/or color.

        Args:
            label_uuid (str): uuid of the label to update.
            name (Optional[str]): New name for the label.
            color (Optional[str]): New color for the label.
        """
        await self._db_client.update_label(label_uuid=label_uuid, name=name, color=color)
        # TODO CJO: await self._db_client.export_labels_from_db_to_json(str(self._labels_json))

    async def get_label_uuid_map(self) -> Dict[str, LabelData]:
        """
        Get a mapping of label UUIDs to their metadata.
        """
        flat_list: List[LabelMetadata] = await self._db_client.list_labels()
        uuid_to_node: Dict[str, LabelData] = {label.uuid: LabelData(metadata=label) for label in flat_list}

        for label in flat_list:
            node = uuid_to_node[label.uuid]
            if label.parent_uuid and label.parent_uuid in uuid_to_node:
                parent_node = uuid_to_node[label.parent_uuid]
                parent_node.children.append(node)

        return uuid_to_node

    async def get_labels(self) -> List[LabelData]:
        """
        Gets a list of all labels
        """
        return list((await self.get_label_uuid_map()).values())

    ################################################################################
    # SOURCES API
    ################################################################################

    async def get_avail_sources(self) -> List[SourceMetadata]:
        """
        List all sources managed by the Manager.
        """
        sources = await self._db_client.get_sources()
        return sources

    async def create_new_source(
        self, image_provider: str, provider_params: Dict[str, Any], source_name: str
    ) -> SourceMetadata:
        """
        Create a new source from an image source as a preset image_provider/params

        Args:
            image_provider (str): The image provider for this source
            params (Dict[str, Any]): The parameters to pass to the image provider
            source_name (str): The name given for the new source
        """
        if image_provider not in image_provider_registry:
            raise ValueError(f"Unknown image provider: {image_provider}")

        ret: SourceMetadata = await self._db_client.add_source(
            name=source_name, typename=image_provider, params=provider_params
        )
        return ret

    async def delete_sources(self, uuid_list: list[str]) -> None:
        """
        Delete sources by their UUIDs.

        Args:
            uuid_list (list[str]): List of source UUIDs to delete.
        """
        await self._db_client.delete_sources(uuid_list)

    async def update_source(
        self, source_uuid: str, image_provider: str, provider_params: Dict[str, Any], source_name: str
    ) -> SourceMetadata:
        """
        Updates sources by their UUIDs.

        Args:
            uuid_list (list[str]): List of source UUIDs to delete.
        """
        return await self._db_client.update_source(
            source_uuid=source_uuid,
            name=source_name,
            typename=image_provider,
            params=provider_params,
        )

    async def list_avail_image_providers(self) -> List[str]:
        """
        List all available image providers.
        """
        return list(image_provider_registry.keys())

    async def get_image_provider_schema(self, image_provider: str) -> Dict[str, Any]:
        """
        Get the schema for a specific image provider.

        Args:
            image_provider (str): The name of the image provider.

        Returns:
            Dict[str, Any]: The schema for the image provider.
        """
        if image_provider not in image_provider_registry:
            raise ValueError(f"Unknown image provider: {image_provider}")

        return image_provider_registry[image_provider].params_schema()
