"""Manages the database"""

import asyncio
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import fnmatch
import aiofiles

from apps.helpers.db.types import ClassData, FileEntry

from apps.helpers.db.types import BoundingBoxMetadata, ImageMetadata, SourceMetadata
from apps.helpers.db.db_client import (
    DbClient,
    ClassMetadata,
    TagMetadata,
    TagKinds,
)
from apps.helpers.fileUtils import get_safe_path
from apps.helpers.imageProviders.Registry import image_provider_registry

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class Manager:
    """Manages interactions with the database"""

    POLLING_INTERVAL: float = 5.0  # Interval in seconds for periodic tasks

    def __init__(self, db_client: DbClient, files_root: Path):
        """Initialize the Manager with a database client"""
        self._db_client = db_client
        self._files_root: Path = files_root
        self._task: Optional[asyncio.Task] = None  # Background task for periodic operations
        self._config = {}

    def start(self):
        """Start the periodic worker task, should be called from the event loop to run on"""
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.get_running_loop().create_task(self._worker_task())

    async def _init(self):
        """Initialization that should run on event loop"""
        await self._db_client.init_db()

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
    # CLASSES API
    ################################################################################

    async def create_new_class(self, name: str, color: str, parent_uuid: Optional[str] = None) -> ClassMetadata:
        """
        Adds a new class to the database

        Returns:
            ClassMetaData: Class metadata added to database
        """
        ret = await self._db_client.add_class(name, color, parent_uuid=parent_uuid)
        # TODO CJO: await self._db_client.export_labels_from_db_to_json(str(self._labels_json))
        return ret

    async def delete_class(self, class_uuid: str) -> None:
        """
        Deletes a class from the database

        Returns:
            ClassMetaData: Class metadata deleted from database
        """
        # Check if the class has a parent
        child_uuids = await self._db_client.get_class_children(class_uuid)
        if child_uuids:
            raise ValueError(f"Cannot delete class '{class_uuid}' because it has child classes.")

        await self._db_client.delete_class(class_uuid)
        # TODO CJO: await self._db_client.export_labels_from_db_to_json(str(self._labels_json))

    async def update_class(self, class_uuid: str, name: Optional[str] = None, color: Optional[str] = None) -> None:
        """
        Update a class's name and/or color.

        Args:
            class_uuid (str): uuid of the class to update.
            name (Optional[str]): New name for the class.
            color (Optional[str]): New color for the class.
        """
        await self._db_client.update_class(class_uuid=class_uuid, name=name, color=color)
        # TODO CJO: await self._db_client.export_labels_from_db_to_json(str(self._labels_json))

    async def get_class_uuid_map(self) -> Dict[str, ClassData]:
        """
        Get a mapping of class UUIDs to their metadata.
        """
        flat_list: List[ClassMetadata] = await self._db_client.list_classes()
        uuid_to_node: Dict[str, ClassData] = {x.uuid: ClassData(metadata=x) for x in flat_list}

        # Set up parent-child relationships
        for node in uuid_to_node.values():
            parent_uuid = node.metadata.parent_uuid if node.metadata else None
            parent_node = uuid_to_node.get(parent_uuid) if parent_uuid else None
            if parent_node:
                parent_node.children.append(node)

        return uuid_to_node

    async def get_classes(self) -> List[ClassData]:
        """
        Gets a list of all classes
        """
        return list((await self.get_class_uuid_map()).values())

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

    ################################################################################
    # TAGS API
    ################################################################################

    async def create_new_tag(
        self,
        name: str,
        color: str,
        protected: bool = False,
        kind: str = TagKinds.GENERIC,
        exclusive_group: Optional[str] = None,
    ) -> TagMetadata:
        """
        Adds a new class to the database

        Returns:
            TagMetadata: Tag metadata added to database
        """
        ret = await self._db_client.add_tag(
            name=name, color=color, protected=protected, kind=kind, exclusive_group=exclusive_group
        )
        # TODO CJO: await self._db_client.export_labels_from_db_to_json(str(self._labels_json))
        return ret

    async def delete_tag(self, tag_uuid: str) -> None:
        """
        Deletes a tag from the database

        Returns:
            TagMetadata: Tag metadata deleted from database
        """
        await self._db_client.delete_tag(tag_uuid)
        # TODO CJO: await self._db_client.export_labels_from_db_to_json(str(self._labels_json))

    async def update_tag(
        self,
        tag_uuid: str,
        name: Optional[str] = None,
        color: Optional[str] = None,
        protected: bool = False,
        kind: Optional[str] = None,
        exclusive_group: Optional[str] = None,
    ) -> None:
        """
        Update a tag's name and/or color.

        Args:
            tag_uuid (str): uuid of the tag to update.
            name (Optional[str]): New name for the tag.
            color (Optional[str]): New color for the tag.
            protected (bool): Whether the tag is protected.
            kind (Optional[str]): The kind of tag.
            exclusive_group (Optional[str]): The exclusive group of the tag.
        """
        await self._db_client.update_tag(
            tag_uuid=tag_uuid,
            name=name,
            color=color,
            protected=protected,
            kind=kind,
            exclusive_group=exclusive_group,
        )
        # TODO CJO: await self._db_client.export_labels_from_db_to_json(str(self._labels_json))

    async def get_tag_uuid_map(self) -> Dict[str, TagMetadata]:
        """
        Get a mapping of tag UUIDs to their metadata.
        """
        flat_list: List[TagMetadata] = await self._db_client.list_tags()
        uuid_to_node: Dict[str, TagMetadata] = {tag.uuid: tag for tag in flat_list}
        return uuid_to_node

    async def get_tags(self) -> List[TagMetadata]:
        """
        Gets a list of all tags
        """
        return list((await self.get_tag_uuid_map()).values())

    ################################################################################
    # BBOXES API
    ################################################################################

    async def get_bboxes_info(self, bbox_uuids: List[str]) -> List[BoundingBoxMetadata]:
        """
        Get metadata for image by resolving image ID from filename.

        Args:
            bbox_uuids (List[str]): List of bounding box UUIDs.

        Returns:
            List[BoundingBoxMetadata]: Metadata or empty list
        """
        bboxes_info = await self._db_client.get_bboxes_info(bbox_uuids)
        return bboxes_info

    ################################################################################
    # IMAGES API
    ################################################################################

    async def get_image_metadata(self, image_rel_path: str) -> Optional[ImageMetadata]:
        """
        Get metadata for image by resolving image ID from filename.

        Args:
            image_rel_path (str): Relative image path (filename).

        Returns:
            Optional[ImageMetadata]: Full metadata object or None.
        """
        safe_path = get_safe_path(self._files_root, image_rel_path)
        if not safe_path or not safe_path.exists():
            return None
        img_path = str(safe_path)
        image_id = await self._db_client.get_image_uuid_by_filename(img_path)
        if image_id is None:
            image_id = await self._db_client.add_image(img_path)
            await self._db_client.read_image_metadata_json_to_db(img_path)
        return await self._db_client.read_image_metadata_from_db(image_id)

    async def update_image_metadata(self, image_rel_path: str, metadata: ImageMetadata) -> None:
        """
        Update metadata for image identified by filename.

        Args:
            image_rel_path (str): Relative image path (filename).
            metadata (ImageMetadata): New metadata to write.
        """
        safe_path = get_safe_path(self._files_root, image_rel_path)
        if safe_path is None or not safe_path.exists():
            logger.warning(f"Path not found or inaccessible: {image_rel_path}")
            return
        img_path = str(safe_path)
        image_id = await self._db_client.add_image(img_path)
        await self._db_client.write_image_metadata_to_db(image_id, metadata)
        await self._db_client.save_image_metadata_db_to_json(img_path)

    async def list_files(
        self,
        rel_path: str = "/",
        patterns: str | List[str] = "*",
        exclude_patterns: str | List[str] = "*.json",
        recursive: bool = False,
    ) -> List[FileEntry]:
        """
        List all files and directories in the given relative path under the root directory.

        Args:
            rel_path (str): Relative path from the root directory (default: "/").
            include_patterns (str or List[str]): Glob pattern(s) to include (default: "*").
            exclude_patterns (str or List[str]): Glob pattern(s) to exclude (default: "").
            recursive (bool): Whether to list files recursively (default: False).

        Returns:
            List[FileEntry]: List of FileEntry instances.
        """
        if not self._files_root or not os.path.isdir(self._files_root):
            logger.warning("Root directory is not set or does not exist.")
            return []

        safe_path = get_safe_path(self._files_root, rel_path)

        if not os.path.exists(safe_path) or not os.path.isdir(safe_path):
            logger.warning(f"Directory not found: {safe_path}")
            return []

        # Normalize patterns
        if isinstance(patterns, str):
            patterns = [patterns]
        if isinstance(exclude_patterns, str):
            exclude_patterns = [exclude_patterns]

        # Get matching files
        glob_path = os.path.join(safe_path, "**", "*") if recursive else os.path.join(safe_path, "*")
        matched_paths = glob.glob(glob_path, recursive=recursive)

        entries: List[FileEntry] = []
        for full_path in sorted(matched_paths):
            basename = os.path.basename(full_path)

            # Skip hidden files/folders
            if basename.startswith("."):
                continue

            # Skip if outside root
            if not os.path.commonpath([self._files_root, full_path]).startswith(str(self._files_root)):
                continue

            # Apply include/exclude pattern matching
            rel_entry_path = os.path.relpath(full_path, self._files_root)
            matched = any(fnmatch.fnmatch(rel_entry_path, pat) for pat in patterns)
            excluded = any(fnmatch.fnmatch(rel_entry_path, pat) for pat in exclude_patterns)

            if matched and not excluded:
                entry_type = "dir" if os.path.isdir(full_path) else "file"
                entries.append(FileEntry(name=basename, type=entry_type, path=rel_entry_path))

        return entries

    async def open_file(self, rel_path: str) -> Optional[Tuple[aiofiles.threadpool.binary.AsyncBufferedReader, str]]:
        """
        Securely open a file under the root and return an aiofiles stream and filename.

        Args:
            rel_path (str): Relative path under the image root.

        Returns:
            Tuple[aiofiles.AsyncBufferedReader, str] or None
        """
        if not self._files_root or not os.path.isdir(self._files_root):
            logger.warning("Root directory not set or invalid")
            return None

        try:
            abs_path = os.path.normpath(os.path.join(self._files_root, rel_path.lstrip("/")))

            # Prevent directory traversal
            if not abs_path.startswith(str(self._files_root)):
                logger.warning(f"Blocked directory traversal attempt: {rel_path}")
                return None

            if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
                logger.warning(f"File not found: {abs_path}")
                return None

            f = await aiofiles.open(abs_path, mode="rb")
            filename = os.path.basename(abs_path)
            return f, filename

        except Exception as e:
            logger.error(f"Failed to open file {rel_path}: {e}", exc_info=True)
            return None
