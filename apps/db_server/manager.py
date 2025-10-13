"""Manages the database"""

import asyncio
import glob
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import fnmatch
import aiofiles

from apps.helpers.db.db_client import (
    BoundingBoxMetadata,
    DbClient,
    ImageMetadata,
    LabelData,
    LabelMetadata,
    SourceMetadata,
)
from apps.helpers.fileUtils import get_safe_path
from apps.helpers.imageProviders.Registry import image_provider_registry

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@dataclass
class FileEntry:
    name: str
    type: str  # "file" or "dir"
    path: str  # relative path from root


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
        image_id = await self._db_client.get_image_id_by_filename(img_path)
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
            logger.error(f"Failed to open file {rel_path}: {e}")
            return None
