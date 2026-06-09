"""Tests for DatabaseMaintenanceTask file-reconcile logic.

The old FileEntry dataclass had a `type: str` field ("file"/"dir").
FileEntryRead (the model returned by the generated db_client) replaced it with
`is_dir: bool`.  DatabaseMaintenanceTask must filter on `not e.is_dir`, not on
`e.type == "file"` which raises AttributeError.
"""
import asyncio
from unittest.mock import MagicMock, patch

import db_client.models as db_models
import pytest

import apps.tasks_server.tasks  # noqa: F401  — registers all task types
from apps.tasks_server.tasks.Registry import task_registry


_MOCK_MANAGER = MagicMock()


def _make_task(params=None):
    return task_registry["DatabaseMaintenanceTask"](
        task_config_uuid="test-config-uuid",
        params=params or {},
        manager=_MOCK_MANAGER,
    )


def _images_api_mock(file_entries):
    mock = MagicMock()
    mock.list_images.return_value = MagicMock(files=file_entries)
    mock.reconcile_image.return_value = MagicMock(action="synced")
    mock.prune_missing_images.return_value = MagicMock(deleted_count=0)
    return mock


def _tags_api_mock():
    mock = MagicMock()
    mock.clear_orphan_generated_tags.return_value = MagicMock(deleted_count=0)
    return mock


async def test_task_counts_only_file_entries():
    """Only non-directory entries should be counted and reconciled."""
    entries = [
        db_models.FileEntryRead(name="cat.jpg", path="cat.jpg", isDir=False),
        db_models.FileEntryRead(name="dog.jpg", path="subdir/dog.jpg", isDir=False),
        db_models.FileEntryRead(name="subdir", path="subdir", isDir=True),
    ]

    images_api = _images_api_mock(entries)
    tags_api = _tags_api_mock()

    task = _make_task({"sync_jsons": False, "remove_deleted": False, "clear_orphan_tags": False})
    await task._init(task._params, None)

    with (
        patch("apps.tasks_server.tasks.DatabaseMaintenanceTask.ImagesApi", return_value=images_api),
        patch("apps.tasks_server.tasks.DatabaseMaintenanceTask.TagsApi", return_value=tags_api),
    ):
        result = await task._run()

    assert result["files_found"] == 2


async def test_task_empty_dir_returns_zero():
    """An empty listing should produce zero files_found without crashing."""
    images_api = _images_api_mock([])
    tags_api = _tags_api_mock()

    task = _make_task({"sync_jsons": False, "remove_deleted": False, "clear_orphan_tags": False})
    await task._init(task._params, None)

    with (
        patch("apps.tasks_server.tasks.DatabaseMaintenanceTask.ImagesApi", return_value=images_api),
        patch("apps.tasks_server.tasks.DatabaseMaintenanceTask.TagsApi", return_value=tags_api),
    ):
        result = await task._run()

    assert result["files_found"] == 0


async def test_task_directory_only_returns_zero():
    """A listing with only directories should yield files_found == 0."""
    entries = [
        db_models.FileEntryRead(name="a", path="a", isDir=True),
        db_models.FileEntryRead(name="b", path="b", isDir=True),
    ]

    images_api = _images_api_mock(entries)
    tags_api = _tags_api_mock()

    task = _make_task({"sync_jsons": False, "remove_deleted": False, "clear_orphan_tags": False})
    await task._init(task._params, None)

    with (
        patch("apps.tasks_server.tasks.DatabaseMaintenanceTask.ImagesApi", return_value=images_api),
        patch("apps.tasks_server.tasks.DatabaseMaintenanceTask.TagsApi", return_value=tags_api),
    ):
        result = await task._run()

    assert result["files_found"] == 0
