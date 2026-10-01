"""The dataset-import HTTP surface: staging, inspection, and registration.

Three endpoints, each with one thing that has to hold:

* ``/api/datasets/import/archive`` must unpack outside FILES_ROOT. A dataset
  the user is still deciding about, sitting in the image library, would be
  picked up by directory scans and collector globs before they agreed to
  import any of it.
* ``/api/datasets/import/detect`` must refuse a path that escapes FILES_ROOT,
  and must report the format it found.
* ``/api/images/register`` must create rows for files already on disk, because
  without it an unannotated in-place import is invisible — a record only
  appears on its own once something writes annotations.

The format parsing itself is covered in ``tests/helpers/test_dataset_import``;
what is asserted here is the boundary those functions are reached through.
"""

import io
import zipfile

import pytest

from tests.conftest import make_png

# ⚠️ Request and response bodies are camelCase: ApiPayload/ApiModel carry
# alias_generator=to_camel, so staging_uuid crosses the wire as stagingUuid.


@pytest.fixture(autouse=True)
def staging_dir(manager, tmp_path):
    """Point staging at the test's own tree.

    The default is under CACHE_DIR, which is a real shared volume: left alone,
    every worker in a parallel run would stage into the same directory and the
    sweep would delete another test's upload mid-assertion.
    """
    manager.DATASET_STAGING_DIR = tmp_path / "staging"
    return manager.DATASET_STAGING_DIR


def _zip_bytes(entries: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def _yolo_zip() -> bytes:
    return _zip_bytes(
        {
            "ds/data.yaml": "nc: 1\nnames:\n  0: cat\n",
            "ds/images/train/a.png": make_png(),
            "ds/labels/train/a.txt": "0 0.5 0.5 0.2 0.2\n",
        }
    )


async def _stage(http_client, data: bytes, name: str = "ds.zip"):
    return await http_client.post(
        "/api/datasets/import/archive",
        files=[("file", (name, data, "application/zip"))],
    )


async def _detect(http_client, **body):
    return await http_client.post("/api/datasets/import/detect", json=body)


class TestImportFormats:
    async def test_lists_the_formats_the_importer_reads(self, http_client):
        r = await http_client.get("/api/datasets/import-formats")
        assert r.status_code == 200
        ids = {f["id"] for f in r.json()["formats"]}
        # "images" is the one that has no export counterpart, so its presence
        # is what distinguishes this list from the export list.
        assert {"yolo", "coco", "voc", "images"} == ids


class TestStageArchive:
    async def test_unpacks_outside_the_image_library(self, http_client, files_root):
        r = await _stage(http_client, _yolo_zip())
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["stagingUuid"]
        # The whole point of staging: nothing appears under FILES_ROOT until
        # the import task runs.
        assert list(files_root.iterdir()) == []

    async def test_a_staged_archive_can_then_be_inspected(self, http_client):
        staging_uuid = (await _stage(http_client, _yolo_zip())).json()["stagingUuid"]
        detected = (await _detect(http_client, stagingUuid=staging_uuid)).json()
        assert detected["status"] == "success"
        assert detected["format"] == "yolo"
        assert detected["imageCount"] == 1
        assert detected["annotationCount"] == 1
        assert detected["classNames"] == ["cat"]

    async def test_a_file_that_is_not_an_archive_is_refused(self, http_client):
        r = await _stage(http_client, b"just some text", name="notes.txt")
        assert r.json()["status"] == "failure"

    async def test_a_traversing_entry_is_refused_and_leaves_nothing_staged(self, http_client, manager):
        r = await _stage(http_client, _zip_bytes({"../escaped.png": make_png()}))
        assert r.json()["status"] == "failure"
        staged = manager.DATASET_STAGING_DIR
        assert not staged.exists() or list(staged.iterdir()) == []

    async def test_discarding_makes_the_upload_unreachable(self, http_client):
        staging_uuid = (await _stage(http_client, _yolo_zip())).json()["stagingUuid"]
        assert (await http_client.post("/api/datasets/import/discard", json={"stagingUuid": staging_uuid})).json()[
            "status"
        ] == "success"
        # Gone, not merely empty: inspection must fail rather than report a
        # zero-image dataset the user would read as "the archive was empty".
        assert (await _detect(http_client, stagingUuid=staging_uuid)).json()["status"] == "failure"

    async def test_discarding_an_unknown_id_is_not_an_error(self, http_client):
        # Called from the wizard's close handler, which cannot know whether the
        # id it is holding was already consumed by a started import.
        r = await http_client.post("/api/datasets/import/discard", json={"stagingUuid": "not-a-uuid"})
        assert r.json()["status"] == "success"


class TestDetectPath:
    async def test_reads_a_folder_under_the_image_root(self, http_client, files_root):
        (files_root / "shots").mkdir()
        (files_root / "shots" / "a.png").write_bytes(make_png())
        (files_root / "shots" / "b.png").write_bytes(make_png())
        body = (await _detect(http_client, path="shots")).json()
        assert body["format"] == "images"
        assert body["imageCount"] == 2
        # Reported relative to FILES_ROOT, so it is a handle the caller can
        # send back rather than a container path.
        assert body["root"] == "shots"

    async def test_a_path_escaping_the_image_root_is_refused(self, http_client, files_root, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "a.png").write_bytes(make_png())
        body = (await _detect(http_client, path="../outside")).json()
        assert body["status"] == "failure"
        assert body["imageCount"] == 0

    async def test_an_explicit_format_overrides_the_guess(self, http_client, files_root):
        root = files_root / "ds"
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        (root / "images" / "a.png").write_bytes(make_png())
        (root / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        assert (await _detect(http_client, path="ds")).json()["annotationCount"] == 1
        assert (await _detect(http_client, path="ds", format="images")).json()["annotationCount"] == 0

    async def test_an_unreadable_coco_file_is_a_warning_not_a_failure(self, http_client, files_root):
        root = files_root / "ds"
        root.mkdir()
        (root / "a.png").write_bytes(make_png())
        # Shaped like COCO to the detector's pre-filter, but not parseable.
        (root / "broken.json").write_text('{"images": [], "annotations": [ truncated')
        body = (await _detect(http_client, path="ds", format="coco")).json()
        assert body["status"] == "success"
        assert body["imageCount"] == 1
        assert body["warnings"]

    async def test_a_folder_with_no_images_is_refused(self, http_client, files_root):
        (files_root / "empty").mkdir()
        assert (await _detect(http_client, path="empty")).json()["status"] == "failure"


class TestRegisterImages:
    async def test_registers_a_file_already_on_disk(self, http_client, files_root, db_client):
        (files_root / "ds").mkdir()
        (files_root / "ds" / "a.png").write_bytes(make_png())

        r = await http_client.post("/api/images/register", json={"paths": ["ds/a.png"]})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert [row["path"] for row in body["registered"]] == ["ds/a.png"]
        assert body["registered"][0]["uuid"]

        # The reason the endpoint exists: the image is now findable without
        # anything having written an annotation to it.
        listed = await http_client.post("/api/images/list-by-directory", json={"directories": ["ds"]})
        assert [img["path"] for img in listed.json()["images"]] == ["ds/a.png"]

    async def test_registering_twice_does_not_duplicate_the_row(self, http_client, files_root):
        (files_root / "a.png").write_bytes(make_png())
        first = await http_client.post("/api/images/register", json={"paths": ["a.png"]})
        second = await http_client.post("/api/images/register", json={"paths": ["a.png"]})
        assert first.json()["registered"][0]["uuid"] == second.json()["registered"][0]["uuid"]

    async def test_a_missing_file_is_reported_without_failing_the_batch(self, http_client, files_root):
        (files_root / "real.png").write_bytes(make_png())
        r = await http_client.post("/api/images/register", json={"paths": ["real.png", "ghost.png"]})
        body = r.json()
        assert [row["path"] for row in body["registered"]] == ["real.png"]
        assert [row["path"] for row in body["failed"]] == ["ghost.png"]

    async def test_a_path_escaping_the_image_root_is_reported_not_registered(self, http_client, files_root, tmp_path):
        outside = tmp_path / "secret.png"
        outside.write_bytes(make_png())
        r = await http_client.post("/api/images/register", json={"paths": ["../secret.png"]})
        body = r.json()
        assert body["registered"] == []
        assert [row["path"] for row in body["failed"]] == ["../secret.png"]
