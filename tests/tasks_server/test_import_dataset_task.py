"""Importing a dataset: what the task does with what the parser hands it.

The parsing is covered in ``tests/helpers/test_dataset_import``; what is only
reachable here is the glue, and three parts of it fail silently:

* **A box must land on the image the server actually stored.** The upload
  endpoint uniquifies a colliding name, so the path that comes back is not
  always the one that went out. Attaching boxes to the sent name instead would
  put somebody else's annotations on an unrelated picture — and every count
  the run reports would still say it worked.
* **An in-place import must register its images.** A record only appears on
  its own once something writes annotations, so an unannotated dataset
  imported where it sits would be invisible with nothing raising.
* **An unmapped class must be counted, not dropped.** Silently losing a third
  of a dataset's boxes looks identical to importing it.

The API is faked; the filesystem is not. Which files exist, and what the
provider-shaped paths resolve to, is the thing being asserted.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from apps.helpers.db.types import UNVERIFIED_TAG_UUID, VERIFIED_TAG_UUID
from apps.tasks_server.tasks.ImportDatasetTask import ImportDatasetTask
from tests.conftest import make_png

_LABEL_CAT = "11111111-1111-1111-1111-111111111111"
_LABEL_DOG = "22222222-2222-2222-2222-222222222222"


class _FakeImagesApi:
    """Records what the task asked for, and answers the way db_server does."""

    def __init__(self):
        self.uploaded = []
        self.registered = []
        self.boxes = []
        #: Set to a path to simulate the server renaming around a collision.
        self.upload_path_override = None

    def upload_images(self, files, path, overwrite, source_uuid):
        name = files[0][0]
        self.uploaded.append((name, path))
        stored = self.upload_path_override or (f"{path}/{name}" if path else name)
        return SimpleNamespace(
            status="success",
            message=None,
            results=[
                SimpleNamespace(ok=True, path=stored, message=None, duplicate_of=None, uuid="img-1"),
            ],
        )

    def register_images(self, payload):
        self.registered.extend(payload.paths)
        return SimpleNamespace(
            status="success",
            registered=[SimpleNamespace(path=p, uuid=f"uuid-{p}") for p in payload.paths],
            failed=[],
        )

    def add_bbox(self, payload):
        self.boxes.append(payload)
        return SimpleNamespace(bbox_uuid="bbox-1")


class _FakeLabelsApi:
    def __init__(self, existing=()):
        self.existing = list(existing)
        self.created = []

    def list_labels(self):
        return SimpleNamespace(status="success", labels=list(self.existing))

    def add_label(self, payload):
        self.created.append(payload.name)
        uuid = f"created-{payload.name}"
        self.existing.append(SimpleNamespace(name=payload.name, uuid=uuid))
        return SimpleNamespace(label=SimpleNamespace(uuid=uuid), message=None)


class _FakeDatasetsApi:
    def __init__(self):
        self.created = []

    def create_dataset(self, payload):
        self.created.append(payload)
        return SimpleNamespace(dataset=SimpleNamespace(uuid="dataset-1"), message=None)


def _yolo_tree(root: Path, *, label_text="0 0.5 0.4 0.2 0.2\n", names="nc: 1\nnames:\n  0: cat\n") -> Path:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "images" / "a.png").write_bytes(make_png())
    (root / "labels" / "a.txt").write_text(label_text)
    (root / "data.yaml").write_text(names)
    return root


async def _run_task(files_root: Path, monkeypatch, *, apis=None, **params):
    monkeypatch.setenv("FILES_ROOT", str(files_root))
    manager = MagicMock()
    manager.update_task_heartbeat = AsyncMock()
    task = ImportDatasetTask(task_config_uuid="cfg", params=params, manager=manager)
    task.set_instance_uuid("33333333-3333-3333-3333-333333333333")

    apis = apis or {}
    images = apis.setdefault("images", _FakeImagesApi())
    labels = apis.setdefault("labels", _FakeLabelsApi())
    datasets = apis.setdefault("datasets", _FakeDatasetsApi())
    fakes = {"ImagesApi": images, "LabelsApi": labels, "DatasetsApi": datasets}
    monkeypatch.setattr(task, "_api", lambda cls: fakes[cls.__name__])

    await task._init(params, None)
    result = await task._run()
    return result, apis


class TestInPlaceImport:
    async def test_registers_the_images_where_they_already_are(self, tmp_path, monkeypatch):
        _yolo_tree(tmp_path / "ds")
        result, apis = await _run_task(
            tmp_path,
            monkeypatch,
            source_path="ds",
            copy_files=False,
            class_labels={"cat": _LABEL_CAT},
        )
        # Registered, not uploaded: the bytes were already in the library.
        assert apis["images"].registered == ["ds/images/a.png"]
        assert apis["images"].uploaded == []
        assert result["images_imported"] == 1

    async def test_writes_the_box_at_the_parsed_coordinates(self, tmp_path, monkeypatch):
        _yolo_tree(tmp_path / "ds")
        _, apis = await _run_task(
            tmp_path,
            monkeypatch,
            source_path="ds",
            copy_files=False,
            class_labels={"cat": _LABEL_CAT},
        )
        (box,) = apis["images"].boxes
        # YOLO's centre form (0.5, 0.4, 0.2, 0.2) is top-left (0.4, 0.3) here.
        assert box.x == pytest.approx(0.4)
        assert box.y == pytest.approx(0.3)
        assert box.width == pytest.approx(0.2)
        assert box.height == pytest.approx(0.2)
        assert box.image_path == "ds/images/a.png"
        assert box.label_uuid == _LABEL_CAT


class TestVerificationState:
    """Which verification tag an imported box carries.

    The two tags share ``exclusive_group="verification"``, so this is a choice
    between them and never both. Getting it wrong is invisible at import time
    and shows up much later — as a review queue full of boxes nobody needed to
    look at, or as unreviewed third-party labels silently counted as
    confirmed training data.
    """

    async def test_boxes_await_review_by_default(self, tmp_path, monkeypatch):
        _yolo_tree(tmp_path / "ds")
        _, apis = await _run_task(
            tmp_path,
            monkeypatch,
            source_path="ds",
            copy_files=False,
            class_labels={"cat": _LABEL_CAT},
        )
        assert apis["images"].boxes[0].tag_uuids == [UNVERIFIED_TAG_UUID]

    async def test_mark_verified_files_them_as_reviewed_instead(self, tmp_path, monkeypatch):
        _yolo_tree(tmp_path / "ds")
        _, apis = await _run_task(
            tmp_path,
            monkeypatch,
            source_path="ds",
            copy_files=False,
            class_labels={"cat": _LABEL_CAT},
            mark_verified=True,
        )
        tags = apis["images"].boxes[0].tag_uuids
        assert tags == [VERIFIED_TAG_UUID]
        # Both at once is the failure the exclusive group would reject.
        assert UNVERIFIED_TAG_UUID not in tags


class TestCopyImport:
    async def test_boxes_follow_the_path_the_server_stored_not_the_one_sent(self, tmp_path, monkeypatch):
        # The upload endpoint uniquifies a colliding name. A box attached to
        # the name that was sent would land on a different image entirely.
        _yolo_tree(tmp_path / "ds")
        images = _FakeImagesApi()
        images.upload_path_override = "imported/images/a_1.png"
        _, apis = await _run_task(
            tmp_path,
            monkeypatch,
            apis={"images": images},
            source_path="ds",
            copy_files=True,
            dest_path="imported",
            class_labels={"cat": _LABEL_CAT},
        )
        assert apis["images"].boxes[0].image_path == "imported/images/a_1.png"

    async def test_keeps_the_source_folder_shape_under_the_destination(self, tmp_path, monkeypatch):
        _yolo_tree(tmp_path / "ds")
        _, apis = await _run_task(
            tmp_path,
            monkeypatch,
            source_path="ds",
            copy_files=True,
            dest_path="imported",
            class_labels={"cat": _LABEL_CAT},
        )
        assert apis["images"].uploaded == [("a.png", "imported/images")]
        # Copying means nothing is registered separately — the upload endpoint
        # already created the row.
        assert apis["images"].registered == []


class TestClassMapping:
    async def test_an_unmapped_class_is_counted_not_silently_dropped(self, tmp_path, monkeypatch):
        _yolo_tree(
            tmp_path / "ds",
            label_text="0 0.5 0.4 0.2 0.2\n1 0.2 0.2 0.1 0.1\n",
            names="nc: 2\nnames:\n  0: cat\n  1: dog\n",
        )
        result, apis = await _run_task(
            tmp_path,
            monkeypatch,
            source_path="ds",
            copy_files=False,
            class_labels={"cat": _LABEL_CAT},
            create_missing_labels=False,
        )
        assert len(apis["images"].boxes) == 1
        assert result["boxes_imported"] == 1
        assert result["boxes_skipped"] == 1
        assert result["unmapped_classes"] == ["dog"]
        assert apis["labels"].created == []

    async def test_an_existing_label_of_the_same_name_is_reused_not_duplicated(self, tmp_path, monkeypatch):
        # Importing a second dataset that also calls it "cat" must not split
        # the class across two labels, which would halve it for training.
        _yolo_tree(tmp_path / "ds")
        labels = _FakeLabelsApi(existing=[SimpleNamespace(name="Cat", uuid=_LABEL_CAT)])
        _, apis = await _run_task(
            tmp_path,
            monkeypatch,
            apis={"labels": labels},
            source_path="ds",
            copy_files=False,
            create_missing_labels=True,
        )
        assert apis["labels"].created == []
        assert apis["images"].boxes[0].label_uuid == _LABEL_CAT

    async def test_creates_a_label_only_when_asked_to(self, tmp_path, monkeypatch):
        _yolo_tree(tmp_path / "ds")
        result, apis = await _run_task(
            tmp_path,
            monkeypatch,
            source_path="ds",
            copy_files=False,
            create_missing_labels=True,
        )
        assert apis["labels"].created == ["cat"]
        assert result["created_labels"] == {"cat": "created-cat"}


class TestDatasetCreation:
    async def test_no_dataset_is_created_without_a_name(self, tmp_path, monkeypatch):
        _yolo_tree(tmp_path / "ds")
        result, apis = await _run_task(
            tmp_path,
            monkeypatch,
            source_path="ds",
            copy_files=False,
            class_labels={"cat": _LABEL_CAT},
        )
        assert apis["datasets"].created == []
        assert result["dataset_uuid"] is None

    async def test_the_dataset_filters_on_the_folder_the_images_landed_in(self, tmp_path, monkeypatch):
        _yolo_tree(tmp_path / "ds")
        result, apis = await _run_task(
            tmp_path,
            monkeypatch,
            source_path="ds",
            copy_files=True,
            dest_path="imported",
            class_labels={"cat": _LABEL_CAT},
            dataset_name="Imported cats",
        )
        (payload,) = apis["datasets"].created
        assert payload.name == "Imported cats"
        assert payload.label_uuids == [_LABEL_CAT]
        (term,) = payload.filter_json.terms
        assert term.actual_instance.patterns == ["imported/"]
        assert result["dataset_uuid"] == "dataset-1"


class TestValidation:
    async def test_a_source_path_escaping_the_image_root_is_refused(self, tmp_path, monkeypatch):
        outside = tmp_path / "outside"
        _yolo_tree(outside / "ds")
        files_root = tmp_path / "files"
        files_root.mkdir()
        with pytest.raises(ValueError, match="outside the image root"):
            await _run_task(files_root, monkeypatch, source_path="../outside/ds", copy_files=False)

    async def test_a_staging_id_that_is_a_path_is_refused(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="staging id"):
            await _run_task(tmp_path, monkeypatch, staging_uuid="../../etc")

    async def test_neither_handle_is_refused(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="source_path or staging_uuid"):
            await _run_task(tmp_path, monkeypatch)
