"""ImportDatasetTask — bring an existing annotated dataset into the library.

The read-side counterpart to the dataset export. ``apps.helpers.datasetImport``
has already decided what the directory holds; this task is the part that
touches the database: resolving class names to labels, getting the pixels into
FILES_ROOT, writing the boxes, and optionally defining a Dataset over the
result.

**Two ways in, and only one of them copies.** A folder already under FILES_ROOT
is registered where it sits — copying a 40 GB dataset onto the same volume to
give it a second name helps nobody. An uploaded archive was unpacked outside
FILES_ROOT (so a half-inspected dataset never appears in the library), and is
therefore always copied in.

**Class names are mapped, never guessed silently.** The wizard sends an
explicit name→label map. A name the user left unmapped creates a label only if
they asked for that; otherwise its boxes are counted as skipped and reported,
because quietly inventing project labels from a stranger's ``data.yaml`` is how
a label tree becomes unusable.

**Whether imported boxes count as reviewed is the caller's call.** A scraped
or auto-labelled export is a proposal and arrives UNVERIFIED, which is what the
review pipeline keys off. A dataset somebody actually annotated by hand is
already reviewed, and re-reviewing thousands of correct boxes to say so is
busywork — `mark_verified` files those as VERIFIED instead. The two tags are
mutually exclusive per bbox (`exclusive_group="verification"`), so exactly one
is applied.

Re-running is safe but not free: the upload path reports near-duplicates and
imports them anyway (matching `/api/images/upload`), so a second run of the
same dataset doubles it rather than merging. The wizard says so before starting.
"""

from __future__ import annotations

import asyncio
import colorsys
import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from db_client.api.datasets_api import DatasetsApi
from db_client.api.images_api import ImagesApi
from db_client.api.labels_api import LabelsApi

from apps import CACHE_DIR
from apps.helpers.consts import JsonValues, TaskAccent
from apps.helpers.datasetImport import (
    IMPORT_FORMAT_IDS,
    DatasetImportError,
    ImportedImage,
    detect,
    iter_images,
)
from apps.helpers.db.types import UNVERIFIED_TAG_UUID, VERIFIED_TAG_UUID
from apps.helpers.logUtils import LOG_LEVEL

from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)

#: Where the archive-upload path unpacks to. Mirrors
#: `db_server.Manager.DATASET_STAGING_DIR`; the two services share the volume,
#: so the task reads the staged tree directly rather than streaming it back
#: through the API it was uploaded to.
_STAGING_DIR = CACHE_DIR / "dataset_imports"

#: Default destination under FILES_ROOT for a copied import.
DEFAULT_IMPORT_DIR = "imported"

#: Images per progress/heartbeat update. Small enough that a stalled import is
#: visible, large enough that a 50k-image run is not mostly bookkeeping.
_PROGRESS_EVERY = 25

#: How many failures are quoted in the result. The rest are counted.
_MAX_REPORTED_FAILURES = 10


def _color_for(name: str) -> str:
    """A stable, well-spread colour for an auto-created label.

    Deterministic in the name so re-importing the same dataset into a fresh
    database produces the same palette, and so two labels created in the same
    run never collide by accident of ordering.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.85)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


@register_task()
class ImportDatasetTask(Task):
    # Gathering data — same family as CollectTrainingImages and the blip harvest.
    CARD_ACCENT = TaskAccent.GREEN

    # Started from the Import Dataset wizard, which has already resolved the
    # class map and the destination. There is nothing useful to fill in by hand.
    SYSTEM_TASK = True

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "meta": {
                "order": [
                    "source_path",
                    "staging_uuid",
                    "format",
                    "copy_files",
                    "dest_path",
                    "class_labels",
                    "create_missing_labels",
                    "mark_verified",
                    "dataset_name",
                ]
            },
            "source_path": {
                "type": "str",
                "default": "",
                "help": "Folder under the image root to import from. Leave empty when importing an upload.",
            },
            "staging_uuid": {
                "type": "str",
                "default": "",
                "help": "Id of an uploaded archive already unpacked into staging.",
            },
            "format": {
                "type": "str",
                "default": "",
                "help": "yolo, coco, voc or images. Empty detects the layout.",
            },
            "copy_files": {
                "type": "bool",
                "default": False,
                "help": (
                    "Copy the images into the destination folder. Off imports them where they "
                    "already are, which is only possible for a folder under the image root."
                ),
            },
            "dest_path": {
                "type": "str",
                "default": DEFAULT_IMPORT_DIR,
                "help": "Folder under the image root that copied images are written to.",
            },
            "class_labels": {
                "type": "dict",
                "default": {},
                "help": "Map of the dataset's class names to label uuids in this project.",
            },
            "create_missing_labels": {
                "type": "bool",
                "default": True,
                "help": "Create a top-level label for any class name the map does not cover.",
            },
            "mark_verified": {
                "type": "bool",
                "default": False,
                "help": (
                    "File imported boxes as reviewed rather than awaiting review. On for a dataset "
                    "a person annotated; off for anything auto-labelled or of unknown provenance."
                ),
            },
            "dataset_name": {
                "type": "str",
                "default": "",
                "help": "Create a Dataset over the imported folder under this name. Empty creates none.",
            },
        }

    async def _init(self, params: Dict[str, Any], resume_data: Optional[Dict[str, Any]]) -> None:
        # Params arrive raw; everything below is validated here rather than trusted.
        self._source_path: str = str(params.get("source_path") or "").strip()
        self._staging_uuid: str = str(params.get("staging_uuid") or "").strip()
        if not self._source_path and not self._staging_uuid:
            raise ValueError("one of source_path or staging_uuid is required")

        fmt = str(params.get("format") or "").strip()
        if fmt and fmt not in IMPORT_FORMAT_IDS:
            raise ValueError(f"Unknown format {fmt!r}; expected one of {sorted(IMPORT_FORMAT_IDS)}")
        self._format: Optional[str] = fmt or None

        self._files_root = Path(os.environ.get("FILES_ROOT", "/app/image_datasets"))
        self._root: Path = self._resolve_root()

        # A staged upload lives outside FILES_ROOT, so there is nothing to
        # register in place; the copy is not optional there.
        self._copy_files: bool = bool(params.get("copy_files", False)) or bool(self._staging_uuid)
        self._dest_path: str = str(params.get("dest_path") or DEFAULT_IMPORT_DIR).strip("/")
        if self._copy_files and not self._dest_path:
            raise ValueError("dest_path is required when copying images in")

        raw_map = params.get("class_labels") or {}
        if not isinstance(raw_map, dict):
            raise ValueError("class_labels must be a map of class name to label uuid")
        self._class_labels: Dict[str, str] = {str(k): str(v) for k, v in raw_map.items() if v}
        self._create_missing_labels: bool = bool(params.get("create_missing_labels", True))
        self._mark_verified: bool = bool(params.get("mark_verified", False))
        self._dataset_name: str = str(params.get("dataset_name") or "").strip()

        resume = resume_data or {}
        self._images_done: int = int(resume.get("images_done", 0))
        self._images_imported: int = int(resume.get("images_imported", 0))
        self._boxes_imported: int = int(resume.get("boxes_imported", 0))
        self._boxes_skipped: int = int(resume.get("boxes_skipped", 0))
        self._duplicates: int = int(resume.get("duplicates", 0))
        self._failed: int = int(resume.get("failed", 0))
        self._failures: List[str] = list(resume.get("failures", []))
        self._created_labels: Dict[str, str] = dict(resume.get("created_labels", {}))
        self._dataset_uuid: Optional[str] = resume.get("dataset_uuid")
        self._unmapped: set = set(resume.get("unmapped", []))
        self._total: int = 0

    def _resolve_root(self) -> Path:
        if self._staging_uuid:
            # Validated as a uuid by db_server before it ever reaches params;
            # re-checked here because a task's params can also be posted directly.
            if "/" in self._staging_uuid or "\\" in self._staging_uuid or ".." in self._staging_uuid:
                raise ValueError(f"Not a staging id: {self._staging_uuid!r}")
            root = _STAGING_DIR / self._staging_uuid
        else:
            candidate = (self._files_root / self._source_path).resolve()
            files_root = self._files_root.resolve()
            if candidate != files_root and files_root not in candidate.parents:
                raise ValueError(f"source_path is outside the image root: {self._source_path!r}")
            root = candidate
        if not root.is_dir():
            raise ValueError(f"Not a folder: {self._source_path or self._staging_uuid}")
        return root

    def _api(self, cls):
        return cls(self._manager.get_db_api_client())

    async def _run(self) -> Dict[str, Any]:
        detected = await asyncio.to_thread(detect, self._root, self._format)
        self._format = detected.format
        # detect() descends past a single wrapping directory, which is what an
        # archive almost always has; parse from where it landed, not where the
        # user pointed.
        self._root = detected.root
        self._total = detected.image_count
        self.log(
            f"Importing {detected.image_count} image(s) and {detected.annotation_count} "
            f"annotation(s) as {detected.format}"
        )
        for warning in detected.warnings:
            self.log(f"Warning: {warning}")

        label_map = await self._resolve_labels(detected.class_names)

        warnings: List[str] = []
        records = await asyncio.to_thread(lambda: list(iter_images(self._root, self._format, warnings)))
        if self._images_done:
            self.log(f"Resuming after {self._images_done} image(s)")
            records = records[self._images_done :]

        pending_registration: List[str] = []
        for record in records:
            if self._cancel_flag.is_set():
                break
            await self._check_pause()
            try:
                await self._import_one(record, label_map, pending_registration)
            except Exception as exc:  # noqa: BLE001 - one bad image never fails the run
                logger.exception("Import failed for %s", record.path)
                self._note_failure(f"{record.rel_path}: {exc}")
            self._images_done += 1
            if self._images_done % _PROGRESS_EVERY == 0:
                await self._flush_registrations(pending_registration)
                self._update_progress(100.0 * self._images_done / max(1, self._total))
                # Checkpointed on the same beat as the flush: resuming from a
                # point where registrations were still queued would re-import
                # images the previous run had already written.
                self._resume_data = self._build_resume_data()
                await self.heartbeat()

        await self._flush_registrations(pending_registration)
        self._resume_data = self._build_resume_data()

        if not self._cancel_flag.is_set():
            await self._maybe_create_dataset(label_map)
            await self._discard_staging()

        self._update_progress(100.0)
        return self._build_result()

    # ── labels ───────────────────────────────────────────────────────────

    async def _resolve_labels(self, class_names: List[str]) -> Dict[str, str]:
        """Class name → label uuid, creating labels only where asked to."""
        api = self._api(LabelsApi)
        resolved: Dict[str, str] = dict(self._class_labels)
        resolved.update(self._created_labels)  # a resumed run keeps what it made

        missing = [n for n in class_names if n not in resolved]
        if not missing:
            return resolved

        if not self._create_missing_labels:
            self._unmapped.update(missing)
            self.log(
                f"{len(missing)} class name(s) are not mapped to a label and label creation is off; "
                f"their boxes will be skipped: {', '.join(sorted(missing)[:8])}"
            )
            return resolved

        # Match an existing label by name before making a new one: importing a
        # second dataset that also calls it "person" must not produce "person"
        # twice, which would split the class across two labels for training.
        by_name: Dict[str, str] = {}
        try:
            existing = await asyncio.to_thread(api.list_labels)
            if existing.status == JsonValues.SUCCESS:
                for label in existing.labels or []:
                    by_name.setdefault(label.name.strip().lower(), label.uuid)
        except Exception as exc:  # noqa: BLE001 - fall back to creating
            logger.warning("ImportDatasetTask: could not list labels: %s", exc)

        from db_client.models.add_label_payload import AddLabelPayload

        for name in missing:
            match = by_name.get(name.strip().lower())
            if match:
                resolved[name] = match
                continue
            try:
                response = await asyncio.to_thread(api.add_label, AddLabelPayload(name=name, color=_color_for(name)))
            except Exception as exc:  # noqa: BLE001 - reported, not fatal
                self._note_failure(f"could not create label {name!r}: {exc}")
                self._unmapped.add(name)
                continue
            uuid = getattr(getattr(response, "label", None), "uuid", None)
            if not uuid:
                self._note_failure(f"could not create label {name!r}: {response.message or 'no uuid returned'}")
                self._unmapped.add(name)
                continue
            resolved[name] = uuid
            self._created_labels[name] = uuid

        if self._created_labels:
            self.log(f"Created {len(self._created_labels)} label(s): {', '.join(sorted(self._created_labels))}")
        return resolved

    # ── images ───────────────────────────────────────────────────────────

    async def _import_one(
        self,
        record: ImportedImage,
        label_map: Dict[str, str],
        pending_registration: List[str],
    ) -> None:
        if self._copy_files:
            dest_path = await self._copy_in(record)
            if dest_path is None:
                return
        else:
            dest_path = self._in_place_path(record)
            # Registration is batched: one call per image would be thousands of
            # round trips for a dataset that is already on disk.
            pending_registration.append(dest_path)
            self._images_imported += 1

        await self._write_boxes(record, dest_path, label_map)

    def _in_place_path(self, record: ImportedImage) -> str:
        return record.path.resolve().relative_to(self._files_root.resolve()).as_posix()

    async def _copy_in(self, record: ImportedImage) -> Optional[str]:
        """Upload one image into the destination; return its stored path.

        Goes through `/api/images/upload` rather than writing the file and
        registering it separately, so a copied import gets exactly the same
        name-collision handling, decode check and duplicate report a manual
        upload does.
        """
        images = self._api(ImagesApi)
        # The source tree's shape is kept under the destination: a YOLO export's
        # train/ and val/ directories are meaningful to a human reading the
        # folder later, even though the split itself is not imported.
        sub_dir = Path(record.rel_path).parent.as_posix()
        target_dir = self._dest_path if sub_dir in ("", ".") else f"{self._dest_path}/{sub_dir}"

        data = await asyncio.to_thread(record.path.read_bytes)
        response = await asyncio.to_thread(
            images.upload_images,
            files=[(record.path.name, data)],
            path=target_dir,
            overwrite=False,
            source_uuid="",
        )
        if response.status != JsonValues.SUCCESS or not response.results:
            self._note_failure(f"{record.rel_path}: {response.message or 'the upload was refused'}")
            return None
        result = response.results[0]
        if not result.ok:
            self._note_failure(f"{record.rel_path}: {result.message or 'refused'}")
            return None
        if result.duplicate_of:
            self._duplicates += 1
        self._images_imported += 1
        return result.path

    async def _flush_registrations(self, pending: List[str]) -> None:
        """Register a batch of in-place images, then clear the batch."""
        if not pending:
            return
        images = self._api(ImagesApi)
        from db_client.models.register_images_payload import RegisterImagesPayload

        try:
            response = await asyncio.to_thread(images.register_images, RegisterImagesPayload(paths=list(pending)))
        except Exception as exc:  # noqa: BLE001 - counted, never fatal
            self._note_failure(f"registering {len(pending)} image(s) failed: {exc}")
            pending.clear()
            return
        for failure in response.failed or []:
            self._note_failure(f"{failure.path}: {failure.message}")
            self._images_imported = max(0, self._images_imported - 1)
        pending.clear()

    async def _write_boxes(self, record: ImportedImage, dest_path: str, label_map: Dict[str, str]) -> None:
        if not record.boxes:
            return
        images = self._api(ImagesApi)
        from db_client.models.add_bbox_payload import AddBboxPayload

        # One or the other, never both: the two tags share
        # exclusive_group="verification" and the DB would reject the pair.
        verification_tag = VERIFIED_TAG_UUID if self._mark_verified else UNVERIFIED_TAG_UUID

        for box in record.boxes:
            label_uuid = label_map.get(box.class_name)
            if not label_uuid:
                self._boxes_skipped += 1
                self._unmapped.add(box.class_name)
                continue
            await asyncio.to_thread(
                images.add_bbox,
                AddBboxPayload(
                    image_path=dest_path,
                    label_uuid=label_uuid,
                    x=box.x,
                    y=box.y,
                    width=box.w,
                    height=box.h,
                    tag_uuids=[verification_tag],
                    metadata_json={"imported_from": self._format, "imported_class": box.class_name},
                ),
            )
            self._boxes_imported += 1

    # ── dataset + cleanup ────────────────────────────────────────────────

    async def _maybe_create_dataset(self, label_map: Dict[str, str]) -> None:
        if not self._dataset_name or self._dataset_uuid:
            return
        if not self._copy_files and not self._source_path:
            return

        from db_client.models.dataset_create import DatasetCreate
        from db_client.models.dataset_filter_criteria import DatasetFilterCriteria
        from db_client.models.filename_term import FilenameTerm
        from db_client.models.terms_inner import TermsInner

        # A Dataset here is a saved filter, not a fixed list of images, so the
        # import's own folder is what defines it. That means images added to
        # that folder later join the dataset — which is the behaviour every
        # other dataset has, and the wizard says so before creating one.
        #
        # TermsInner is the generated wrapper for the discriminated union; the
        # criteria model rejects a bare term variant.
        folder = self._dest_path if self._copy_files else self._source_path
        term = TermsInner(FilenameTerm(kind="filename", patterns=[f"{folder}/"], match_mode="all"))
        api = self._api(DatasetsApi)
        try:
            response = await asyncio.to_thread(
                api.create_dataset,
                DatasetCreate(
                    name=self._dataset_name,
                    description=f"Imported from {folder} ({self._format})",
                    filter_json=DatasetFilterCriteria(terms=[term]),
                    label_uuids=sorted(set(label_map.values())),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - the images are already in
            self._note_failure(f"could not create dataset {self._dataset_name!r}: {exc}")
            return
        dataset = getattr(response, "dataset", None)
        if dataset is not None:
            self._dataset_uuid = dataset.uuid
            self.log(f"Created dataset '{self._dataset_name}'")
        else:
            self._note_failure(f"could not create dataset: {response.message or 'no dataset returned'}")

    async def _discard_staging(self) -> None:
        """Delete the staged upload once its images are safely in the library."""
        if not self._staging_uuid:
            return
        await asyncio.to_thread(shutil.rmtree, _STAGING_DIR / self._staging_uuid, True)

    # ── reporting ────────────────────────────────────────────────────────

    def _note_failure(self, message: str) -> None:
        self._failed += 1
        if len(self._failures) < _MAX_REPORTED_FAILURES:
            self._failures.append(message)

    def _build_result(self) -> Dict[str, Any]:
        return {
            "format": self._format,
            "images_imported": self._images_imported,
            "boxes_imported": self._boxes_imported,
            "boxes_skipped": self._boxes_skipped,
            "boxes_verified": self._mark_verified,
            "duplicates": self._duplicates,
            "failed": self._failed,
            "failures": list(self._failures),
            "created_labels": dict(self._created_labels),
            "unmapped_classes": sorted(self._unmapped),
            "dataset_uuid": self._dataset_uuid,
        }

    def get_status_message(self) -> str:
        if not self._total:
            return "Inspecting the dataset"
        parts = [f"{self._images_imported}/{self._total} images", f"{self._boxes_imported} boxes"]
        if self._boxes_skipped:
            parts.append(f"{self._boxes_skipped} boxes skipped")
        if self._failed:
            parts.append(f"{self._failed} failed")
        return ", ".join(parts)

    def _build_resume_data(self) -> Dict[str, Any]:
        return {
            "images_done": self._images_done,
            "images_imported": self._images_imported,
            "boxes_imported": self._boxes_imported,
            "boxes_skipped": self._boxes_skipped,
            "duplicates": self._duplicates,
            "failed": self._failed,
            "failures": list(self._failures),
            "created_labels": dict(self._created_labels),
            "dataset_uuid": self._dataset_uuid,
            "unmapped": sorted(self._unmapped),
        }

    async def _deinit(self) -> None:
        return None


__all__ = ["ImportDatasetTask", "DEFAULT_IMPORT_DIR", "DatasetImportError"]
