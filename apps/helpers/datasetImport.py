"""Dataset import — read a standard on-disk layout into image/box records.

The mirror of ``datasetExport``: the same three detection formats, read instead
of written, plus a fourth entry for a directory of bare images with no
annotations at all.

* ``yolo`` — ``images/`` + ``labels/*.txt`` (normalised ``class cx cy w h``),
  class names from ``data.yaml``.
* ``coco`` — ``annotations/instances_*.json`` (or any COCO-shaped ``*.json``),
  pixel ``[x, y, w, h]``.
* ``voc`` — ``Annotations/*.xml`` with pixel corners, 1-based and inclusive.
* ``images`` — every image found, no annotations.

Splits are deliberately not carried across. A Dataset here has no train/val
distinction of its own; ``datasetExport.split_images`` derives one at export
time from uuid order, so honouring an incoming split would invent a field the
rest of the app cannot store or round-trip.

Nothing in this module writes, copies, or decodes an image. Boxes come back
normalised and clamped, which is what ``add_bbox`` stores; pixel formats are
converted using the dimensions their own metadata declares, so no image needs
to be opened to parse a dataset. Doing the copying and the DB writes is
``ImportDatasetTask``'s job, and keeping that out of here is what makes the
format logic testable against a fixture tree.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence
from xml.etree import ElementTree

from apps.helpers.logUtils import LOG_LEVEL

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)

#: Advertised to the UI by ``GET /api/datasets/import-formats``. Mirrors the
#: shape of ``datasetExport.EXPORT_FORMATS`` so the two modals can share a
#: renderer.
IMPORT_FORMATS: List[dict] = [
    {
        "id": "yolo",
        "name": "YOLO (Ultralytics / darknet)",
        "description": "images/ + labels/*.txt + data.yaml",
    },
    {
        "id": "coco",
        "name": "COCO",
        "description": "images/ + annotations/instances_*.json",
    },
    {
        "id": "voc",
        "name": "Pascal VOC",
        "description": "JPEGImages/ + Annotations/*.xml",
    },
    {
        "id": "images",
        "name": "Images only (no annotations)",
        "description": "Every image found under the folder, imported unannotated",
    },
]

IMPORT_FORMAT_IDS = frozenset(f["id"] for f in IMPORT_FORMATS)

#: Matches ``Cv2VideoImageProvider.IMAGE_EXTENSIONS`` — a dataset whose images
#: this app can read is one the collector could also have produced.
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"})

#: Directory names skipped wholesale during discovery. ``__MACOSX`` is the one
#: that matters in practice: unzipping a Mac-made archive otherwise yields a
#: shadow tree of ``._name`` resource forks that look like images by extension.
_SKIP_DIRS = frozenset({"__MACOSX", ".git", ".ipynb_checkpoints"})

#: Cap on how many warnings a detect/parse pass accumulates. A dataset with a
#: systematic problem produces one per file, and a 40k-entry list helps nobody.
_MAX_WARNINGS = 25


@dataclass
class ImportedBox:
    """One annotation, in the coordinate system ``bboxes`` stores.

    Top-left origin, extents as fractions of the image, clamped to the frame —
    not the centre-relative form YOLO and this app's own exporter use. The
    conversion happens once, here, so the import task writes what it is handed.
    """

    class_name: str
    x: float
    y: float
    w: float
    h: float


@dataclass
class ImportedImage:
    """One image and everything the source format said about it."""

    path: Path
    #: Path relative to the dataset root, used to lay the copy out under
    #: FILES_ROOT. Keeps the source tree's shape instead of flattening it.
    rel_path: str
    boxes: List[ImportedBox] = field(default_factory=list)


@dataclass
class DetectedDataset:
    """What a scan concluded about a directory, before anything is imported."""

    format: str
    #: The real dataset root. May be below the directory that was handed in —
    #: an archive of "mydataset/" unpacks to a single child holding everything.
    root: Path
    class_names: List[str] = field(default_factory=list)
    image_count: int = 0
    annotation_count: int = 0
    warnings: List[str] = field(default_factory=list)


class DatasetImportError(ValueError):
    """A directory that cannot be imported at all, with a user-facing reason."""


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _box_from_corners(x0: float, y0: float, x1: float, y1: float) -> Optional[tuple]:
    """Normalised ``(x, y, w, h)`` from normalised corners, or None if empty.

    Corners are clamped before the extent is derived, not after: clamping an
    origin while keeping the stored width leaves an overhanging box
    overhanging. This is the read-side twin of ``datasetExport._clamped_box``.
    """
    x0, x1 = _clamp01(min(x0, x1)), _clamp01(max(x0, x1))
    y0, y1 = _clamp01(min(y0, y1)), _clamp01(max(y0, y1))
    w, h = x1 - x0, y1 - y0
    if w <= 0.0 or h <= 0.0:
        return None
    return x0, y0, w, h


def _is_image(path: Path) -> bool:
    # A leading dot is how both macOS resource forks and editor swap files
    # present, and neither is an image however the extension reads.
    return path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".")


def _walk_files(root: Path) -> Iterator[Path]:
    """Every file under ``root``, skipping the directories nothing wants."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        base = Path(dirpath)
        for name in filenames:
            yield base / name


def _find_images(root: Path) -> List[Path]:
    return sorted(p for p in _walk_files(root) if _is_image(p))


#: Directory names that are part of a dataset's own layout. Descending into
#: one would throw the structure away — ``root/images`` read as the root has
#: no ``labels`` sibling, so a YOLO dataset reads as loose images.
_STRUCTURAL_DIRS = frozenset(
    {
        "images",
        "labels",
        "annotations",
        "jpegimages",
        "imagesets",
        "train",
        "val",
        "valid",
        "test",
    }
)


def descend_to_root(start: Path) -> Path:
    """Follow a chain of single-child directories down to the real root.

    An archive almost always contains one top-level folder, so the directory
    it was unpacked into holds nothing but that folder. Descending means the
    wizard reports "COCO" rather than "a directory with one subdirectory".

    Stops at a level holding a file or more than one entry, and refuses to
    descend into a directory that is part of the dataset layout itself.
    """
    current = start
    for _ in range(8):  # a runaway symlink loop is not worth chasing forever
        try:
            entries = [e for e in current.iterdir() if e.name not in _SKIP_DIRS and not e.name.startswith(".")]
        except OSError:
            return current
        if len(entries) == 1 and entries[0].is_dir() and entries[0].name.lower() not in _STRUCTURAL_DIRS:
            current = entries[0]
            continue
        return current
    return current


# ──────────────────────────────────────────────────────────────────────────
# data.yaml
# ──────────────────────────────────────────────────────────────────────────

_YAML_LIST_ITEM = re.compile(r"^\s*-\s*(.+?)\s*$")
_YAML_INDEXED_ITEM = re.compile(r"^\s*(\d+)\s*:\s*(.+?)\s*$")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse_yolo_names(text: str) -> List[str]:
    """Class names from a ``data.yaml``, in index order.

    Hand-parsed rather than handed to PyYAML because PyYAML is a dev-only
    dependency here, and the three shapes Ultralytics and Roboflow actually
    emit — an inline list, a block list, and an index-keyed map — are a few
    lines to read directly. An index-keyed map is honoured as a map: Roboflow
    writes them in order but nothing guarantees it.
    """
    lines = text.splitlines()
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped.startswith("names:"):
            continue
        inline = stripped[len("names:") :].strip()
        if inline.startswith("["):
            body = inline[1 : inline.rindex("]")] if "]" in inline else inline[1:]
            return [_unquote(p) for p in body.split(",") if _unquote(p)]

        indent = len(raw) - len(raw.lstrip())
        indexed: dict[int, str] = {}
        ordered: List[str] = []
        for follower in lines[i + 1 :]:
            if not follower.strip():
                continue
            if len(follower) - len(follower.lstrip()) <= indent:
                break  # dedented back out of the names block
            if (m := _YAML_INDEXED_ITEM.match(follower)) is not None:
                indexed[int(m.group(1))] = _unquote(m.group(2))
            elif (m := _YAML_LIST_ITEM.match(follower)) is not None:
                ordered.append(_unquote(m.group(1)))
        if indexed:
            return [indexed[k] for k in sorted(indexed)]
        return ordered
    return []


def _read_yolo_class_names(root: Path) -> List[str]:
    for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        candidate = root / name
        if candidate.is_file():
            try:
                names = parse_yolo_names(candidate.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
            if names:
                return names
    return []


# ──────────────────────────────────────────────────────────────────────────
# Format detection
# ──────────────────────────────────────────────────────────────────────────


def _coco_json_files(root: Path) -> List[Path]:
    """Every COCO-shaped JSON under ``root``, annotation files first.

    A COCO export names its files ``instances_<split>.json`` under
    ``annotations/``, but plenty of real exports drop a single
    ``_annotations.coco.json`` next to the images instead, so the shape of the
    document decides rather than its name.
    """
    found: List[Path] = []
    for path in _walk_files(root):
        if path.suffix.lower() != ".json" or path.name.startswith("."):
            continue
        try:
            with path.open("rb") as fh:
                head = fh.read(4096).decode("utf-8", errors="replace")
        except OSError:
            continue
        # Cheap pre-filter: a full json.load of every sidecar in a 50k-image
        # tree would make detection cost as much as the import.
        if '"annotations"' in head and '"images"' in head:
            found.append(path)
    return sorted(found, key=lambda p: (p.parent.name != "annotations", str(p)))


def _looks_like_yolo(root: Path) -> bool:
    if any((root / n).is_file() for n in ("data.yaml", "data.yml")):
        return True
    labels = root / "labels"
    return labels.is_dir() and any(p.suffix.lower() == ".txt" for p in _walk_files(labels))


def _looks_like_voc(root: Path) -> bool:
    annotations = root / "Annotations"
    return annotations.is_dir() and any(p.suffix.lower() == ".xml" for p in _walk_files(annotations))


def detect(directory: Path, fmt: Optional[str] = None) -> DetectedDataset:
    """Identify the dataset in ``directory`` and count what it holds.

    ``fmt`` overrides the guess, for a tree that carries two formats at once —
    a Roboflow export ships YOLO labels and a COCO json side by side — or one
    the user wants read as bare images regardless of what else is in it. The
    root is still located the same way, so an override does not cost the
    single-child descent.

    Raises ``DatasetImportError`` when the directory holds no images at all —
    every format here is ultimately a pile of images, so that is the one
    condition no format choice can rescue.
    """
    if not directory.is_dir():
        raise DatasetImportError(f"Not a directory: {directory}")

    root = descend_to_root(directory)
    images = _find_images(root)
    if not images:
        raise DatasetImportError("No images found in that folder")

    if fmt:
        return summarize(root, fmt, image_paths=images)

    if _looks_like_voc(root):
        fmt = "voc"
    elif _coco_json_files(root):
        fmt = "coco"
    elif _looks_like_yolo(root):
        fmt = "yolo"
    else:
        fmt = "images"

    return summarize(root, fmt, image_paths=images)


def summarize(root: Path, fmt: str, *, image_paths: Optional[Sequence[Path]] = None) -> DetectedDataset:
    """Parse ``root`` as ``fmt`` and report counts, classes and warnings.

    This is a full parse, not an estimate. The wizard shows the numbers it
    returns as the thing the user is about to import, and a count that turns
    out to be wrong once the task runs is worse than a slower dialog.
    """
    if fmt not in IMPORT_FORMAT_IDS:
        raise DatasetImportError(f"Unknown import format: {fmt!r}")

    images = list(image_paths) if image_paths is not None else _find_images(root)
    if not images:
        raise DatasetImportError("No images found in that folder")

    warnings: List[str] = []
    records = list(_parse(root, fmt, images, warnings))
    class_names = sorted({box.class_name for rec in records for box in rec.boxes})
    if fmt == "yolo":
        # Keep data.yaml's own order where it has one: a YOLO class list is
        # index-ordered by definition and the indices are what the .txt files
        # reference, so reporting it alphabetised misdescribes the dataset.
        declared = [n for n in _read_yolo_class_names(root) if n in set(class_names)]
        class_names = declared + [n for n in class_names if n not in set(declared)]

    return DetectedDataset(
        format=fmt,
        root=root,
        class_names=class_names,
        image_count=len(records),
        annotation_count=sum(len(r.boxes) for r in records),
        warnings=warnings[:_MAX_WARNINGS],
    )


# ──────────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────────


def iter_images(root: Path, fmt: str, warnings: Optional[List[str]] = None) -> Iterator[ImportedImage]:
    """Yield every importable image under ``root``, read as ``fmt``."""
    if fmt not in IMPORT_FORMAT_IDS:
        raise DatasetImportError(f"Unknown import format: {fmt!r}")
    yield from _parse(root, fmt, _find_images(root), warnings if warnings is not None else [])


def _parse(root: Path, fmt: str, images: Sequence[Path], warnings: List[str]) -> Iterator[ImportedImage]:
    if fmt == "images":
        for path in images:
            yield ImportedImage(path=path, rel_path=_rel(root, path))
    elif fmt == "yolo":
        yield from _parse_yolo(root, images, warnings)
    elif fmt == "coco":
        yield from _parse_coco(root, images, warnings)
    else:
        yield from _parse_voc(root, images, warnings)


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _warn(warnings: List[str], message: str) -> None:
    # Bounded, but still counted: the caller shows "and N more" from the
    # difference between this list and the real total where it matters.
    if len(warnings) < _MAX_WARNINGS:
        warnings.append(message)


# ── YOLO ──────────────────────────────────────────────────────────────────


def _yolo_label_candidates(root: Path, image: Path) -> List[Path]:
    """Where a YOLO label file for ``image`` could live, best guess first.

    The canonical layout swaps the ``images`` path component for ``labels``;
    the two fallbacks cover a flat ``labels/`` directory and a ``.txt`` sitting
    directly beside the image, both of which occur in real exports.
    """
    rel = Path(_rel(root, image))
    stem_txt = rel.with_suffix(".txt")
    candidates: List[Path] = []

    parts = list(rel.parts)
    for i, part in enumerate(parts):
        if part == "images":
            swapped = parts[:i] + ["labels"] + parts[i + 1 :]
            candidates.append(root.joinpath(*swapped).with_suffix(".txt"))
            break

    candidates.append(root / "labels" / stem_txt.name)
    candidates.append(image.with_suffix(".txt"))
    return candidates


def _parse_yolo(root: Path, images: Sequence[Path], warnings: List[str]) -> Iterator[ImportedImage]:
    names = _read_yolo_class_names(root)
    for image in images:
        label_file = next((c for c in _yolo_label_candidates(root, image) if c.is_file()), None)
        boxes: List[ImportedBox] = []
        if label_file is not None:
            boxes = _read_yolo_label(label_file, names, warnings)
        yield ImportedImage(path=image, rel_path=_rel(root, image), boxes=boxes)


def _read_yolo_label(path: Path, names: Sequence[str], warnings: List[str]) -> List[ImportedBox]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        _warn(warnings, f"{path.name}: could not be read ({exc})")
        return []

    boxes: List[ImportedBox] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        # Segmentation labels carry a polygon instead of a box; they are a
        # different task, not a malformed detection label, so say so once.
        if len(parts) > 5:
            _warn(warnings, f"{path.name}:{lineno}: polygon/segmentation label skipped")
            continue
        if len(parts) < 5:
            _warn(warnings, f"{path.name}:{lineno}: expected 'class cx cy w h'")
            continue
        try:
            cls_idx = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            _warn(warnings, f"{path.name}:{lineno}: non-numeric label values")
            continue

        clamped = _box_from_corners(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        if clamped is None:
            _warn(warnings, f"{path.name}:{lineno}: zero-area box skipped")
            continue
        name = names[cls_idx] if 0 <= cls_idx < len(names) else f"class_{cls_idx}"
        boxes.append(ImportedBox(name, *clamped))
    return boxes


# ── COCO ──────────────────────────────────────────────────────────────────


def _index_by_name(images: Sequence[Path], root: Path) -> dict:
    """Lookup for resolving a declared file name to a file on disk.

    Keyed by both the basename and the root-relative path: COCO and VOC both
    record a bare ``file_name`` most of the time but a relative one sometimes,
    and basenames collide across split directories.
    """
    index: dict = {}
    for path in images:
        index.setdefault(path.name, path)
        index.setdefault(_rel(root, path), path)
        index.setdefault(path.stem, path)
    return index


def _parse_coco(root: Path, images: Sequence[Path], warnings: List[str]) -> Iterator[ImportedImage]:
    by_name = _index_by_name(images, root)
    boxes_by_path: dict = {}
    seen_docs = 0

    for doc_path in _coco_json_files(root):
        try:
            doc = json.loads(doc_path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError) as exc:
            _warn(warnings, f"{doc_path.name}: not readable as JSON ({exc})")
            continue
        if not isinstance(doc, dict):
            continue
        seen_docs += 1

        categories = {
            c.get("id"): str(c.get("name", c.get("id")))
            for c in doc.get("categories", [])
            if isinstance(c, dict) and c.get("id") is not None
        }
        # An image id is only unique within its own document, so resolve to a
        # real path here rather than carrying ids across files.
        resolved: dict = {}
        for entry in doc.get("images", []):
            if not isinstance(entry, dict):
                continue
            declared = str(entry.get("file_name") or "")
            path = by_name.get(declared) or by_name.get(Path(declared).name) or by_name.get(Path(declared).stem)
            if path is None:
                _warn(warnings, f"{doc_path.name}: no image file for {declared!r}")
                continue
            width = float(entry.get("width") or 0)
            height = float(entry.get("height") or 0)
            resolved[entry.get("id")] = (path, width, height)

        for ann in doc.get("annotations", []):
            if not isinstance(ann, dict):
                continue
            target = resolved.get(ann.get("image_id"))
            if target is None:
                continue
            path, width, height = target
            if width <= 0 or height <= 0:
                _warn(warnings, f"{doc_path.name}: {path.name} declares no usable width/height")
                continue
            bbox = ann.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
                continue
            try:
                x, y, w, h = (float(v) for v in bbox[:4])
            except (TypeError, ValueError):
                _warn(warnings, f"{doc_path.name}: non-numeric bbox on image {path.name}")
                continue
            clamped = _box_from_corners(x / width, y / height, (x + w) / width, (y + h) / height)
            if clamped is None:
                continue
            name = categories.get(ann.get("category_id"), f"class_{ann.get('category_id')}")
            boxes_by_path.setdefault(path, []).append(ImportedBox(name, *clamped))

    if not seen_docs:
        _warn(warnings, "No COCO annotation file was readable; importing images only")

    for image in images:
        yield ImportedImage(path=image, rel_path=_rel(root, image), boxes=boxes_by_path.get(image, []))


# ── Pascal VOC ────────────────────────────────────────────────────────────


def _voc_text(node, tag: str) -> str:
    child = node.find(tag)
    return (child.text or "").strip() if child is not None and child.text else ""


def _parse_voc(root: Path, images: Sequence[Path], warnings: List[str]) -> Iterator[ImportedImage]:
    by_name = _index_by_name(images, root)
    boxes_by_path: dict = {}

    annotations_dir = root / "Annotations"
    xml_files = sorted(p for p in _walk_files(annotations_dir) if p.suffix.lower() == ".xml")
    for xml_path in xml_files:
        try:
            doc = ElementTree.parse(xml_path).getroot()
        except (OSError, ElementTree.ParseError) as exc:
            _warn(warnings, f"{xml_path.name}: not readable as XML ({exc})")
            continue

        declared = _voc_text(doc, "filename")
        path = by_name.get(declared) or by_name.get(Path(declared).name) or by_name.get(xml_path.stem)
        if path is None:
            _warn(warnings, f"{xml_path.name}: no image file for {declared or xml_path.stem!r}")
            continue

        size = doc.find("size")
        try:
            width = float(_voc_text(size, "width")) if size is not None else 0.0
            height = float(_voc_text(size, "height")) if size is not None else 0.0
        except ValueError:
            width = height = 0.0
        if width <= 0 or height <= 0:
            _warn(warnings, f"{xml_path.name}: declares no usable <size>")
            continue

        for obj in doc.findall("object"):
            box = obj.find("bndbox")
            if box is None:
                continue
            try:
                # VOC corners are 1-based and inclusive of both ends, which is
                # the inverse of what the exporter writes; undo it here so a
                # round-trip lands back on the same pixels.
                xmin = float(_voc_text(box, "xmin")) - 1
                ymin = float(_voc_text(box, "ymin")) - 1
                xmax = float(_voc_text(box, "xmax"))
                ymax = float(_voc_text(box, "ymax"))
            except ValueError:
                _warn(warnings, f"{xml_path.name}: non-numeric <bndbox>")
                continue
            clamped = _box_from_corners(xmin / width, ymin / height, xmax / width, ymax / height)
            if clamped is None:
                continue
            name = _voc_text(obj, "name") or "object"
            boxes_by_path.setdefault(path, []).append(ImportedBox(name, *clamped))

    for image in images:
        yield ImportedImage(path=image, rel_path=_rel(root, image), boxes=boxes_by_path.get(image, []))


def detected_to_dict(detected: DetectedDataset, *, root_relative_to: Optional[Path] = None) -> dict[str, Any]:
    """Serialise a ``DetectedDataset`` for the API, hiding absolute paths.

    The wizard works in FILES_ROOT-relative paths throughout; handing it an
    absolute container path would both leak the layout and be unusable as the
    value it has to send back.
    """
    root = detected.root
    if root_relative_to is not None:
        try:
            root = root.relative_to(root_relative_to)
        except ValueError:
            pass
    return {
        "format": detected.format,
        "root": root.as_posix() if root_relative_to is not None else str(root),
        "class_names": list(detected.class_names),
        "image_count": detected.image_count,
        "annotation_count": detected.annotation_count,
        "warnings": list(detected.warnings),
    }
