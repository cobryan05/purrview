"""Dataset export — turn a Dataset's annotations into a standard on-disk layout.

Three detection formats, chosen because between them they cover every trainer
anyone is likely to point at this data:

* ``yolo`` — Ultralytics/darknet: ``images/<split>/`` + ``labels/<split>/*.txt``
  (one ``class cx cy w h`` line per box, all normalised) + ``data.yaml``.
* ``coco`` — one ``annotations/instances_<split>.json`` per split, pixel
  ``[x, y, w, h]`` boxes. What detectron2/mmdetection/torchvision eat.
* ``voc`` — Pascal VOC: ``JPEGImages/`` + one ``Annotations/*.xml`` per image +
  ``ImageSets/Main/<split>.txt``, pixel corner coordinates.

Everything here is pure: it takes bytes and annotation tuples and produces tar
members. Reading image files and deciding *which* images are in scope belongs to
``db_server.manager``; keeping the two apart is what makes the format logic
testable without a filesystem.

Annotation input is the shape ``DbClient._assemble_detection_dataset`` already
returns — ``[class_idx, cx, cy, w, h]`` with all four coordinates as fractions of
the image — so the export path and the training path can never disagree about
what a Dataset contains.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
import time
from typing import List, Sequence, Tuple
from xml.sax.saxutils import escape

#: Advertised to the UI by ``GET /api/datasets/export-formats`` rather than
#: duplicated in JS — a hardcoded copy in the frontend would drift the first time
#: a format is added here.
EXPORT_FORMATS: List[dict] = [
    {
        "id": "yolo",
        "name": "YOLO (Ultralytics / darknet)",
        "description": "images/<split>/ + labels/<split>/*.txt + data.yaml",
    },
    {
        "id": "coco",
        "name": "COCO",
        "description": "images/<split>/ + annotations/instances_<split>.json",
    },
    {
        "id": "voc",
        "name": "Pascal VOC",
        "description": "JPEGImages/ + Annotations/*.xml + ImageSets/Main/<split>.txt",
    },
]

EXPORT_FORMAT_IDS = frozenset(f["id"] for f in EXPORT_FORMATS)

_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_stem(filename: str, image_uuid: str) -> str:
    """Return a tar-safe, collision-free base name for one image.

    The stem keeps the original name readable, and the uuid prefix makes it
    unique: dataset images come from many directories and their basenames
    collide freely, while YOLO/VOC both pair an image with its annotation *by
    stem*, so a collision silently overwrites one image's labels with another's.
    """
    stem = filename.replace("\\", "/").rsplit("/", 1)[-1]
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    stem = _UNSAFE_NAME_CHARS.sub("_", stem).strip("._-")
    return f"{stem}_{image_uuid[:8]}" if stem else image_uuid


def image_extension(filename: str) -> str:
    ext = filename.replace("\\", "/").rsplit("/", 1)[-1]
    ext = "." + ext.rsplit(".", 1)[1].lower() if "." in ext else ".jpg"
    return ext if _UNSAFE_NAME_CHARS.sub("", ext[1:]) == ext[1:] else ".jpg"


def split_images(images: Sequence[dict], val_fraction: float) -> List[Tuple[str, dict]]:
    """Assign every image to ``"train"`` or ``"val"``, deterministically.

    Sorted by uuid then sliced — the same rule ``training_server.dataset`` uses,
    so an exported split is reproducible and matches what training would have
    seen. ``val_fraction <= 0`` puts everything in train.
    """
    ordered = sorted(images, key=lambda r: r["image_uuid"])
    if val_fraction <= 0 or len(ordered) < 2:
        return [("train", img) for img in ordered]
    n_val = max(1, int(len(ordered) * min(val_fraction, 1.0)))
    n_train = max(0, len(ordered) - n_val)
    return [("train", img) for img in ordered[:n_train]] + [("val", img) for img in ordered[n_train:]]


class _ChunkSink:
    """File-like object ``tarfile`` writes into; the caller drains it as chunks."""

    def __init__(self) -> None:
        self._parts: List[bytes] = []

    def write(self, data) -> int:
        self._parts.append(bytes(data))
        return len(data)

    def flush(self) -> None:  # pragma: no cover - tarfile calls this on close
        pass

    def drain(self) -> bytes:
        out = b"".join(self._parts)
        self._parts.clear()
        return out


class TarStream:
    """Incremental, non-seekable tar writer.

    Uncompressed on purpose: the payload is overwhelmingly JPEG/PNG bytes, which
    gzip cannot shrink, so compressing would cost a core per download and save
    nothing.
    """

    def __init__(self) -> None:
        self._sink = _ChunkSink()
        self._tar = tarfile.open(fileobj=self._sink, mode="w|")

    def add(self, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mtime = int(time.time())
        info.mode = 0o644
        self._tar.addfile(info, io.BytesIO(data))

    def add_text(self, name: str, text: str) -> None:
        self.add(name, text.encode("utf-8"))

    def drain(self) -> bytes:
        return self._sink.drain()

    def close(self) -> None:
        self._tar.close()


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


class _Writer:
    """Common surface: per-image members, then whatever index files close it out."""

    #: True when the format stores pixel coordinates and the exporter therefore
    #: has to decode each image's header to learn its size.
    needs_dimensions = False

    def __init__(self, class_names: Sequence[str], dataset_name: str) -> None:
        self._class_names = list(class_names)
        self._dataset_name = dataset_name

    def add_image(self, tar: TarStream, split: str, base: str, ext: str, data: bytes, size, annotations) -> None:
        raise NotImplementedError

    def finalize(self, tar: TarStream, splits: Sequence[str]) -> None:
        raise NotImplementedError


class YoloWriter(_Writer):
    def add_image(self, tar, split, base, ext, data, size, annotations):
        tar.add(f"images/{split}/{base}{ext}", data)
        lines = []
        for cls_idx, cx, cy, w, h in annotations:
            lines.append(
                f"{int(cls_idx)} {_clamp01(cx):.6f} {_clamp01(cy):.6f} " f"{_clamp01(w):.6f} {_clamp01(h):.6f}"
            )
        # A negative image still gets its (empty) label file: YOLO treats a
        # missing .txt as "unlabelled", an empty one as "nothing here".
        tar.add_text(f"labels/{split}/{base}.txt", "\n".join(lines) + ("\n" if lines else ""))

    def finalize(self, tar, splits):
        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(self._class_names))
        lines = ["path: .", "train: images/train"]
        if "val" in splits:
            lines.append("val: images/val")
        lines += [f"nc: {len(self._class_names)}", "names:", names]
        tar.add_text("data.yaml", "\n".join(lines) + "\n")


class CocoWriter(_Writer):
    needs_dimensions = True

    def __init__(self, class_names, dataset_name):
        super().__init__(class_names, dataset_name)
        self._images: dict = {}
        self._annotations: dict = {}
        self._next_image_id = 1
        self._next_ann_id = 1

    def add_image(self, tar, split, base, ext, data, size, annotations):
        width, height = size
        image_id = self._next_image_id
        self._next_image_id += 1
        tar.add(f"images/{split}/{base}{ext}", data)
        self._images.setdefault(split, []).append(
            {"id": image_id, "file_name": f"{base}{ext}", "width": width, "height": height}
        )
        for cls_idx, cx, cy, w, h in annotations:
            # Clamp the *corners* and derive the extent from them. Clamping the
            # origin while keeping the stored width would push `x + w` past the
            # right edge for any box that overhangs the frame — which detector
            # output routinely does — and COCO validators reject an out-of-image
            # box outright.
            x0 = _clamp01(cx - w / 2) * width
            y0 = _clamp01(cy - h / 2) * height
            px_w = max(0.0, _clamp01(cx + w / 2) * width - x0)
            px_h = max(0.0, _clamp01(cy + h / 2) * height - y0)
            self._annotations.setdefault(split, []).append(
                {
                    "id": self._next_ann_id,
                    "image_id": image_id,
                    "category_id": int(cls_idx) + 1,  # COCO category ids are 1-based
                    "bbox": [round(x0, 2), round(y0, 2), round(px_w, 2), round(px_h, 2)],
                    "area": round(px_w * px_h, 2),
                    "iscrowd": 0,
                }
            )
            self._next_ann_id += 1

    def finalize(self, tar, splits):
        categories = [{"id": i + 1, "name": n, "supercategory": "none"} for i, n in enumerate(self._class_names)]
        for split in splits:
            doc = {
                "info": {"description": self._dataset_name, "version": "1.0"},
                "licenses": [],
                "images": self._images.get(split, []),
                "annotations": self._annotations.get(split, []),
                "categories": categories,
            }
            tar.add_text(f"annotations/instances_{split}.json", json.dumps(doc, indent=1))


class VocWriter(_Writer):
    needs_dimensions = True

    def __init__(self, class_names, dataset_name):
        super().__init__(class_names, dataset_name)
        self._sets: dict = {}

    def add_image(self, tar, split, base, ext, data, size, annotations):
        width, height = size
        tar.add(f"JPEGImages/{base}{ext}", data)
        self._sets.setdefault(split, []).append(base)

        objects = []
        for cls_idx, cx, cy, w, h in annotations:
            name = self._class_names[int(cls_idx)] if int(cls_idx) < len(self._class_names) else str(cls_idx)
            # VOC pixel coordinates are 1-based and inclusive of both corners.
            xmin = int(round(_clamp01(cx - w / 2) * width)) + 1
            ymin = int(round(_clamp01(cy - h / 2) * height)) + 1
            xmax = max(xmin, int(round(_clamp01(cx + w / 2) * width)))
            ymax = max(ymin, int(round(_clamp01(cy + h / 2) * height)))
            objects.append(
                "\t<object>\n"
                f"\t\t<name>{escape(name)}</name>\n"
                "\t\t<pose>Unspecified</pose>\n"
                "\t\t<truncated>0</truncated>\n"
                "\t\t<difficult>0</difficult>\n"
                "\t\t<bndbox>\n"
                f"\t\t\t<xmin>{xmin}</xmin>\n\t\t\t<ymin>{ymin}</ymin>\n"
                f"\t\t\t<xmax>{xmax}</xmax>\n\t\t\t<ymax>{ymax}</ymax>\n"
                "\t\t</bndbox>\n"
                "\t</object>"
            )

        xml = (
            "<annotation>\n"
            f"\t<folder>JPEGImages</folder>\n"
            f"\t<filename>{escape(base + ext)}</filename>\n"
            f"\t<source>\n\t\t<database>{escape(self._dataset_name)}</database>\n\t</source>\n"
            f"\t<size>\n\t\t<width>{width}</width>\n\t\t<height>{height}</height>\n"
            "\t\t<depth>3</depth>\n\t</size>\n"
            "\t<segmented>0</segmented>\n" + ("\n".join(objects) + "\n" if objects else "") + "</annotation>\n"
        )
        tar.add_text(f"Annotations/{base}.xml", xml)

    def finalize(self, tar, splits):
        for split in splits:
            tar.add_text(f"ImageSets/Main/{split}.txt", "\n".join(self._sets.get(split, [])) + "\n")
        tar.add_text("labels.txt", "\n".join(self._class_names) + "\n")


_WRITERS = {"yolo": YoloWriter, "coco": CocoWriter, "voc": VocWriter}


def make_writer(fmt: str, class_names: Sequence[str], dataset_name: str) -> _Writer:
    """Return the writer for ``fmt``; raises ``KeyError`` for an unknown format."""
    return _WRITERS[fmt](class_names, dataset_name)
