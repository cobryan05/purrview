"""Dataset import: format detection, coordinate conversion, and archive safety.

The centrepiece is the round trip. ``datasetExport`` and ``datasetImport`` are
two independent readings of the same three on-disk layouts, with no shared
code between the writer and the reader — exactly the arrangement where a
convention drifts on one side and nothing notices. Exporting real boxes and
importing them back asserts the two agree about what a box *is*: the centre-vs-
corner origin, the 1-based inclusive VOC corners, and the pixel-vs-fraction
scaling are each a place a silent off-by-one lives.
"""

import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from apps.helpers.archiveUtils import ArchiveError, extract_archive
from apps.helpers.datasetExport import TarStream, make_writer
from apps.helpers.datasetImport import (
    DatasetImportError,
    detect,
    iter_images,
    parse_yolo_names,
)

# A 1x1 PNG is enough: nothing in the import path decodes pixels, and the
# formats that need dimensions read them from their own metadata.
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6300010000050001"
    "0d0a2db4"
    "0000000049454e44ae426082"
)

#: (class_idx, cx, cy, w, h) — the shape the exporter takes. Deliberately
#: off-centre and non-square so a transposed axis or a centre/corner mix-up
#: cannot pass by symmetry.
_ANNOTATIONS = [
    (0, 0.25, 0.40, 0.10, 0.20),
    (1, 0.75, 0.60, 0.30, 0.10),
]
_CLASS_NAMES = ["cat", "dog"]

#: Image size declared to the pixel formats. Chosen non-square for the same
#: reason the boxes are.
_SIZE = (200, 100)


def _export_tree(tmp_path: Path, fmt: str) -> Path:
    """Write a one-image dataset in ``fmt`` and unpack it into a directory."""
    tar = TarStream()
    writer = make_writer(fmt, _CLASS_NAMES, "round-trip")
    chunks = [tar.drain()]
    writer.add_image(tar, "train", "frame", ".png", _PNG, _SIZE, _ANNOTATIONS)
    chunks.append(tar.drain())
    writer.finalize(tar, ["train"])
    chunks.append(tar.drain())
    tar.close()
    chunks.append(tar.drain())

    archive = tmp_path / f"{fmt}.tar"
    archive.write_bytes(b"".join(chunks))
    dest = tmp_path / fmt
    extract_archive(archive, dest)
    return dest


def _corners(box) -> tuple:
    return (box.x, box.y, box.x + box.w, box.y + box.h)


def _expected_corners() -> list:
    return [(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2) for _cls, cx, cy, w, h in _ANNOTATIONS]


class TestRoundTrip:
    """Export a dataset, import it back, and compare the boxes."""

    @pytest.mark.parametrize("fmt", ["yolo", "coco", "voc"])
    def test_detects_the_format_it_was_written_as(self, tmp_path, fmt):
        detected = detect(_export_tree(tmp_path, fmt))
        assert detected.format == fmt

    @pytest.mark.parametrize("fmt", ["yolo", "coco", "voc"])
    def test_boxes_survive_the_round_trip(self, tmp_path, fmt):
        root = _export_tree(tmp_path, fmt)
        records = list(iter_images(root, fmt))
        assert len(records) == 1

        got = sorted(_corners(b) for b in records[0].boxes)
        want = sorted(_expected_corners())
        assert len(got) == len(want)
        for actual, expected in zip(got, want):
            # VOC rounds to whole pixels on the way out, so the tolerance is one
            # pixel on the smaller axis rather than float noise.
            assert actual == pytest.approx(expected, abs=1.0 / min(_SIZE))

    @pytest.mark.parametrize("fmt", ["yolo", "coco", "voc"])
    def test_class_names_survive_the_round_trip(self, tmp_path, fmt):
        root = _export_tree(tmp_path, fmt)
        records = list(iter_images(root, fmt))
        assert sorted(b.class_name for b in records[0].boxes) == sorted(_CLASS_NAMES)

    def test_yolo_reports_classes_in_index_order_not_alphabetical(self, tmp_path):
        # "dog" sorts after "cat" but is class 1; a YOLO class list is ordered
        # by the index its label files reference, and reporting it sorted would
        # misdescribe which name the wizard is mapping.
        root = _export_tree(tmp_path, "yolo")
        (root / "data.yaml").write_text("path: .\ntrain: images/train\nnc: 2\nnames:\n  0: zebra\n  1: ant\n")
        (root / "labels" / "train" / "frame.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
        assert detect(root).class_names == ["zebra", "ant"]


class TestFormatDetection:
    def test_bare_images_are_their_own_format(self, tmp_path):
        (tmp_path / "a.png").write_bytes(_PNG)
        (tmp_path / "b.png").write_bytes(_PNG)
        detected = detect(tmp_path)
        assert detected.format == "images"
        assert detected.image_count == 2
        assert detected.annotation_count == 0

    def test_a_folder_with_no_images_is_refused(self, tmp_path):
        (tmp_path / "notes.txt").write_text("nothing here")
        with pytest.raises(DatasetImportError):
            detect(tmp_path)

    def test_descends_past_a_single_wrapping_directory(self, tmp_path):
        # What every archive of "mydataset/" looks like once unpacked.
        inner = tmp_path / "mydataset"
        (inner / "images").mkdir(parents=True)
        (inner / "images" / "a.png").write_bytes(_PNG)
        assert detect(tmp_path).root == inner

    def test_an_explicit_format_overrides_the_guess(self, tmp_path):
        # A Roboflow export ships YOLO labels the user may not want read.
        root = _export_tree(tmp_path, "yolo")
        assert detect(root).annotation_count == len(_ANNOTATIONS)
        assert detect(root, "images").annotation_count == 0

    def test_macos_resource_forks_are_not_images(self, tmp_path):
        (tmp_path / "real.png").write_bytes(_PNG)
        shadow = tmp_path / "__MACOSX"
        shadow.mkdir()
        (shadow / "._real.png").write_bytes(b"not an image")
        assert detect(tmp_path).image_count == 1


class TestYoloLabelReading:
    def _tree(self, tmp_path, label_text):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        (tmp_path / "images" / "a.png").write_bytes(_PNG)
        (tmp_path / "labels" / "a.txt").write_text(label_text)
        return tmp_path

    def test_an_empty_label_file_means_no_objects_not_no_annotation(self, tmp_path):
        root = self._tree(tmp_path, "")
        records = list(iter_images(root, "yolo"))
        assert len(records) == 1 and records[0].boxes == []

    def test_segmentation_labels_are_skipped_with_a_warning(self, tmp_path):
        root = self._tree(tmp_path, "0 0.1 0.1 0.2 0.2 0.3 0.3 0.4 0.4\n")
        warnings = []
        records = list(iter_images(root, "yolo", warnings))
        assert records[0].boxes == []
        assert any("polygon" in w for w in warnings)

    def test_a_zero_area_box_is_dropped(self, tmp_path):
        root = self._tree(tmp_path, "0 0.5 0.5 0.0 0.3\n")
        assert list(iter_images(root, "yolo"))[0].boxes == []

    def test_an_overhanging_box_is_clamped_without_moving_its_far_edge(self, tmp_path):
        # cx=0.95 w=0.2 reaches 1.05. Clamping the centre and keeping the width
        # would leave it overhanging; clamping the corners must not drag the
        # left edge along with the right.
        root = self._tree(tmp_path, "0 0.95 0.5 0.2 0.2\n")
        box = list(iter_images(root, "yolo"))[0].boxes[0]
        assert box.x == pytest.approx(0.85)
        assert box.x + box.w == pytest.approx(1.0)

    def test_an_unknown_class_index_still_produces_a_named_box(self, tmp_path):
        root = self._tree(tmp_path, "7 0.5 0.5 0.2 0.2\n")
        assert list(iter_images(root, "yolo"))[0].boxes[0].class_name == "class_7"


class TestYoloNames:
    def test_inline_list(self):
        assert parse_yolo_names("nc: 2\nnames: ['cat', 'dog']\n") == ["cat", "dog"]

    def test_block_list(self):
        assert parse_yolo_names("names:\n  - cat\n  - dog\nnc: 2\n") == ["cat", "dog"]

    def test_index_map_is_ordered_by_index_not_by_file_order(self):
        assert parse_yolo_names("names:\n  1: dog\n  0: cat\n") == ["cat", "dog"]

    def test_a_following_key_ends_the_block(self):
        assert parse_yolo_names("names:\n  - cat\npath: .\ntrain: images\n") == ["cat"]


class TestCoco:
    def _tree(self, tmp_path, *, width=200, height=100, file_name="a.png"):
        (tmp_path / "images").mkdir()
        (tmp_path / "images" / "a.png").write_bytes(_PNG)
        doc = {
            "images": [{"id": 1, "file_name": file_name, "width": width, "height": height}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 5, "bbox": [20, 10, 40, 30]}],
            "categories": [{"id": 5, "name": "cat"}],
        }
        (tmp_path / "annotations").mkdir()
        (tmp_path / "annotations" / "instances_train.json").write_text(json.dumps(doc))
        return tmp_path

    def test_pixel_boxes_are_scaled_by_the_declared_size(self, tmp_path):
        box = list(iter_images(self._tree(tmp_path), "coco"))[0].boxes[0]
        assert (box.x, box.y, box.w, box.h) == pytest.approx((0.1, 0.1, 0.2, 0.3))
        assert box.class_name == "cat"

    def test_an_image_declaring_no_size_is_skipped_not_scaled_by_zero(self, tmp_path):
        warnings = []
        root = self._tree(tmp_path, width=0, height=0)
        records = list(iter_images(root, "coco", warnings))
        assert records[0].boxes == []
        assert any("width/height" in w for w in warnings)

    def test_a_declared_file_that_does_not_exist_is_reported(self, tmp_path):
        warnings = []
        root = self._tree(tmp_path, file_name="missing.png")
        list(iter_images(root, "coco", warnings))
        assert any("missing.png" in w for w in warnings)


class TestVoc:
    def _tree(self, tmp_path, xml):
        (tmp_path / "JPEGImages").mkdir()
        (tmp_path / "Annotations").mkdir()
        (tmp_path / "JPEGImages" / "a.png").write_bytes(_PNG)
        (tmp_path / "Annotations" / "a.xml").write_text(xml)
        return tmp_path

    def test_one_based_inclusive_corners_are_undone(self, tmp_path):
        # xmin=21 in VOC is pixel index 20 counting from zero.
        root = self._tree(
            tmp_path,
            "<annotation><filename>a.png</filename>"
            "<size><width>200</width><height>100</height></size>"
            "<object><name>cat</name><bndbox>"
            "<xmin>21</xmin><ymin>11</ymin><xmax>60</xmax><ymax>40</ymax>"
            "</bndbox></object></annotation>",
        )
        box = list(iter_images(root, "voc"))[0].boxes[0]
        assert (box.x, box.y) == pytest.approx((0.1, 0.1))
        assert (box.x + box.w, box.y + box.h) == pytest.approx((0.3, 0.4))

    def test_an_annotation_with_no_size_is_skipped(self, tmp_path):
        warnings = []
        root = self._tree(
            tmp_path,
            "<annotation><filename>a.png</filename>"
            "<object><name>cat</name><bndbox>"
            "<xmin>1</xmin><ymin>1</ymin><xmax>9</xmax><ymax>9</ymax>"
            "</bndbox></object></annotation>",
        )
        assert list(iter_images(root, "voc", warnings))[0].boxes == []
        assert any("size" in w for w in warnings)


class TestArchiveSafety:
    def test_a_traversing_zip_entry_is_refused(self, tmp_path):
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../escaped.txt", "pwned")
        with pytest.raises(ArchiveError):
            extract_archive(archive, tmp_path / "out")
        assert not (tmp_path / "escaped.txt").exists()

    def test_an_absolute_zip_entry_is_written_inside_the_destination(self, tmp_path):
        archive = tmp_path / "abs.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("/etc/passwd", "nope")
        dest = tmp_path / "out"
        extract_archive(archive, dest)
        assert (dest / "etc" / "passwd").is_file()

    def test_a_traversing_tar_entry_is_refused(self, tmp_path):
        archive = tmp_path / "evil.tar"
        with tarfile.open(archive, "w") as tf:
            data = b"pwned"
            info = tarfile.TarInfo("../escaped.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        with pytest.raises(ArchiveError):
            extract_archive(archive, tmp_path / "out")
        assert not (tmp_path / "escaped.txt").exists()

    def test_a_tar_symlink_member_is_dropped_rather_than_followed(self, tmp_path):
        archive = tmp_path / "link.tar"
        with tarfile.open(archive, "w") as tf:
            info = tarfile.TarInfo("link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tf.addfile(info)
            data = b"x"
            real = tarfile.TarInfo("real.txt")
            real.size = len(data)
            tf.addfile(real, io.BytesIO(data))
        dest = tmp_path / "out"
        assert extract_archive(archive, dest) == 1
        assert not (dest / "link").exists()

    def test_a_declared_size_over_the_budget_is_refused_before_writing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.helpers.archiveUtils.MAX_EXTRACTED_BYTES", 8)
        archive = tmp_path / "big.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("big.bin", "x" * 1024)
        with pytest.raises(ArchiveError):
            extract_archive(archive, tmp_path / "out")

    def test_a_non_archive_is_refused(self, tmp_path):
        plain = tmp_path / "notes.txt"
        plain.write_text("just text")
        with pytest.raises(ArchiveError):
            extract_archive(plain, tmp_path / "out")
