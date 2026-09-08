"""Tests for the dataset export endpoint (`GET /api/datasets/{uuid}/export`).

The interesting behaviour is what lands *inside* the tar: which images are in
scope, which class index a box gets, and whether the annotation numbers survive
the conversion out of the DB's normalised coordinates. Those are the things a
refactor can silently break, so they are what these assert — not the presence of
the endpoint.
"""

import io
import json
import tarfile
import uuid

from PIL import Image

from apps.helpers.db.types import UNVERIFIED_TAG_UUID, VERIFIED_TAG_UUID

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _png_bytes(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


async def _create_label(http_client, name: str, parent_uuid: str | None = None) -> str:
    payload: dict = {"name": name, "color": "#ff0000"}
    if parent_uuid is not None:
        payload["parentUuid"] = parent_uuid
    r = await http_client.post("/api/labels", json=payload)
    assert r.status_code == 200, r.text
    return r.json()["label"]["uuid"]


async def _create_dataset(http_client, name: str, label_uuids: list) -> dict:
    r = await http_client.post(
        "/api/datasets",
        json={"name": name, "filter_json": {}, "label_uuids": label_uuids},
    )
    assert r.status_code == 200, r.text
    return r.json()["dataset"]


async def _register_image(http_client, files_root, name: str, *, width: int = 100, height: int = 50) -> str:
    (files_root / name).write_bytes(_png_bytes(width, height))
    r = await http_client.post("/api/images/metadata/update", json={"imagePath": name, "boxes": []})
    assert r.status_code == 200, r.text
    return name


async def _add_bbox(http_client, image_path, label_uuid, tag_uuids, *, x=0.1, y=0.2, width=0.4, height=0.2) -> str:
    box_uuid = str(uuid.uuid4())
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": image_path,
            "boxes": [
                {
                    "uuid": box_uuid,
                    "labelUuid": label_uuid,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "tagUuids": tag_uuids,
                }
            ],
        },
    )
    assert r.status_code == 200, r.text
    return box_uuid


async def _set_image_label(http_client, image_path, label_uuid, status) -> None:
    r = await http_client.post(
        "/api/images/labels/set",
        json={"imagePath": image_path, "labelUuid": label_uuid, "status": status},
    )
    assert r.status_code == 200, r.text


async def _export(http_client, dataset_uuid: str, **params) -> tarfile.TarFile:
    r = await http_client.get(f"/api/datasets/{dataset_uuid}/export", params=params)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/x-tar")
    return tarfile.open(fileobj=io.BytesIO(r.content), mode="r:")


def _text(tar: tarfile.TarFile, name: str) -> str:
    return tar.extractfile(name).read().decode()


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDatasetExport:
    async def test_yolo_export_pairs_image_with_normalised_label_file(self, http_client, files_root):
        cat = await _create_label(http_client, "cat")
        ds = await _create_dataset(http_client, "export-yolo", [cat])
        img = await _register_image(http_client, files_root, "yolo_cat.png")
        await _add_bbox(http_client, img, cat, [VERIFIED_TAG_UUID], x=0.1, y=0.2, width=0.4, height=0.2)
        await _set_image_label(http_client, img, cat, "present")

        tar = await _export(http_client, ds["uuid"], format="yolo", val_fraction=0)
        names = tar.getnames()

        images = [n for n in names if n.startswith("images/train/")]
        assert len(images) == 1
        # The label file must share the image's stem — that pairing *is* the format.
        stem = images[0].rsplit("/", 1)[1].rsplit(".", 1)[0]
        assert f"labels/train/{stem}.txt" in names

        cls, cx, cy, w, h = _text(tar, f"labels/train/{stem}.txt").split()
        assert cls == "0"
        # (x, y, w, h) top-left → YOLO centre form: cx = 0.1 + 0.4/2, cy = 0.2 + 0.2/2
        assert abs(float(cx) - 0.30) < 1e-4
        assert abs(float(cy) - 0.30) < 1e-4
        assert abs(float(w) - 0.40) < 1e-4
        assert abs(float(h) - 0.20) < 1e-4

        data_yaml = _text(tar, "data.yaml")
        assert "nc: 1" in data_yaml
        assert "0: cat" in data_yaml
        # No val split was requested, so data.yaml must not point at a val dir
        # that doesn't exist in the archive.
        assert "val:" not in data_yaml

    async def test_val_fraction_splits_images_into_both_dirs(self, http_client, files_root):
        cat = await _create_label(http_client, "cat")
        ds = await _create_dataset(http_client, "export-split", [cat])
        for i in range(4):
            img = await _register_image(http_client, files_root, f"split_{i}.png")
            await _add_bbox(http_client, img, cat, [VERIFIED_TAG_UUID])
            await _set_image_label(http_client, img, cat, "present")

        tar = await _export(http_client, ds["uuid"], format="yolo", val_fraction=0.25)
        names = tar.getnames()
        assert len([n for n in names if n.startswith("images/train/")]) == 3
        assert len([n for n in names if n.startswith("images/val/")]) == 1
        assert "val: images/val" in _text(tar, "data.yaml")

    async def test_child_label_box_rolls_up_into_selected_parent_class(self, http_client, files_root):
        """On an exported image, a 'tabby' box is a 'cat' box when the dataset picks 'cat'.

        Dropping it would be the silent failure: the image still exports, just
        with one animal unannotated, and the model learns cats are background.
        'tabby' must not become a second class either — the dataset named one.
        """
        cat = await _create_label(http_client, "cat")
        tabby = await _create_label(http_client, "tabby", parent_uuid=cat)
        ds = await _create_dataset(http_client, "export-rollup", [cat])
        img = await _register_image(http_client, files_root, "rollup.png")
        # One update call: /images/metadata/update replaces the image's box list.
        r = await http_client.post(
            "/api/images/metadata/update",
            json={
                "imagePath": img,
                "boxes": [
                    {
                        "uuid": str(uuid.uuid4()),
                        "labelUuid": lbl,
                        "x": x,
                        "y": x,
                        "width": 0.2,
                        "height": 0.2,
                        "tagUuids": [VERIFIED_TAG_UUID],
                    }
                    for lbl, x in ((cat, 0.1), (tabby, 0.5))
                ],
            },
        )
        assert r.status_code == 200, r.text
        await _set_image_label(http_client, img, cat, "present")

        tar = await _export(http_client, ds["uuid"], format="yolo", val_fraction=0)
        label_files = [n for n in tar.getnames() if n.startswith("labels/train/")]
        assert len(label_files) == 1
        lines = _text(tar, label_files[0]).strip().splitlines()
        assert len(lines) == 2
        assert {line.split()[0] for line in lines} == {"0"}
        assert "nc: 1" in _text(tar, "data.yaml")

    async def test_unverified_image_is_excluded_unless_verification_is_waived(self, http_client, files_root):
        """The default export is training-shaped; require_verification=false is raw."""
        cat = await _create_label(http_client, "cat")
        ds = await _create_dataset(http_client, "export-unverified", [cat])
        img = await _register_image(http_client, files_root, "unverified.png")
        await _add_bbox(http_client, img, cat, [UNVERIFIED_TAG_UUID])

        strict = await _export(http_client, ds["uuid"], format="yolo", val_fraction=0)
        assert [n for n in strict.getnames() if n.startswith("images/")] == []

        loose = await _export(http_client, ds["uuid"], format="yolo", val_fraction=0, require_verification="false")
        loose_images = [n for n in loose.getnames() if n.startswith("images/train/")]
        assert len(loose_images) == 1
        # The image comes along, but an unverified box is still not an annotation.
        stem = loose_images[0].rsplit("/", 1)[1].rsplit(".", 1)[0]
        assert _text(loose, f"labels/train/{stem}.txt").strip() == ""

    async def test_coco_export_converts_boxes_to_pixels_of_the_real_image(self, http_client, files_root):
        cat = await _create_label(http_client, "cat")
        ds = await _create_dataset(http_client, "export-coco", [cat])
        img = await _register_image(http_client, files_root, "coco.png", width=200, height=100)
        await _add_bbox(http_client, img, cat, [VERIFIED_TAG_UUID], x=0.1, y=0.2, width=0.4, height=0.2)
        await _set_image_label(http_client, img, cat, "present")

        tar = await _export(http_client, ds["uuid"], format="coco", val_fraction=0)
        doc = json.loads(_text(tar, "annotations/instances_train.json"))

        assert [c["name"] for c in doc["categories"]] == ["cat"]
        assert doc["images"][0]["width"] == 200 and doc["images"][0]["height"] == 100
        ann = doc["annotations"][0]
        assert ann["category_id"] == doc["categories"][0]["id"]
        assert ann["image_id"] == doc["images"][0]["id"]
        # 0.1*200, 0.2*100, 0.4*200, 0.2*100
        assert ann["bbox"] == [20.0, 20.0, 80.0, 20.0]
        assert ann["area"] == 1600.0

    async def test_voc_export_writes_corner_pixels_and_an_image_set(self, http_client, files_root):
        cat = await _create_label(http_client, "cat")
        ds = await _create_dataset(http_client, "export-voc", [cat])
        img = await _register_image(http_client, files_root, "voc.png", width=200, height=100)
        await _add_bbox(http_client, img, cat, [VERIFIED_TAG_UUID], x=0.1, y=0.2, width=0.4, height=0.2)
        await _set_image_label(http_client, img, cat, "present")

        tar = await _export(http_client, ds["uuid"], format="voc", val_fraction=0)
        xml_name = [n for n in tar.getnames() if n.startswith("Annotations/")][0]
        xml = _text(tar, xml_name)

        assert "<width>200</width>" in xml and "<height>100</height>" in xml
        assert "<name>cat</name>" in xml
        # x1 = 0.1*200 = 20 (+1, VOC is 1-based), x2 = 0.5*200 = 100
        assert "<xmin>21</xmin>" in xml and "<xmax>100</xmax>" in xml
        assert "<ymin>21</ymin>" in xml and "<ymax>40</ymax>" in xml

        stem = xml_name.rsplit("/", 1)[1][: -len(".xml")]
        assert _text(tar, "ImageSets/Main/train.txt").split() == [stem]

    async def test_missing_image_file_is_reported_not_silently_dropped(self, http_client, files_root):
        cat = await _create_label(http_client, "cat")
        ds = await _create_dataset(http_client, "export-missing", [cat])
        img = await _register_image(http_client, files_root, "gone.png")
        await _add_bbox(http_client, img, cat, [VERIFIED_TAG_UUID])
        await _set_image_label(http_client, img, cat, "present")
        (files_root / "gone.png").unlink()

        tar = await _export(http_client, ds["uuid"], format="yolo", val_fraction=0)
        assert [n for n in tar.getnames() if n.startswith("images/")] == []
        assert "gone.png" in _text(tar, "SKIPPED.txt")

    async def test_unknown_format_is_rejected_and_missing_dataset_is_404(self, http_client, files_root):
        cat = await _create_label(http_client, "cat")
        ds = await _create_dataset(http_client, "export-errors", [cat])

        r = await http_client.get(f"/api/datasets/{ds['uuid']}/export", params={"format": "pascal-ish"})
        assert r.status_code == 400

        r = await http_client.get("/api/datasets/does-not-exist/export", params={"format": "yolo"})
        assert r.status_code == 404


class TestDatasetExportFormats:
    async def test_format_list_matches_what_export_accepts(self, http_client, files_root):
        """The picker is built from this list — every id it advertises must work."""
        r = await http_client.get("/api/datasets/export-formats")
        assert r.status_code == 200, r.text
        formats = r.json()["formats"]
        assert formats

        cat = await _create_label(http_client, "cat")
        ds = await _create_dataset(http_client, "export-formats", [cat])
        for fmt in formats:
            resp = await http_client.get(f"/api/datasets/{ds['uuid']}/export", params={"format": fmt["id"]})
            assert resp.status_code == 200, f"{fmt['id']}: {resp.text}"
