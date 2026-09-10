"""Readiness and the detection dataset must gate on per-class coverage.

Both used to gate on a *count* of verified ImageLabel rows: the detection dataset
required `len(classes)` rows drawn from the whole in-scope label set (classes plus
their rollup children), and readiness required just 1 row whenever rollup was
active. With rollup, those two sets differ in size, so an image could clear the
count while leaving whole classes undecided — the count says "8 of 15 labels
decided", which is silent about *which* 8.

That is not a rounding error. On the dataset this was found against, 2113 of the
2545 images the trainer accepted had three of the eight classes untouched; every
one of those that actually contained the missing class taught the model to miss it.

The gate is now "every class is accounted for" — a class being accounted for on an
image when any label that speaks for it (itself or a rollup child) carries a
verified bbox or an explicit PRESENT/ABSENT.
"""

import uuid

from apps.helpers.db.types import VERIFIED_TAG_UUID
from tests.conftest import make_png


async def _create_label(http_client, name: str, parent_uuid: str | None = None) -> str:
    payload: dict = {"name": name, "color": "#ff0000"}
    if parent_uuid is not None:
        payload["parentUuid"] = parent_uuid
    r = await http_client.post("/api/labels", json=payload)
    assert r.status_code == 200, r.text
    return r.json()["label"]["uuid"]


async def _create_dataset(http_client, name: str) -> str:
    r = await http_client.post("/api/datasets", json={"name": name, "filter_json": {}, "label_uuids": []})
    assert r.status_code == 200, r.text
    return r.json()["dataset"]["uuid"]


async def _create_model_config(http_client, name: str, nodes: list) -> dict:
    """Create a config whose nodes are preceded by a root node (label_uuid absent).

    The root is what makes this a *multi-class* target: a training config pinned to
    a label's own node covers one class and cannot express the bug, which needs a
    second class to leave undecided.
    """
    root = {"model_type": "object_detection", "architecture": "yolo_v9_c"}
    r = await http_client.post("/api/model-configs", json={"name": name, "nodes": [root] + nodes})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["config"]


def _root_node_uuid(mc: dict) -> str:
    root = [n for n in mc["nodes"] if not n.get("labelUuid") and not n.get("label_uuid")]
    assert len(root) == 1, f"expected exactly one root node, got {mc['nodes']}"
    return root[0]["uuid"]


async def _create_training_config(http_client, name: str, dataset_uuid: str, node_uuid: str) -> str:
    r = await http_client.post(
        "/api/training-configs",
        json={"name": name, "datasetUuids": [dataset_uuid], "modelConfigNodeUuid": node_uuid},
    )
    assert r.status_code == 200, r.text
    return r.json()["config"]["uuid"]


async def _register_image(http_client, files_root, name: str) -> str:
    (files_root / name).write_bytes(make_png())
    r = await http_client.post("/api/images/metadata/update", json={"imagePath": name, "boxes": []})
    assert r.status_code == 200, r.text
    return name


async def _add_verified_bbox(http_client, image_path: str, label_uuid: str) -> None:
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": image_path,
            "boxes": [
                {
                    "uuid": str(uuid.uuid4()),
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [VERIFIED_TAG_UUID],
                }
            ],
        },
    )
    assert r.status_code == 200, r.text


async def _set_image_label(http_client, image_path: str, label_uuid: str, status: str) -> None:
    r = await http_client.post(
        "/api/images/labels/set",
        json={"imagePath": image_path, "labelUuid": label_uuid, "status": status},
    )
    assert r.status_code == 200, r.text


async def _readiness(http_client, tc_uuid: str) -> dict:
    r = await http_client.get(f"/api/training-configs/{tc_uuid}/readiness")
    assert r.status_code == 200, r.text
    return r.json()


async def _detection_dataset(http_client, tc_uuid: str) -> dict:
    r = await http_client.get(f"/api/training-configs/{tc_uuid}/detection-dataset")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body


async def _ready_filenames(http_client, tc_uuid: str) -> set[str]:
    r = await http_client.post(f"/api/training-configs/{tc_uuid}/ready-images", json={"limit": 50})
    assert r.status_code == 200, r.text
    return {img["filename"].rsplit("/", 1)[-1] for img in r.json()["images"]}


class TestClassCoverageGate:
    """Two classes, one of which (`vehicle`) rolls its children up.

    In-scope labels number four (vehicle, car, truck, dog) against two classes, so
    an image can carry more verified rows than there are classes while `dog` has
    never been looked at — the shape the old count-based gate could not see.
    """

    async def _fixture(self, http_client, files_root, prefix: str):
        vehicle = await _create_label(http_client, f"{prefix}-vehicle")
        car = await _create_label(http_client, f"{prefix}-car", parent_uuid=vehicle)
        await _create_label(http_client, f"{prefix}-truck", parent_uuid=vehicle)
        dog = await _create_label(http_client, f"{prefix}-dog")

        mc = await _create_model_config(
            http_client,
            f"mc-{prefix}",
            nodes=[
                {
                    "label_uuid": vehicle,
                    "model_type": "object_detection",
                    "architecture": "yolo_v9_c",
                    "roll_up_unchecked": True,
                },
                {"label_uuid": dog, "model_type": "object_detection", "architecture": "yolo_v9_c"},
            ],
        )
        ds = await _create_dataset(http_client, f"ds-{prefix}")
        tc = await _create_training_config(http_client, f"tc-{prefix}", ds, _root_node_uuid(mc))

        # covered.png: a verified car box (so `vehicle` is accounted for) AND an
        # explicit decision on `dog`. Every class has been looked at.
        covered = await _register_image(http_client, files_root, f"{prefix}_covered.png")
        await _add_verified_bbox(http_client, covered, car)
        await _set_image_label(http_client, covered, vehicle, "present")
        await _set_image_label(http_client, covered, dog, "absent")

        # gap.png: the same verified car box and *two* verified rows (vehicle, car),
        # which clears a threshold of 2 — but nobody has ever decided about `dog`.
        gap = await _register_image(http_client, files_root, f"{prefix}_gap.png")
        await _add_verified_bbox(http_client, gap, car)
        await _set_image_label(http_client, gap, vehicle, "present")
        await _set_image_label(http_client, gap, car, "present")

        return tc

    async def test_detection_dataset_excludes_an_image_with_an_undecided_class(self, http_client, files_root):
        tc = await self._fixture(http_client, files_root, "dsgate")

        filenames = {
            img["filename"].rsplit("/", 1)[-1] for img in (await _detection_dataset(http_client, tc))["images"]
        }
        assert "dsgate_covered.png" in filenames
        assert "dsgate_gap.png" not in filenames, (
            "gap.png has no decision at all about `dog`, so a `dog` standing in it "
            "would be trained as background — it must not reach the trainer"
        )

    async def test_readiness_does_not_count_an_image_with_an_undecided_class(self, http_client, files_root):
        tc = await self._fixture(http_client, files_root, "rdgate")

        readiness = await _readiness(http_client, tc)
        assert readiness["rollUpActive"] is True
        assert readiness["readyCount"] == 1, (
            "rollup used to drop the threshold to a single verified label, so any "
            "one decision made an image ready no matter how many classes were left"
        )
        assert readiness["notStartedCount"] + readiness["needsReviewCount"] == 1

    async def test_ready_count_equals_what_the_trainer_receives(self, http_client, files_root):
        """The invariant the two gates exist to keep: the readiness panel's headline
        number is the size of the manifest a run started now would train on."""
        tc = await self._fixture(http_client, files_root, "agree")

        readiness = await _readiness(http_client, tc)
        dataset = await _detection_dataset(http_client, tc)
        assert readiness["readyCount"] == len(dataset["images"])

    async def test_ready_image_list_matches_the_ready_count(self, http_client, files_root):
        """The count and the browse-this-bucket list are computed separately; a gate
        change that moves one without the other makes the panel unclickable-through."""
        tc = await self._fixture(http_client, files_root, "listagree")

        readiness = await _readiness(http_client, tc)
        assert await _ready_filenames(http_client, tc) == {"listagree_covered.png"}
        assert readiness["readyCount"] == 1

    async def test_positive_split_counts_images_carrying_a_box(self, http_client, files_root):
        """`ready_positive` is what the trainer sees as a positive sample: an image
        with at least one verified in-scope box, not merely one marked PRESENT."""
        tc = await self._fixture(http_client, files_root, "split")

        readiness = await _readiness(http_client, tc)
        dataset = await _detection_dataset(http_client, tc)
        with_boxes = [img for img in dataset["images"] if img["annotations"]]
        assert readiness["readyPositiveCount"] == len(with_boxes)
        assert readiness["readyNegativeCount"] == len(dataset["images"]) - len(with_boxes)


class TestEitherMediumAccountsForAClass:
    """A class is accounted for by a verified bbox *or* an explicit PRESENT/ABSENT.

    Scope is not the discriminator it looks like — object-scope labels are routinely
    trained by an image_classifier node with no boxes anywhere — so a box cannot be
    required to call a class decided.

    The known cost of the looser rule: for a *detection* target, an image marked
    PRESENT with nothing drawn is ready and reaches the trainer carrying no
    annotation for that class, which is what asserting absence looks like. The
    Dataset Coverage panel's co-label gaps are what surface that; tightening this
    gate to demand a box would be a deliberate change, and would flip the first
    test below.
    """

    async def test_present_without_a_box_accounts_for_the_class(self, http_client, files_root):
        dog = await _create_label(http_client, "presonly-dog")
        mc = await _create_model_config(
            http_client,
            "mc-presonly",
            nodes=[{"label_uuid": dog, "model_type": "object_detection", "architecture": "yolo_v9_c"}],
        )
        ds = await _create_dataset(http_client, "ds-presonly")
        tc = await _create_training_config(http_client, "tc-presonly", ds, _root_node_uuid(mc))

        img = await _register_image(http_client, files_root, "presonly.png")
        await _set_image_label(http_client, img, dog, "present")

        readiness = await _readiness(http_client, tc)
        assert readiness["readyCount"] == 1
        assert readiness["readyPositiveCount"] == 1
        assert (await _detection_dataset(http_client, tc))["images"][0]["annotations"] == []

    async def test_absent_alone_is_a_background_sample(self, http_client, files_root):
        dog = await _create_label(http_client, "absonly-dog")
        mc = await _create_model_config(
            http_client,
            "mc-absonly",
            nodes=[{"label_uuid": dog, "model_type": "object_detection", "architecture": "yolo_v9_c"}],
        )
        ds = await _create_dataset(http_client, "ds-absonly")
        tc = await _create_training_config(http_client, "tc-absonly", ds, _root_node_uuid(mc))

        img = await _register_image(http_client, files_root, "absonly.png")
        await _set_image_label(http_client, img, dog, "absent")

        readiness = await _readiness(http_client, tc)
        assert readiness["readyCount"] == 1
        assert readiness["readyNegativeCount"] == 1
        assert (await _detection_dataset(http_client, tc))["images"][0]["annotations"] == []
