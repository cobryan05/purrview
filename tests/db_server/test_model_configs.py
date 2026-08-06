"""Tests for the model configurations feature — CRUD, validation, and cascade behavior."""

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


async def _create_label(http_client, name: str, parent_uuid: str | None = None) -> str:
    """Create a label and return its uuid."""
    payload: dict = {"name": name, "color": "#ff0000"}
    if parent_uuid is not None:
        payload["parentUuid"] = parent_uuid
    r = await http_client.post("/api/labels", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["label"]["uuid"]


async def _create_config(
    http_client,
    name: str,
    nodes: list | None = None,
    description: str | None = None,
) -> dict:
    """Create a model config and return the config dict."""
    payload: dict = {"name": name, "nodes": nodes or []}
    if description is not None:
        payload["description"] = description
    r = await http_client.post("/api/model-configs", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["config"]


# Default valid node fields for use across tests.
_VALID_NODE = {
    "label_uuid": None,
    "model_type": "object_detection",
    "architecture": "yolo_v9_c",
}


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


async def test_create_and_list(http_client):
    cfg = await _create_config(http_client, "MyConfig")
    assert cfg["name"] == "MyConfig"
    assert cfg["uuid"]

    r = await http_client.get("/api/model-configs")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    uuids = [c["uuid"] for c in body["configs"]]
    assert cfg["uuid"] in uuids


async def test_get_single(http_client):
    cfg = await _create_config(http_client, "SingleConfig", description="desc")
    r = await http_client.get(f"/api/model-configs/{cfg['uuid']}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["config"]["uuid"] == cfg["uuid"]
    assert body["config"]["description"] == "desc"


async def test_create_with_nodes(http_client):
    label_uuid = await _create_label(http_client, "Cat")
    nodes = [
        {
            "label_uuid": None,
            "model_type": "object_detection",
            "architecture": "yolo_v9_c",
        },
        {
            "label_uuid": label_uuid,
            "model_type": "image_classifier",
            "architecture": "efficientnet_b0",
        },
    ]
    cfg = await _create_config(http_client, "WithNodes", nodes=nodes)
    assert len(cfg["nodes"]) == 2
    model_types = {n["modelType"] for n in cfg["nodes"]}
    assert model_types == {"object_detection", "image_classifier"}
    root_node = next(n for n in cfg["nodes"] if n["labelUuid"] is None)
    assert root_node["modelType"] == "object_detection"
    assert root_node["architecture"] == "yolo_v9_c"


async def test_get_model_config_node_includes_parent_config_uuid(http_client):
    """GET /api/model-config-nodes/{uuid} must echo model_config_uuid so callers

    (e.g. TrainModelTask) can look up bundle training targets for the node's
    parent config without a separate list-and-search round trip.
    """
    label_uuid = await _create_label(http_client, "Cat")
    nodes = [
        {"label_uuid": None, "model_type": "object_detection", "architecture": "yolo_v9_c"},
        {"label_uuid": label_uuid, "model_type": "image_classifier", "architecture": "efficientnet_b0"},
    ]
    cfg = await _create_config(http_client, "NodeParentLookup", nodes=nodes)
    node_uuid = next(n["uuid"] for n in cfg["nodes"] if n["labelUuid"] == label_uuid)

    r = await http_client.get(f"/api/model-config-nodes/{node_uuid}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["modelConfigUuid"] == cfg["uuid"]


async def test_update_scalar_fields(http_client):
    cfg = await _create_config(http_client, "OldName", description="old")
    r = await http_client.patch(
        f"/api/model-configs/{cfg['uuid']}",
        json={"name": "NewName", "description": "new"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["config"]["name"] == "NewName"
    assert body["config"]["description"] == "new"
    # Nodes unchanged (none were set)
    assert body["config"]["nodes"] == cfg["nodes"]


async def test_update_nodes_wholesale(http_client):
    label_uuid = await _create_label(http_client, "Dog")
    cfg = await _create_config(
        http_client,
        "NodeReplace",
        nodes=[
            {
                "label_uuid": None,
                "model_type": "object_detection",
                "architecture": "yolo_v9_c",
            }
        ],
    )
    assert len(cfg["nodes"]) == 1

    new_nodes = [
        {
            "label_uuid": None,
            "model_type": "image_classifier",
            "architecture": "resnet_18",
        },
        {
            "label_uuid": label_uuid,
            "model_type": "crop_classifier",
            "architecture": "mobilenet_v3_small",
        },
    ]
    r = await http_client.patch(
        f"/api/model-configs/{cfg['uuid']}",
        json={"nodes": new_nodes},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert len(body["config"]["nodes"]) == 2
    model_types = {n["modelType"] for n in body["config"]["nodes"]}
    assert model_types == {"image_classifier", "crop_classifier"}


async def test_update_nodes_clear(http_client):
    cfg = await _create_config(
        http_client,
        "ClearNodes",
        nodes=[
            {
                "label_uuid": None,
                "model_type": "object_detection",
                "architecture": "yolo_v9_c",
            }
        ],
    )
    assert len(cfg["nodes"]) == 1

    r = await http_client.patch(
        f"/api/model-configs/{cfg['uuid']}",
        json={"nodes": []},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["config"]["nodes"] == []


async def test_delete(http_client):
    cfg = await _create_config(http_client, "ToDelete")
    r = await http_client.delete(f"/api/model-configs/{cfg['uuid']}")
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    r2 = await http_client.get("/api/model-configs")
    uuids = [c["uuid"] for c in r2.json()["configs"]]
    assert cfg["uuid"] not in uuids


async def test_nodes_cascade_on_delete(http_client):
    label_uuid = await _create_label(http_client, "Bird")
    cfg = await _create_config(
        http_client,
        "CascadeTest",
        nodes=[
            {
                "label_uuid": None,
                "model_type": "object_detection",
                "architecture": "yolo_v9_c",
            },
            {
                "label_uuid": label_uuid,
                "model_type": "image_classifier",
                "architecture": "efficientnet_b0",
            },
        ],
    )
    assert len(cfg["nodes"]) == 2

    r = await http_client.delete(f"/api/model-configs/{cfg['uuid']}")
    assert r.json()["status"] == "success"

    # Config is gone; no 404-type error but config field is null
    r2 = await http_client.get(f"/api/model-configs/{cfg['uuid']}")
    body = r2.json()
    assert body["config"] is None


async def test_label_delete_cascades_node(http_client):
    """Deleting a label removes its node row; the config itself survives."""
    label_uuid = await _create_label(http_client, "Fish")
    cfg = await _create_config(
        http_client,
        "LabelCascade",
        nodes=[
            {
                "label_uuid": None,
                "model_type": "object_detection",
                "architecture": "yolo_v9_c",
            },
            {
                "label_uuid": label_uuid,
                "model_type": "image_classifier",
                "architecture": "efficientnet_b0",
            },
        ],
    )
    assert len(cfg["nodes"]) == 2

    # Delete the label
    r = await http_client.delete(f"/api/labels/{label_uuid}")
    assert r.json()["status"] == "success"

    # Config should still exist with only the root node remaining
    r2 = await http_client.get(f"/api/model-configs/{cfg['uuid']}")
    body = r2.json()
    assert body["status"] == "success"
    assert body["config"] is not None
    remaining_nodes = body["config"]["nodes"]
    assert len(remaining_nodes) == 1
    assert remaining_nodes[0]["labelUuid"] is None


async def test_reparent_label_preserves_model_config_nodes(http_client):
    """Constructive counterpart to test_label_delete_cascades_node: re-parenting
    a label (the only way to 'move a model in the hierarchy' today) must
    preserve every ModelConfigNode hanging off the moved label AND its
    descendants — same uuids, same model_type, same architecture, same
    roll_up_unchecked. ModelConfigNode is keyed by label_uuid, not by tree
    position, so a true re-parent (as opposed to delete+recreate) should not
    touch these rows at all.
    """
    old_parent = await _create_label(http_client, "OldParent")
    new_parent = await _create_label(http_client, "NewParent")
    moved_label = await _create_label(http_client, "MovedLabel", parent_uuid=old_parent)
    descendant = await _create_label(http_client, "Descendant", parent_uuid=moved_label)

    nodes = [
        {
            "label_uuid": None,
            "model_type": "object_detection",
            "architecture": "yolo_v9_c",
        },
        {
            # A sub-model boundary on the label being moved.
            "label_uuid": moved_label,
            "model_type": "crop_classifier",
            "architecture": "efficientnet_b0",
            "roll_up_unchecked": True,
        },
        {
            "label_uuid": descendant,
            "model_type": "image_classifier",
            "architecture": "resnet_18",
            "roll_up_unchecked": False,
        },
    ]
    cfg = await _create_config(http_client, "ReparentSurvives", nodes=nodes)
    assert len(cfg["nodes"]) == 3
    before_by_uuid = {n["uuid"]: n for n in cfg["nodes"]}

    # Re-parent the label — the whole point of the feature: no destructive
    # delete+recreate should be necessary, and none happens here.
    r = await http_client.patch(f"/api/labels/{moved_label}", json={"parentUuid": new_parent})
    assert r.status_code == 200
    assert r.json()["status"] == "success", r.json()

    r2 = await http_client.get(f"/api/model-configs/{cfg['uuid']}")
    body = r2.json()
    assert body["status"] == "success"
    after_nodes = body["config"]["nodes"]
    assert len(after_nodes) == 3

    after_by_uuid = {n["uuid"]: n for n in after_nodes}
    assert set(after_by_uuid) == set(before_by_uuid), "every node uuid must survive the move untouched"
    for uuid, before in before_by_uuid.items():
        after = after_by_uuid[uuid]
        assert after["labelUuid"] == before["labelUuid"]
        assert after["modelType"] == before["modelType"]
        assert after["architecture"] == before["architecture"]
        assert after["rollUpUnchecked"] == before["rollUpUnchecked"]

    # And the label tree itself actually moved.
    labels_r = await http_client.get("/api/labels")
    moved = next(l for l in labels_r.json()["labels"] if l["uuid"] == moved_label)
    assert moved["parentUuid"] == new_parent


async def test_invalid_model_type_rejected(http_client):
    r = await http_client.post(
        "/api/model-configs",
        json={
            "name": "BadType",
            "nodes": [
                {
                    "label_uuid": None,
                    "model_type": "nonexistent_type",
                    "architecture": "yolo_v9_c",
                }
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failure"
    assert "nonexistent_type" in body.get("message", "")


async def test_invalid_architecture_rejected(http_client):
    """Architecture that doesn't belong to the given model_type is rejected."""
    r = await http_client.post(
        "/api/model-configs",
        json={
            "name": "BadArch",
            "nodes": [
                {
                    "label_uuid": None,
                    "model_type": "object_detection",
                    "architecture": "efficientnet_b0",
                }
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failure"
    assert "efficientnet_b0" in body.get("message", "")


async def test_duplicate_label_uuid_rejected(http_client):
    label_uuid = await _create_label(http_client, "Rabbit")
    r = await http_client.post(
        "/api/model-configs",
        json={
            "name": "DupLabel",
            "nodes": [
                {
                    "label_uuid": label_uuid,
                    "model_type": "object_detection",
                    "architecture": "yolo_v9_c",
                },
                {
                    "label_uuid": label_uuid,
                    "model_type": "image_classifier",
                    "architecture": "resnet_18",
                },
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failure"


async def test_root_node_label_uuid_none(http_client):
    """label_uuid=null is valid as the root node."""
    cfg = await _create_config(
        http_client,
        "RootNull",
        nodes=[
            {
                "label_uuid": None,
                "model_type": "object_detection",
                "architecture": "yolo_v9_c",
            }
        ],
    )
    assert len(cfg["nodes"]) == 1
    assert cfg["nodes"][0]["labelUuid"] is None
    assert cfg["nodes"][0]["modelType"] == "object_detection"
    assert cfg["nodes"][0]["architecture"] == "yolo_v9_c"


# ─────────────────────────────────────────────────────────────────────────────
# Bundle training targets (GET /api/model-configs/{uuid}/training-targets)
# ─────────────────────────────────────────────────────────────────────────────

_ROOT_SPEC = {"model_type": "object_detection", "architecture": "yolo_v9_c"}
_SUB_SPEC = {"model_type": "crop_classifier", "architecture": "resnet_18"}
_SUB_SPEC_2 = {"model_type": "crop_classifier", "architecture": "efficientnet_b0"}


def _node(label_uuid, spec, roll_up_unchecked=False):
    return {"label_uuid": label_uuid, "roll_up_unchecked": roll_up_unchecked, **spec}


async def _fetch_targets(http_client, config_uuid: str) -> list:
    r = await http_client.get(f"/api/model-configs/{config_uuid}/training-targets")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["targets"]


async def test_targets_config_not_found(http_client):
    r = await http_client.get("/api/model-configs/no-such-uuid/training-targets")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failure"
    assert "not found" in body["message"].lower()


async def test_targets_config_without_nodes_yields_no_targets(http_client):
    cfg = await _create_config(http_client, "TargetsEmpty")
    assert await _fetch_targets(http_client, cfg["uuid"]) == []


async def test_targets_flat_config_single_top_target(http_client):
    cat = await _create_label(http_client, "Cat")
    dog = await _create_label(http_client, "Dog")
    cfg = await _create_config(
        http_client,
        "TargetsFlat",
        nodes=[_node(None, _ROOT_SPEC), _node(cat, _ROOT_SPEC), _node(dog, _ROOT_SPEC)],
    )

    targets = await _fetch_targets(http_client, cfg["uuid"])
    assert len(targets) == 1
    top = targets[0]
    assert top["role"] == "top"
    assert top["labelUuid"] is None
    assert top["modelType"] == "object_detection"
    assert top["architecture"] == "yolo_v9_c"
    assert top["classNames"] == ["Cat", "Dog"]
    assert top["labelUuids"] == [cat, dog]
    assert top["classRemap"] == {}
    root_node = next(n for n in cfg["nodes"] if n["labelUuid"] is None)
    assert top["nodeUuid"] == root_node["uuid"]


async def test_targets_boundary_produces_submodel(http_client):
    """A node whose spec differs from its inherited spec is a sub-model boundary;
    its assigned descendants roll up into it for the top model."""
    cat = await _create_label(http_client, "Cat")
    cat_a = await _create_label(http_client, "CatA", parent_uuid=cat)
    cat_b = await _create_label(http_client, "CatB", parent_uuid=cat)
    dog = await _create_label(http_client, "Dog")
    cfg = await _create_config(
        http_client,
        "TargetsBoundary",
        nodes=[
            _node(None, _ROOT_SPEC),
            _node(cat, _SUB_SPEC),  # spec differs from root -> boundary
            _node(cat_a, _SUB_SPEC),  # equals inherited (Cat's) spec -> NOT a boundary
            _node(cat_b, _SUB_SPEC),
            _node(dog, _ROOT_SPEC),
        ],
    )

    targets = await _fetch_targets(http_client, cfg["uuid"])
    assert [t["role"] for t in targets] == ["top", "submodel"]

    top, sub = targets
    assert top["classNames"] == ["Cat", "Dog"]
    assert top["classRemap"] == {cat_a: cat, cat_b: cat}

    assert sub["labelUuid"] == cat
    assert sub["modelType"] == "crop_classifier"
    assert sub["architecture"] == "resnet_18"
    assert sub["classNames"] == ["CatA", "CatB"]
    assert sub["labelUuids"] == [cat_a, cat_b]
    assert sub["classRemap"] == {}
    node_by_label = {n["labelUuid"]: n["uuid"] for n in cfg["nodes"]}
    assert sub["nodeUuid"] == node_by_label[cat]


async def test_targets_nested_boundaries_collapse_to_outermost(http_client):
    """With nested boundaries the top model sees only the outermost class; each
    boundary's sub-model distinguishes the labels that roll up directly to it."""
    animal = await _create_label(http_client, "Animal")
    cat = await _create_label(http_client, "Cat", parent_uuid=animal)
    cat_a = await _create_label(http_client, "CatA", parent_uuid=cat)
    cfg = await _create_config(
        http_client,
        "TargetsNested",
        nodes=[
            _node(None, _ROOT_SPEC),
            _node(animal, _SUB_SPEC),  # boundary (differs from root)
            _node(cat, _SUB_SPEC_2),  # boundary (differs from Animal's spec)
            _node(cat_a, _SUB_SPEC_2),  # equals inherited (Cat's) spec
        ],
    )

    targets = await _fetch_targets(http_client, cfg["uuid"])
    assert [t["role"] for t in targets] == ["top", "submodel", "submodel"]

    top, sub_animal, sub_cat = targets
    assert top["labelUuids"] == [animal]
    assert top["classRemap"] == {cat: animal, cat_a: animal}

    assert sub_animal["labelUuid"] == animal
    assert sub_animal["labelUuids"] == [cat]
    assert sub_animal["classRemap"] == {cat_a: cat}

    assert sub_cat["labelUuid"] == cat
    assert sub_cat["labelUuids"] == [cat_a]
    assert sub_cat["classRemap"] == {}


async def test_targets_rollup_unchecked_descendants_remap(http_client):
    """Unassigned hierarchy descendants of a roll_up_unchecked node remap to
    wherever their rollup parent resolves in each target."""
    cat = await _create_label(http_client, "Cat")
    cat_a = await _create_label(http_client, "CatA", parent_uuid=cat)  # no node
    cat_b = await _create_label(http_client, "CatB", parent_uuid=cat)  # no node
    cfg = await _create_config(
        http_client,
        "TargetsRollup",
        nodes=[_node(None, _ROOT_SPEC), _node(cat, _ROOT_SPEC, roll_up_unchecked=True)],
    )

    targets = await _fetch_targets(http_client, cfg["uuid"])
    assert len(targets) == 1
    top = targets[0]
    assert top["labelUuids"] == [cat]
    assert top["classRemap"] == {cat_a: cat, cat_b: cat}


async def test_targets_childless_boundary_has_empty_classes(http_client):
    """A boundary with no assigned children yields a sub-model target with an
    empty class list (nothing to distinguish) — callers decide how to handle it.
    Its roll_up_unchecked descendants train the TOP model as the boundary class,
    not the boundary's own sub-model."""
    cat = await _create_label(http_client, "Cat")
    cat_a = await _create_label(http_client, "CatA", parent_uuid=cat)  # no node
    cfg = await _create_config(
        http_client,
        "TargetsChildless",
        nodes=[_node(None, _ROOT_SPEC), _node(cat, _SUB_SPEC, roll_up_unchecked=True)],
    )

    targets = await _fetch_targets(http_client, cfg["uuid"])
    assert [t["role"] for t in targets] == ["top", "submodel"]

    top, sub = targets
    assert top["labelUuids"] == [cat]
    assert top["classRemap"] == {cat_a: cat}

    assert sub["labelUuid"] == cat
    assert sub["labelUuids"] == []
    assert sub["classNames"] == []
    assert sub["classRemap"] == {}


async def test_targets_db_client_returns_none_for_missing_config(db_client):
    assert await db_client.derive_training_targets("no-such-uuid") is None
