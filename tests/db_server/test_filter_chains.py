"""Tests for the filter chains feature — CRUD, step replace semantics, and cascade behavior."""

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_STEP_A = {
    "order": 0,
    "model_name": "corrupt_classifier",
    "confidence": 0.8,
    "iou_threshold": 0.5,
    "action": "discard_image",
    "action_params": {},
    "enabled": True,
}

_STEP_B = {
    "order": 1,
    "model_name": "water_valves",
    "confidence": 0.6,
    "iou_threshold": 0.4,
    "action": "relabel_bbox",
    "action_params": {"target_label_uuid": "label-cat"},
    "enabled": True,
}


async def _create_chain(
    http_client,
    name: str,
    steps: list | None = None,
    description: str | None = None,
) -> dict:
    payload: dict = {"name": name, "steps": steps or []}
    if description is not None:
        payload["description"] = description
    r = await http_client.post("/api/filter-chains", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["chain"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFilterChains:
    async def test_create_and_list(self, http_client):
        chain = await _create_chain(http_client, "MyChain")
        assert chain["name"] == "MyChain"
        assert chain["uuid"]

        r = await http_client.get("/api/filter-chains")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        uuids = [c["uuid"] for c in body["chains"]]
        assert chain["uuid"] in uuids

    async def test_get_single(self, http_client):
        chain = await _create_chain(http_client, "SingleChain", description="desc")
        r = await http_client.get(f"/api/filter-chains/{chain['uuid']}")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["chain"]["uuid"] == chain["uuid"]
        assert body["chain"]["description"] == "desc"

    async def test_get_missing_returns_failure(self, http_client):
        r = await http_client.get("/api/filter-chains/no-such-uuid")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "failure"

    async def test_create_with_steps_preserves_order_and_fields(self, http_client):
        chain = await _create_chain(http_client, "WithSteps", steps=[_STEP_A, _STEP_B])
        assert len(chain["steps"]) == 2
        steps_by_order = {s["order"]: s for s in chain["steps"]}
        assert steps_by_order[0]["modelName"] == "corrupt_classifier"
        assert steps_by_order[0]["action"] == "discard_image"
        assert steps_by_order[1]["modelName"] == "water_valves"
        assert steps_by_order[1]["action"] == "relabel_bbox"
        # action_params is a raw JSON blob, not a camelCased pydantic model.
        assert steps_by_order[1]["actionParams"] == {"target_label_uuid": "label-cat"}

    async def test_invalid_action_rejected(self, http_client):
        bad_step = {**_STEP_A, "action": "delete_everything"}
        r = await http_client.post(
            "/api/filter-chains",
            json={"name": "BadAction", "steps": [bad_step]},
        )
        assert r.status_code == 422

    async def test_update_scalar_fields(self, http_client):
        chain = await _create_chain(http_client, "OldName", description="old")
        r = await http_client.patch(
            f"/api/filter-chains/{chain['uuid']}",
            json={"name": "NewName", "description": "new"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["chain"]["name"] == "NewName"
        assert body["chain"]["description"] == "new"
        # Steps unchanged (none were set)
        assert body["chain"]["steps"] == chain["steps"]

    async def test_update_steps_wholesale_replace(self, http_client):
        chain = await _create_chain(http_client, "StepReplace", steps=[_STEP_A])
        assert len(chain["steps"]) == 1

        r = await http_client.patch(
            f"/api/filter-chains/{chain['uuid']}",
            json={"steps": [_STEP_B]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert len(body["chain"]["steps"]) == 1
        assert body["chain"]["steps"][0]["modelName"] == "water_valves"

    async def test_update_steps_none_leaves_unchanged(self, http_client):
        chain = await _create_chain(http_client, "StepsUntouched", steps=[_STEP_A])
        r = await http_client.patch(
            f"/api/filter-chains/{chain['uuid']}",
            json={"description": "just a description change"},
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["chain"]["steps"]) == 1
        assert body["chain"]["steps"][0]["modelName"] == "corrupt_classifier"

    async def test_update_steps_empty_list_clears(self, http_client):
        chain = await _create_chain(http_client, "ClearSteps", steps=[_STEP_A, _STEP_B])
        assert len(chain["steps"]) == 2

        r = await http_client.patch(
            f"/api/filter-chains/{chain['uuid']}",
            json={"steps": []},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["chain"]["steps"] == []

    async def test_update_missing_returns_failure(self, http_client):
        r = await http_client.patch(
            "/api/filter-chains/no-such-uuid",
            json={"name": "Whatever"},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "failure"

    async def test_delete(self, http_client):
        chain = await _create_chain(http_client, "ToDelete")
        r = await http_client.delete(f"/api/filter-chains/{chain['uuid']}")
        assert r.status_code == 200
        assert r.json()["status"] == "success"

        r2 = await http_client.get("/api/filter-chains")
        uuids = [c["uuid"] for c in r2.json()["chains"]]
        assert chain["uuid"] not in uuids

    async def test_delete_missing_returns_failure(self, http_client):
        r = await http_client.delete("/api/filter-chains/no-such-uuid")
        assert r.status_code == 200
        assert r.json()["status"] == "failure"

    async def test_steps_cascade_on_delete(self, http_client):
        chain = await _create_chain(http_client, "CascadeTest", steps=[_STEP_A, _STEP_B])
        assert len(chain["steps"]) == 2

        r = await http_client.delete(f"/api/filter-chains/{chain['uuid']}")
        assert r.json()["status"] == "success"

        r2 = await http_client.get(f"/api/filter-chains/{chain['uuid']}")
        body = r2.json()
        assert body["chain"] is None

    async def test_duplicate_name_rejected(self, http_client):
        await _create_chain(http_client, "DupName")
        r = await http_client.post(
            "/api/filter-chains",
            json={"name": "DupName", "steps": []},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "failure"


class TestFilterChainsDbClient:
    async def test_create_and_replace_steps_unit_level(self, db_client):
        from apps.helpers.db.types import (
            ActionKind,
            FilterChainCreate,
            FilterChainStepWrite,
            FilterChainUpdate,
        )

        created = await db_client.create_filter_chain(
            FilterChainCreate(
                name="UnitChain",
                steps=[
                    FilterChainStepWrite(
                        order=0,
                        model_name="m1",
                        action=ActionKind.TAG_IMAGE,
                        action_params={"tag_uuid": "tag-1"},
                    )
                ],
            )
        )
        assert len(created.steps) == 1
        assert created.steps[0].action == "tag_image"

        updated = await db_client.update_filter_chain(
            created.uuid,
            FilterChainUpdate(
                steps=[
                    FilterChainStepWrite(order=0, model_name="m2", action=ActionKind.TAG_BBOX),
                ]
            ),
        )
        assert updated is not None
        assert len(updated.steps) == 1
        assert updated.steps[0].model_name == "m2"
        assert updated.steps[0].action == "tag_bbox"

    async def test_get_returns_none_for_missing(self, db_client):
        assert await db_client.get_filter_chain("no-such-uuid") is None

    async def test_delete_returns_false_for_missing(self, db_client):
        assert await db_client.delete_filter_chain("no-such-uuid") is False
