"""Tests for the model groups feature — CRUD, nesting, cycle rejection, and cascade behavior."""

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


async def _create_group(
    http_client,
    name: str,
    members: list | None = None,
    description: str | None = None,
) -> dict:
    payload: dict = {"name": name, "members": members or []}
    if description is not None:
        payload["description"] = description
    r = await http_client.post("/api/model-groups", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["group"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModelGroups:
    async def test_create_and_list(self, http_client):
        group = await _create_group(http_client, "MyGroup")
        assert group["name"] == "MyGroup"
        assert group["uuid"]

        r = await http_client.get("/api/model-groups")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        uuids = [g["uuid"] for g in body["groups"]]
        assert group["uuid"] in uuids

    async def test_get_single(self, http_client):
        group = await _create_group(http_client, "SingleGroup", description="desc")
        r = await http_client.get(f"/api/model-groups/{group['uuid']}")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["group"]["uuid"] == group["uuid"]
        assert body["group"]["description"] == "desc"

    async def test_get_missing_returns_failure(self, http_client):
        r = await http_client.get("/api/model-groups/no-such-uuid")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "failure"

    async def test_create_with_model_members_preserves_order(self, http_client):
        members = [
            {"order": 0, "model_name": "yolov8n"},
            {"order": 1, "model_name": "yolov8s"},
        ]
        group = await _create_group(http_client, "WithModels", members=members)
        assert len(group["members"]) == 2
        members_by_order = {m["order"]: m for m in group["members"]}
        assert members_by_order[0]["modelName"] == "yolov8n"
        assert members_by_order[1]["modelName"] == "yolov8s"

    async def test_nested_group_member(self, http_client):
        inner = await _create_group(http_client, "Inner", members=[{"order": 0, "model_name": "yolov8n"}])
        outer = await _create_group(
            http_client,
            "Outer",
            members=[
                {"order": 0, "model_name": "yolov8s"},
                {"order": 1, "child_group_uuid": inner["uuid"]},
            ],
        )
        assert len(outer["members"]) == 2
        by_order = {m["order"]: m for m in outer["members"]}
        assert by_order[1]["childGroupUuid"] == inner["uuid"]
        assert by_order[1]["modelName"] is None

    async def test_member_requires_exactly_one_kind_both_set_rejected(self, http_client):
        r = await http_client.post(
            "/api/model-groups",
            json={
                "name": "BadBoth",
                "members": [{"order": 0, "model_name": "yolov8n", "child_group_uuid": "some-uuid"}],
            },
        )
        assert r.status_code == 422

    async def test_member_requires_exactly_one_kind_neither_set_rejected(self, http_client):
        r = await http_client.post(
            "/api/model-groups",
            json={"name": "BadNeither", "members": [{"order": 0}]},
        )
        assert r.status_code == 422

    async def test_duplicate_member_order_rejected(self, http_client):
        r = await http_client.post(
            "/api/model-groups",
            json={
                "name": "DupOrder",
                "members": [
                    {"order": 0, "model_name": "yolov8n"},
                    {"order": 0, "model_name": "yolov8s"},
                ],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "failure"

    async def test_update_scalar_fields(self, http_client):
        group = await _create_group(http_client, "OldName", description="old")
        r = await http_client.patch(
            f"/api/model-groups/{group['uuid']}",
            json={"name": "NewName", "description": "new"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["group"]["name"] == "NewName"
        assert body["group"]["description"] == "new"
        # Members unchanged (none were set)
        assert body["group"]["members"] == group["members"]

    async def test_update_members_wholesale_replace(self, http_client):
        group = await _create_group(http_client, "MemberReplace", members=[{"order": 0, "model_name": "yolov8n"}])
        assert len(group["members"]) == 1

        r = await http_client.patch(
            f"/api/model-groups/{group['uuid']}",
            json={"members": [{"order": 0, "model_name": "yolov8s"}]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert len(body["group"]["members"]) == 1
        assert body["group"]["members"][0]["modelName"] == "yolov8s"

    async def test_update_members_none_leaves_unchanged(self, http_client):
        group = await _create_group(http_client, "MembersUntouched", members=[{"order": 0, "model_name": "yolov8n"}])
        r = await http_client.patch(
            f"/api/model-groups/{group['uuid']}",
            json={"description": "just a description change"},
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["group"]["members"]) == 1
        assert body["group"]["members"][0]["modelName"] == "yolov8n"

    async def test_update_members_empty_list_clears(self, http_client):
        group = await _create_group(
            http_client,
            "ClearMembers",
            members=[{"order": 0, "model_name": "yolov8n"}, {"order": 1, "model_name": "yolov8s"}],
        )
        assert len(group["members"]) == 2

        r = await http_client.patch(f"/api/model-groups/{group['uuid']}", json={"members": []})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["group"]["members"] == []

    async def test_update_missing_returns_failure(self, http_client):
        r = await http_client.patch("/api/model-groups/no-such-uuid", json={"name": "Whatever"})
        assert r.status_code == 200
        assert r.json()["status"] == "failure"

    async def test_direct_self_reference_cycle_rejected(self, http_client):
        group = await _create_group(http_client, "SelfRef")
        r = await http_client.patch(
            f"/api/model-groups/{group['uuid']}",
            json={"members": [{"order": 0, "child_group_uuid": group["uuid"]}]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "failure"
        assert "cycle" in body["message"].lower()

    async def test_transitive_cycle_rejected(self, http_client):
        # A contains B; now try to make B contain A — a cycle two hops back.
        a = await _create_group(http_client, "CycleA")
        b = await _create_group(http_client, "CycleB", members=[{"order": 0, "child_group_uuid": a["uuid"]}])

        r = await http_client.patch(
            f"/api/model-groups/{a['uuid']}",
            json={"members": [{"order": 0, "child_group_uuid": b["uuid"]}]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "failure"
        assert "cycle" in body["message"].lower()

    async def test_delete(self, http_client):
        group = await _create_group(http_client, "ToDelete")
        r = await http_client.delete(f"/api/model-groups/{group['uuid']}")
        assert r.status_code == 200
        assert r.json()["status"] == "success"

        r2 = await http_client.get("/api/model-groups")
        uuids = [g["uuid"] for g in r2.json()["groups"]]
        assert group["uuid"] not in uuids

    async def test_delete_missing_returns_failure(self, http_client):
        r = await http_client.delete("/api/model-groups/no-such-uuid")
        assert r.status_code == 200
        assert r.json()["status"] == "failure"

    async def test_members_cascade_on_delete(self, http_client):
        group = await _create_group(
            http_client,
            "CascadeTest",
            members=[{"order": 0, "model_name": "yolov8n"}, {"order": 1, "model_name": "yolov8s"}],
        )
        assert len(group["members"]) == 2

        r = await http_client.delete(f"/api/model-groups/{group['uuid']}")
        assert r.json()["status"] == "success"

        r2 = await http_client.get(f"/api/model-groups/{group['uuid']}")
        body = r2.json()
        assert body["group"] is None

    async def test_delete_referenced_as_nested_member_rejected(self, http_client):
        inner = await _create_group(http_client, "Referenced")
        await _create_group(http_client, "Referencer", members=[{"order": 0, "child_group_uuid": inner["uuid"]}])

        r = await http_client.delete(f"/api/model-groups/{inner['uuid']}")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "failure"

        # Still present afterward — the delete must not have gone through.
        r2 = await http_client.get(f"/api/model-groups/{inner['uuid']}")
        assert r2.json()["group"] is not None

    async def test_duplicate_name_rejected(self, http_client):
        await _create_group(http_client, "DupName")
        r = await http_client.post("/api/model-groups", json={"name": "DupName", "members": []})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "failure"


class TestModelGroupsDbClient:
    async def test_create_and_replace_members_unit_level(self, db_client):
        from apps.helpers.db.types import (
            ModelGroupCreate,
            ModelGroupMemberWrite,
            ModelGroupUpdate,
        )

        created = await db_client.create_model_group(
            ModelGroupCreate(
                name="UnitGroup",
                members=[ModelGroupMemberWrite(order=0, model_name="m1")],
            )
        )
        assert len(created.members) == 1
        assert created.members[0].model_name == "m1"

        updated = await db_client.update_model_group(
            created.uuid,
            ModelGroupUpdate(members=[ModelGroupMemberWrite(order=0, model_name="m2")]),
        )
        assert updated is not None
        assert len(updated.members) == 1
        assert updated.members[0].model_name == "m2"

    async def test_get_returns_none_for_missing(self, db_client):
        assert await db_client.get_model_group("no-such-uuid") is None

    async def test_delete_returns_false_for_missing(self, db_client):
        assert await db_client.delete_model_group("no-such-uuid") is False
