"""Tests for task-config and task-instance CRUD through db_server.

These hit the db_server's /api/task-configs and /api/task-instances endpoints
with a real SQLite DB, exercising the persistence layer used by tasks_server.
"""

import pytest

from apps.helpers.consts import TaskStatus

# ──────────────────────────────────────────────────────────────────────────────
# Task Config CRUD
# ──────────────────────────────────────────────────────────────────────────────


async def test_list_task_configs_empty(http_client):
    r = await http_client.get("/api/task-configs")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["configs"] == {}


async def test_create_task_config(http_client):
    r = await http_client.post(
        "/api/task-configs",
        json={"name": "my-task", "typename": "CollectImages", "params": {"steps": 10}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    config = body["config"]
    assert config["name"] == "my-task"
    assert config["typename"] == "CollectImages"
    assert config["params_json"]["steps"] == 10
    assert "uuid" in config


async def test_list_task_configs_after_create(http_client):
    create_r = await http_client.post(
        "/api/task-configs",
        json={"name": "listed-task", "typename": "SomeTask", "params": {}},
    )
    config_uuid = create_r.json()["config"]["uuid"]

    list_r = await http_client.get("/api/task-configs")
    assert config_uuid in list_r.json()["configs"]


async def test_get_task_config_by_uuid(http_client):
    create_r = await http_client.post(
        "/api/task-configs",
        json={"name": "specific-task", "typename": "ATask", "params": {"x": 1}},
    )
    config_uuid = create_r.json()["config"]["uuid"]

    r = await http_client.get("/api/task-configs", params={"uuids": config_uuid})
    assert r.status_code == 200
    configs = r.json()["configs"]
    assert config_uuid in configs
    assert configs[config_uuid]["name"] == "specific-task"


async def test_update_task_config_name(http_client):
    create_r = await http_client.post(
        "/api/task-configs",
        json={"name": "old-name", "typename": "T", "params": {}},
    )
    config_uuid = create_r.json()["config"]["uuid"]

    patch_r = await http_client.patch(
        f"/api/task-configs/{config_uuid}",
        json={"name": "new-name"},
    )
    assert patch_r.status_code == 200
    assert patch_r.json()["status"] == "success"

    list_r = await http_client.get("/api/task-configs")
    assert list_r.json()["configs"][config_uuid]["name"] == "new-name"


async def test_update_task_config_params(http_client):
    create_r = await http_client.post(
        "/api/task-configs",
        json={"name": "p-task", "typename": "T", "params": {"a": 1}},
    )
    config_uuid = create_r.json()["config"]["uuid"]

    await http_client.patch(
        f"/api/task-configs/{config_uuid}",
        json={"params": {"a": 99, "b": "new"}},
    )

    list_r = await http_client.get("/api/task-configs")
    params = list_r.json()["configs"][config_uuid]["params_json"]
    assert params["a"] == 99
    assert params["b"] == "new"


async def test_update_task_config_marked_for_delete(http_client):
    create_r = await http_client.post(
        "/api/task-configs",
        json={"name": "del-task", "typename": "T", "params": {}},
    )
    config_uuid = create_r.json()["config"]["uuid"]

    await http_client.patch(
        f"/api/task-configs/{config_uuid}",
        json={"markedForDelete": True},
    )

    list_r = await http_client.get("/api/task-configs")
    assert list_r.json()["configs"][config_uuid]["marked_for_delete"] is True


async def test_delete_task_configs(http_client):
    r1 = await http_client.post("/api/task-configs", json={"name": "d1", "typename": "T", "params": {}})
    r2 = await http_client.post("/api/task-configs", json={"name": "d2", "typename": "T", "params": {}})
    uuid1 = r1.json()["config"]["uuid"]
    uuid2 = r2.json()["config"]["uuid"]

    del_r = await http_client.post(
        "/api/task-configs/delete",
        json={"configUuids": [uuid1, uuid2]},
    )
    assert del_r.status_code == 200
    assert del_r.json()["status"] == "success"

    list_r = await http_client.get("/api/task-configs")
    configs = list_r.json()["configs"]
    assert uuid1 not in configs
    assert uuid2 not in configs


# ──────────────────────────────────────────────────────────────────────────────
# Task Instance CRUD
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
async def config_uuid(http_client):
    r = await http_client.post(
        "/api/task-configs",
        json={"name": "inst-cfg", "typename": "TestTask", "params": {}},
    )
    return r.json()["config"]["uuid"]


async def test_create_task_instance(http_client, config_uuid):
    r = await http_client.post(
        "/api/task-instances",
        json={"configUuid": config_uuid},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    instance = body["instance"]
    assert instance["config_uuid"] == config_uuid
    assert "uuid" in instance


async def test_list_task_instances_empty(http_client):
    r = await http_client.get("/api/task-instances")
    assert r.status_code == 200
    assert r.json()["instances"] == []


async def test_list_task_instances_after_create(http_client, config_uuid):
    create_r = await http_client.post("/api/task-instances", json={"configUuid": config_uuid})
    instance_uuid = create_r.json()["instance"]["uuid"]

    list_r = await http_client.get("/api/task-instances")
    uuids = [i["uuid"] for i in list_r.json()["instances"]]
    assert instance_uuid in uuids


async def test_set_task_instance_status(http_client, config_uuid):
    create_r = await http_client.post("/api/task-instances", json={"configUuid": config_uuid})
    inst_uuid = create_r.json()["instance"]["uuid"]

    patch_r = await http_client.patch(
        f"/api/task-instances/{inst_uuid}/status",
        json={"status": TaskStatus.RUNNING},
    )
    assert patch_r.status_code == 200
    assert patch_r.json()["status"] == "success"

    list_r = await http_client.get("/api/task-instances")
    instances = {i["uuid"]: i for i in list_r.json()["instances"]}
    assert instances[inst_uuid]["status"] == TaskStatus.RUNNING


async def test_set_task_instance_result(http_client, config_uuid):
    create_r = await http_client.post("/api/task-instances", json={"configUuid": config_uuid})
    inst_uuid = create_r.json()["instance"]["uuid"]

    r = await http_client.patch(
        f"/api/task-instances/{inst_uuid}/result",
        json={"result": {"status": "completed", "count": 42}},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"


async def test_get_task_results(http_client, config_uuid):
    create_r = await http_client.post("/api/task-instances", json={"configUuid": config_uuid})
    inst_uuid = create_r.json()["instance"]["uuid"]

    await http_client.patch(
        f"/api/task-instances/{inst_uuid}/result",
        json={"result": {"answer": 99}},
    )

    results_r = await http_client.post(
        "/api/task-instances/results",
        json={"taskUuids": [inst_uuid]},
    )
    assert results_r.status_code == 200
    results = results_r.json()["results"]
    assert inst_uuid in results
    assert results[inst_uuid]["answer"] == 99


async def test_set_and_get_resume_data(http_client, config_uuid):
    create_r = await http_client.post("/api/task-instances", json={"configUuid": config_uuid})
    inst_uuid = create_r.json()["instance"]["uuid"]

    r = await http_client.patch(
        f"/api/task-instances/{inst_uuid}/resume-data",
        json={"resumeData": {"checkpoint": 7, "items": [1, 2, 3]}},
    )
    assert r.status_code == 200

    list_r = await http_client.get("/api/task-instances")
    instances = {i["uuid"]: i for i in list_r.json()["instances"]}
    resume = instances[inst_uuid]["resume_data_json"]
    assert resume["checkpoint"] == 7
    assert resume["items"] == [1, 2, 3]


async def test_delete_task_instances(http_client, config_uuid):
    r1 = await http_client.post("/api/task-instances", json={"configUuid": config_uuid})
    r2 = await http_client.post("/api/task-instances", json={"configUuid": config_uuid})
    u1 = r1.json()["instance"]["uuid"]
    u2 = r2.json()["instance"]["uuid"]

    del_r = await http_client.post(
        "/api/task-instances/delete",
        json={"taskUuids": [u1, u2]},
    )
    assert del_r.status_code == 200
    assert del_r.json()["status"] == "success"

    list_r = await http_client.get("/api/task-instances")
    uuids = [i["uuid"] for i in list_r.json()["instances"]]
    assert u1 not in uuids
    assert u2 not in uuids


async def test_delete_config_with_instance_marks_for_delete(http_client, config_uuid):
    """Deleting a config that has active instances marks it for_delete instead of hard-deleting."""
    await http_client.post("/api/task-instances", json={"configUuid": config_uuid})
    r = await http_client.post(
        "/api/task-configs/delete",
        json={"configUuids": [config_uuid]},
    )
    # The delete is accepted (not an error); the config is marked for deletion.
    assert r.json()["status"] == "success"

    list_r = await http_client.get("/api/task-configs")
    configs = list_r.json()["configs"]
    # Config still present but marked for delete.
    assert config_uuid in configs
    assert configs[config_uuid]["marked_for_delete"] is True
