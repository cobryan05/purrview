import asyncio

from apps import APPS
from apps.training_server.manager import Manager

# ---------------------------------------------------------------------------
# Web endpoints
# ---------------------------------------------------------------------------


async def test_server_config_returns_service_name(http_client):
    resp = await http_client.get("/server-config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["config"]["service"] == APPS.TRAINING_SERVER.value


async def test_server_config_failure_returns_error_status(http_client, manager, monkeypatch):
    async def boom():
        raise RuntimeError("config unavailable")

    monkeypatch.setattr(manager, "get_server_config", boom)

    resp = await http_client.get("/server-config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failure"
    assert "config unavailable" in data["message"]


async def test_training_status_returns_running_message(http_client):
    resp = await http_client.get("/api/training/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["message"] == "Training server running."


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


async def test_get_server_config_returns_a_copy(manager):
    """Mutating the returned config must not affect the manager's internal state."""
    config = await manager.get_server_config()
    assert config == {"service": APPS.TRAINING_SERVER}
    config["service"] = "tampered"

    config_again = await manager.get_server_config()
    assert config_again["service"] == APPS.TRAINING_SERVER


async def test_start_schedules_worker_task(manager):
    manager.start()
    try:
        assert manager._task is not None
        assert not manager._task.done()
        await asyncio.sleep(0.05)
        assert manager._task.done()
        assert manager._task.exception() is None
    finally:
        if manager._task and not manager._task.done():
            manager._task.cancel()


async def test_start_cancels_previous_unfinished_task(manager):
    """Calling start() again must cancel any still-running prior worker task."""
    slow_task_started = asyncio.Event()

    async def _slow_worker():
        slow_task_started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise

    manager._task = asyncio.get_running_loop().create_task(_slow_worker())
    await slow_task_started.wait()
    old_task = manager._task

    manager.start()
    await asyncio.sleep(0.02)

    assert old_task.cancelled()
    assert manager._task is not old_task
    try:
        await asyncio.sleep(0.05)
    finally:
        if manager._task and not manager._task.done():
            manager._task.cancel()

