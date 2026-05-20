import os
import pathlib
import socket
import subprocess
import time

import httpx
import pytest

os.environ.setdefault("PORT_OFFSET", "10000")

from apps import APPS, APPS_CONFIG

_SERVICES = [
    APPS.DB_SERVER,
    APPS.INFERENCE_SERVER,
    APPS.TASKS_SERVER,
    APPS.RELAY_BUFFER_SERVER,
    APPS.IMAGE_TAGGING_SERVER,
]

_SMOKE_DIR = pathlib.Path(__file__).parent


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-smoke"):
        skip = pytest.mark.skip(reason="Pass --run-smoke to run smoke tests")
        for item in items:
            if str(item.fspath).startswith(str(_SMOKE_DIR)):
                item.add_marker(skip)


def _wait_for_port(port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"Port {port} did not become ready within {timeout}s")


@pytest.fixture(scope="session")
def smoke_services(tmp_path_factory):
    db_dir = tmp_path_factory.mktemp("smoke_db")
    env = {**os.environ, "DB_DIR": str(db_dir)}

    procs = []
    for app_key in _SERVICES:
        module = app_key.value
        proc = subprocess.Popen(
            ["uvicorn", f"apps.{module}.main:app", "--host", "0.0.0.0", "--port", str(APPS_CONFIG[app_key].port)],
            env=env,
            cwd="/app",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(proc)

    try:
        for app_key in _SERVICES:
            _wait_for_port(APPS_CONFIG[app_key].port)
    except TimeoutError:
        for proc in procs:
            proc.terminate()
        raise

    yield {app_key: f"http://localhost:{APPS_CONFIG[app_key].port}" for app_key in _SERVICES}

    for proc in procs:
        proc.terminate()
    for proc in procs:
        proc.wait(timeout=10)


@pytest.fixture(scope="session")
def db_url(smoke_services):        return smoke_services[APPS.DB_SERVER]

@pytest.fixture(scope="session")
def inference_url(smoke_services):  return smoke_services[APPS.INFERENCE_SERVER]

@pytest.fixture(scope="session")
def tasks_url(smoke_services):     return smoke_services[APPS.TASKS_SERVER]

@pytest.fixture(scope="session")
def relay_url(smoke_services):     return smoke_services[APPS.RELAY_BUFFER_SERVER]

@pytest.fixture(scope="session")
def tagging_url(smoke_services):   return smoke_services[APPS.IMAGE_TAGGING_SERVER]


@pytest.fixture
async def db(db_url):
    async with httpx.AsyncClient(base_url=db_url, timeout=10) as c:
        yield c

@pytest.fixture
async def inference(inference_url):
    async with httpx.AsyncClient(base_url=inference_url, timeout=10) as c:
        yield c

@pytest.fixture
async def tasks(tasks_url):
    async with httpx.AsyncClient(base_url=tasks_url, timeout=10) as c:
        yield c

@pytest.fixture
async def relay(relay_url):
    async with httpx.AsyncClient(base_url=relay_url, timeout=10) as c:
        yield c

@pytest.fixture
async def tagging(tagging_url):
    async with httpx.AsyncClient(base_url=tagging_url, timeout=10) as c:
        yield c
