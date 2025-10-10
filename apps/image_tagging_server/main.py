import argparse
import os
from pathlib import Path

import inference_client
import tasks_client
import db_client
import uvicorn

from apps.helpers.db.db_client import DbClient


from .manager import Manager
from .web import WebApp

from apps import APPS_CONFIG, DB_DIR

config = APPS_CONFIG[Path(__file__).parent.name]
inference_host = APPS_CONFIG["inference_server"].host
inference_port = APPS_CONFIG["inference_server"].port
tasks_host = APPS_CONFIG["tasks_server"].host
tasks_port = APPS_CONFIG["tasks_server"].port
db_host = APPS_CONFIG["db_server"].host
db_port = APPS_CONFIG["db_server"].port

FILES_ROOT = Path(os.environ.get("FILES_ROOT", "/app/image_datasets"))

inference_client_conf = inference_client.Configuration(f"http://{inference_host}:{inference_port}")
inference_api_client = inference_client.ApiClient(inference_client_conf)
tasks_client_conf = tasks_client.Configuration(f"http://{tasks_host}:{tasks_port}")
tasks_api_client = tasks_client.ApiClient(tasks_client_conf)
db_client_conf = db_client.Configuration(f"http://{db_host}:{db_port}")
db_api_client = db_client.ApiClient(db_client_conf)

legacy_db_client = DbClient(DB_DIR)

manager = Manager(
    inference_api_client=inference_api_client,
    db_api_client=db_api_client,
    legacy_db_client=legacy_db_client,
    tasks_api_client=tasks_api_client,
    files_root=FILES_ROOT,
)
web_app = WebApp(app_name=config.name, manager=manager)

app = web_app.app()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.port)
