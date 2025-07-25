import argparse
import os
from pathlib import Path

import inference_client
import tasks_client
import uvicorn

from apps.helpers.db.db_client import DbClient

from .manager import Manager
from .web import WebApp

from apps import APPS_CONFIG, DB_PATH

config = APPS_CONFIG[Path(__file__).parent.name]
inference_port = APPS_CONFIG["inference_server"].port
tasks_port = APPS_CONFIG["tasks_server"].port

FILES_ROOT = Path(os.environ.get("FILES_ROOT", "/app/image_datasets"))
LABEL_JSON_PATH = Path(os.environ.get("LABELS_JSON", "/data/db/labels.json"))

inference_client_conf = inference_client.Configuration(f"http://localhost:{inference_port}")
inference_api_client = inference_client.ApiClient(inference_client_conf)
tasks_client_conf = tasks_client.Configuration(f"http://localhost:{tasks_port}")
tasks_api_client = tasks_client.ApiClient(tasks_client_conf)
db_client = DbClient(DB_PATH)

manager = Manager(inference_api_client=inference_api_client, db_client=db_client, tasks_api_client=tasks_api_client, files_root=FILES_ROOT, labels_json=LABEL_JSON_PATH)
web_app = WebApp(app_name=config.name, manager=manager)

app = web_app.app()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.port)
