import argparse
import os
from pathlib import Path

import inference_client
import uvicorn

from apps.helpers.db.db_client import DbClient
from apps.inference_server.main import app_port as inference_port

from .manager import Manager
from .web import WebApp

# See configuration.py for a list of all supported configuration parameters.
APP_NAME = "Image Tagging Server"
APP_PORT = 8002

DB_PATH = Path(os.environ.get("DB_PATH", "/data/db/db.sqlite"))
FILES_ROOT = Path(os.environ.get("FILES_ROOT", "/app/image_datasets"))
LABEL_JSON_PATH = Path(os.environ.get("LABELS_JSON", "/data/db/labels.json"))

inference_client_conf = inference_client.Configuration(f"http://localhost:{inference_port}")
api_client = inference_client.ApiClient(inference_client_conf)
db_client = DbClient(str(DB_PATH))
manager = Manager(api_client=api_client, db_client=db_client, files_root=FILES_ROOT, labels_json=LABEL_JSON_PATH)
web_app = WebApp(app_name=APP_NAME, manager=manager)

app = web_app.app()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)
