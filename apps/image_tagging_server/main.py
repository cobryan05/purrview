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

FILES_ROOT = Path(os.environ.get("FILES_ROOT", "/app/image_datasets"))

inference_client_conf = inference_client.Configuration(f"http://localhost:{inference_port}")
api_client = inference_client.ApiClient(inference_client_conf)
db_client = DbClient("/storage/db/db.sqlite")
manager = Manager(api_client=api_client, db_client=db_client, files_root=FILES_ROOT)
web_app = WebApp(app_name=APP_NAME, manager=manager)

app = web_app.app()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)
