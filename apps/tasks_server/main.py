import os
from pathlib import Path

import db_client
import inference_client
import uvicorn

from apps import APPS_CONFIG, DB_DIR
from apps.helpers.db.db_client import DbClient

from .manager import Manager
from .web import WebApp

config = APPS_CONFIG[Path(__file__).parent.name]
db_host = APPS_CONFIG["db_server"].host
db_port = APPS_CONFIG["db_server"].port
inference_host = APPS_CONFIG["inference_server"].host
inference_port = APPS_CONFIG["inference_server"].port

inference_client_conf = inference_client.Configuration(f"http://{inference_host}:{inference_port}")
inference_api_client = inference_client.ApiClient(inference_client_conf)

db_client_conf = db_client.Configuration(f"http://{db_host}:{db_port}")
db_api_client = db_client.ApiClient(db_client_conf)

legacy_db_client = DbClient(DB_DIR)
manager = Manager(
    db_api_client=db_api_client, legacy_db_client=legacy_db_client, inference_api_client=inference_api_client
)
web_app = WebApp(app_name=config.name, manager=manager)

# Get the uvicorn app
app = web_app.app()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.port)
