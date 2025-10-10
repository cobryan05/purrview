"""Retrieve the openapi.json files for generating the API clients"""

import json
import os

os.environ["GENERATING_OPENAPI_CLIENTS"] = "1"

from .inference_server.main import app as inference_server_app
from .relay_buffer_server.main import app as relay_server_app
from .tasks_server.main import app as tasks_server_app
from .db_server.main import app as db_server_app

with open("relay_buffer_server_openapi.json", "w", encoding="utf-8") as f:
    json.dump(relay_server_app.openapi(), f, indent=2)

with open("inference_server_openapi.json", "w", encoding="utf-8") as f:
    json.dump(inference_server_app.openapi(), f, indent=2)

with open("tasks_server_openapi.json", "w", encoding="utf-8") as f:
    json.dump(tasks_server_app.openapi(), f, indent=2)

with open("db_server_openapi.json", "w", encoding="utf-8") as f:
    json.dump(db_server_app.openapi(), f, indent=2)
