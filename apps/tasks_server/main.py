import os
from pathlib import Path

import uvicorn

import apps.helpers.tasks as _register_tasks  # pylint: disable=unused-import
from apps.helpers.db.db_client import DbClient

from .manager import Manager
from .web import WebApp

# Defining the host is optional and defaults to http://localhost:9997
# See configuration.py for a list of all supported configuration parameters.
APP_NAME = "Tasks Server"
APP_PORT = 8003
DB_PATH = Path(os.environ.get("DB_PATH", "/data/db/db.sqlite"))

db_client = DbClient(str(DB_PATH))
manager = Manager(db_client=db_client)
web_app = WebApp(app_name=APP_NAME, manager=manager)

# Get the uvicorn app
app = web_app.app()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)
