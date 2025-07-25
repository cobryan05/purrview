import os
from pathlib import Path

import uvicorn

import apps.helpers.tasks as _register_tasks  # pylint: disable=unused-import
from apps import APPS_CONFIG, DB_PATH
from apps.helpers.db.db_client import DbClient

from .manager import Manager
from .web import WebApp

config = APPS_CONFIG[Path(__file__).parent.name]

db_client = DbClient(DB_PATH)
manager = Manager(db_client=db_client)
web_app = WebApp(app_name=config.name, manager=manager)

# Get the uvicorn app
app = web_app.app()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.port)
