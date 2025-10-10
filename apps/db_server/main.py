from pathlib import Path

import uvicorn

from apps import APPS_CONFIG, DB_DIR

from apps.helpers.db.db_client import DbClient

from .manager import Manager
from .web import WebApp

config = APPS_CONFIG[Path(__file__).parent.name]
db_client = DbClient(DB_DIR)

manager = Manager(db_client)
web_app = WebApp(manager=manager, app_name=config.name)

# Get the uvicorn app
app = web_app.app()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.port)
