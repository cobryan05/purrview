import os
from pathlib import Path

import uvicorn

from apps import APPS_CONFIG, DB_DIR
from apps.helpers.db.db_client import DbClient
from apps.helpers.logUtils import configure_app_loggers, quiet_poll_access_logs

from .manager import Manager
from .web import WebApp

config = APPS_CONFIG[Path(__file__).parent.name]
FILES_ROOT = Path(os.environ.get("FILES_ROOT", "/app/image_datasets"))
db_client = DbClient(DB_DIR, files_root=FILES_ROOT)

manager = Manager(db_client=db_client, files_root=FILES_ROOT)
web_app = WebApp(manager=manager, app_name=config.name)

# Poll endpoints log one access line per tab per tick; quiet the successful
# ones so the log stays readable (see helpers/logUtils).
configure_app_loggers()
quiet_poll_access_logs()

app = web_app.app()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.port)
