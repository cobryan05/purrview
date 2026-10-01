from pathlib import Path

import uvicorn

from apps import APPS_CONFIG
from apps.helpers.logUtils import configure_app_loggers, configure_log_format, quiet_poll_access_logs

from .manager import Manager
from .web import WebApp

config = APPS_CONFIG[Path(__file__).parent.name]

manager = Manager("/app/models/")
web_app = WebApp(app_name=config.name, manager=manager)

# Poll endpoints log one access line per tab per tick; quiet the successful
# ones so the log stays readable (see helpers/logUtils).
configure_log_format()
configure_app_loggers()
quiet_poll_access_logs()

app = web_app.app()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.port)
