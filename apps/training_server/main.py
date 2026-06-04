from pathlib import Path

import uvicorn

from apps import APPS_CONFIG

from .manager import Manager
from .web import WebApp

config = APPS_CONFIG[Path(__file__).parent.name]

manager = Manager()
web_app = WebApp(app_name=config.name, manager=manager)

app = web_app.app()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.port)
