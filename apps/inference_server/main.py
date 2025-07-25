import uvicorn

from apps import APPS_CONFIG, DB_PATH

from .manager import Manager
from .web import WebApp
from pathlib import Path
config = APPS_CONFIG[Path(__file__).parent.name]

manager = Manager("/app/models/")
web_app = WebApp(app_name=config.name, manager=manager)

# Get the uvicorn app
app = web_app.app()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.port)
