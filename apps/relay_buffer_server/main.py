from pathlib import Path

import mediamtx_client
import uvicorn

from apps import APPS_CONFIG

from .manager import Manager
from .web import WebApp

config = APPS_CONFIG[Path(__file__).parent.name]

# Defining the host is optional and defaults to http://localhost:9997
# See configuration.py for a list of all supported configuration parameters.
mediamtx_conf = mediamtx_client.Configuration(host="http://localhost:9997")
api_client = mediamtx_client.ApiClient(mediamtx_conf)

manager = Manager(api_client)
web_app = WebApp(manager=manager, app_name=config.name)

# Get the uvicorn app
app = web_app.app()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.port)
