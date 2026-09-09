from pathlib import Path

import db_client
import inference_client
import uvicorn

from apps import APPS, APPS_CONFIG
from apps.helpers.apiCall import bind_default_timeout
from apps.helpers.logUtils import quiet_poll_access_logs

from .manager import Manager
from .web import WebApp

config = APPS_CONFIG[Path(__file__).parent.name]
db_host = APPS_CONFIG[APPS.DB_SERVER].host
db_port = APPS_CONFIG[APPS.DB_SERVER].port
inference_host = APPS_CONFIG[APPS.INFERENCE_SERVER].host
inference_port = APPS_CONFIG[APPS.INFERENCE_SERVER].port

inference_client_conf = inference_client.Configuration(f"http://{inference_host}:{inference_port}")
inference_api_client = bind_default_timeout(inference_client.ApiClient(inference_client_conf))

db_client_conf = db_client.Configuration(f"http://{db_host}:{db_port}")
db_api_client = bind_default_timeout(db_client.ApiClient(db_client_conf))

manager = Manager(db_api_client=db_api_client, inference_api_client=inference_api_client)
web_app = WebApp(app_name=config.name, manager=manager)

# Poll endpoints log one access line per tab per tick; quiet the successful
# ones so the log stays readable (see helpers/logUtils).
quiet_poll_access_logs()

app = web_app.app()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.port)
