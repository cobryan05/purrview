import inference_client
import uvicorn

from apps.inference_server.main import app_port as inference_port

from .manager import Manager
from .web import WebApp

# See configuration.py for a list of all supported configuration parameters.
APP_NAME = "Image Tagging Server"
APP_PORT = 8002
inference_client_conf = inference_client.Configuration(f"http://localhost:{inference_port}")
api_client = inference_client.ApiClient(inference_client_conf)
manager = Manager(api_client=api_client)
web_app = WebApp(app_name=APP_NAME, manager=manager)

# Get the uvicorn app
app = web_app.app()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)
