import uvicorn
import mediamtx_client

from apps.stream_manager.web import WebApp
from apps.stream_manager.manager import Manager

 # Defining the host is optional and defaults to http://localhost:9997
# See configuration.py for a list of all supported configuration parameters.
app_name = "Stream Manager"
mediamtx_conf = mediamtx_client.Configuration(
    host = "http://localhost:9997"
)
api_client = mediamtx_client.ApiClient(mediamtx_conf)

manager = Manager(api_client)
web_app = WebApp(manager=manager, app_name=app_name)

# Get the uvicorn app
app = web_app.app()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)