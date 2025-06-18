import uvicorn

from .web import WebApp

# Defining the host is optional and defaults to http://localhost:9997
# See configuration.py for a list of all supported configuration parameters.
app_name = "Image Tagging Server"
app_port = 8002
web_app = WebApp(app_name=app_name)

# Get the uvicorn app
app = web_app.app()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=app_port)
