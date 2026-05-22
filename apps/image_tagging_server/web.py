import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from apps import APPS, APPS_CONFIG
from apps.helpers.reverseProxy import ReverseProxy

from .manager import Manager

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


IMAGE_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host, port=APPS_CONFIG[APPS.DB_SERVER].port, path="/api/images"
)
LABEL_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host, port=APPS_CONFIG[APPS.DB_SERVER].port, path="/api/labels"
)
MODELS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.INFERENCE_SERVER].host, port=APPS_CONFIG[APPS.INFERENCE_SERVER].port, path="/api/models"
)
SOURCES_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host, port=APPS_CONFIG[APPS.DB_SERVER].port, path="/api/sources"
)
TAG_PROXY = ReverseProxy(host=APPS_CONFIG[APPS.DB_SERVER].host, port=APPS_CONFIG[APPS.DB_SERVER].port, path="/api/tags")
TASKS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.TASKS_SERVER].host, port=APPS_CONFIG[APPS.TASKS_SERVER].port, path="/api/tasks"
)


class WebApp:
    SUCCESS_KEY = "success"
    FAILURE_KEY = "failure"

    def __init__(self, app_name: str, manager: Manager):
        """Initialize the WebApp with the application name"""
        self._app_name: str = app_name
        self._app: FastAPI = FastAPI(lifespan=self._lifespan)
        self._templates: Jinja2Templates = Jinja2Templates(directory=os.path.join(SCRIPT_DIR, "templates"))
        self._manager: Manager = manager

        # Mount static files
        self._app.mount("/static", StaticFiles(directory="/app/static"), name="static")
        self._app.mount("/app-static", StaticFiles(directory=os.path.join(SCRIPT_DIR, "static")), name="app-static")

        # Register routes
        self._register_routes()

    def app(self) -> FastAPI:
        """Return the FastAPI application instance"""
        return self._app

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """
        Lifespan event handler for startup and shutdown logic.

        Args:
            app (FastAPI): The FastAPI application instance.
        """
        logging.info("Application is starting up...")

        yield  # This allows the app to run

        logging.info("Application is shutting down...")

    def _register_routes(self):
        """Register all routes for the application"""

        @self._app.get("/", response_class=HTMLResponse)
        async def get_home(request: Request):
            """Render the home page"""
            return self._templates.TemplateResponse(request=request, name="index.html")

        @self._app.get(
            "/server-config",
            operation_id="server_config",
            response_class=JSONResponse,
        )
        @self._app.get("/server-config", response_class=JSONResponse)
        async def server_config(request: Request):
            """
            Get server configuration params

            Args:
                request (Request): The incoming request.

            Returns:
                JSONResponse: Server configuration parameters.
            """
            try:
                server_config = await self._manager.get_server_config()
                response_data = {"status": WebApp.SUCCESS_KEY, "config": server_config}
            except Exception as e:
                response_data = {"status": WebApp.FAILURE_KEY, "message": str(e)}
            return JSONResponse(content=response_data)

        @self._app.api_route("/api/images/{path:path}", methods=["GET", "POST", "PUT",  "PATCH", "DELETE"])
        async def proxy_images(request: Request, path: str):
            return await IMAGE_PROXY.handle_request(path, request)

        @self._app.api_route("/api/labels/{path:path}", methods=["GET", "POST", "PUT",  "PATCH", "DELETE"])
        async def proxy_labels(request: Request, path: str):
            return await LABEL_PROXY.handle_request(path, request)

        @self._app.api_route("/api/models/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
        async def proxy_models(request: Request, path: str):
            return await MODELS_PROXY.handle_request(path, request)

        @self._app.api_route("/api/sources/{path:path}", methods=["GET", "POST", "PUT",  "PATCH", "DELETE"])
        async def proxy_sources(request: Request, path: str):
            return await SOURCES_PROXY.handle_request(path, request)

        @self._app.api_route("/api/tags/{path:path}", methods=["GET", "POST", "PUT",  "PATCH", "DELETE"])
        async def proxy_tags(request: Request, path: str):
            return await TAG_PROXY.handle_request(path, request)

        @self._app.api_route("/api/tasks/{path:path}", methods=["GET", "POST", "PATCH", "DELETE"])
        async def proxy_tasks(request: Request, path: str):
            return await TASKS_PROXY.handle_request(path, request)
