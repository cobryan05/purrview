import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from apps import APPS, APPS_CONFIG
from apps.helpers.reverseProxy import ReverseProxy

from .manager import Manager

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


IMAGE_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/images",
)
LABEL_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/labels",
)
SOURCES_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/sources",
)
TAG_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/tags",
)
TASKS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.TASKS_SERVER].host,
    port=APPS_CONFIG[APPS.TASKS_SERVER].port,
    path="/api/tasks",
)
DATASETS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/datasets",
)
MODEL_CONFIGS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/model-configs",
)
TRAINING_CONFIGS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/training-configs",
)
FILTER_CHAINS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/filter-chains",
)
PREVIEW_RUNS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/preview-runs",
)
MODEL_GROUPS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/model-groups",
)
WATCH_PROFILES_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/watch-profiles",
)
WATCH_EVENTS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/watch-events",
)
RECORDINGS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/recordings",
)
SETTINGS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/settings",
)


class TestServerPayload(BaseModel):
    url: str
    server_type: str  # "inference_server" or "training_server"


# Task SSE polling cadence. The stream is a poll loop behind a push-shaped API, so
# every connected browser costs a full image_tagging → tasks → db round trip per tick,
# forever. 2 Hz is only worth paying while something is actually moving: a task that
# is running or about to run changes its progress between ticks. Nothing live means
# nothing to animate, and the next change is user-initiated, so a slower tick just
# costs a little start-up latency on a card that was idle anyway.
TASKS_STREAM_ACTIVE_POLL_S = 0.5
TASKS_STREAM_IDLE_POLL_S = 3.0
_TASKS_STREAM_ACTIVE_STATUSES = frozenset({"new", "pending", "running"})


def _tasks_poll_interval(data: dict) -> float:
    """Poll fast while any task is live, slowly when the task list is at rest.

    Paused/completed/error instances are deliberately not "live" — they hold still
    until the user acts, and a user action re-renders from its own response.
    """
    try:
        tasks = (data or {}).get("tasks") or {}
        for entry in tasks.values():
            status = ((entry or {}).get("instance") or {}).get("status")
            if status in _TASKS_STREAM_ACTIVE_STATUSES:
                return TASKS_STREAM_ACTIVE_POLL_S
    except (AttributeError, TypeError):
        # A shape we don't recognise shouldn't silently pin the loop at 2 Hz.
        return TASKS_STREAM_IDLE_POLL_S
    return TASKS_STREAM_IDLE_POLL_S


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
        self._app.mount("/static", StaticFiles(directory="/app/static", check_dir=False), name="static")
        self._app.mount(
            "/app-static",
            StaticFiles(directory=os.path.join(SCRIPT_DIR, "static"), check_dir=False),
            name="app-static",
        )

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
        await self._manager.start()

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

        @self._app.api_route("/api/images/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
        async def proxy_images(request: Request, path: str):
            return await IMAGE_PROXY.handle_request(path, request)

        @self._app.api_route("/api/labels/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
        async def proxy_labels(request: Request, path: str):
            return await LABEL_PROXY.handle_request(path, request)

        @self._app.api_route("/api/models/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
        async def proxy_models(request: Request, path: str):
            return await self._manager.inference_proxy.handle_request(path, request)

        @self._app.api_route(
            "/api/sources/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_sources(request: Request, path: str):
            return await SOURCES_PROXY.handle_request(path, request)

        @self._app.api_route("/api/tags/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
        async def proxy_tags(request: Request, path: str):
            return await TAG_PROXY.handle_request(path, request)

        _SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

        @self._app.get("/api/tasks/stream")
        async def tasks_stream(request: Request) -> StreamingResponse:
            """SSE stream — pushes a data frame whenever task state changes."""

            async def _gen():
                last_json: str | None = None
                last_hb = asyncio.get_running_loop().time()
                try:
                    while True:
                        try:
                            data = await self._manager.get_tasks_for_stream()
                            j = json.dumps(data)
                        except Exception:
                            data = {}
                            j = "{}"
                        now = asyncio.get_running_loop().time()
                        if j != last_json:
                            last_json = j
                            last_hb = now
                            yield f"data: {j}\n\n"
                        elif now - last_hb > 15:
                            last_hb = now
                            yield ":\n\n"  # keep-alive comment
                        await asyncio.sleep(_tasks_poll_interval(data))
                except asyncio.CancelledError:
                    pass

            return StreamingResponse(_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)

        @self._app.api_route("/api/tasks/{path:path}", methods=["GET", "POST", "PATCH", "DELETE"])
        async def proxy_tasks(request: Request, path: str):
            return await TASKS_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/datasets/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_datasets(request: Request, path: str):
            return await DATASETS_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/model-configs/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_model_configs(request: Request, path: str):
            return await MODEL_CONFIGS_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/training-configs/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_training_configs(request: Request, path: str):
            return await TRAINING_CONFIGS_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/filter-chains/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_filter_chains(request: Request, path: str):
            return await FILTER_CHAINS_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/preview-runs/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_preview_runs(request: Request, path: str):
            return await PREVIEW_RUNS_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/model-groups/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_model_groups(request: Request, path: str):
            return await MODEL_GROUPS_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/watch-profiles/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_watch_profiles(request: Request, path: str):
            return await WATCH_PROFILES_PROXY.handle_request(path, request)

        @self._app.api_route("/api/watch-events/{path:path}", methods=["GET"])
        async def proxy_watch_events(request: Request, path: str):
            return await WATCH_EVENTS_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/recordings/{path:path}",
            methods=["GET", "DELETE"],
        )
        async def proxy_recordings(request: Request, path: str):
            # NOTE: clip *files* must not be fetched through here — ReverseProxy buffers
            # whole responses, so a video would be loaded into memory and lose range
            # support. The <video> element hits db_server's /file route directly.
            return await RECORDINGS_PROXY.handle_request(path, request)

        @self._app.api_route("/api/settings/{path:path}", methods=["GET", "PUT"])
        async def proxy_settings(request: Request, path: str):
            return await SETTINGS_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/training/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_training(request: Request, path: str):
            return await self._manager.training_proxy.handle_request(path, request)

        @self._app.get("/api/service-status", response_class=JSONResponse)
        async def service_status() -> JSONResponse:
            """Probe all backend services and return their status."""
            try:
                results = await self._manager.get_all_service_status()
                return JSONResponse(content={"status": WebApp.SUCCESS_KEY, "services": results})
            except Exception as e:
                logger.error(e, exc_info=True)
                return JSONResponse(content={"status": WebApp.FAILURE_KEY, "message": str(e)})

        @self._app.get("/api/notices", response_class=JSONResponse)
        async def get_notices() -> JSONResponse:
            """Return aggregated advisory notices from all backend services."""
            try:
                notices = await self._manager.get_notices()
                return JSONResponse(content={"notices": notices})
            except Exception as e:
                logger.error(e, exc_info=True)
                return JSONResponse(content={"notices": []})

        @self._app.get("/api/notices/stream")
        async def notices_stream(request: Request) -> StreamingResponse:
            """SSE stream — pushes a data frame whenever the aggregated notice list changes."""

            async def _gen():
                last_json: str | None = None
                last_hb = asyncio.get_running_loop().time()
                try:
                    while True:
                        try:
                            notices = await self._manager.get_raw_notices()
                            j = json.dumps(notices)
                        except Exception:
                            j = "[]"
                        now = asyncio.get_running_loop().time()
                        if j != last_json:
                            last_json = j
                            last_hb = now
                            yield f"data: {j}\n\n"
                        elif now - last_hb > 15:
                            last_hb = now
                            yield ":\n\n"  # keep-alive comment
                        await asyncio.sleep(2)
                except asyncio.CancelledError:
                    pass

            return StreamingResponse(_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)

        @self._app.post("/api/test-server", response_class=JSONResponse)
        async def test_server(payload: TestServerPayload) -> JSONResponse:
            """Test connectivity to an inference or training server URL."""
            result = await self._manager.test_server(payload.url, payload.server_type)
            return JSONResponse(content=result)

        @self._app.post("/api/reload-server-config", response_class=JSONResponse)
        async def reload_server_config() -> JSONResponse:
            """Reload server URL overrides from DB settings and rebuild proxies."""
            try:
                await self._manager.reload_url_overrides()
                config = await self._manager.get_server_config()
                return JSONResponse(content={"status": WebApp.SUCCESS_KEY, "config": config})
            except Exception as e:
                logger.error(e, exc_info=True)
                return JSONResponse(content={"status": WebApp.FAILURE_KEY, "message": str(e)})
