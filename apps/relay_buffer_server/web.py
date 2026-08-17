"""Web API for Relay Buffer Server"""

import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mediamtx_client.api_client import ApiClient

from apps.helpers.logUtils import LOG_LEVEL

from .manager import Manager

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)


class WebApp:
    SUCCESS_KEY = "success"
    ERROR_KEY = "error"
    RESULT_KEY = "results"

    STREAM_API_TAG_NAME = "streams"
    CONFIG_API_TAG_NAME = "config"

    def __init__(self, app_name: str, manager: Manager):
        """Initialize the WebApp with the application name and manager"""
        self._manager: Manager = manager
        self._app_name: str = app_name
        self._app: FastAPI = FastAPI(lifespan=self._lifespan)
        self._templates: Jinja2Templates = Jinja2Templates(directory="/app/templates")
        self._dflt_args: dict[str, str] = {"app_name": self._app_name}
        self._api_client: ApiClient = manager._media_mtx_api_client

        # Mount static files
        self._app.mount("/static", StaticFiles(directory="/app/static", check_dir=False), name="static")

        # Register routes
        self._register_routes()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Lifespan event handler for startup and shutdown logic"""
        logging.info("Application is starting up...")
        self._manager.start()

        yield

        logging.info("Application is shutting down...")
        if self._manager._task:
            self._manager._task.cancel()
            await self._manager._task

    def app(self) -> FastAPI:
        """Return the FastAPI application instance"""
        return self._app

    def _error_response(self, request: Request, message: str):
        """Render an error response using the dynamic response template"""
        return self._templates.TemplateResponse(
            "dynamic_response.html",
            {
                "request": request,
                "title": "Error",
                "response_data": {self.ERROR_KEY: message},
                **self._dflt_args,
            },
        )

    def _register_routes(self):
        """Register all routes for the application"""

        @self._app.get("/", response_class=HTMLResponse)
        def index(request: Request):
            """Render the index page using the dynamic index template"""
            buttons = [
                {"label": "Create Stream", "action": "/create-stream-form"},
                {"label": "List Streams", "action": "/list-streams"},
                {"label": "Show Configuration", "action": "/list-config"},
            ]
            return self._templates.TemplateResponse(
                "dynamic_index.html",
                {"request": request, "buttons": buttons, **self._dflt_args},
            )

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
                response_data = {"status": "success", "config": server_config}
            except Exception as e:
                response_data = {"status": "failure", "message": str(e)}
            return JSONResponse(content=response_data)

        @self._app.get(
            "/status",
            operation_id="relay_buffer_server_status",
            response_class=JSONResponse,
        )
        async def relay_buffer_server_status():
            return JSONResponse(content=self._manager.get_status())

        @self._app.post(
            "/api/streams/create",
            response_class=JSONResponse,
            operation_id="createStream",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def create_stream_api(
            request: Request,
            source_url: str = Form(..., description="RTSP source URL"),
            stream_name: str = Form(..., description="Name for relay stream"),
            delay: float = Form(0.0, description="Delay in seconds"),
        ) -> JSONResponse:
            """Create a stream and return success status"""
            try:
                logging.info(
                    f"Creating stream with Source URL: {source_url}, Destination Name: {stream_name}, Delay: {delay}"
                )
                if delay > 0:
                    await self._manager.create_delay_stream(rtsp_url=source_url, stream_name=stream_name, delay=delay)
                else:
                    await self._manager.create_rtsp_relay_stream(rtsp_url=source_url, stream_name=stream_name)
                return JSONResponse(content={WebApp.SUCCESS_KEY: True})
            except Exception as e:
                logging.exception(f"Stream creation failed: {str(e)}")
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})

        @self._app.post("/create-stream", response_class=HTMLResponse, include_in_schema=False)
        async def create_stream_html(
            request: Request,
            source_url: str = Form(...),
            stream_name: str = Form(...),
            delay: float = Form(0.0),
        ) -> HTMLResponse:
            """Handle stream creation via HTML form"""
            response = await create_stream_api(
                request=request,
                source_url=source_url,
                stream_name=stream_name,
                delay=delay,
            )
            result = json.loads(response.body.decode("utf-8"))
            if result.get(WebApp.SUCCESS_KEY):
                return await list_streams_html(request)
            else:
                return self._error_response(
                    request,
                    f"Failed to create stream: {result.get(WebApp.ERROR_KEY, 'Unknown error')}",
                )

        @self._app.get("/create-stream-form", response_class=HTMLResponse, include_in_schema=False)
        async def create_stream_form_html(request: Request) -> HTMLResponse:
            """Render the form for creating a new stream"""
            fields = {
                "source_url": {"label": "RTSP Source URL", "type": "text"},
                "stream_name": {"label": "Stream Name", "type": "text"},
                "delay": {"label": "Delay (seconds)", "type": "text"},
            }
            return self._templates.TemplateResponse(
                "dynamic_form.html",
                {
                    "request": request,
                    "title": "Create Stream",
                    "action_url": "/create-stream",
                    "fields": fields,
                    "submit_label": "Create Stream",
                },
            )

        @self._app.post(
            "/api/streams/ensure",
            response_class=JSONResponse,
            operation_id="ensureStream",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def ensure_stream_api(
            source_url: str = Form(..., description="Source URL to relay (rtsp://, rtmp://, srt://, http(s):// …)"),
            holder: str = Form("", description="Identity of the consumer taking the lease"),
            retention_s: float = Form(
                -1.0, description="Seconds of rolling buffer to keep; negative uses the server default"
            ),
            record: bool = Form(True, description="Whether to maintain a rolling buffer for this source"),
        ) -> JSONResponse:
            """Idempotently arm a relay path for a source and take a lease on it.

            Consumers read the returned master path directly — there is no per-consumer
            path, because fanning out from the master is the whole point of the relay.
            The response carries `recording`/`retentionS` so a caller can tell the user
            "this source has no rewind buffer" instead of silently cutting empty clips.
            """
            try:
                result = await self._manager.ensure_stream(
                    source_url=source_url,
                    holder=holder,
                    retention_s=None if retention_s < 0 else retention_s,
                    record=record,
                )
                return JSONResponse(content={WebApp.SUCCESS_KEY: True, **result.to_dict()})
            except Exception as e:
                logging.exception(f"Stream ensure failed: {e}")
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})

        @self._app.post(
            "/api/streams/release",
            response_class=JSONResponse,
            operation_id="releaseStream",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def release_stream_api(
            lease_id: str = Form(..., description="Lease id returned by ensure"),
        ) -> JSONResponse:
            """Release one lease; the relay path is torn down when it was the last."""
            try:
                released = await self._manager.release_lease(lease_id)
                return JSONResponse(content={WebApp.SUCCESS_KEY: True, "released": released})
            except Exception as e:
                logging.exception(f"Stream release failed: {e}")
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})

        @self._app.post(
            "/api/streams/renew",
            response_class=JSONResponse,
            operation_id="renewStream",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def renew_stream_api(
            lease_id: str = Form(..., description="Lease id returned by ensure"),
        ) -> JSONResponse:
            """Heartbeat: tell the relay this holder is still alive.

            A lease that stops being renewed is reaped, so a holder that intends to keep
            reading must call this well inside the TTL. `renewed: false` means the relay
            has forgotten the lease and the caller should re-`ensure`.
            """
            try:
                renewed = self._manager.renew_lease(lease_id)
                return JSONResponse(content={WebApp.SUCCESS_KEY: True, "renewed": renewed})
            except Exception as e:
                logging.exception(f"Stream renew failed: {e}")
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})

        @self._app.get(
            "/api/streams/{name}/ranges",
            response_class=JSONResponse,
            operation_id="streamRanges",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def stream_ranges_api(name: str) -> JSONResponse:
            """Time ranges currently held in the rolling buffer for a path.

            This is the honest answer to "how much rewind do I have", which is not the
            configured retention: the buffer is RAM-backed, so a relay restart shortens
            it without warning.
            """
            try:
                ranges = await self._manager.get_ranges(name)
                return JSONResponse(content={WebApp.SUCCESS_KEY: True, WebApp.RESULT_KEY: ranges})
            except Exception as e:
                logging.exception(f"Fetching ranges for {name} failed: {e}")
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})

        @self._app.post(
            "/api/streams/clip",
            response_class=JSONResponse,
            operation_id="extractClip",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def extract_clip_api(
            path: str = Form(..., description="Master path name to cut from"),
            start: str = Form(..., description="Clip start as an ISO-8601 timestamp"),
            duration_s: float = Form(..., description="Clip duration in seconds"),
            dest: str = Form(..., description="Absolute destination file path"),
        ) -> JSONResponse:
            """Cut a window out of the rolling buffer into a file.

            Retroactive by design: pre-roll is whatever the caller asks for at event
            time, not something fixed when the stream was created. The result reports
            `partialPreroll` when the requested window had partly aged out, so a
            truncated clip is never presented as a complete one.
            """
            try:
                started_at = datetime.fromisoformat(start.replace("Z", "+00:00"))
                if started_at.tzinfo is None:
                    started_at = started_at.replace(tzinfo=timezone.utc)
                result = await self._manager.extract_clip(
                    path=path, start=started_at, duration_s=duration_s, dest=Path(dest)
                )
                return JSONResponse(content={WebApp.SUCCESS_KEY: True, **result})
            except Exception as e:
                logging.exception(f"Clip extraction failed: {e}")
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})

        @self._app.get(
            "/api/streams/buffer",
            response_class=JSONResponse,
            operation_id="bufferUsage",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def buffer_usage_api() -> JSONResponse:
            """RAM buffer budget: cap, committed bytes, and actual mount usage."""
            try:
                usage = self._manager.buffer_usage()
                return JSONResponse(content={WebApp.SUCCESS_KEY: True, **usage.to_dict()})
            except Exception as e:
                logging.exception(f"Buffer usage failed: {e}")
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})

        @self._app.post(
            "/api/streams/destroy",
            response_class=JSONResponse,
            operation_id="destroyStream",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def destroy_stream_api(
            request: Request,
            stream_name: str = Form(..., description="Name of stream to destroy"),
        ) -> JSONResponse:
            """Handle the destruction of a stream"""
            try:
                logging.info(f"Destroying stream {stream_name}")
                await self._manager.destroy_stream(stream_name=stream_name)
                return JSONResponse(content={WebApp.SUCCESS_KEY: True})
            except Exception as e:
                logging.exception(f"Stream destruction failed: {str(e)}")
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})

        @self._app.post("/destroy-stream", response_class=HTMLResponse, include_in_schema=False)
        async def destroy_stream_html(
            request: Request,
            stream_name: str = Form(..., description="Name of stream to destroy"),
        ) -> HTMLResponse:
            """Handle the destruction of a stream"""
            response = await destroy_stream_api(request=request, stream_name=stream_name)
            result = json.loads(response.body.decode("utf-8"))
            if result.get(WebApp.SUCCESS_KEY):
                return await list_streams_html(request)
            else:
                return self._error_response(
                    request,
                    f"Failed to destroy stream: {result.get(WebApp.ERROR_KEY, 'Unknown error')}",
                )

        @self._app.get(
            "/api/config/list",
            response_class=JSONResponse,
            operation_id="listConfig",
            tags=[WebApp.CONFIG_API_TAG_NAME],
        )
        async def list_configs_api(request: Request) -> JSONResponse:
            """List the global configuration"""
            try:
                config = await self._manager.get_config()
                return JSONResponse(content={WebApp.SUCCESS_KEY: True, WebApp.RESULT_KEY: config})
            except Exception as e:
                logging.exception(f"Config listing failed: {str(e)}")
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})

        @self._app.get("/list-config", response_class=HTMLResponse, include_in_schema=False)
        async def list_configs_html(request: Request) -> HTMLResponse:
            """List the global configuration"""
            response = await list_configs_api(request)
            result = json.loads(response.body.decode("utf-8"))
            if result.get(WebApp.SUCCESS_KEY):
                return self._templates.TemplateResponse(
                    "dynamic_response.html",
                    {
                        "request": request,
                        "title": "Configuration",
                        "response_data": {self.RESULT_KEY: result[self.RESULT_KEY]},
                        **self._dflt_args,
                    },
                )
            else:
                return self._error_response(
                    request,
                    f"Failed to list configuration: {result.get(WebApp.ERROR_KEY, 'Unknown error')}",
                )

        @self._app.get(
            "/api/streams/list",
            response_class=JSONResponse,
            operation_id="listStreams",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def list_streams_api(request: Request) -> JSONResponse:
            """List all available streams"""
            try:
                await self._manager.refresh_streams()
                streams = self._manager.get_streams()
                stream_info: List[Dict[str, Any]] = [
                    {
                        "name": info["name"],
                        "info": info,
                        "actions": [
                            {
                                "label": "Stop Stream",
                                "url": "/destroy-stream",
                                "params": {"stream_name": info["name"]},
                            }
                        ],
                    }
                    for info in streams.values()
                ]
                return JSONResponse(content={WebApp.SUCCESS_KEY: True, WebApp.RESULT_KEY: stream_info})
            except Exception as e:
                logging.exception(f"Fetching streams failed: {str(e)}")
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})

        @self._app.get("/list-streams", response_class=HTMLResponse, include_in_schema=False)
        async def list_streams_html(request: Request) -> HTMLResponse:
            """List all available streams"""
            response = await list_streams_api(request)
            result = json.loads(response.body.decode("utf-8"))
            if result.get(WebApp.SUCCESS_KEY):
                return self._templates.TemplateResponse(
                    "dynamic_response.html",
                    {
                        "request": request,
                        "title": "Stream List",
                        "response_data": {self.RESULT_KEY: result[self.RESULT_KEY]},
                        **self._dflt_args,
                    },
                )
            else:
                return self._error_response(
                    request,
                    f"Failed to list streams: {result.get(WebApp.ERROR_KEY, 'Unknown error')}",
                )
