"""Web API for Relay Buffer Server"""

import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mediamtx_client.api_client import ApiClient

from .manager import Manager

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


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
        self._templates: Jinja2Templates = Jinja2Templates(directory="templates")
        self._dflt_args: dict[str, str] = {"app_name": self._app_name}
        self._api_client: ApiClient = manager._api_client

        # Mount static files
        self._app.mount("/static", StaticFiles(directory="static"), name="static")

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
                request=request, source_url=source_url, stream_name=stream_name, delay=delay
            )
            result = json.loads(response.body.decode("utf-8"))
            if result.get(WebApp.SUCCESS_KEY):
                return await list_streams_html(request)
            else:
                return self._error_response(
                    request, f"Failed to create stream: {result.get(WebApp.ERROR_KEY, 'Unknown error')}"
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
            "/api/streams/destroy",
            response_class=JSONResponse,
            operation_id="destroyStream",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def destroy_stream_api(
            request: Request, stream_name: str = Form(..., description="Name of stream to destroy")
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
            request: Request, stream_name: str = Form(..., description="Name of stream to destroy")
        ) -> HTMLResponse:
            """Handle the destruction of a stream"""
            response = await destroy_stream_api(request=request, stream_name=stream_name)
            result = json.loads(response.body.decode("utf-8"))
            if result.get(WebApp.SUCCESS_KEY):
                return await list_streams_html(request)
            else:
                return self._error_response(
                    request, f"Failed to destroy stream: {result.get(WebApp.ERROR_KEY, 'Unknown error')}"
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
                    request, f"Failed to list configuration: {result.get(WebApp.ERROR_KEY, 'Unknown error')}"
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
                        "name": info.name,
                        "text": str(info),
                        "actions": [
                            {
                                "label": "Stop Stream",
                                "url": "/destroy-stream",
                                "params": {"stream_name": info.name},
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
                    request, f"Failed to list streams: {result.get(WebApp.ERROR_KEY, 'Unknown error')}"
                )
