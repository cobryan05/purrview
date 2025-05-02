"""Web API for Relay Buffer Server"""

from .manager import Manager
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mediamtx_client.api_client import ApiClient
from pydantic import BaseModel, Field
from typing import Dict, List, Any
import logging
import sys

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
        self._manager: Manager = manager  # Manager instance for handling streams
        self._app_name: str = app_name  # Name of the application
        self._app: FastAPI = FastAPI(lifespan=self._lifespan)  # FastAPI instance with lifespan events
        self._templates: Jinja2Templates = Jinja2Templates(directory="templates")  # Template engine for rendering HTML
        self._dflt_args: dict[str, str] = {"app_name": self._app_name}  # Default arguments for templates
        self._api_client: ApiClient = manager._api_client  # API client for interacting with MediaMTX

        # Mount static files
        self._app.mount("/static", StaticFiles(directory="static"), name="static")

        # Register routes
        self._register_routes()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Lifespan event handler for startup and shutdown logic"""
        logging.info("Application is starting up...")
        self._manager.start()  # Start the manager's periodic task

        yield  # This allows the app to run

        logging.info("Application is shutting down...")
        if self._manager._task:
            self._manager._task.cancel()  # Cancel the periodic task
            await self._manager._task  # Wait for the task to finish

    def app(self) -> FastAPI:
        """Return the FastAPI application instance"""
        return self._app

    def _error_response(self, request: Request, message: str):
        """Render an error response using a text template"""
        return self._templates.TemplateResponse(
            "display_text.html",
            {
                "request": request,
                "text": message,
                **self._dflt_args,
            },
        )

    def _register_routes(self):
        """Register all routes for the application"""

        @self._app.get("/", response_class=HTMLResponse)
        def index(request: Request):
            """Render the index page"""
            return self._templates.TemplateResponse("index.html", {"request": request, **self._dflt_args})

        @self._app.post(
            "/api/create-stream",
            response_model=Dict[str, Any],
            operation_id="createStream",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def create_stream_api(
            request: Request,
            source_url: str = Form(..., description="rtsp source url"),
            stream_name: str = Form(..., description="name for relay stream"),
            delay: float = Form(0.0, description="Delay in seconds"),
        ) -> Dict[str, Any]:
            """Create a stream and return success status"""

            class CreateStreamData(BaseModel):
                source_url: str = Field(...)
                stream_name: str = Field(...)
                delay: float = Field(0.0)

            try:
                data = CreateStreamData(source_url=source_url, stream_name=stream_name, delay=delay)
                logging.info(
                    f"Creating stream with Source URL: {data.source_url}, Destination Name: {data.stream_name}, Delay: {data.delay}"
                )
                await self._manager.create_rtsp_relay_stream(rtsp_url=data.source_url, stream_name=data.stream_name)

                if data.delay > 0:
                    streams = self._manager.get_streams()
                    stream = streams.get(data.stream_name)
                    if stream is None:
                        raise Exception(f"Failed to create delayed stream: stream '{data.stream_name}' was not found")
                    await self._manager.create_delay_stream(stream.url, f"{data.stream_name}_delayed", data.delay)

                return {WebApp.SUCCESS_KEY: True}

            except Exception as e:
                logging.exception(f"Stream creation failed: {str(e)}")
                return {WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)}

        @self._app.post("/create-stream", response_class=HTMLResponse, include_in_schema=False)
        async def create_stream_html(
            request: Request,
            source_url: str = Form(...),
            stream_name: str = Form(...),
            delay: float = Form(0.0),
        ) -> HTMLResponse:
            """Handle stream creation via HTML form"""
            # Call the API version internally
            result = await create_stream_api(
                request=request, source_url=source_url, stream_name=stream_name, delay=delay
            )

            if result.get(WebApp.SUCCESS_KEY):
                return await list_streams_html(request)
            else:
                return self._error_response(
                    request, f"Failed to create stream: {result.get(WebApp.ERROR_KEY, 'Unknown error')}"
                )

        @self._app.get("/create-stream-form", response_class=HTMLResponse, include_in_schema=False)
        async def create_stream_form_html(request: Request) -> HTMLResponse:
            """Render the form for creating a new stream"""
            try:
                return self._templates.TemplateResponse(
                    "create_stream_form.html",
                    {"request": request, **self._dflt_args},
                )
            except Exception as e:
                return self._error_response(request, f"Failed to create stream form: {str(e)}")

        @self._app.post(
            "/api/destroy-stream",
            response_model=Dict[str, Any],
            operation_id="destroyStream",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def destroy_stream_api(
            request: Request, stream_name: str = Form(..., description="name of stream to destroy")
        ) -> Dict[str, Any]:
            """Handle the destruction of a stream"""

            class DestroyStreamData(BaseModel):
                stream_name: str = Field(...)

            try:
                # Parse form data
                data = DestroyStreamData(stream_name=stream_name)

                # Log or process the data
                logging.info(f"Destroying stream {data.stream_name}")
                await self._manager.destroy_stream(stream_name=data.stream_name)
                return {WebApp.SUCCESS_KEY: True}
            except Exception as e:
                return {WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)}

        @self._app.post("/destroy-stream", response_class=HTMLResponse, include_in_schema=False)
        async def destroy_stream_html(
            request: Request, stream_name: str = Form(..., description="name of stream to destroy")
        ) -> HTMLResponse:
            """Handle the destruction of a stream"""
            # Call the API version internally
            result = await destroy_stream_api(request=request, stream_name=stream_name)

            if result.get(WebApp.SUCCESS_KEY):
                return await list_streams_html(request)
            else:
                return self._error_response(
                    request, f"Failed to create stream: {result.get(WebApp.ERROR_KEY, 'Unknown error')}"
                )

        @self._app.get(
            "/api/list-config",
            response_model=Dict[str, Any],
            operation_id="listConfig",
            tags=[WebApp.CONFIG_API_TAG_NAME],
        )
        async def list_configs_api(request: Request) -> Dict[str, Any]:
            """List the global configuration"""
            try:
                config = await self._manager.get_config()  # Fetch the global configuration
                return {WebApp.SUCCESS_KEY: True, WebApp.RESULT_KEY: config}
            except Exception as e:
                logging.exception(f"Config listing failed: {str(e)}")
                return {WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)}

        @self._app.get("/list-config", response_class=HTMLResponse, include_in_schema=False)
        async def list_configs_html(request: Request) -> HTMLResponse:
            """List the global configuration"""
            # Call the API version internally
            result = await list_configs_api(request)

            if result.get(WebApp.SUCCESS_KEY):
                return self._templates.TemplateResponse(
                    "display_text.html",
                    {
                        "request": request,
                        "text": result.get(WebApp.RESULT_KEY, {}),
                        **self._dflt_args,
                    },
                )
            else:
                return self._error_response(
                    request, f"Failed to destroy stream: {result.get(WebApp.ERROR_KEY, 'Unknown error')}"
                )

        @self._app.get(
            "/api/list-streams",
            response_model=Dict[str, Any],
            operation_id="listStreams",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def list_streams_api(request: Request) -> Dict[str, Any]:
            """List all available streams"""
            try:
                await self._manager.refresh_streams()
                streams = self._manager.get_streams()  # Fetch the list of streams
                stream_info: List[Dict[str, str]] = []
                for info in streams.values():
                    stream_info.append({"name": info.name, "text": str(info)})  # Format stream information
                return {WebApp.SUCCESS_KEY: True, WebApp.RESULT_KEY: stream_info}
            except Exception as e:
                logging.exception(f"Fetching streams failed: {str(e)}")
                return {WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)}

        @self._app.get("/list-streams", response_class=HTMLResponse, include_in_schema=False)
        async def list_streams_html(request: Request) -> HTMLResponse:
            """List all available streams"""
            result = await list_streams_api(request)

            if result.get(WebApp.SUCCESS_KEY):
                return self._templates.TemplateResponse(
                    "streams_list.html",
                    {
                        "request": request,
                        "streams": result[WebApp.RESULT_KEY],
                        **self._dflt_args,
                    },
                )
            else:
                return self._error_response(
                    request, f"Failed to list streams: {result.get(WebApp.ERROR_KEY, 'Unknown error')}"
                )
