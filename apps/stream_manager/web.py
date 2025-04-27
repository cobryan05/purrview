from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from distutils.util import strtobool

from mediamtx_client.api_client import ApiClient

from apps.stream_manager.manager import Manager

from typing import Dict, List

import sys
import logging

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class WebApp:
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

        @self._app.get("/create-stream-form", response_class=HTMLResponse)
        async def create_stream_form(request: Request):
            """Render the form for creating a new stream"""
            try:
                return self._templates.TemplateResponse(
                    "create_stream_form.html",
                    {"request": request, **self._dflt_args},
                )
            except Exception as e:
                return self._error_response(request, f"Error fetching create stream form: {str(e)}")

        @self._app.post("/create-stream", response_class=HTMLResponse)
        async def create_stream(request: Request):
            """Handle the creation of a new stream"""
            try:
                # Parse form data
                form_data = await request.form()
                source_url: str = form_data.get("source_url")  # Source URL for the stream
                destination_name: str = form_data.get("destination_name")  # Destination name for the stream
                delay: float = float(form_data.get("delay", "0"))  # Delay for the stream

                logging.info(
                    f"Creating stream with Source URL: {source_url}, Destination Name: {destination_name}, Delay: {delay}"
                )
                await self._manager.create_rtsp_relay_stream(rtsp_url=source_url, stream_name=destination_name)
                if delay > 0:
                    streams = self._manager.get_streams()
                    stream = streams.get(destination_name, None)
                    if stream is None:
                        raise Exception(f"Failed to create delated stream, stream {destination_name} was not in stream list")
                    await self._manager.create_delay_stream(stream.url, f"{destination_name}_delayed", delay)


            except Exception as e:
                return self._error_response(request, f"Failed to create stream: {e}")
            return await list_streams(request)

        @self._app.post("/destroy-stream", response_class=HTMLResponse)
        async def destroy_stream(request: Request):
            """Handle the destruction of a stream"""
            try:
                # Parse form data
                form_data = await request.form()
                stream_name: str = form_data.get("stream_name")  # Name of the stream to destroy

                # Log or process the data
                logging.info(f"Destroying stream {stream_name}")

                await self._manager.destroy_stream(stream_name=stream_name)
                return await list_streams(request)
            except Exception as e:
                return self._error_response(request, f"Failed to destroy stream: {e}")

        @self._app.get("/list-streams", response_class=HTMLResponse)
        async def list_streams(request: Request):
            """List all available streams"""
            try:
                streams = self._manager.get_streams()  # Fetch the list of streams
                stream_info: List[Dict[str, str]] = []
                for info in streams.values():
                    stream_info.append({"name": info.name, "text": str(info)})  # Format stream information

                return self._templates.TemplateResponse(
                    "streams_list.html",
                    {
                        "request": request,
                        "streams": stream_info,
                        **self._dflt_args,
                    },
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error fetching streams: {str(e)}")

        @self._app.get("/list-config", response_class=HTMLResponse)
        async def list_configs(request: Request):
            """List the global configuration"""
            try:
                config = await self._manager.get_config()  # Fetch the global configuration
                return self._templates.TemplateResponse(
                    "display_text.html",
                    {
                        "request": request,
                        "text": config,
                        **self._dflt_args,
                    },
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error fetching streams: {str(e)}")
