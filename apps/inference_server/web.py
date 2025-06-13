"""Web API for Inference Server"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mediamtx_client.api_client import ApiClient
from pydantic import BaseModel, Field

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

    def __init__(self, app_name:str, manager: Manager):
        """Initialize the WebApp with the application name and manager"""
        self._manager: Manager = manager  # Manager instance
        self._app_name: str = app_name  # Name of the application
        self._app: FastAPI = FastAPI(lifespan=self._lifespan)  # FastAPI instance with lifespan events
        self._templates: Jinja2Templates = Jinja2Templates(directory="templates")  # Template engine for rendering HTML
        self._dflt_args: dict[str, str] = {"app_name": self._app_name}  # Default arguments for templates

        # Mount static files
        self._app.mount("/static", StaticFiles(directory="static"), name="static")

        # Register routes
        self._register_routes()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Lifespan event handler for startup and shutdown logic"""
        logging.info("Application is starting up...")

        yield  # This allows the app to run

        logging.info("Application is shutting down...")

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
            return self._templates.TemplateResponse(
                "inference_server_index.html", {"request": request, **self._dflt_args}
            )
