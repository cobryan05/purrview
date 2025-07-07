"""Web API for Tasks Server"""

import base64
import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from apps.helpers.inferenceProviders.inferenceProvider import InferenceResult

from .manager import Manager

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class WebApp:
    """Web application for managing inference models and running recognitions"""

    SUCCESS_KEY = "success"
    ERROR_KEY = "error"
    RESULT_KEY = "results"

    TASKS_API_TAG_NAME = "tasks"

    def __init__(self, app_name: str, manager: Manager):
        """
        Initialize the WebApp with the application name and manager.

        Args:
            app_name (str): Name of the application.
            manager (Manager): Instance of the Manager class
        """
        self._manager: Manager = manager  # Manager instance
        self._app_name: str = app_name  # Name of the application
        self._app: FastAPI = FastAPI(lifespan=self._lifespan)  # FastAPI instance with lifespan events
        self._templates: Jinja2Templates = Jinja2Templates(directory="templates")  # Template engine for rendering HTML
        self._dflt_args: dict[str, str] = {"app_name": self._app_name}  # Default arguments for templates

        # Mount static files
        self._app.mount("/static", StaticFiles(directory="static"), name="static")

        # Register routes
        self._register_routes()
        self._register_exception()

    def _register_exception(self):
        @self._app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):

            exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
            logger.error(request, exc_str)
            content = {"status_code": 10422, "message": exc_str, "data": None}
            return JSONResponse(content=content, status_code=422)

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """
        Lifespan event handler for startup and shutdown logic.

        Args:
            app (FastAPI): The FastAPI application instance.
        """
        logging.info("Application is starting up...")

        self._manager.start()  # Start the manager's periodic task
        yield  # This allows the app to run

        logging.info("Application is shutting down...")

    def app(self) -> FastAPI:
        """
        Return the FastAPI application instance.

        Returns:
            FastAPI: The FastAPI application instance.
        """
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
        """Register all routes for the application."""

        @self._app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            """Home Page"""
            buttons = [
                {"label": "List Tasks", "action": "/list-tasks"},
            ]
            return self._templates.TemplateResponse(
                "dynamic_index.html",
                {
                    "request": request,
                    "app_name": "Tasks Server",
                    "buttons": buttons,
                },
            )

        @self._app.get(
            "/api/tasks/list-active",
            response_class=JSONResponse,
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="list_active_tasks_api",
        )
        async def list_active_tasks_api(request: Request) -> JSONResponse:
            """
            API endpoint to return a list of active tasks.

            Args:
                request (Request): The FastAPI request object.

            Returns:
                JSONResponse: A JSON response containing the list of models.
            """
            try:
                task_list = await self._manager.list_active_tasks()
                response_data = {"status": "success", "tasks": task_list}
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={"status": "failure", "message": str(e)},
                    status_code=500,
                )

        @self._app.get(
            "/api/tasks/list-avail",
            response_class=JSONResponse,
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="list_avail_tasks_api",
        )
        async def list_avail_tasks_api(request: Request) -> JSONResponse:
            """
            API endpoint to return a list of active tasks.

            Args:
                request (Request): The FastAPI request object.

            Returns:
                JSONResponse: A JSON response containing the list of models.
            """
            try:
                task_list = await self._manager.list_avail_tasks()
                response_data = {"status": "success", "tasks": task_list}
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={"status": "failure", "message": str(e)},
                    status_code=500,
                )

        @self._app.post("/list-tasks", response_class=HTMLResponse, include_in_schema=False)
        async def list_tasks(
            request: Request, model_name: str = Form(...), duration: float = Form(...)
        ) -> HTMLResponse:
            """
            HTML endpoint to list tasks

            Args:
                request (Request): The FastAPI request object.
                model_name (str): Name of the model to pin.
                duration (str): Duration for which the model should be pinned.

            Returns:
                HTMLResponse: Rendered HTML response containing the pin ID or an error message.
            """
            try:
                # Call the API function to list tasks
                response: JSONResponse = await list_tasks_api(request=request)
                response_data = json.loads(response.body.decode("utf-8"))

                # Render the HTML response using the dynamic template
                return self._templates.TemplateResponse(
                    "dynamic_response.html",
                    {
                        "request": request,
                        "title": "List Tasks Results",
                        "response_data": response_data,
                    },
                )
            except Exception as e:
                return self._error_response(request, f"Internal server error: {str(e)}")
