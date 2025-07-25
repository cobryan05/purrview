"""Web API for Tasks Server"""

import html
import json
import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional, List

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .manager import Manager, TaskInfo

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
            """
            Custom exception handler for validation errors.

            Args:
                request (Request): The incoming request.
                exc (RequestValidationError): The validation exception.

            Returns:
                JSONResponse: A JSON-formatted error response.
            """
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
        """
        Render an error response using the dynamic response template.

        Args:
            request (Request): The request object.
            message (str): The error message to display.

        Returns:
            TemplateResponse: Rendered HTML error response.
        """
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
            """
            Render the home page with navigation buttons.

            Args:
                request (Request): The incoming request.

            Returns:
                HTMLResponse: Rendered homepage with buttons.
            """
            buttons = [
                {"label": "Task Status", "action": "/task-status-form"},
                {"label": "List Available Tasks", "action": "/api/tasks/list-avail"},
                {"label": "Create Task", "action": "/create-task-form"},
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
                response_data = {"status": "success", "config": json.dumps(server_config)}
            except Exception as e:
                response_data = {"status": "failure", "message": str(e)}
            return JSONResponse(content=response_data)

        class TaskStatusRequest(BaseModel):
            """Request model for creating a new task."""

            task_ids: Optional[List[int]] = None

        @self._app.post(
            "/api/tasks/status",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="get_task_status",
            response_class=JSONResponse,
        )
        async def task_status_api(req: TaskStatusRequest) -> JSONResponse:
            """
            API endpoint for getting status of one or more tasks.

            Args:
                req (TaskStatusRequest): Request object with task IDs.

            Returns:
                JSONResponse: A JSON response containing task statuses.
            """
            try:
                tasks_status = await self._manager.get_tasks_status(req.task_ids)
                response_data = {"status": "success", "tasks": json.dumps(tasks_status)}
            except Exception as e:
                response_data = {"status": "failure", "message": str(e)}
            return JSONResponse(content=response_data)

        @self._app.post("/task-status", response_class=HTMLResponse, include_in_schema=False)
        async def task_status(
            task_ids: str = Form(...),
            request: Request = None,
        ):
            """
            HTML form handler for displaying task statuses.

            Args:
                task_ids (str): Comma-separated list of task IDs.
                request (Request): The request object.

            Returns:
                HTMLResponse: Rendered HTML form with task statuses and actions.
            """
            try:
                # Parse task_ids: empty string -> None, else list of ints
                if not task_ids.strip():
                    parsed_task_ids = None
                else:
                    parsed_task_ids = [int(tid.strip()) for tid in task_ids.split(",") if tid.strip()]

                req: TaskStatusRequest = TaskStatusRequest(task_ids=parsed_task_ids)
                reponse_json = await task_status_api(req)
                reponse_data = json.loads(reponse_json.body)
                tasks = json.loads(reponse_data.get("tasks", "{}"))

                # Prepare fields for dynamic_form.html
                fields = {
                    "selected_task": {
                        "label": "Select Tasks",
                        "type": "checkbox",
                        "options": list(tasks.keys()),
                        "option_labels": {
                            k: f"Task {k} — {v.get('typename', '')} (progress: {v.get('progress', 0)}%, status: {v.get('status', {})}), result: {v.get('result', {})})"
                            for k, v in tasks.items()
                        },
                    }
                }

                # Define extra buttons (actions) at the end of the form
                extra_buttons = [
                    {"label": "Refresh", "action": "/task-status", "method": "post"},
                    {"label": "Pause", "action": "/task-pause", "method": "post"},
                    {"label": "Resume", "action": "/task-resume", "method": "post"},
                    {"label": "Delete", "action": "/task-delete", "method": "post"},
                ]

                return self._templates.TemplateResponse(
                    "dynamic_form.html",
                    {
                        "request": request,
                        "title": "Task Status",
                        "action_url": "/task-status",  # Default action for the form
                        "fields": fields,
                        "hidden_fields": {"task_ids": task_ids},
                        "submit_label": "Submit",
                        "extra_buttons": extra_buttons,
                    },
                )
            except Exception as e:
                return self._error_response(request, f"Internal server error: {str(e)}")

        @self._app.get("/task-status-form", response_class=HTMLResponse)
        async def task_status_form(request: Request) -> HTMLResponse:
            """
            Display the form for checking task statuses.

            Args:
                request (Request): The request object.

            Returns:
                HTMLResponse: Rendered form for entering task IDs.
            """
            fields = {
                "task_ids": {"type": "text", "label": "Task ids (comma separated) or empty", "optional": True},
            }
            return self._templates.TemplateResponse(
                "dynamic_form.html",
                {
                    "request": request,
                    "title": "Task Status",
                    "action_url": "/task-status",
                    "fields": fields,
                    "submit_label": "Submit",
                },
            )

        class DeleteTasksRequest(BaseModel):
            task_ids: List[int]

        @self._app.post(
            "/api/tasks/delete",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="delete_tasks",
            response_class=JSONResponse,
        )
        async def delete_tasks_api(req: DeleteTasksRequest) -> JSONResponse:
            """
            API endpoint to delete tasks.

            Args:
                req (DeleteTasksRequest): Request object with task IDs to delete.

            Returns:
                JSONResponse: JSON response with deletion result.
            """
            try:
                await self._manager.delete_tasks(req.task_ids)
                response_data = {"status": "success", "deleted": req.task_ids}
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={"status": "failure", "message": str(e)},
                    status_code=500,
                )

        @self._app.post("/task-delete", response_class=HTMLResponse, include_in_schema=False)
        async def delete_tasks_form(
            request: Request,
            selected_task: Optional[List[str]] = Form(None),
        ):
            """
            HTML form handler to delete selected tasks.

            Args:
                request (Request): The request object.
                selected_task (Optional[List[str]]): List of selected task IDs as strings.

            Returns:
                HTMLResponse: Rendered confirmation of deleted tasks.
            """
            try:
                if not selected_task:
                    return self._error_response(request, "No tasks selected for deletion.")

                task_ids = [int(tid) for tid in selected_task]
                api_request = DeleteTasksRequest(task_ids=task_ids)
                response = await delete_tasks_api(api_request)
                return await task_status(task_ids="", request=request)

            except Exception as e:
                return self._error_response(request, f"Internal server error: {str(e)}")

        class PauseTasksRequest(BaseModel):
            task_ids: List[int]

        @self._app.post(
            "/api/tasks/pause",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="pause_tasks",
            response_class=JSONResponse,
        )
        async def pause_tasks_api(req: PauseTasksRequest) -> JSONResponse:
            """
            API endpoint to pause tasks.

            Args:
                req (PauseTasksRequest): Request object with task IDs to pause.

            Returns:
                JSONResponse: JSON response with pause result.
            """
            try:
                await self._manager.pause_tasks(req.task_ids)
                response_data = {"status": "success", "paused": req.task_ids}
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={"status": "failure", "message": str(e)},
                    status_code=500,
                )

        @self._app.post("/task-pause", response_class=HTMLResponse, include_in_schema=False)
        async def pause_tasks_form(
            request: Request,
            selected_task: Optional[List[str]] = Form(None),
        ):
            """
            HTML form handler to pause selected tasks.

            Args:
                request (Request): The request object.
                selected_task (Optional[List[str]]): List of selected task IDs as strings.

            Returns:
                HTMLResponse: Rendered confirmation of paused tasks.
            """
            try:
                if not selected_task:
                    return self._error_response(request, "No tasks selected for pausing.")

                task_ids = [int(tid) for tid in selected_task]
                api_request = PauseTasksRequest(task_ids=task_ids)
                response = await pause_tasks_api(api_request)
                return await task_status(task_ids="", request=request)

            except Exception as e:
                return self._error_response(request, f"Internal server error: {str(e)}")

        class ResumeTasksRequest(BaseModel):
            task_ids: List[int]

        @self._app.post(
            "/api/tasks/resume",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="resume_tasks",
            response_class=JSONResponse,
        )
        async def resume_tasks_api(req: ResumeTasksRequest) -> JSONResponse:
            """
            API endpoint to resume tasks.

            Args:
                req (ResumeTasksRequest): Request object with task IDs to resume.

            Returns:
                JSONResponse: JSON response with resume result.
            """
            try:
                await self._manager.resume_tasks(req.task_ids)
                response_data = {"status": "success", "resumed": req.task_ids}
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={"status": "failure", "message": str(e)},
                    status_code=500,
                )

        @self._app.post("/task-resume", response_class=HTMLResponse, include_in_schema=False)
        async def resume_tasks_form(
            request: Request,
            selected_task: Optional[List[str]] = Form(None),
        ):
            """
            HTML form handler to resume selected tasks.

            Args:
                request (Request): The request object.
                selected_task (Optional[List[str]]): List of selected task IDs as strings.

            Returns:
                HTMLResponse: Rendered confirmation of paused tasks.
            """
            try:
                if not selected_task:
                    return self._error_response(request, "No tasks selected for resuming.")

                task_ids = [int(tid) for tid in selected_task]
                api_request = ResumeTasksRequest(task_ids=task_ids)
                response = await resume_tasks_api(api_request)
                return await task_status(task_ids="", request=request)

            except Exception as e:
                return self._error_response(request, f"Internal server error: {str(e)}")

        @self._app.get(
            "/api/tasks/list-avail",
            response_class=JSONResponse,
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="list_avail_tasks_api",
        )
        async def list_avail_tasks_api(request: Request) -> JSONResponse:
            """
            API endpoint to return a list of available tasks.

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

        class CreateTaskRequest(BaseModel):
            """Request model for creating a new task."""

            typename: str
            params: dict

        @self._app.post(
            "/api/tasks/create",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="create_task",
            response_class=JSONResponse,
        )
        async def create_task_api(req: CreateTaskRequest) -> JSONResponse:
            """
            API endpoint for creating a new task.

            Args:
                req (CreateTaskRequest): Request object with task type and parameters.

            Returns:
                JSONResponse: Response with new task ID or error.
            """
            try:
                task_id: int = await self._manager.create_new_task(typename=req.typename, params=req.params)
                response_data = {"status": "success", "task_id": task_id}
            except Exception as e:
                response_data = {"status": "failure", "message": str(e)}
            return JSONResponse(content=response_data)

        @self._app.post("/create-task", response_class=HTMLResponse, include_in_schema=False)
        async def create_task(
            task_name: str = Form(...),
            params: str = Form(""),
            request: Request = None,
        ):
            """
            HTML form handler for creating a task.

            Args:
                task_name (str): Task type name.
                params (str): JSON-formatted parameters.
                request (Request): The request object.

            Returns:
                HTMLResponse: Rendered result of task creation.
            """
            try:
                req: CreateTaskRequest = CreateTaskRequest(
                    typename=task_name, params=json.loads(params) if params else {}
                )
                reponse_json = await create_task_api(req)
                reponse_data = json.loads(reponse_json.body)
                return self._templates.TemplateResponse(
                    "dynamic_response.html",
                    {
                        "request": request,
                        "title": "Create Task Results",
                        "response_data": reponse_data,
                    },
                )
            except Exception as e:
                return self._error_response(request, f"Internal server error: {str(e)}")

        @self._app.get("/create-task-form", response_class=HTMLResponse)
        async def create_task_form(request: Request) -> HTMLResponse:
            """
            Display the form for creating a new task.

            Args:
                request (Request): The request object.

            Returns:
                HTMLResponse: Rendered task creation form.
            """
            task_list = await self._manager.list_avail_tasks()
            fields = {
                "task_name": {
                    "label": "Task Name",
                    "type": "radio",
                    "options": task_list,
                },
                "params": {"label": "Parameters (JSON)", "type": "text", "optional": True},
            }
            return self._templates.TemplateResponse(
                "dynamic_form.html",
                {
                    "request": request,
                    "title": "Create Task",
                    "action_url": "/create-task",
                    "fields": fields,
                    "submit_label": "Submit",
                    "params_schema_url": "/task_schema",
                    "enable_dynamic_help": True,
                },
            )

        @self._app.post(
            "/task_schema",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="schema",
            response_class=HTMLResponse,
        )
        async def get_task_schema(typename: str = Form(...)) -> HTMLResponse:
            """
            Endpoint to retrieve and display the schema for a given task type.

            Args:
                typename (str): The name of the task type to retrieve the schema for.

            Returns:
                HTMLResponse: Rendered HTML content of the JSON schema or error message.
            """
            try:
                schema = await self._manager.get_task_schema(typename=typename)
                pretty = json.dumps(schema, indent=2)
                escaped = html.escape(pretty)
                html_content = f"<pre>{escaped}</pre>"
            except Exception as e:
                html_content = f"<pre>Error: {html.escape(str(e))}</pre>"
            return HTMLResponse(content=html_content)
