"""Web API for Inference Server"""

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
from apps.helpers.consts import ApiTags, JsonKeys, JsonValues

from .manager import Manager


class AssociateLabelWithModelClassPayload(BaseModel):
    model_name: str
    model_class: str
    label_uuid: Optional[str]


class GetModelLabelsPayload(BaseModel):
    model_name: str


class PinModelPayload(BaseModel):
    model_name: str
    duration: int


class RecognizePayload(BaseModel):
    model_name: str
    conf_thresh: float
    return_annotated: bool = False
    pin_id: Optional[str] = None
    image_base64: str  # base64 encoded image string


logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class WebApp:
    """Web application for managing inference models and running recognitions"""

    def __init__(self, app_name: str, manager: Manager):
        """
        Initialize the WebApp with the application name and manager.

        Args:
            app_name (str): Name of the application.
            manager (Manager): Instance of the Manager class for handling models.
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
            content = {"status_code": 10422, JsonKeys.MESSAGE: exc_str, "data": None}
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
                "response_data": {JsonKeys.ERROR: message},
                **self._dflt_args,
            },
        )

    def _register_routes(self):
        """Register all routes for the application."""

        @self._app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            """Home Page"""
            buttons = []
            return self._templates.TemplateResponse(
                "dynamic_index.html",
                {
                    "request": request,
                    "app_name": "Inference Server",
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
                response_data = {JsonKeys.STATUS: JsonValues.SUCCESS, JsonKeys.CONFIG: server_config}
            except Exception as e:
                response_data = {JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)}
            return JSONResponse(content=response_data)

        @self._app.get(
            "/api/models/list",
            response_class=JSONResponse,
            tags=[ApiTags.MODELS],
            operation_id="list_models",
        )
        async def list_models_api(request: Request) -> JSONResponse:
            """
            API endpoint to return a list of models.

            Args:
                request (Request): The FastAPI request object.

            Returns:
                JSONResponse: A JSON response containing the list of models.
            """
            try:
                model_list = await self._manager.list_models()
                response_data = {JsonKeys.STATUS: JsonValues.SUCCESS, "models": model_list}
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                    status_code=500,
                )

        @self._app.post(
            "/api/models/labels/get",
            response_class=JSONResponse,
            tags=[ApiTags.MODELS],
            operation_id="get_model_labels",
        )
        async def get_model_labels_api(payload: GetModelLabelsPayload) -> JSONResponse:
            """
            API endpoint to list labels for a given model.

            Returns:
                JSONResponse: A dictionary mapping label names to label IDs.
            """
            try:
                label_map: dict[str, str] = await self._manager.get_model_labels(payload.model_name)
            except Exception as e:
                logger.exception(e)
                return JSONResponse(content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)})

            return JSONResponse(
                content={
                    JsonKeys.STATUS: JsonValues.SUCCESS,
                    "labels": label_map,
                },
                status_code=200,
            )

        @self._app.post(
            "/api/models/labels/associate",
            tags=[ApiTags.MODELS],
            operation_id="associate_label_with_model_class",
            response_class=JSONResponse,
        )
        async def associate_label_with_model_class_api(payload: AssociateLabelWithModelClassPayload) -> JSONResponse:
            """
            API endpoint to associate a model's class with a label

            Returns:
                JSONResponse: A JSON response containing the list of labels for the model
            """
            try:
                ret = await self._manager.set_model_label_uuid(
                    model_name=payload.model_name, model_class=payload.model_class, label_uuid=payload.label_uuid
                )
                response_data = {JsonKeys.STATUS: JsonValues.SUCCESS, "label_set": ret}
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                    status_code=500,
                )

        @self._app.post(
            "/api/models/pin",
            response_class=JSONResponse,
            tags=[ApiTags.MODELS],
            operation_id="pin_model",
        )
        async def pin_model_api(payload: PinModelPayload) -> JSONResponse:
            """
            API endpoint to pin a model in memory.

            Args:
                request (Request): The FastAPI request object.
                model_name (str): Name of the model to pin.
                duration (str): Duration in seconds for which the model should be pinned.

            Returns:
                JSONResponse: JSON response containing the pin ID for use in unpin_model.
            """
            try:
                # Use the payload fields
                pin_id = await self._manager.pin_model(payload.model_name, payload.duration)
            except Exception as e:
                logger.exception(e)
                return JSONResponse(content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)})

            return JSONResponse(
                content={
                    JsonKeys.STATUS: JsonValues.SUCCESS,
                    JsonKeys.MESSAGE: "Model pinned successfully.",
                    "pin_id": pin_id,
                },
                status_code=200,
            )

        @self._app.post(
            "/api/recognize",
            tags=[ApiTags.INFERENCE],
            operation_id="recognize",
            response_class=JSONResponse,
        )
        async def recognize_api(payload: RecognizePayload):
            try:
                np_bytes = base64.b64decode(payload.image_base64)
                np_image = np.frombuffer(np_bytes, np.uint8)
                image_array = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

                inference_result: InferenceResult = await self._manager.recognize(
                    model_name=payload.model_name,
                    image=image_array,
                    conf_thresh=payload.conf_thresh,
                    return_annotated=payload.return_annotated,
                    pin_id=payload.pin_id,
                )
                return JSONResponse(
                    content={
                        JsonKeys.STATUS: JsonValues.SUCCESS,
                        **inference_result.serialize(include_annotated=payload.return_annotated),
                    }
                )
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                    status_code=500,
                )
