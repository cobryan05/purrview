"""Web API for Inference Server"""

import base64
import logging
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from apps.helpers.consts import ApiTags, JsonKeys, JsonValues
from apps.helpers.types import (
    DetectionResult,
    InferenceResult,
    InferenceResultModel,
    StatusResponse,
)

from .manager import Manager


class AssociateLabelPayload(BaseModel):
    label_uuid: Optional[str]


class ModelBulkClassesPayload(BaseModel):
    model_names: List[str]


class ModelBulkClassesResponse(StatusResponse):
    models: Dict[str, Dict[str, Optional[str]]] = Field(default_factory=dict)


class PinModelPayload(BaseModel):
    model_name: str
    duration: int


class InferencePayload(BaseModel):
    model_name: str
    conf_thresh: float
    return_source_img: bool = False
    return_annotated_img: bool = False
    pin_id: Optional[str] = None
    image_base64: str  # base64 encoded image string


class InferenceResponse(StatusResponse):
    result: Optional[InferenceResultModel] = None


class DetectionResultModel(BaseModel):
    bounding_box: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_str: Optional[str] = None
    label_uuid: Optional[str] = None


class ModelListResponse(StatusResponse):
    models: List[str] = Field(default_factory=list)


class ModelClassesResponse(StatusResponse):
    # Mapping of { class_name: label_uuid }
    classes: Dict[str, Optional[str]] = Field(default_factory=dict)


class AssociateLabelResponse(StatusResponse):
    label_set: bool = False


class PinModelResponse(StatusResponse):
    pin_id: Optional[str] = None


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
        self._templates: Jinja2Templates = Jinja2Templates(directory="/app/templates")  # Template engine for rendering HTML
        self._dflt_args: dict[str, str] = {"app_name": self._app_name}  # Default arguments for templates

        # Mount static files
        self._app.mount("/static", StaticFiles(directory="/app/static"), name="static")

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
            "/api/models",
            response_model=ModelListResponse,
            tags=[ApiTags.MODELS],
        )
        async def list_models_api():
            try:
                model_list = await self._manager.list_models()
                return ModelListResponse(status=JsonValues.SUCCESS, models=model_list)
            except Exception as e:
                return ModelListResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post("/api/models/classes/bulk", response_model=ModelBulkClassesResponse, tags=[ApiTags.MODELS])
        async def get_bulk_model_classes_api(payload: ModelBulkClassesPayload):
            try:
                # The manager already supports a list of names
                model_class_maps = await self._manager.get_models_classes(payload.model_names)
                return ModelBulkClassesResponse(
                    status=JsonValues.SUCCESS, models=model_class_maps  # Dict[str, Dict[str, Optional[str]]]
                )
            except Exception as e:
                return ModelBulkClassesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/models/{model_name}/classes",
            response_model=ModelClassesResponse,
            tags=[ApiTags.MODELS],
        )
        async def get_model_classes_api(model_name: str):
            try:
                res = await self._manager.get_models_classes([model_name])
                class_map = res.get(model_name, {})
                return ModelClassesResponse(status=JsonValues.SUCCESS, classes=class_map)
            except Exception as e:
                return ModelClassesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.put(
            "/api/models/{model_name}/classes/{class_name}/label",
            response_model=AssociateLabelResponse,
            tags=[ApiTags.MODELS],
        )
        async def associate_label_api(model_name: str, class_name: str, payload: AssociateLabelPayload):
            try:
                ret = await self._manager.set_model_class_label_uuid(
                    model_name=model_name, model_class=class_name, label_uuid=payload.label_uuid
                )
                return AssociateLabelResponse(status=JsonValues.SUCCESS, label_set=ret)
            except Exception as e:
                return AssociateLabelResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/models/{model_name}/pin",
            response_model=PinModelResponse,
            tags=[ApiTags.MODELS],
        )
        async def pin_model_api(model_name: str, payload: PinModelPayload):
            try:
                pin_id = await self._manager.pin_model(model_name, payload.duration)
                return PinModelResponse(status=JsonValues.SUCCESS, pin_id=pin_id)
            except Exception as e:
                return PinModelResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/models/inference",
            tags=[ApiTags.INFERENCE],
            operation_id="inference",
            response_model=InferenceResponse,
        )
        async def inference_api(payload: InferencePayload) -> InferenceResponse:
            try:
                np_bytes = base64.b64decode(payload.image_base64)
                np_image = np.frombuffer(np_bytes, np.uint8)
                image_array: np.ndarray = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

                inference_result: InferenceResult = await self._manager.recognize(
                    model_name=payload.model_name,
                    image=image_array,
                    conf_thresh=payload.conf_thresh,
                    return_annotated_img=payload.return_annotated_img,
                    pin_id=payload.pin_id,
                )

                if not payload.return_source_img:
                    inference_result.source_image = None

                model = InferenceResultModel.from_dataclass(inference_result)
                return InferenceResponse(result=model)
            except Exception as e:
                logger.error(e, exc_info=True)
                return InferenceResponse(status=JsonValues.FAILURE, message=str(e))
