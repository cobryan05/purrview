"""Web API for Inference Server"""

import asyncio
import base64
import logging
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from apps.helpers.consts import ApiTags, JsonKeys, JsonValues
from apps.helpers.curatedModels import CURATED_BASE_MODELS
from apps.helpers.imageUtils import base64_decode_image
from apps.helpers.types import (
    InferenceResult,
    InferenceResultModel,
    Payload,
    StatusResponse,
)

from .manager import Manager


class AssociateLabelPayload(Payload):
    label_uuid: Optional[str]


class ModelBulkClassesPayload(Payload):
    model_names: List[str]


class ModelBulkClassesResponse(StatusResponse):
    models: Dict[str, Dict[str, Optional[str]]] = Field(default_factory=dict)


class PinModelPayload(Payload):
    model_name: str
    duration: int = Field(gt=0, description="Pin lifetime in seconds; must be positive")


class InferencePayload(Payload):
    conf_thresh: float = Field(ge=0.0, le=1.0, description="Confidence threshold; must be in [0, 1]")
    return_source_img: bool = False
    return_annotated_img: bool = False
    pin_id: Optional[str] = None
    image_base64: str  # base64 encoded image string


class ClassifyPayload(Payload):
    image_base64: str


class ClassifyPrediction(Payload):
    label_uuid: str
    confidence: float
    is_present: bool


class ClassifyResult(Payload):
    model_kind: str
    predictions: List[ClassifyPrediction]


class ClassifyResponse(StatusResponse):
    result: Optional[ClassifyResult] = None


class ModelMetaResponse(StatusResponse):
    model_kind: Optional[str] = None
    label_uuid: Optional[str] = None
    label_uuids: Optional[List[str]] = None
    device: str = "auto"


class SetModelDevicePayload(Payload):
    device: str


class SetModelDeviceResponse(StatusResponse):
    device_set: bool = False


class ImportModelUploadPayload(Payload):
    model_name: str
    onnx_base64: str
    overwrite: bool = False


class ImportModelUploadResponse(StatusResponse):
    model_name: Optional[str] = None
    classes: Optional[List[str]] = None


class DeleteModelResponse(StatusResponse):
    model_name: Optional[str] = None


class ImportModelDetectPayload(Payload):
    filename: str
    header_base64: str


class ImportModelDetectResponse(StatusResponse):
    format: Optional[str] = None
    is_onnx: bool = False
    conversion_supported: bool = False


class StageSourceModelPayload(Payload):
    source_base64: str


class StageSourceModelResponse(StatusResponse):
    staging_id: Optional[str] = None


class CuratedModelInfo(Payload):
    key: str
    display_name: str
    requires_conversion: bool = False


class CuratedModelsResponse(StatusResponse):
    models: List[CuratedModelInfo] = Field(default_factory=list)


class ModelTrainingImagesResponse(StatusResponse):
    train_image_uuids: List[str] = Field(default_factory=list)
    val_image_uuids: List[str] = Field(default_factory=list)
    # Provenance, so a caller that guessed this model name from a training config
    # can confirm it really is that config's last run before showing its numbers.
    trained_at: Optional[str] = None
    model_config_node_uuid: Optional[str] = None


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
        self._templates: Jinja2Templates = Jinja2Templates(
            directory="/app/templates"
        )  # Template engine for rendering HTML
        self._dflt_args: dict[str, str] = {"app_name": self._app_name}  # Default arguments for templates

        # Mount static files
        self._app.mount("/static", StaticFiles(directory="/app/static", check_dir=False), name="static")

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
                response_data = {
                    JsonKeys.STATUS: JsonValues.SUCCESS,
                    JsonKeys.CONFIG: server_config,
                }
            except Exception as e:
                response_data = {
                    JsonKeys.STATUS: JsonValues.FAILURE,
                    JsonKeys.MESSAGE: str(e),
                }
            return JSONResponse(content=response_data)

        @self._app.get(
            "/status",
            operation_id="inference_server_status",
            response_class=JSONResponse,
        )
        async def inference_server_status():
            return JSONResponse(content=self._manager.get_status())

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

        @self._app.post(
            "/api/models/classes/bulk",
            response_model=ModelBulkClassesResponse,
            tags=[ApiTags.MODELS],
        )
        async def get_bulk_model_classes_api(payload: ModelBulkClassesPayload):
            try:
                # The manager already supports a list of names
                model_class_maps = await self._manager.get_models_classes(payload.model_names)
                return ModelBulkClassesResponse(
                    status=JsonValues.SUCCESS,
                    models=model_class_maps,  # Dict[str, Dict[str, Optional[str]]]
                )
            except Exception as e:
                return ModelBulkClassesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/models/{model_name}/meta",
            response_model=ModelMetaResponse,
            tags=[ApiTags.MODELS],
            operation_id="get_model_meta",
        )
        async def get_model_meta_api(model_name: str) -> ModelMetaResponse:
            """Return the model kind and label UUID(s) from its companion JSON config."""
            try:
                config = self._manager.get_model_config(model_name)
                if config is None:
                    return ModelMetaResponse(status=JsonValues.FAILURE, message=f"Model '{model_name}' not found")
                model_kind = config.get("model_kind", "detector")
                return ModelMetaResponse(
                    status=JsonValues.SUCCESS,
                    model_kind=model_kind,
                    label_uuid=config.get("label_uuid"),
                    label_uuids=config.get("label_uuids"),
                    device=config.get("device", "auto"),
                )
            except Exception as e:
                return ModelMetaResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/models/{model_name}/device",
            response_model=SetModelDeviceResponse,
            tags=[ApiTags.MODELS],
            operation_id="set_model_device",
        )
        async def set_model_device_api(model_name: str, payload: SetModelDevicePayload):
            try:
                ret = await self._manager.set_model_device(model_name=model_name, device=payload.device)
                return SetModelDeviceResponse(status=JsonValues.SUCCESS, device_set=ret)
            except Exception as e:
                return SetModelDeviceResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/models/curated",
            response_model=CuratedModelsResponse,
            tags=[ApiTags.MODELS],
            operation_id="get_curated_models",
        )
        async def get_curated_models_api() -> CuratedModelsResponse:
            """List the curated base models offered for cold-start import (key + display
            name only — internal fields like onnx_url/model_family stay server-side)."""
            try:
                models = [
                    CuratedModelInfo(
                        key=entry["key"],
                        display_name=entry["display_name"],
                        requires_conversion=bool(entry.get("requires_conversion", False)),
                    )
                    for entry in CURATED_BASE_MODELS
                ]
                return CuratedModelsResponse(status=JsonValues.SUCCESS, models=models)
            except Exception as e:
                return CuratedModelsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/models/import/upload",
            response_model=ImportModelUploadResponse,
            tags=[ApiTags.MODELS],
            operation_id="import_model_upload",
        )
        async def import_model_upload_api(payload: ImportModelUploadPayload) -> ImportModelUploadResponse:
            """Import an uploaded ONNX file: decode + write it + auto-generate its sidecar."""
            try:
                onnx_bytes = base64.b64decode(payload.onnx_base64, validate=True)
            except Exception as e:
                return ImportModelUploadResponse(status=JsonValues.FAILURE, message=f"Invalid onnx_base64: {e}")

            try:
                result = await self._manager.import_model_upload(
                    model_name=payload.model_name,
                    onnx_bytes=onnx_bytes,
                    overwrite=payload.overwrite,
                )
                return ImportModelUploadResponse(
                    status=JsonValues.SUCCESS,
                    model_name=result["model_name"],
                    classes=result["classes"],
                )
            except Exception as e:
                return ImportModelUploadResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/models/detect",
            response_model=ImportModelDetectResponse,
            tags=[ApiTags.MODELS],
            operation_id="detect_model_format",
        )
        async def detect_model_format_api(payload: ImportModelDetectPayload) -> ImportModelDetectResponse:
            """Detect a model file's format from its filename + leading bytes (torch-free)."""
            try:
                header_bytes = base64.b64decode(payload.header_base64, validate=True)
            except Exception as e:
                return ImportModelDetectResponse(status=JsonValues.FAILURE, message=f"Invalid header_base64: {e}")

            try:
                result = self._manager.detect_uploaded_model(payload.filename, header_bytes)
                return ImportModelDetectResponse(
                    status=JsonValues.SUCCESS,
                    format=result["format"],
                    is_onnx=result["is_onnx"],
                    conversion_supported=result["conversion_supported"],
                )
            except Exception as e:
                return ImportModelDetectResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/models/import/stage",
            response_model=StageSourceModelResponse,
            tags=[ApiTags.MODELS],
            operation_id="stage_source_model",
        )
        async def stage_source_model_api(payload: StageSourceModelPayload) -> StageSourceModelResponse:
            """Stage raw uploaded model bytes to a shared dir for later conversion (e.g. pytorch->onnx)."""
            try:
                source_bytes = base64.b64decode(payload.source_base64, validate=True)
            except Exception as e:
                return StageSourceModelResponse(status=JsonValues.FAILURE, message=f"Invalid source_base64: {e}")

            try:
                staging_id = await self._manager.stage_source_model(source_bytes)
                return StageSourceModelResponse(status=JsonValues.SUCCESS, staging_id=staging_id)
            except Exception as e:
                return StageSourceModelResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/models/{model_name}/training-images",
            response_model=ModelTrainingImagesResponse,
            tags=[ApiTags.MODELS],
            operation_id="get_model_training_images",
        )
        async def get_model_training_images_api(model_name: str) -> ModelTrainingImagesResponse:
            """Return the train/val image UUIDs recorded in the model's companion JSON."""
            try:
                config = self._manager.get_model_config(model_name)
                if config is None:
                    return ModelTrainingImagesResponse(
                        status=JsonValues.FAILURE, message=f"Model '{model_name}' not found"
                    )
                return ModelTrainingImagesResponse(
                    status=JsonValues.SUCCESS,
                    train_image_uuids=config.get("train_image_uuids") or [],
                    val_image_uuids=config.get("val_image_uuids") or [],
                    trained_at=config.get("trained_at"),
                    model_config_node_uuid=config.get("model_config_node_uuid"),
                )
            except Exception as e:
                return ModelTrainingImagesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/models/{model_name}/classify",
            response_model=ClassifyResponse,
            tags=[ApiTags.INFERENCE],
            operation_id="classify",
        )
        async def classify_api(model_name: str, payload: ClassifyPayload) -> ClassifyResponse:
            """Run an ONNX classifier on the supplied image and return per-label predictions."""
            try:
                config = self._manager.get_model_config(model_name)
                if config is None:
                    return ClassifyResponse(status=JsonValues.FAILURE, message=f"Model '{model_name}' not found")

                model_kind = config.get("model_kind")
                if model_kind not in ("image_classifier", "multi_label_classifier"):
                    return ClassifyResponse(
                        status=JsonValues.FAILURE,
                        message=f"Model '{model_name}' is not a classifier (kind={model_kind!r})",
                    )

                image_bgr: np.ndarray = base64_decode_image(payload.image_base64)
                # conf_thresh=0.0 → provider returns all labels; is_present is derived here.
                inference_result = await self._manager.recognize(model_name, image_bgr, conf_thresh=0.0)

                predictions = [
                    ClassifyPrediction(
                        label_uuid=det.bbox.label_uuid,
                        confidence=det.confidence,
                        is_present=det.confidence >= 0.5,
                    )
                    for det in inference_result.detections
                ]

                return ClassifyResponse(
                    status=JsonValues.SUCCESS,
                    result=ClassifyResult(model_kind=model_kind, predictions=predictions),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return ClassifyResponse(status=JsonValues.FAILURE, message=str(e))

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
                    model_name=model_name,
                    model_class=class_name,
                    label_uuid=payload.label_uuid,
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

        @self._app.delete(
            "/api/models/{model_name}/pin/{pin_id}",
            response_model=StatusResponse,
            tags=[ApiTags.MODELS],
            operation_id="unpin_model",
        )
        async def unpin_model_api(model_name: str, pin_id: str):
            try:
                await self._manager.unpin_model(model_name, pin_id)
                return StatusResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.delete(
            "/api/models/{model_name}",
            response_model=DeleteModelResponse,
            tags=[ApiTags.MODELS],
            operation_id="delete_model",
        )
        async def delete_model_api(model_name: str) -> DeleteModelResponse:
            try:
                await self._manager.delete_model(model_name)
                return DeleteModelResponse(status=JsonValues.SUCCESS, model_name=model_name)
            except Exception as e:
                return DeleteModelResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/models/{model_name}/detect",
            tags=[ApiTags.INFERENCE],
            operation_id="detect",
            response_model=InferenceResponse,
        )
        async def inference_api(model_name: str, payload: InferencePayload) -> InferenceResponse:
            try:
                image_array: np.ndarray = base64_decode_image(payload.image_base64)

                inference_result: InferenceResult = await self._manager.recognize(
                    model_name=model_name,
                    image=image_array,
                    conf_thresh=payload.conf_thresh,
                    return_annotated_img=payload.return_annotated_img,
                    pin_id=payload.pin_id,
                )

                if not payload.return_source_img:
                    inference_result.source_image = None

                model = InferenceResultModel.model_validate(inference_result)
                return InferenceResponse(result=model)
            except Exception as e:
                logger.error(e, exc_info=True)
                return InferenceResponse(status=JsonValues.FAILURE, message=str(e))
