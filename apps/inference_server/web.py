"""Web API for Inference Server"""

import asyncio
import base64
import logging
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

from apps.helpers.consts import ApiTags, JsonKeys, JsonValues
from apps.helpers.curatedModels import CURATED_BASE_MODELS
from apps.helpers.gpuDevices import list_gpus
from apps.helpers.imageUtils import base64_decode_image
from apps.helpers.landingPage import action, render_landing_page
from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.metadataUtils import default_inference_ordinal
from apps.helpers.modelVersions import is_version_preview
from apps.helpers.types import (
    InferenceResult,
    InferenceResultModel,
    Payload,
    StatusResponse,
)

from .manager import Manager


class UnloadIdleResponse(StatusResponse):
    unloaded: List[str] = Field(default_factory=list)


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
    top_k: int = Field(
        default=0,
        ge=0,
        le=32,
        description=(
            "Return this many best classes per detection instead of just the winner. "
            "0 (default) returns only the winner, exactly as before. The runners-up "
            "come back on each detection's `alternates`; check `alternatesAvailable` "
            "on the result, since NMS-free/E2E models cannot produce them at all."
        ),
    )
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


class CropBox(Payload):
    """One box to classify, in the same relative coordinates detections use."""

    x: float
    y: float
    width: float
    height: float


class ClassifyCropsPayload(Payload):
    image_base64: str
    boxes: List[CropBox] = Field(default_factory=list)
    pin_id: Optional[str] = None


class CropPrediction(Payload):
    label_uuid: str
    confidence: float


class CropResult(Payload):
    """Predictions for one input box, highest confidence first."""

    predictions: List[CropPrediction] = Field(default_factory=list)


class ClassifyCropsResponse(StatusResponse):
    # One entry per input box, in the order they were sent — that ordering is how
    # a caller pairs a result back to the detection it came from.
    results: List[CropResult] = Field(default_factory=list)


class DetectUnitPayload(Payload):
    """Run a whole model config — top model plus its sub-models — over one image.

    Takes exactly one of `modelConfigUuid` (a trained hierarchy) or `modelNames` (an
    explicitly assembled stack). Both cascade identically: which models run is a
    selection question, and the second pass must not depend on how they were chosen.
    """

    model_config_uuid: Optional[str] = None
    #: An explicit stack, in priority order. Crop classifiers in it are aimed at their
    #: own sidecar's boundary label, exactly as a hierarchy's sub-models are.
    model_names: List[str] = Field(default_factory=list)
    image_base64: str
    conf_thresh: float = 0.5
    #: Floor a sub-model's verdict must clear to replace the coarse label. 0 keeps the
    #: server's default of always taking the classifier's best guess; a detection that
    #: keeps its parent label ("a cat, not sure which") is a correct answer, so where to
    #: draw this line is the caller's policy.
    min_sub_confidence: float = 0.0
    #: {label_uuid: [ancestor label uuids]}. Optional and additive — the unit's own
    #: artifacts already describe the labels it can emit. A caller holding the real label
    #: tree passes it so detections from FOREIGN models in the same stack are eligible
    #: for refinement too. inference_server never fetches this; it stays db-free.
    label_ancestry: Dict[str, List[str]] = Field(default_factory=dict)
    pin_ids: Dict[str, str] = Field(default_factory=dict)
    #: Runner-up classes per detection, as on detect-batch. Note these are the FIRST
    #: pass's alternates: they are the detector's next-best classes for the box, not the
    #: sub-model's, and a refined detection's winner therefore does not come from this
    #: list. The training collector's cross-model ambiguity scan needs them, and they
    #: describe exactly the decision it is scanning — which detector saw what.
    top_k: Union[int, Dict[str, int]] = 0


class UnitPartModel(Payload):
    """One trained artifact that was found for the config."""

    model_name: str
    role: str
    node_uuid: Optional[str] = None
    label_uuid: Optional[str] = None
    trained_at: Optional[str] = None


class ModelUnitModel(Payload):
    """One model config that has trained artifacts on this server."""

    model_config_uuid: str
    detector_names: List[str] = Field(default_factory=list)
    sub_model_names: List[str] = Field(default_factory=list)
    parts: List[UnitPartModel] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ModelUnitListResponse(StatusResponse):
    units: List[ModelUnitModel] = Field(default_factory=list)


class DetectUnitResponse(StatusResponse):
    # Keyed by detector name, exactly like detect-batch, and with the second pass
    # already applied to the detections inside — a caller applies these the same way it
    # applies any detections, which is the point of the unit.
    results: Dict[str, InferenceResultModel] = Field(default_factory=dict)
    errors: Dict[str, str] = Field(default_factory=dict)
    # What this server actually found and ran. The caller owns the config and diffs
    # this against it to report an untrained part — inference_server cannot know how
    # many parts there *should* be, and guessing would report a half-trained hierarchy
    # as healthy.
    parts: List[UnitPartModel] = Field(default_factory=list)
    detector_names: List[str] = Field(default_factory=list)
    sub_model_names: List[str] = Field(default_factory=list)
    # Artifacts found but not run, and sub-models that failed mid-request.
    notes: List[str] = Field(default_factory=list)


class ModelMetaResponse(StatusResponse):
    model_kind: Optional[str] = None
    label_uuid: Optional[str] = None
    label_uuids: Optional[List[str]] = None
    # crop_classifier only: the boundary label whose detections this sub-model
    # refines. A cascade reads it to decide which boxes to hand this model, so it
    # is served here rather than requiring filesystem access to the sidecar —
    # inference_server can be a remote host (see the split-container topology).
    parent_label_uuid: Optional[str] = None
    # The hierarchy this model belongs to, if it was trained as part of one
    # (TrainHierarchyTask / a model config with sub-models) — every trainer
    # writes both into its sidecar. `role` is "top" for the root detector,
    # "submodel" for a crop classifier refining one of its labels; absent
    # (both None) for a model trained standalone. Lets a caller tell "this
    # detector has a crop-classifier hierarchy attached" from "this is the
    # whole system" without walking the label tree itself.
    model_config_uuid: Optional[str] = None
    role: Optional[str] = None
    device: str = "auto"
    # Set when this model is loaded right now AND its session had to fall back to
    # the CPU because CUDA could not initialize (see
    # apps.helpers.onnxSession.create_session). None when the model is not loaded,
    # which is not the same as "running on the GPU" — a model is only known to
    # have fallen back while it is in memory.
    device_fallback: Optional[str] = None
    # (height, width) the model's ONNX graph declares; None if the model isn't
    # found or its graph declares a dynamic shape. Lets a caller compute
    # send_width = min(source_native_width, model_input_width) instead of
    # guessing at a fixed downscale.
    input_shape: Optional[Tuple[int, int]] = None


class SetModelDevicePayload(Payload):
    device: str


class SetModelDeviceResponse(StatusResponse):
    device_set: bool = False


class GpuDeviceModel(Payload):
    """One GPU this service's host can see.

    Carries both numbering schemes because they are different numbers: `index` is what
    the driver and `nvidia-smi` report (what a human recognizes), `cuda_ordinal` is the
    position within CUDA_VISIBLE_DEVICES that a device setting must store. See
    apps.helpers.gpuDevices.
    """

    index: int
    name: str
    # PCI bus id — the only short field that distinguishes two identical cards, which
    # have the same name and the same memory.
    bus_id: str
    memory_mb: Optional[int] = None
    uuid: str = ""
    # None when this card is not exposed to this container, so it cannot be selected.
    cuda_ordinal: Optional[int] = None
    label: str = ""


class GpuDevicesResponse(StatusResponse):
    gpus: List[GpuDeviceModel] = Field(default_factory=list)
    # What `auto`/`gpu` resolve to here (INFERENCE_GPU), or None when unconfigured —
    # in which case onnxruntime picks, which is ordinal 0 in practice.
    default_cuda_ordinal: Optional[int] = None


class LabelTreeEntry(Payload):
    """One label's shape of the tree, as needed by suppression-group recomputation.

    Callers already hold the full label map (`fetchLabels()` on the frontend), so
    this is that map's rows narrowed to the three fields the computation actually
    reads — see apps/helpers/suppressionGroups.py.
    """

    uuid: str
    parent_uuid: Optional[str] = None
    exclusive_variants: bool = False
    exclusive_variants_iou_threshold: Optional[float] = None


class PreviewSuppressionGroupsPayload(Payload):
    labels: List[LabelTreeEntry] = Field(default_factory=list)


class SuppressionGroupsImpact(Payload):
    model_name: str
    changed_classes: int
    total_classes: int


class PreviewSuppressionGroupsResponse(StatusResponse):
    models: List[SuppressionGroupsImpact] = Field(default_factory=list)


class ApplySuppressionGroupsPayload(Payload):
    labels: List[LabelTreeEntry] = Field(default_factory=list)
    model_names: List[str] = Field(default_factory=list)


class ApplySuppressionGroupsResponse(StatusResponse):
    updated: List[str] = Field(default_factory=list)


class ImportModelUploadPayload(Payload):
    model_name: str
    onnx_base64: str
    overwrite: bool = False


class ImportModelArtifactPayload(Payload):
    artifact_tar_base64: str
    #: How many previous generations of each model in the tar to keep. Sent by the
    #: pusher because this server may be remote and cannot read db_server's
    #: settings; None falls back to DEFAULT_VERSIONS_KEEP.
    versions_keep: Optional[int] = None


class ModelVersionRead(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    version_id: str
    archived_at: Optional[str] = None
    created_at: Optional[str] = None
    bytes: int = 0
    files: List[str] = Field(default_factory=list)


class ListModelVersionsResponse(StatusResponse):
    model_name: Optional[str] = None
    versions: List[ModelVersionRead] = Field(default_factory=list)


class MaterializeModelVersionResponse(StatusResponse):
    #: The temporary model name the archived generation is runnable under.
    preview_name: Optional[str] = None


class RestoreModelVersionResponse(StatusResponse):
    #: The version now live.
    restored: Optional[str] = None
    #: The generation this replaced, archived on the way out so the restore is
    #: itself undoable. None when there was nothing live to archive.
    archived: Optional[str] = None
    files: List[str] = Field(default_factory=list)


class ImportModelArtifactResponse(StatusResponse):
    files: Optional[List[str]] = None


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


class BatchInferencePayload(Payload):
    """One image, several models — the shape a model stack actually has.

    Carries exactly one copy of the image. Sending it once per model meant the
    server decoded the identical JPEG once per model (3.6 ms each) and
    letterboxed it once per model on top, for a byte-identical result each
    time.
    """

    model_names: List[str] = Field(min_length=1, description="Models to run over this one image")
    conf_thresh: float = Field(ge=0.0, le=1.0, description="Confidence threshold; must be in [0, 1]")
    # model name -> that caller's pin id. Per-model because pins are issued per
    # model, and a stack typically holds one for each.
    pin_ids: Dict[str, str] = Field(default_factory=dict)
    top_k: Union[int, Dict[str, int]] = Field(
        default=0,
        description=(
            "See InferencePayload.top_k. A plain int applies to every model in "
            "model_names, exactly as before. A dict gives each model its own ask; "
            "a model name missing from the dict gets 0 (today's default)."
        ),
    )
    image_base64: str

    @field_validator("top_k")
    @classmethod
    def _bound_top_k(cls, value: Union[int, Dict[str, int]]) -> Union[int, Dict[str, int]]:
        # Field(ge=..., le=...) can't reach inside the dict form, so enforce
        # InferencePayload.top_k's [0, 32] range by hand on both shapes.
        asks = value.values() if isinstance(value, dict) else [value]
        for ask in asks:
            if not 0 <= ask <= 32:
                raise ValueError("top_k must be between 0 and 32")
        return value


class BatchInferenceResponse(StatusResponse):
    #: model name -> that model's result. Present only for models that
    #: succeeded.
    results: Dict[str, InferenceResultModel] = Field(default_factory=dict)
    #: model name -> error string, for models that failed. A broken model in a
    #: stack costs its own detections, not every other model's, so these are
    #: reported alongside the successes rather than replacing them.
    errors: Dict[str, str] = Field(default_factory=dict)


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
logger.setLevel(LOG_LEVEL)


class WebApp:
    """Web application for managing inference models and running recognitions"""

    def __init__(self, app_name: str, manager: Manager):
        self._manager: Manager = manager
        self._app_name: str = app_name
        self._app: FastAPI = FastAPI(lifespan=self._lifespan)
        self._templates: Jinja2Templates = Jinja2Templates(directory="/app/templates")
        self._dflt_args: dict[str, str] = {"app_name": self._app_name}

        self._app.mount("/static", StaticFiles(directory="/app/static", check_dir=False), name="static")

        self._register_routes()
        self._register_exception()

    def _register_exception(self):
        @self._app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):

            exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
            # logger.error(msg, *args) does printf-style formatting: passing the
            # Request as the message and exc_str as an arg made logging try
            # str(request) % (exc_str,) and raise "not all arguments converted",
            # so the validation error never reached the log at all.
            logger.error("Validation error on %s %s: %s", request.method, request.url.path, exc_str)
            content = {"status_code": 10422, JsonKeys.MESSAGE: exc_str, "data": None}
            return JSONResponse(content=content, status_code=422)

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        logging.info("Application is starting up...")

        self._manager.start()
        yield

        logging.info("Application is shutting down...")

    def app(self) -> FastAPI:
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

        @self._app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def index(request: Request):
            """Operator landing page — this service is often alone on a GPU host."""
            return HTMLResponse(
                render_landing_page(
                    title="PurrView Inference Server",
                    subtitle="ONNX model serving",
                    status_url="/status",
                    info_url="/api/info",
                    actions=[
                        action(
                            "Unload idle models",
                            "/api/models/unload-idle",
                            description="Free memory held by keep-warm models nothing is using.",
                        ),
                    ],
                    links=[
                        ("API docs", "/docs"),
                        ("Status JSON", "/status"),
                        ("Models JSON", "/api/models"),
                    ],
                )
            )

        @self._app.get(
            "/api/info",
            operation_id="inference_server_info",
            response_class=JSONResponse,
            tags=[ApiTags.MODELS],
        )
        async def inference_server_info():
            """Runtime/model-storage detail for the landing page.

            Split from /status because it stats every model file on disk, and
            /status is polled every few seconds by the notices loop.
            """
            return JSONResponse(content=await asyncio.to_thread(self._manager.get_info))

        @self._app.post(
            "/api/models/unload-idle",
            operation_id="unload_idle_models",
            response_model=UnloadIdleResponse,
            tags=[ApiTags.MODELS],
        )
        async def unload_idle_models():
            """Unload models that only the implicit keep-warm pin is holding."""
            unloaded = self._manager.unload_idle_models()
            return UnloadIdleResponse(status=JsonValues.SUCCESS, unloaded=unloaded)

        @self._app.get(
            "/server-config",
            operation_id="server_config",
            response_class=JSONResponse,
        )
        @self._app.get("/server-config", response_class=JSONResponse)
        async def server_config(request: Request):
            """Get server configuration params."""
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
            """Models available to pick.

            Materialised version previews are filtered out rather than excluded from
            the manager's cache: they have to stay resolvable by name so a comparison
            can actually run one, but they must never be offerable as a watcher's
            model or a training target — they are deleted when the comparison ends.
            """
            try:
                model_list = [m for m in await self._manager.list_models() if not is_version_preview(m)]
                return ModelListResponse(status=JsonValues.SUCCESS, models=model_list)
            except Exception as e:
                return ModelListResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/models/units",
            response_model=ModelUnitListResponse,
            tags=[ApiTags.MODELS],
            operation_id="list_model_units",
        )
        async def list_model_units_api():
            """Every model config this server holds trained artifacts for.

            What makes a hierarchy selectable as one entry in a picker: the SPA lists
            these, joins the names from its own model-configs API, and sends back a
            `modelConfigUuid` to `detect-unit`. Deliberately does NOT name the configs —
            inference_server has no database and a name it invented here would be one
            more thing that can disagree with the config editor.
            """
            try:
                units = await self._manager.list_model_units()
                return ModelUnitListResponse(
                    status=JsonValues.SUCCESS,
                    units=[
                        ModelUnitModel(
                            model_config_uuid=u.model_config_uuid,
                            detector_names=u.detector_names,
                            sub_model_names=[s.model_name for s in u.sub_models],
                            parts=[UnitPartModel(**vars(p)) for p in u.parts],
                            notes=u.problems,
                        )
                        for u in units
                    ],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return ModelUnitListResponse(status=JsonValues.FAILURE, message=str(e))

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
                input_shape = await self._manager.get_model_input_shape(model_name)
                return ModelMetaResponse(
                    status=JsonValues.SUCCESS,
                    model_kind=model_kind,
                    label_uuid=config.get("label_uuid"),
                    label_uuids=config.get("label_uuids"),
                    parent_label_uuid=config.get("parent_label_uuid"),
                    model_config_uuid=config.get("model_config_uuid"),
                    role=config.get("role"),
                    device=config.get("device", "auto"),
                    device_fallback=self._manager.device_fallback_for(model_name),
                    input_shape=input_shape,
                )
            except Exception as e:
                return ModelMetaResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/models/gpus",
            response_model=GpuDevicesResponse,
            tags=[ApiTags.MODELS],
            operation_id="list_inference_gpus",
        )
        async def list_inference_gpus_api() -> GpuDevicesResponse:
            """The GPUs on the host running *this* inference service.

            Served per-service rather than globally: inference and training may be
            different machines with different cards (see the split-container
            topology), so a single shared GPU list would be wrong on one of them.

            Under /api/models rather than a bare /api/gpus because that prefix is what
            image_tagging_server proxies to this service; a new top-level path would
            not reach it. Registered before the `/api/models/{model_name}` routes for
            the same reason detect-batch is — FastAPI matches in declaration order and
            would otherwise read "gpus" as a model name.
            """
            try:
                devices = await asyncio.to_thread(list_gpus)
                return GpuDevicesResponse(
                    status=JsonValues.SUCCESS,
                    gpus=[
                        GpuDeviceModel(
                            index=d.index,
                            name=d.name,
                            bus_id=d.bus_id,
                            memory_mb=d.memory_mb,
                            uuid=d.uuid,
                            cuda_ordinal=d.cuda_ordinal,
                            label=d.label,
                        )
                        for d in devices
                    ],
                    default_cuda_ordinal=default_inference_ordinal(),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return GpuDevicesResponse(status=JsonValues.FAILURE, message=str(e))

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

        def _label_tree_maps(labels: List[LabelTreeEntry]):
            return (
                {l.uuid: l.parent_uuid for l in labels},
                {l.uuid: l.exclusive_variants for l in labels},
                {l.uuid: l.exclusive_variants_iou_threshold for l in labels},
            )

        @self._app.post(
            "/api/models/suppression-groups/preview",
            response_model=PreviewSuppressionGroupsResponse,
            tags=[ApiTags.MODELS],
            operation_id="preview_suppression_groups",
        )
        async def preview_suppression_groups_api(payload: PreviewSuppressionGroupsPayload):
            """Which currently-available object_detection models would get different

            suppression-group data if `payload.labels` (the caller's current label
            tree — no DB access here, see the manager method) were applied right
            now. Read-only. Powers the label editor's "N trained models affected"
            notice after an exclusive_variants edit.
            """
            try:
                parent_of, exclusive_of, threshold_of = _label_tree_maps(payload.labels)
                impacted = await self._manager.preview_suppression_groups(parent_of, exclusive_of, threshold_of)
                return PreviewSuppressionGroupsResponse(
                    status=JsonValues.SUCCESS,
                    models=[SuppressionGroupsImpact(**m) for m in impacted],
                )
            except Exception as e:
                return PreviewSuppressionGroupsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/models/suppression-groups/apply",
            response_model=ApplySuppressionGroupsResponse,
            tags=[ApiTags.MODELS],
            operation_id="apply_suppression_groups",
        )
        async def apply_suppression_groups_api(payload: ApplySuppressionGroupsPayload):
            """Write freshly-computed suppression groups into the named models' sidecars.

            The model's next pin/recognize call picks this up automatically
            (`Manager._reload_if_stale` also checks the sidecar's mtime) — no
            restart, no retrain.
            """
            try:
                parent_of, exclusive_of, threshold_of = _label_tree_maps(payload.labels)
                updated = await self._manager.apply_suppression_groups(
                    parent_of, exclusive_of, threshold_of, payload.model_names
                )
                return ApplySuppressionGroupsResponse(status=JsonValues.SUCCESS, updated=updated)
            except Exception as e:
                return ApplySuppressionGroupsResponse(status=JsonValues.FAILURE, message=str(e))

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
            "/api/models/artifact/upload",
            response_model=ImportModelArtifactResponse,
            tags=[ApiTags.MODELS],
            operation_id="import_model_artifact",
        )
        async def import_model_artifact_api(payload: ImportModelArtifactPayload) -> ImportModelArtifactResponse:
            """Extract a training-job artifact tar (.onnx + optional .onnx.data + .json
            sidecar) straight into the models dir. Lets tasks_server push a freshly
            trained model here over HTTP instead of requiring a shared models directory."""
            try:
                tar_bytes = base64.b64decode(payload.artifact_tar_base64, validate=True)
            except Exception as e:
                return ImportModelArtifactResponse(
                    status=JsonValues.FAILURE, message=f"Invalid artifact_tar_base64: {e}"
                )

            try:
                result = await self._manager.import_artifact_tar(tar_bytes, versions_keep=payload.versions_keep)
                return ImportModelArtifactResponse(status=JsonValues.SUCCESS, files=result["files"])
            except Exception as e:
                return ImportModelArtifactResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/models/{model_name}/versions",
            response_model=ListModelVersionsResponse,
            tags=[ApiTags.MODELS],
            operation_id="list_model_versions",
        )
        async def list_model_versions_api(model_name: str) -> ListModelVersionsResponse:
            """Previous generations of a model, newest first.

            An empty list is the normal answer for a model that has only ever been
            trained once, or was imported rather than trained — not an error.
            """
            try:
                versions = await self._manager.list_model_versions(model_name)
                return ListModelVersionsResponse(
                    status=JsonValues.SUCCESS,
                    model_name=model_name,
                    versions=[ModelVersionRead(**v) for v in versions],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListModelVersionsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/models/{model_name}/versions/{version_id}/restore",
            response_model=RestoreModelVersionResponse,
            tags=[ApiTags.MODELS],
            operation_id="restore_model_version",
        )
        async def restore_model_version_api(model_name: str, version_id: str) -> RestoreModelVersionResponse:
            """Roll a model back to one of its archived generations.

            The generation being replaced is archived on the way out, so this is
            reversible: restoring the wrong one is undone by restoring the entry
            this call creates.
            """
            try:
                result = await self._manager.restore_model_version(model_name, version_id)
                return RestoreModelVersionResponse(
                    status=JsonValues.SUCCESS,
                    restored=result["restored"],
                    archived=result["archived"],
                    files=result["files"],
                )
            except FileNotFoundError as e:
                return RestoreModelVersionResponse(status=JsonValues.FAILURE, message=str(e))
            except Exception as e:
                logger.error(e, exc_info=True)
                return RestoreModelVersionResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/models/{model_name}/versions/{version_id}/preview",
            response_model=MaterializeModelVersionResponse,
            tags=[ApiTags.MODELS],
            operation_id="materialize_model_version",
        )
        async def materialize_model_version_api(model_name: str, version_id: str) -> MaterializeModelVersionResponse:
            """Materialise an archived generation as a temporary, runnable model.

            For comparing a version against the live model: the returned name can be
            run by anything that takes a model name. Delete it with
            ``DELETE /api/models/{name}/preview`` when the comparison is done — it is
            a full copy of the weights.
            """
            try:
                name = await self._manager.materialize_model_version(model_name, version_id)
                return MaterializeModelVersionResponse(status=JsonValues.SUCCESS, preview_name=name)
            except FileNotFoundError as e:
                return MaterializeModelVersionResponse(status=JsonValues.FAILURE, message=str(e))
            except Exception as e:
                logger.error(e, exc_info=True)
                return MaterializeModelVersionResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.delete(
            "/api/models/{name}/preview",
            response_model=StatusResponse,
            tags=[ApiTags.MODELS],
            operation_id="delete_model_preview",
        )
        async def delete_model_preview_api(name: str) -> StatusResponse:
            """Delete a materialised version.

            A separate route from ``DELETE /api/models/{name}`` on purpose: this one
            refuses anything that is not a preview, so a cleanup path that loses track
            of which name it made cannot delete a real model.
            """
            try:
                removed = await self._manager.delete_model_preview(name)
                if not removed:
                    return StatusResponse(status=JsonValues.FAILURE, message=f"{name!r} is not a version preview")
                return StatusResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

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
                # crop_classifier is served here too, but the caller must have already
                # cut the region out (see CropClassifierInferenceProvider) — this
                # endpoint classifies whatever image it is given, whole frame or crop.
                if model_kind not in ("image_classifier", "multi_label_classifier", "crop_classifier"):
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

        @self._app.post(
            "/api/models/{model_name}/classify-crops",
            response_model=ClassifyCropsResponse,
            tags=[ApiTags.INFERENCE],
            operation_id="classify_crops",
        )
        async def classify_crops_api(model_name: str, payload: ClassifyCropsPayload) -> ClassifyCropsResponse:
            """Second pass of a hierarchical config: classify N boxes of one frame.

            The image is decoded once here and all N crops go through one batched
            session run, which is the point of doing this server-side rather than
            having the caller cut crops and post each one back through /classify.
            """
            try:
                image_bgr: np.ndarray = base64_decode_image(payload.image_base64)
                scored = await self._manager.classify_crops(
                    model_name,
                    image_bgr,
                    [b.model_dump() for b in payload.boxes],
                    pin_id=payload.pin_id,
                )
                return ClassifyCropsResponse(
                    status=JsonValues.SUCCESS,
                    results=[
                        CropResult(
                            predictions=[
                                CropPrediction(label_uuid=label_uuid, confidence=confidence)
                                for label_uuid, confidence in per_box
                            ]
                        )
                        for per_box in scored
                    ],
                )
            except ValueError as e:
                # Wrong model kind / not available — a caller error, not a crash.
                return ClassifyCropsResponse(status=JsonValues.FAILURE, message=str(e))
            except Exception as e:
                logger.error(e, exc_info=True)
                return ClassifyCropsResponse(status=JsonValues.FAILURE, message=str(e))

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
            "/api/models/detect-batch",
            tags=[ApiTags.INFERENCE],
            operation_id="detect_batch",
            response_model=BatchInferenceResponse,
        )
        async def inference_batch_api(payload: BatchInferencePayload) -> BatchInferenceResponse:
            """Run several models over one image, decoding and letterboxing it once.

            Registered BEFORE /api/models/{model_name}/detect deliberately:
            FastAPI matches routes in declaration order, and "detect-batch"
            would otherwise be captured by {model_name} as a model called
            "detect-batch".
            """
            try:
                image_array: np.ndarray = base64_decode_image(payload.image_base64)

                outcomes = await self._manager.recognize_many(
                    model_names=payload.model_names,
                    image=image_array,
                    conf_thresh=payload.conf_thresh,
                    pin_ids=payload.pin_ids,
                    top_k=payload.top_k,
                )

                results: dict = {}
                errors: dict = {}
                for name, outcome in outcomes.items():
                    if isinstance(outcome, str):
                        errors[name] = outcome
                        continue
                    # The batch path never returns images: it exists for a model
                    # stack running over a live frame, where N copies of the
                    # source image in one response is exactly the waste being
                    # removed here.
                    outcome.source_image = None
                    results[name] = InferenceResultModel.model_validate(outcome)
                return BatchInferenceResponse(results=results, errors=errors)
            except Exception as e:
                logger.error(e, exc_info=True)
                return BatchInferenceResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/models/detect-unit",
            tags=[ApiTags.INFERENCE],
            operation_id="detect_unit",
            response_model=DetectUnitResponse,
        )
        async def detect_unit_api(payload: DetectUnitPayload) -> DetectUnitResponse:
            """Run one model config as a single model: detect, then refine, in one call.

            Registered BEFORE /api/models/{model_name}/detect for the same reason
            detect-batch is: FastAPI matches in declaration order, and "detect-unit"
            would otherwise be captured as a model named "detect-unit".

            This is the only place the cascade runs. Callers — the SPA's inference panel,
            the watcher, the review preview — hand over a config uuid and get back
            detections that are already as specific as the hierarchy can make them, so
            none of them reimplements the second pass (and none of them can drift).
            """
            try:
                image_array: np.ndarray = base64_decode_image(payload.image_base64)
                results, unit, notes = await self._manager.detect_unit(
                    model_config_uuid=payload.model_config_uuid or None,
                    model_names=payload.model_names or None,
                    image=image_array,
                    conf_thresh=payload.conf_thresh,
                    label_ancestry={k: set(v) for k, v in payload.label_ancestry.items()},
                    min_sub_confidence=payload.min_sub_confidence,
                    pin_ids=payload.pin_ids,
                    top_k=payload.top_k,
                )

                ok: dict = {}
                errors: dict = {}
                for name, outcome in results.items():
                    if isinstance(outcome, str):
                        errors[name] = outcome
                        continue
                    # Same as detect-batch: never echo the source image back per model.
                    outcome.source_image = None
                    ok[name] = InferenceResultModel.model_validate(outcome)

                return DetectUnitResponse(
                    results=ok,
                    errors=errors,
                    parts=[UnitPartModel(**vars(p)) for p in unit.parts],
                    detector_names=unit.detector_names,
                    sub_model_names=[s.model_name for s in unit.sub_models],
                    notes=notes,
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return DetectUnitResponse(status=JsonValues.FAILURE, message=str(e))

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
                    top_k=payload.top_k,
                )

                if not payload.return_source_img:
                    inference_result.source_image = None

                model = InferenceResultModel.model_validate(inference_result)
                return InferenceResponse(result=model)
            except Exception as e:
                logger.error(e, exc_info=True)
                return InferenceResponse(status=JsonValues.FAILURE, message=str(e))
