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

from .manager import Manager

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class WebApp:
    """Web application for managing inference models and running recognitions"""

    SUCCESS_KEY = "success"
    ERROR_KEY = "error"
    RESULT_KEY = "results"

    MODELS_API_TAG_NAME = "models"
    INFERENCE_API_TAG_NAME = "inference"

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
                {"label": "List Models", "action": "/list-models"},
                {"label": "Get Model Labels", "action": "/get-model-labels-form"},
                {"label": "Pin Model", "action": "/pin-model-form"},
                {"label": "Recognize", "action": "/recognize-form"},
            ]
            return self._templates.TemplateResponse(
                "dynamic_index.html",
                {
                    "request": request,
                    "app_name": "Inference Server",
                    "buttons": buttons,
                },
            )

        @self._app.get(
            "/api/models/list",
            response_class=JSONResponse,
            tags=[WebApp.MODELS_API_TAG_NAME],
            operation_id="list_models_api",
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
                response_data = {"status": "success", "models": model_list}
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={"status": "failure", "message": str(e)},
                    status_code=500,
                )

        @self._app.get("/list-models", response_class=HTMLResponse)
        async def list_models(request: Request) -> HTMLResponse:
            """
            HTML endpoint to render a list of models.

            Args:
                request (Request): The FastAPI request object.

            Returns:
                HTMLResponse: Rendered HTML response containing the list of models.
            """
            try:
                # Call the API function to get the JSON response
                response: JSONResponse = await list_models_api(request)
                response_data = json.loads(response.body.decode("utf-8"))

                # Render the HTML response using the dynamic template
                return self._templates.TemplateResponse(
                    "dynamic_response.html",
                    {
                        "request": request,
                        "title": "Model List",
                        "response_data": response_data,
                    },
                )
            except Exception as e:
                return self._error_response(request, f"Internal server error: {str(e)}")


        class GetModelLabelsRequest(BaseModel):
            model_name: str

        @self._app.post(
            "/api/models/labels/get",
            response_class=JSONResponse,
            tags=[WebApp.MODELS_API_TAG_NAME],
            operation_id="get_model_labels_api",
        )
        async def get_model_labels_api(
            payload: GetModelLabelsRequest
        ) -> JSONResponse:
            """
            API endpoint to list labels for a given model.

            Returns:
                JSONResponse: A dictionary mapping label names to label IDs.
            """
            try:
                label_map: dict[str, str] = await self._manager.get_model_labels(payload.model_name)
            except Exception as e:
                logger.exception(e)
                return JSONResponse(content={"status": "failure", "message": str(e)})

            return JSONResponse(
                content={
                    "status": "success",
                    "labels": label_map,
                },
                status_code=200,
            )

        class AssociateLabelWithModelClassRequest(BaseModel):
            model_name: str
            model_class: str
            label_uuid: str

        @self._app.post(
            "/api/models/labels/associate",
            tags=[WebApp.MODELS_API_TAG_NAME],
            operation_id="associate_label_with_model_class",
            response_class=JSONResponse,
        )
        async def associate_label_with_model_class_api(req: AssociateLabelWithModelClassRequest) -> JSONResponse:
            """
            API endpoint to associate a model's class with a label

            Returns:
                JSONResponse: A JSON response containing the list of labels for the model
            """
            try:
                ret = await self._manager.set_model_label_uuid(
                    model_name=req.model_name,
                    model_class=req.model_class,
                    label_uuid=req.label_uuid
                )
                response_data = {"status": "success", "label_set": ret}
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={"status": "failure", "message": str(e)},
                    status_code=500,
                )


        @self._app.post("/get-model-labels", response_class=HTMLResponse, include_in_schema=False)
        async def get_model_labels(
            request: Request,
            model_name: str = Form(...),
        ) -> HTMLResponse:
            """
            HTML endpoint to render the labels for a selected model.

            Args:
                request (Request): The FastAPI request object.
                model_name (str): Name of the model to query labels for.

            Returns:
                HTMLResponse: Rendered HTML showing label mappings.
            """
            try:
                req: GetModelLabelsRequest = GetModelLabelsRequest(model_name=model_name)
                response: JSONResponse = await get_model_labels_api(req)
                response_data = json.loads(response.body.decode("utf-8"))

                return self._templates.TemplateResponse(
                    "dynamic_response.html",
                    {
                        "request": request,
                        "title": f"Labels for model '{model_name}'",
                        "response_data": response_data,
                    },
                )
            except Exception as e:
                return self._error_response(request, f"Internal server error: {str(e)}")

        @self._app.get("/get-model-labels-form", response_class=HTMLResponse)
        async def get_model_labels_form(request: Request) -> HTMLResponse:
            """
            Render a form to input model name and list its labels.

            Args:
                request (Request): The FastAPI request object.

            Returns:
                HTMLResponse: Rendered form for listing model labels.
            """
            model_list = await self._manager.list_models()
            fields = {
                "model_name": {
                    "label": "Model Name",
                    "type": "radio",
                    "options": model_list,
                },
            }

            return self._templates.TemplateResponse(
                "dynamic_form.html",
                {
                    "request": request,
                    "title": "List Model Labels",
                    "action_url": "/get-model-labels",
                    "fields": fields,
                    "submit_label": "Show Labels",
                },
            )

        @self._app.post(
            "/api/models/pin",
            response_class=JSONResponse,
            tags=[WebApp.MODELS_API_TAG_NAME],
            operation_id="pin_model_api",
        )
        async def pin_model_api(
            request: Request, model_name: str = Body(...), duration: float = Body(...)
        ) -> JSONResponse:
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
                # Convert duration to float
                pin_id = await self._manager.pin_model(model_name, duration)
            except Exception as e:
                logger.exception(e)
                return JSONResponse(content={"status": "failure", "message": str(e)})

            # Return a JSON response with the model ID
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "Model pinned successfully.",
                    "pin_id": pin_id,
                },
                status_code=200,
            )

        @self._app.post("/pin-model", response_class=HTMLResponse, include_in_schema=False)
        async def pin_model(request: Request, model_name: str = Form(...), duration: float = Form(...)) -> HTMLResponse:
            """
            HTML endpoint to pin a model in memory.

            Args:
                request (Request): The FastAPI request object.
                model_name (str): Name of the model to pin.
                duration (str): Duration for which the model should be pinned.

            Returns:
                HTMLResponse: Rendered HTML response containing the pin ID or an error message.
            """
            try:
                # Call the API function to pin the model
                response: JSONResponse = await pin_model_api(request=request, model_name=model_name, duration=duration)
                response_data = json.loads(response.body.decode("utf-8"))

                # Render the HTML response using the dynamic template
                return self._templates.TemplateResponse(
                    "dynamic_response.html",
                    {
                        "request": request,
                        "title": "Pin Model Results",
                        "response_data": response_data,
                    },
                )
            except Exception as e:
                return self._error_response(request, f"Internal server error: {str(e)}")

        @self._app.get("/pin-model-form", response_class=HTMLResponse)
        async def pin_model_form(request: Request) -> HTMLResponse:
            """
            Render the form for pinning a model in memory.

            Args:
                request (Request): The FastAPI request object.

            Returns:
                HTMLResponse: Rendered form for pinning a model.
            """
            model_list = await self._manager.list_models()
            fields = {
                "model_name": {
                    "label": "Model Name",
                    "type": "radio",
                    "options": model_list,
                },
                "duration": {"type": "text", "label": "Duration (min)"},
            }

            return self._templates.TemplateResponse(
                "dynamic_form.html",
                {
                    "request": request,
                    "title": "Pin Model",
                    "action_url": "/pin-model",
                    "fields": fields,
                    "submit_label": "Submit",
                },
            )

        async def recognize_core(
            manager: Manager,
            model_name: str,
            image: np.ndarray,
            conf_thresh: float,
            return_annotated: bool,
            pin_id: Optional[str] = None
        ) -> dict:
            inference_result: InferenceResult = await manager.recognize(
                model_name=model_name,
                image=image,
                conf_thresh=conf_thresh,
                return_annotated=return_annotated,
                pind_id=pin_id
            )

            return inference_result.serialize(include_annotated=return_annotated)

        class RecognizeRequest(BaseModel):
            model_name: str
            conf_thresh: float
            return_annotated: bool = False
            pin_id: Optional[str] = None
            image_base64: str  # base64 encoded image string

        @self._app.post(
            "/api/recognize-json",
            tags=[WebApp.INFERENCE_API_TAG_NAME],
            operation_id="recognize_json",
            response_class=JSONResponse,
        )
        async def recognize_json_api(payload: RecognizeRequest):
            try:
                np_bytes = base64.b64decode(payload.image_base64)
                np_image = np.frombuffer(np_bytes, np.uint8)
                image_array = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

                return JSONResponse(
                    content=await recognize_core(
                        self._manager,
                        model_name=payload.model_name,
                        image=image_array,
                        conf_thresh=payload.conf_thresh,
                        return_annotated=payload.return_annotated,
                        pin_id=payload.pin_id
                    )
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

        @self._app.post(
            "/api/recognize",
            response_class=JSONResponse,
            tags=[WebApp.INFERENCE_API_TAG_NAME],
            operation_id="recognize_formdata",
        )
        async def recognize_api(
            model_name: str = Form(..., description="Name of the model to use for recognition"),
            conf_thresh: float = Form(..., description="Confidence threshold for detections"),
            return_annotated: bool = Form(False, description="Whether to return the annotated image"),
            pin_id: str = Form(None, description="Optional pin id to refresh timeout for"),
            image: UploadFile = File(..., description="Image file to process"),
            request: Request = None,
        ) -> JSONResponse:
            """
            API endpoint to recognize objects in an image using a specified model.

            Args:
                model_name (str): Name of the model to use for recognition.
                conf_thresh (float): Confidence threshold for detections.
                return_annotated (bool): Whether to return the annotated image.
                image (UploadFile): The image file to process.

            Returns:
                JSONResponse: A JSON response containing detected bounding boxes and optionally the annotated image.
            """
            try:
                file_bytes = await image.read()
                np_image = np.frombuffer(file_bytes, np.uint8)
                image_array = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

                detections, annotated_image = await self._manager.recognize(
                    model_name=model_name,
                    image=image_array,
                    conf_thresh=conf_thresh,
                    pin_id=pin_id,
                    return_annotated=return_annotated,
                )

                response = {"detections": detections}
                if return_annotated and annotated_image is not None:
                    _, buffer = cv2.imencode(".png", annotated_image)
                    image_base64 = base64.b64encode(buffer).decode("utf-8")
                    response["annotated_image"] = image_base64

                return JSONResponse(content=response)

            except FileNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

        @self._app.post("/recognize", response_class=HTMLResponse, include_in_schema=False)
        async def recognize_html(
            model_name: str = Form(...),
            conf_thresh: float = Form(...),
            return_annotated: bool = Form(False),
            pin_id: Optional[str] = Form(None, description="Optional pin id to refresh timeout for"),
            image: UploadFile = File(...),
            request: Request = None,
        ):
            try:
                file_bytes = await image.read()
                np_image = np.frombuffer(file_bytes, np.uint8)
                image_array = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

                response_data = await recognize_core(
                    self._manager,
                    model_name=model_name,
                    image=image_array,
                    conf_thresh=conf_thresh,
                    pin_id=pin_id,
                    return_annotated=return_annotated,
                )

                if "annotated_image" in response_data:
                    response_data["annotated_image"] = f"data:image/jpeg;base64,{response_data['annotated_image']}"

                return self._templates.TemplateResponse(
                    "dynamic_response.html",
                    {
                        "request": request,
                        "title": "Recognition Results",
                        "response_data": response_data,
                    },
                )
            except Exception as e:
                return self._error_response(request, f"Internal server error: {str(e)}")

        @self._app.get("/recognize-form", response_class=HTMLResponse)
        async def recognize_form(request: Request) -> HTMLResponse:
            model_list = await self._manager.list_models()
            fields = {
                "model_name": {
                    "label": "Model Name",
                    "type": "radio",
                    "options": model_list,
                },
                "conf_thresh": {"label": "Confidence Threshold", "type": "text"},
                "pin_id": {"label": "Optional Pin ID To refresh", "type": "text", "optional": True},
                "image": {"label": "Upload Image", "type": "file"},
                "return_annotated": {
                    "label": "Return Annotated Image",
                    "type": "checkbox",
                    "options": ["Yes"],
                },
            }
            return self._templates.TemplateResponse(
                "dynamic_form.html",
                {
                    "request": request,
                    "title": "Recognize Image",
                    "action_url": "/recognize",
                    "fields": fields,
                    "submit_label": "Submit",
                },
            )
