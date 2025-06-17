"""Web API for Inference Server"""

import base64
import json
import logging
import sys
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
        """
        Render an error response using a text template.

        Args:
            request (Request): The FastAPI request object.
            message (str): The error message to display.

        Returns:
            TemplateResponse: Rendered error response.
        """
        return self._templates.TemplateResponse(
            "display_text.html",
            {
                "request": request,
                "text": message,
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

        @self._app.get("/api/list-models", response_class=JSONResponse, tags=[WebApp.MODELS_API_TAG_NAME])
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

        @self._app.post("/api/pin-model", response_class=JSONResponse, tags=[WebApp.MODELS_API_TAG_NAME])
        async def pin_model_api(
            request: Request, model_name: str = Body(...), duration: float = Body(...)
        ) -> JSONResponse:
            """
            API endpoint to pin a model in memory.

            Args:
                request (Request): The FastAPI request object.
                model_name (str): Name of the model to pin.
                duration (str): Duration for which the model should be pinned.

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

        @self._app.post("/api/recognize", response_class=JSONResponse, tags=[WebApp.INFERENCE_API_TAG_NAME])
        async def recognize_api(
            model_name: str = Body(..., description="Name of the model to use for recognition"),
            conf_thresh: float = Body(..., description="Confidence threshold for detections"),
            return_annotated: bool = Body(False, description="Whether to return the annotated image"),
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
                # Read the uploaded image
                file_bytes = await image.read()
                np_image = np.frombuffer(file_bytes, np.uint8)
                image_array = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

                # Call the recognize function
                detections, annotated_image = await self._manager.recognize(
                    model_name=model_name,
                    image=image_array,
                    conf_thresh=conf_thresh,
                    return_annotated=return_annotated,
                )

                # Prepare the response
                response = {"detections": detections}
                if return_annotated and annotated_image is not None:
                    _, buffer = cv2.imencode(".jpg", annotated_image)
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
            model_name: str = Form(..., description="Name of the model to use for recognition"),
            conf_thresh: float = Form(..., description="Confidence threshold for detections"),
            return_annotated: bool = Form(False, description="Whether to return the annotated image"),
            image: UploadFile = File(..., description="Image file to process"),
            request: Request = None,
        ) -> HTMLResponse:
            """
            HTML endpoint to recognize objects in an image using a specified model.

            Returns:
                HTMLResponse: An HTML response containing detected bounding boxes and optionally the annotated image.
            """
            try:
                # Call the recognize_api endpoint
                response: JSONResponse = await recognize_api(
                    model_name=model_name,
                    conf_thresh=conf_thresh,
                    return_annotated=return_annotated,
                    image=image,
                    request=request,
                )

                # Extract data from the JSON response
                response_data = json.loads(response.body.decode("utf-8"))

                # Add the correct prefix for the Base64-encoded image if it exists
                if "annotated_image" in response_data:
                    response_data["annotated_image"] = f"data:image/jpeg;base64,{response_data['annotated_image']}"

                # Render the HTML response using the dynamic template
                return self._templates.TemplateResponse(
                    "dynamic_response.html",
                    {
                        "request": request,
                        "title": "Recognition Results",
                        "response_data": response_data,
                    },
                )
            except HTTPException as e:
                return self._error_response(request, f"Error: {e.detail}")
            except Exception as e:
                return self._error_response(request, f"Internal server error: {str(e)}")

        @self._app.get("/recognize-form", response_class=HTMLResponse)
        async def recognize_form(request: Request) -> HTMLResponse:
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
                "conf_thresh": {"label": "Confidence Threshold", "type": "text"},
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
