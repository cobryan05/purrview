import base64
import json
import logging
import os
import sys

from fastapi import Body, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .manager import Manager

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class WebApp:
    SUCCESS_KEY = "success"
    ERROR_KEY = "error"
    RESULT_KEY = "results"

    MODELS_API_TAG_NAME = "models"
    INFERENCE_API_TAG_NAME = "inference"

    def __init__(self, app_name: str, manager: Manager):
        """Initialize the WebApp with the application name"""
        self._app_name: str = app_name
        self._app: FastAPI = FastAPI()
        self._templates: Jinja2Templates = Jinja2Templates(directory=os.path.join(SCRIPT_DIR, "templates"))
        self._manager: Manager = manager

        # Mount static files
        self._app.mount("/static", StaticFiles(directory="static"), name="static")
        self._app.mount("/app-static", StaticFiles(directory=os.path.join(SCRIPT_DIR, "static")), name="app-static")
        self._app.mount("/images", StaticFiles(directory="static/images"), name="images")
        self._app.mount("/metadata", StaticFiles(directory="static/images/metadata"), name="metadata")

        # Register routes
        self._register_routes()

    def app(self) -> FastAPI:
        """Return the FastAPI application instance"""
        return self._app

    def _register_routes(self):
        """Register all routes for the application"""

        class AnnotationRequest(BaseModel):
            image: str
            annotations: list

        @self._app.get("/", response_class=HTMLResponse)
        async def get_home(request: Request):
            """Render the home page"""
            return self._templates.TemplateResponse("index.html", {"request": request})

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

        @self._app.post("/save-annotations", response_class=JSONResponse)
        async def save_annotations(request: AnnotationRequest) -> JSONResponse:
            """Save annotations with associated image name"""
            try:
                logger.info(f"Received annotations for image {request.image}")
                filename = f"{request.image}.json"
                filepath = os.path.join("static/images/metadata", filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, "w") as f:
                    json.dump({"annotations": request.annotations}, f, indent=2)
                return JSONResponse(content={"status": "ok"})
            except Exception as e:
                logger.exception("Failed to save annotations")
                return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

        @self._app.post("/api/recognize", response_class=JSONResponse, tags=[WebApp.INFERENCE_API_TAG_NAME])
        async def recognize_api(
            model_name: str = Form(..., description="Name of the model to use for recognition"),
            conf_thresh: float = Form(..., description="Confidence threshold for detections"),
            return_annotated: bool = Form(False, description="Whether to return the annotated image"),
            image: UploadFile = File(..., description="Image file to process"),
        ):
            try:
                image_bytes = await image.read()

                import cv2
                import numpy as np

                nparr = np.frombuffer(image_bytes, np.uint8)
                image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                results, annotated_image = await self._manager.recognize(
                    model_name=model_name,
                    image=image_array,
                    conf_thresh=conf_thresh,
                    return_annotated=return_annotated,
                )

                response_content = {
                    "status": "success",
                    "results": results,
                }

                if return_annotated and annotated_image is not None:
                    # Encode annotated_image as base64 JPEG string
                    success, buffer = cv2.imencode(".jpg", annotated_image)
                    if success:
                        annotated_base64 = base64.b64encode(buffer).decode("utf-8")
                        response_content["annotated_image"] = annotated_base64

                return JSONResponse(content=response_content)

            except Exception as e:
                logger.exception("Recognition failed")
                return JSONResponse(
                    status_code=500,
                    content={"status": "failure", "message": str(e)},
                )
