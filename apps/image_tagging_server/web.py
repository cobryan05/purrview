import base64
import json
import logging
import os
import sys
import mimetypes

from dataclasses import asdict
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Body, FastAPI, File, Form, Query, Request, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .manager import Manager, ImageMetadata, BoundingBoxMetadata

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class WebApp:
    SUCCESS_KEY = "success"
    ERROR_KEY = "error"
    RESULT_KEY = "results"

    LABELS_API_TAG_NAME = "labels"
    MODELS_API_TAG_NAME = "models"
    INFERENCE_API_TAG_NAME = "inference"
    IMAGES_API_TAG_NAME = "images"

    def __init__(self, app_name: str, manager: Manager):
        """Initialize the WebApp with the application name"""
        self._app_name: str = app_name
        self._app: FastAPI = FastAPI(lifespan=self._lifespan)
        self._templates: Jinja2Templates = Jinja2Templates(directory=os.path.join(SCRIPT_DIR, "templates"))
        self._manager: Manager = manager

        # Mount static files
        self._app.mount("/static", StaticFiles(directory="static"), name="static")
        self._app.mount("/app-static", StaticFiles(directory=os.path.join(SCRIPT_DIR, "static")), name="app-static")

        # Register routes
        self._register_routes()

    def app(self) -> FastAPI:
        """Return the FastAPI application instance"""
        return self._app

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

    def _register_routes(self):
        """Register all routes for the application"""

        class AnnotationRequest(BaseModel):
            image: str
            annotations: list

        @self._app.get("/", response_class=HTMLResponse)
        async def get_home(request: Request):
            """Render the home page"""
            return self._templates.TemplateResponse("index.html", {"request": request})

        @self._app.get("/api/images/list", response_class=JSONResponse, tags=[WebApp.MODELS_API_TAG_NAME])
        async def list_files(
            request: Request,
            rel_path: str = Query("/", alias="path"),
            pattern: str = Query("*", alias="pattern"),
            recursive: bool = Query(False, alias="recursive"),
        ) -> JSONResponse:
            """
            API endpoint to return a list of files.

            Query Parameters:
                rel_path (str): Relative path under the root directory (default: "/").
                glob (str): Glob pattern to filter files (default: "*").
                recursive (bool): Whether to search directories recursively (default: False).

            Returns:
                JSONResponse: A JSON response containing the list of files.
            """
            try:
                file_list: List[Manager.FileEntry] = await self._manager.list_files(
                    rel_path=rel_path, patterns=pattern, recursive=recursive
                )
                response_data = {"status": "success", "files": [entry.__dict__ for entry in file_list]}
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={"status": "failure", "message": str(e)},
                    status_code=500,
                )

        @self._app.get("/api/images/get", tags=[WebApp.MODELS_API_TAG_NAME])
        async def get_file(
            rel_path: str = Query(..., alias="path", description="Relative path to the file under root"),
            raw: bool = Query(False, description="If true, return as raw file response instead of base64 JSON"),
        ):
            """
            Retrieve a file either as base64 JSON (default) or as a browser-friendly image file (raw=true).

            Args:
                rel_path (str): Relative file path.
                raw (bool): If true, return as image; otherwise return base64 JSON.

            Returns:
                JSONResponse or Response
            """
            try:
                result = await self._manager.open_file(rel_path)
                if not result:
                    if raw:
                        raise HTTPException(status_code=404, detail="File not found")
                    else:
                        return JSONResponse({"status": "failure", "message": f"File {rel_path} not found"})

                file_obj, filename = result
                content = await file_obj.read()
                await file_obj.close()

                mime_type, _ = mimetypes.guess_type(filename)
                mime_type = mime_type or "application/octet-stream"

                if raw:
                    return Response(
                        content=content,
                        media_type=mime_type,
                        headers={"Content-Disposition": f'inline; filename="{filename}"'},
                    )

                metadata: ImageMetadata = await self._manager.get_image_metadata(rel_path)
                encoded = base64.b64encode(content).decode("utf-8")
                return JSONResponse(
                    {
                        "status": "success",
                        "filename": filename,
                        "mime_type": mime_type,
                        "content": encoded,
                        **asdict(metadata),
                    }
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.exception("Error retrieving file")
                return JSONResponse(
                    status_code=500,
                    content={"status": "failure", "message": str(e)},
                )

        @self._app.get("/api/models/list", response_class=JSONResponse, tags=[WebApp.IMAGES_API_TAG_NAME])
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

        class AddLabelRequest(BaseModel):
            name: str
            color: str

        class DeleteLabelRequest(BaseModel):
            label_id: int

        class BoundingBoxInput(BaseModel):
            id: Optional[int]
            x: float
            y: float
            width: float
            height: float
            labels: Optional[List[int]] = []  # List of label IDs
            extra: Optional[dict] = {}

        class UpdateMetadataRequest(BaseModel):
            image_path: str
            boxes: List[BoundingBoxInput]
            extra: Optional[dict] = {}

        @self._app.get("/api/images/metadata/get", response_class=JSONResponse, tags=[WebApp.IMAGES_API_TAG_NAME])
        async def get_metadata(rel_path: str = Query("/", alias="path")):
            metadata: Optional[ImageMetadata] = await self._manager.get_image_metadata(rel_path)
            if metadata is None:
                raise HTTPException(status_code=404, detail="Image not found")
            # convert to dict for JSONResponse
            return JSONResponse(content=asdict(metadata))

        @self._app.post("/api/images/metadata/update", response_class=JSONResponse, tags=[WebApp.IMAGES_API_TAG_NAME])
        async def update_metadata(req: UpdateMetadataRequest) -> JSONResponse:
            image_path = req.image_path
            metadata = await self._manager.get_image_metadata(image_path)
            if metadata is None:
                raise HTTPException(status_code=404, detail="Image not found")

            # Build new bounding box list from input
            new_boxes = []
            for b in req.boxes:
                # Resolve label info for each label ID
                labels = []
                for label_id in b.labels or []:
                    # Here, you might want to fetch label info by ID from DB or cache
                    # Let's assume manager._db_client has get_label_by_id
                    label_row = await self._manager._db_client.get_label_by_id(label_id)
                    if label_row:
                        labels.append(Label(id=label_row["id"], name=label_row["name"], color=label_row["color"]))
                new_boxes.append(
                    BoundingBoxMetadata(
                        id=b.id,
                        x=b.x,
                        y=b.y,
                        width=b.width,
                        height=b.height,
                        labels=labels,
                        extra=b.extra or {},
                    )
                )

            metadata.boxes = new_boxes
            metadata.extra = req.extra or {}

            await self._manager.update_image_metadata(image_path, metadata)
            return JSONResponse(content={"status": "success"})

        @self._app.post("/api/labels/add", response_class=JSONResponse, tags=[WebApp.LABELS_API_TAG_NAME])
        async def add_label_api(request: AddLabelRequest) -> JSONResponse:
            """
            API endpoint to add a new label.
            """
            try:
                label = await self._manager.create_new_label(request.name, request.color)
                return JSONResponse(content={"status": "success", "label": label.__dict__})
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={"status": "failure", "message": str(e)},
                    status_code=500,
                )

        @self._app.post("/api/labels/delete", response_class=JSONResponse, tags=[WebApp.LABELS_API_TAG_NAME])
        async def delete_label_api(request: DeleteLabelRequest) -> JSONResponse:
            """
            API endpoint to add a new label.
            """
            try:
                await self._manager.delete_label(request.label_id)
                return JSONResponse(content={"status": "success"})
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={"status": "failure", "message": str(e)},
                    status_code=500,
                )

        @self._app.get("/api/labels/list", response_class=JSONResponse, tags=[WebApp.LABELS_API_TAG_NAME])
        async def list_labels_api(request: Request) -> JSONResponse:
            """
            API endpoint to return a list of labels with metadata.

            Args:
                request (Request): The FastAPI request object.

            Returns:
                JSONResponse: A JSON response containing the list of labels.
            """
            try:
                label_list = await self._manager.list_labels()
                # Convert dataclass objects to dicts
                labels = [label.__dict__ for label in label_list]
                response_data = {"status": "success", "labels": labels}
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={"status": "failure", "message": str(e)},
                    status_code=500,
                )

        class UpdateLabelRequest(BaseModel):
            label_id: int
            name: Optional[str] = None
            color: Optional[str] = None

        @self._app.post("/api/labels/update", response_class=JSONResponse, tags=[WebApp.LABELS_API_TAG_NAME])
        async def update_label_api(request: UpdateLabelRequest) -> JSONResponse:
            """
            API endpoint to update an existing label's name and/or color.
            """
            try:
                await self._manager.update_label(label_id=request.label_id, name=request.name, color=request.color)
                return JSONResponse(content={"status": "success"})
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
                    success, buffer = cv2.imencode(".png", annotated_image)
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
