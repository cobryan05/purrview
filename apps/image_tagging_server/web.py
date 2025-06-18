from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import logging
import os
import sys
import json

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class WebApp:
    def __init__(self, app_name: str):
        """Initialize the WebApp with the application name"""
        self._app_name: str = app_name
        self._app: FastAPI = FastAPI()
        self._templates: Jinja2Templates = Jinja2Templates(directory=os.path.join(SCRIPT_DIR, "templates"))

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
