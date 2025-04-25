from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from mediamtx_client.api_client import ApiClient
from mediamtx_client.api.paths_api import PathsApi


from apps.stream_manager.manager import Manager

from typing import Dict, List

from pydantic import BaseModel, ValidationError


class WebApp:
    def __init__(self, app_name: str, manager: Manager):
        self._manager: Manager = manager
        self._app_name: str = app_name
        self._app: FastAPI = FastAPI(lifespan=self._lifespan)
        self._templates: Jinja2Templates = Jinja2Templates(directory="templates")
        self._dflt_args: dict[str, str] = {"app_name": self._app_name}
        self._api_client: ApiClient = manager._api_client

        # Mount static files
        self._app.mount("/static", StaticFiles(directory="static"), name="static")

        # Register routes
        self._register_routes()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        # Startup logic
        print("Application is starting up...")
        self._manager.start()

        yield  # This allows the app to run

        # Shutdown logic
        print("Application is shutting down...")
        if self._manager._task:
            self._manager._task.cancel()
            await self._manager._task

    def app(self) -> FastAPI:
        return self._app

    def _error_response(self, request: Request, message: str):
        return self._templates.TemplateResponse(
            "display_text.html",
            {
                "request": request,
                "text": message,
                **self._dflt_args,
            },
        )

    def _register_routes(self):
        @self._app.get("/", response_class=HTMLResponse)
        def index(request: Request):
            return self._templates.TemplateResponse("index.html", {"request": request, **self._dflt_args})

        @self._app.get("/create-stream-form", response_class=HTMLResponse)
        async def create_stream_form(request: Request):
            try:
                return self._templates.TemplateResponse(
                    "create_stream_form.html",
                    {"request": request, **self._dflt_args},
                )
            except Exception as e:
                return self._error_response(request, f"Error fetching create stream form: {str(e)}")

        @self._app.post("/create-stream", response_class=HTMLResponse)
        async def create_stream(request: Request):
            try:
                # Parse form data
                form_data = await request.form()
                source_url: str = form_data.get("source_url")
                destination_name: str = form_data.get("destination_name")
                delay: float = float(form_data.get("delay"))

                print(
                    f"Creating stream with Source URL: {source_url}, Destination Name: {destination_name}, Delay: {delay}"
                )
                await self._manager.create_stream(src=source_url, stream_name=destination_name, delay=delay)
            except Exception as e:
                return self._error_response(request, f"Failed to create stream: {e}")

            # Return a success message or redirect to another page
            return self._templates.TemplateResponse(
                "display_text.html",
                {
                    "request": request,
                    "text": "Stream created successfully",
                    **self._dflt_args,
                },
            )

        @self._app.post("/destroy-stream", response_class=HTMLResponse)
        async def destroy_stream(request: Request):
            try:
                # Parse form data
                form_data = await request.form()
                stream_name: str = form_data.get("stream_name")

                # Example: Log or process the data (replace with actual logic)
                print(f"Destroying stream {stream_name}")

                await self._manager.destroy_stream(stream_name=stream_name)

                # Return a success message or redirect to another page
                return self._templates.TemplateResponse(
                    "display_text.html",
                    {
                        "request": request,
                        "text": "Stream destroyed successfully",
                        **self._dflt_args,
                    },
                )
            except Exception as e:
                return self._error_response(request, f"Failed to destroy stream: {e}")

        @self._app.get("/list-streams", response_class=HTMLResponse)
        async def list_streams(request: Request):
            try:
                streams = await self._manager.get_streams()
                stream_info: List[Dict[str, str]] = []
                for info in streams.values():
                    stream_info.append({"name": info.name, "text": str(info)})

                return self._templates.TemplateResponse(
                    "streams_list.html",
                    {
                        "request": request,
                        "streams": stream_info,
                        **self._dflt_args,
                    },
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error fetching streams: {str(e)}")

        @self._app.get("/list-config", response_class=HTMLResponse)
        async def list_configs(request: Request):
            try:
                config = await self._manager.get_config()
                return self._templates.TemplateResponse(
                    "display_text.html",
                    {
                        "request": request,
                        "text": config,
                        **self._dflt_args,
                    },
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error fetching streams: {str(e)}")
