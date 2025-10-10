"""Web API for DB Server"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional, List, Dict

from dataclasses import dataclass, asdict
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


from .manager import Manager, SourceMetadata

from apps.helpers.consts import JsonKeys, JsonValues, ApiTags

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class AddLabelPayload(BaseModel):
    name: str
    color: str
    parent_uuid: Optional[str] = None


class DeleteLabelPayload(BaseModel):
    label_uuid: str


class UpdateLabelPayload(BaseModel):
    label_uuid: str
    name: Optional[str] = None
    color: Optional[str] = None


class GetImageProviderSchemaPayload(BaseModel):
    image_provider: str


class CreateSourcePayload(BaseModel):
    source_name: str
    image_provider: str
    provider_params: dict


class DeleteSourcePayload(BaseModel):
    source_uuids: List[str]


class UpdateSourcePayload(BaseModel):
    source_name: str
    source_uuid: str
    image_provider: str
    provider_params: dict


class WebApp:
    """Web application for managing the database"""

    def __init__(self, app_name: str, manager: Manager):
        """
        Initialize the WebApp with the application name and manager.

        Args:
            app_name (str): Name of the application.
            manager (Manager): Instance of the Manager class for handling models.
        """
        self._manager: Manager = manager
        self._app_name: str = app_name
        self._app: FastAPI = FastAPI(lifespan=self._lifespan)
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

    def _register_routes(self):
        """Register all routes for the application."""

        @self._app.post("/api/labels/add", response_class=JSONResponse, tags=[ApiTags.LABELS], operation_id="add_label")
        async def add_label_api(request: AddLabelPayload) -> JSONResponse:
            """
            API endpoint to add a new label.
            """
            try:
                label = await self._manager.create_new_label(
                    request.name, request.color, parent_uuid=request.parent_uuid
                )
                return JSONResponse(content={JsonKeys.STATUS: JsonValues.SUCCESS, "label": asdict(label)})
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                    status_code=500,
                )

        @self._app.post(
            "/api/labels/delete", response_class=JSONResponse, tags=[ApiTags.LABELS], operation_id="delete_label"
        )
        async def delete_label_api(request: DeleteLabelPayload) -> JSONResponse:
            """
            API endpoint to delete an existing abel.
            """
            try:
                await self._manager.delete_label(request.label_uuid)
                return JSONResponse(content={JsonKeys.STATUS: JsonValues.SUCCESS})
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                    status_code=500,
                )

        @self._app.get(
            "/api/labels/list", response_class=JSONResponse, tags=[ApiTags.LABELS], operation_id="list_labels"
        )
        async def list_labels_api(request: Request) -> JSONResponse:
            """
            API endpoint to return a list of labels with metadata.

            Args:
                request (Request): The FastAPI request object.

            Returns:
                JSONResponse: A JSON response containing the list of labels.
            """
            try:
                label_list = await self._manager.get_labels()
                labels = [asdict(label) for label in label_list]
                response_data = {JsonKeys.STATUS: JsonValues.SUCCESS, "labels": labels}
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                    status_code=500,
                )

        @self._app.post(
            "/api/labels/update", response_class=JSONResponse, tags=[ApiTags.LABELS], operation_id="update_label"
        )
        async def update_label_api(request: UpdateLabelPayload) -> JSONResponse:
            """
            API endpoint to update an existing label's name and/or color.
            """
            try:
                await self._manager.update_label(label_uuid=request.label_uuid, name=request.name, color=request.color)
                return JSONResponse(content={JsonKeys.STATUS: JsonValues.SUCCESS})
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                    status_code=500,
                )

        @self._app.get(
            "/api/sources/image-providers/list",
            response_class=JSONResponse,
            tags=[ApiTags.SOURCES],
            operation_id="list_image_providers",
        )
        async def list_image_providers_api(request: Request) -> JSONResponse:
            """
            API endpoint to return a list of available image provider types

            Args:
                request (Request): The FastAPI request object.

            Returns:
                JSONResponse: A JSON response containing the list of models.
            """
            try:
                providers_list = await self._manager.list_avail_image_providers()
                response_data = {JsonKeys.STATUS: JsonValues.SUCCESS, "providers": providers_list}
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                    status_code=500,
                )

        @self._app.post(
            "/api/sources/image-providers/schema",
            tags=[ApiTags.SOURCES],
            operation_id="get_image_provider_schema",
            response_class=JSONResponse,
        )
        async def get_image_provider_schema_api(payload: GetImageProviderSchemaPayload) -> JSONResponse:
            """
            API endpoint to get the schema for a specific image provider.
            """
            try:
                schema = await self._manager.get_image_provider_schema(payload.image_provider)
                return JSONResponse(content={JsonKeys.STATUS: JsonValues.SUCCESS, "schema": schema})
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                    status_code=500,
                )

        @self._app.post(
            "/api/sources/update",
            tags=[ApiTags.SOURCES],
            operation_id="update_source",
            response_class=JSONResponse,
        )
        async def update_source_api(payload: UpdateSourcePayload) -> JSONResponse:
            """
            API endpoint for updating an existing source.
            """
            try:
                source_metadata: SourceMetadata = await self._manager.update_source(
                    payload.source_uuid,
                    image_provider=payload.image_provider,
                    provider_params=payload.provider_params,
                    source_name=payload.source_name,
                )
                return JSONResponse(content={JsonKeys.STATUS: JsonValues.SUCCESS, "source": asdict(source_metadata)})
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                    status_code=500,
                )

        @self._app.post(
            "/api/sources/create",
            tags=[ApiTags.SOURCES],
            operation_id="create_source",
            response_class=JSONResponse,
        )
        async def create_source_api(payload: CreateSourcePayload) -> JSONResponse:
            """
            API endpoint for creating a new source.

            Args:
                req (CreateSourceRequest): Request object with source type and parameters.

            Returns:
                JSONResponse: Response with new source ID or error.
            """
            try:
                source_metadata: SourceMetadata = await self._manager.create_new_source(
                    image_provider=payload.image_provider,
                    provider_params=payload.provider_params,
                    source_name=payload.source_name,
                )
                response_data = {JsonKeys.STATUS: JsonValues.SUCCESS, "source": asdict(source_metadata)}
            except Exception as e:
                response_data = {JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)}
            return JSONResponse(content=response_data)

        @self._app.post(
            "/api/sources/delete",
            tags=[ApiTags.SOURCES],
            operation_id="delete_sources",
            response_class=JSONResponse,
        )
        async def delete_sources_api(payload: DeleteSourcePayload) -> JSONResponse:
            """
            API endpoint to delete sources.
            """
            try:
                await self._manager.delete_sources(payload.source_uuids)
                return JSONResponse(content={JsonKeys.STATUS: JsonValues.SUCCESS})
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                    status_code=500,
                )

        @self._app.get(
            "/api/sources/get",
            response_class=JSONResponse,
            tags=[ApiTags.SOURCES],
            operation_id="get_sources",
        )
        async def get_sources_api(request: Request) -> JSONResponse:
            """
            API endpoint to return a list of available sources
            Args:
                request (Request): The FastAPI request object.

            Returns:
                JSONResponse: A JSON response containing the list of sources
            """
            try:
                sources: List[SourceMetadata] = await self._manager.get_avail_sources()
                sources_dict = {source.uuid: asdict(source) for source in sources}
                response_data = {JsonKeys.STATUS: JsonValues.SUCCESS, "sources": sources_dict}
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                    status_code=500,
                )
