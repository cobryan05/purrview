"""Web API for Training Server"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from apps.helpers.consts import JsonKeys, JsonValues

from .manager import Manager

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class WebApp:
    def __init__(self, app_name: str, manager: Manager):
        self._manager = manager
        self._app_name = app_name
        self._app: FastAPI = FastAPI(lifespan=self._lifespan)
        self._register_routes()

    def app(self) -> FastAPI:
        return self._app

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        self._manager.start()
        yield

    def _register_routes(self):
        @self._app.get(
            "/server-config",
            operation_id="server_config",
            response_class=JSONResponse,
        )
        async def server_config(request: Request):
            try:
                config = await self._manager.get_server_config()
                return JSONResponse({JsonKeys.STATUS: JsonValues.SUCCESS, JsonKeys.CONFIG: config})
            except Exception as e:
                return JSONResponse({JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)})

        @self._app.get(
            "/api/training/status",
            operation_id="training_status",
            response_class=JSONResponse,
        )
        async def training_status(request: Request):
            return JSONResponse({JsonKeys.STATUS: JsonValues.SUCCESS, JsonKeys.MESSAGE: "Training server running."})
