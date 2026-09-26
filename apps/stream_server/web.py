"""Web API for Stream Server.

JSON only. This service used to serve its own Jinja2/HTMX pages — an index, a
create-stream form and HTML renderings of the stream and config listings — from before
the SPA existed. They were the only route to the delayed-stream path and had no other
caller, so they went with it. Anything a human needs to see about this service is on
the Operations status panel, which reads `/status`.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from mediamtx_client.api_client import ApiClient

from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.recordings.playbackClient import PlaybackError
from apps.helpers.timeUtils import parse_iso_utc

from .manager import Manager

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)


class WebApp:
    SUCCESS_KEY = "success"
    ERROR_KEY = "error"
    RESULT_KEY = "results"

    STREAM_API_TAG_NAME = "streams"
    CONFIG_API_TAG_NAME = "config"

    def __init__(self, app_name: str, manager: Manager):
        """Initialize the WebApp with the application name and manager"""
        self._manager: Manager = manager
        self._app_name: str = app_name
        self._app: FastAPI = FastAPI(lifespan=self._lifespan)
        self._api_client: ApiClient = manager._media_mtx_api_client

        # Register routes
        self._register_routes()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Lifespan event handler for startup and shutdown logic"""
        logger.info("Application is starting up...")
        self._manager.start()

        yield

        logger.info("Application is shutting down...")
        if self._manager._task:
            self._manager._task.cancel()
            await self._manager._task

    def app(self) -> FastAPI:
        """Return the FastAPI application instance"""
        return self._app

    def _register_routes(self):
        """Register all routes for the application"""

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
                response_data = {"status": "success", "config": server_config}
            except Exception as e:
                response_data = {"status": "failure", "message": str(e)}
            return JSONResponse(content=response_data)

        @self._app.get(
            "/status",
            operation_id="stream_server_status",
            response_class=JSONResponse,
        )
        async def stream_server_status():
            return JSONResponse(content=self._manager.get_status())

        @self._app.post(
            "/api/streams/ensure",
            response_class=JSONResponse,
            operation_id="ensureStream",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def ensure_stream_api(
            source_url: str = Form(..., description="Source URL to relay (rtsp://, rtmp://, srt://, http(s):// …)"),
            holder: str = Form("", description="Identity of the consumer taking the lease"),
            retention_s: float = Form(
                -1.0, description="Seconds of rolling buffer to keep; negative uses the server default"
            ),
            record: bool = Form(True, description="Whether to maintain a rolling buffer for this source"),
        ) -> JSONResponse:
            """Idempotently arm a relay path for a source and take a lease on it.

            Consumers read the returned master path directly — there is no per-consumer
            path, because fanning out from the master is the whole point of the relay.
            The response carries `recording`/`retentionS` so a caller can tell the user
            "this source has no rewind buffer" instead of silently cutting empty clips.
            """
            try:
                result = await self._manager.ensure_stream(
                    source_url=source_url,
                    holder=holder,
                    retention_s=None if retention_s < 0 else retention_s,
                    record=record,
                )
                return JSONResponse(content={WebApp.SUCCESS_KEY: True, **result.to_dict()})
            except Exception as e:
                logger.exception(f"Stream ensure failed: {e}")
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})

        @self._app.post(
            "/api/streams/release",
            response_class=JSONResponse,
            operation_id="releaseStream",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def release_stream_api(
            lease_id: str = Form(..., description="Lease id returned by ensure"),
        ) -> JSONResponse:
            """Release one lease; the relay path is torn down when it was the last."""
            try:
                released = await self._manager.release_lease(lease_id)
                return JSONResponse(content={WebApp.SUCCESS_KEY: True, "released": released})
            except Exception as e:
                logger.exception(f"Stream release failed: {e}")
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})

        @self._app.post(
            "/api/streams/renew",
            response_class=JSONResponse,
            operation_id="renewStream",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def renew_stream_api(
            lease_id: str = Form(..., description="Lease id returned by ensure"),
        ) -> JSONResponse:
            """Heartbeat: tell the relay this holder is still alive.

            A lease that stops being renewed is reaped, so a holder that intends to keep
            reading must call this well inside the TTL. `renewed: false` means the relay
            has forgotten the lease and the caller should re-`ensure`.
            """
            try:
                renewed = self._manager.renew_lease(lease_id)
                return JSONResponse(content={WebApp.SUCCESS_KEY: True, "renewed": renewed})
            except Exception as e:
                logger.exception(f"Stream renew failed: {e}")
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})

        @self._app.get(
            "/api/streams/{name}/ranges",
            response_class=JSONResponse,
            operation_id="streamRanges",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def stream_ranges_api(name: str) -> JSONResponse:
            """Time ranges currently held in the rolling buffer for a path.

            This is the honest answer to "how much rewind do I have", which is not the
            configured retention: the buffer is RAM-backed, so a relay restart shortens
            it without warning.
            """
            try:
                ranges = await self._manager.get_ranges(name)
                return JSONResponse(content={WebApp.SUCCESS_KEY: True, WebApp.RESULT_KEY: ranges})
            except Exception as e:
                logger.exception(f"Fetching ranges for {name} failed: {e}")
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})

        @self._app.post(
            "/api/streams/clip",
            response_class=JSONResponse,
            operation_id="extractClip",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def extract_clip_api(
            path: str = Form(..., description="Master path name to cut from"),
            start: str = Form(..., description="Clip start as an ISO-8601 timestamp"),
            duration_s: float = Form(..., description="Clip duration in seconds"),
            dest: str = Form(..., description="Absolute destination file path"),
        ) -> JSONResponse:
            """Cut a window out of the rolling buffer into a file.

            Retroactive by design: pre-roll is whatever the caller asks for at event
            time, not something fixed when the stream was created. The result reports
            `partialPreroll` when the requested window had partly aged out, so a
            truncated clip is never presented as a complete one.
            """
            try:
                started_at = parse_iso_utc(start)
                result = await self._manager.extract_clip(
                    path=path, start=started_at, duration_s=duration_s, dest=Path(dest)
                )
                return JSONResponse(content={WebApp.SUCCESS_KEY: True, **result})
            except PlaybackError as e:
                # The requested window has aged out of the rolling buffer, which is
                # expected whenever a drain lags the buffer. recordClipAction emits the
                # operator-facing line for this, so no traceback is warranted here.
                logger.debug("Clip extraction failed for %r: %s", path, e)
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})
            except Exception as e:
                logger.exception("Clip extraction failed for %r", path)
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})

        @self._app.get(
            "/api/streams/buffer",
            response_class=JSONResponse,
            operation_id="bufferUsage",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def buffer_usage_api() -> JSONResponse:
            """RAM buffer budget: cap, committed bytes, and actual mount usage."""
            try:
                usage = self._manager.buffer_usage()
                return JSONResponse(content={WebApp.SUCCESS_KEY: True, **usage.to_dict()})
            except Exception as e:
                logger.exception(f"Buffer usage failed: {e}")
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})

        @self._app.post(
            "/api/streams/destroy",
            response_class=JSONResponse,
            operation_id="destroyStream",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def destroy_stream_api(
            request: Request,
            stream_name: str = Form(..., description="Name of stream to destroy"),
        ) -> JSONResponse:
            """Handle the destruction of a stream"""
            try:
                logger.info(f"Destroying stream {stream_name}")
                await self._manager.destroy_stream(stream_name=stream_name)
                return JSONResponse(content={WebApp.SUCCESS_KEY: True})
            except Exception as e:
                logger.exception(f"Stream destruction failed: {str(e)}")
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})

        @self._app.get(
            "/api/config/list",
            response_class=JSONResponse,
            operation_id="listConfig",
            tags=[WebApp.CONFIG_API_TAG_NAME],
        )
        async def list_configs_api(request: Request) -> JSONResponse:
            """List the global configuration"""
            try:
                config = await self._manager.get_config()
                return JSONResponse(content={WebApp.SUCCESS_KEY: True, WebApp.RESULT_KEY: config})
            except Exception as e:
                logger.exception(f"Config listing failed: {str(e)}")
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})

        @self._app.get(
            "/api/streams/list",
            response_class=JSONResponse,
            operation_id="listStreams",
            tags=[WebApp.STREAM_API_TAG_NAME],
        )
        async def list_streams_api(request: Request) -> JSONResponse:
            """List all available streams"""
            try:
                await self._manager.refresh_streams()
                streams = self._manager.get_streams()
                stream_info: List[Dict[str, Any]] = [
                    {
                        "name": info["name"],
                        "info": info,
                        "actions": [
                            {
                                "label": "Stop Stream",
                                "url": "/destroy-stream",
                                "params": {"stream_name": info["name"]},
                            }
                        ],
                    }
                    for info in streams.values()
                ]
                return JSONResponse(content={WebApp.SUCCESS_KEY: True, WebApp.RESULT_KEY: stream_info})
            except Exception as e:
                logger.exception(f"Fetching streams failed: {str(e)}")
                return JSONResponse(content={WebApp.SUCCESS_KEY: False, WebApp.ERROR_KEY: str(e)})
