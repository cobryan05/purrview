"""Async adapter over the generated stream_client, shaped for clip extraction.

`PlaybackClipBackend` wants one thing from the relay: an awaitable
``extract_clip(path, start: datetime, duration_s, dest: Path) -> dict``. The generated
OpenAPI client gives neither half of that — it is synchronous (so calling it directly
would block the watcher's event loop for the length of an ffmpeg cut), it takes the
timestamp and destination as strings, and it returns the raw ``{success, error, ...}``
envelope rather than raising. This wraps all three differences in one place so the
backend can stay ignorant of which transport it is talking over, and so tests can
substitute any object with the same method.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import stream_client

from apps.helpers.timeUtils import ensure_utc

logger = logging.getLogger(__name__)


class StreamClipClient:
    """Cut clips out of the relay's rolling buffer over the stream_server API."""

    def __init__(self, api_client: stream_client.ApiClient) -> None:
        self._api_client = api_client

    @property
    def api_client(self) -> stream_client.ApiClient:
        return self._api_client

    async def extract_clip(self, path: str, start: datetime, duration_s: float, dest: Path) -> Dict[str, Any]:
        """Cut ``[start, start+duration_s)`` from ``path`` into ``dest``.

        ``dest`` is a path on the *relay's* filesystem — today that is the same
        container, which is why the clip is written there rather than streamed back.
        """
        start = ensure_utc(start)
        api = stream_client.StreamsApi(self._api_client)
        result = await asyncio.to_thread(
            api.extract_clip,
            path=path,
            start=start.isoformat(),
            duration_s=float(duration_s),
            dest=str(dest),
        )
        result = dict(result or {})
        # The endpoint reports failure in the body, not the status code, so an
        # unchecked caller would happily record a clip file that was never written.
        if not result.get("success"):
            raise RuntimeError(result.get("error") or f"Relay could not extract a clip from {path!r}")
        result.pop("success", None)
        return result
