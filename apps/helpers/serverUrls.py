"""Shared plumbing for the "point a service at another host" settings.

`inference_server` and `training_server` are the two services with no db_server
dependency of their own — they take work over HTTP and answer it — so they are
the two that can legitimately run on a different machine. Settings → Servers
stores that machine's base URL in db_server under the keys below, and every
service that *calls* one of them resolves the override through this module.

Deliberately tiny and httpx-only: the two consumers (image_tagging_server's
proxy layer and tasks_server's task runners) sit on opposite sides of the app
and must agree on the key names and the "empty string means no override"
convention, which is exactly the kind of thing that silently drifts when it is
duplicated. See `read_url_overrides`.
"""

import asyncio
import logging
import sys
from typing import Optional, Tuple

import httpx

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)

INFERENCE_URL_SETTING_KEY = "inference_server_url"
TRAINING_URL_SETTING_KEY = "training_server_url"


async def read_url_override(client: httpx.AsyncClient, db_server_url: str, key: str) -> Optional[str]:
    """Read one server-URL override from db_server's settings table.

    Returns None when the setting is absent, empty, or unreadable — all of which
    mean "no override, use the compiled-in default". An unreachable db_server is
    deliberately not an error: services start concurrently, so the first read can
    lose a boot race, and callers re-read on a later pass.

    Args:
        client (httpx.AsyncClient): Client to issue the request with.
        db_server_url (str): Base URL of db_server.
        key (str): Setting key to read.

    Returns:
        Optional[str]: The override URL, or None for "use the default".
    """
    try:
        res = await client.get(f"{db_server_url.rstrip('/')}/api/settings/{key}")
        if res.status_code == 200:
            value = res.json().get("value")
            # Saving an empty input is how the UI clears an override, so "" and
            # None have to collapse to the same thing here.
            return (value or "").strip() or None
        return None
    except Exception as e:
        logger.debug(f"Could not read setting '{key}': {e}")
        return None


async def read_url_overrides(client: httpx.AsyncClient, db_server_url: str) -> Tuple[Optional[str], Optional[str]]:
    """Read both server-URL overrides.

    Args:
        client (httpx.AsyncClient): Client to issue the requests with.
        db_server_url (str): Base URL of db_server.

    Returns:
        Tuple[Optional[str], Optional[str]]: (inference override, training override).
    """
    return await asyncio.gather(  # type: ignore[return-value]
        read_url_override(client, db_server_url, INFERENCE_URL_SETTING_KEY),
        read_url_override(client, db_server_url, TRAINING_URL_SETTING_KEY),
    )
