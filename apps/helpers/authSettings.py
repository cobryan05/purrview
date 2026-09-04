"""Shared plumbing for the optional shared-secret auth gate.

Settings → Security stores one JSON blob under `AUTH_SETTING_KEY`:
`{"require_auth": bool, "token": str}`. Read here the same way `serverUrls.py`
reads server-URL overrides — this module and `image_tagging_server`'s Manager
must agree on the key name and the JSON shape, which is exactly the kind of
thing that silently drifts when duplicated.
"""

import json
import logging
import sys
from typing import Tuple

import httpx

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)

AUTH_SETTING_KEY = "auth_config"


async def read_auth_config(client: httpx.AsyncClient, db_server_url: str) -> Tuple[bool, str]:
    """Read the (require_auth, token) pair from db_server's settings table.

    Returns (False, "") when the setting is absent, malformed, or unreadable — all of
    which mean "auth off", matching the feature's documented default. An unreachable
    db_server is deliberately not an error here, for the same boot-race reason
    `read_url_override` treats it that way: the caller re-reads on a later pass.

    Args:
        client (httpx.AsyncClient): Client to issue the request with.
        db_server_url (str): Base URL of db_server.

    Returns:
        Tuple[bool, str]: (require_auth, token).
    """
    try:
        res = await client.get(f"{db_server_url.rstrip('/')}/api/settings/{AUTH_SETTING_KEY}")
        if res.status_code != 200:
            return False, ""
        value = res.json().get("value")
        if not value:
            return False, ""
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            return False, ""
        return bool(parsed.get("require_auth")), str(parsed.get("token") or "")
    except Exception as e:
        logger.debug(f"Could not read auth settings: {e}")
        return False, ""
