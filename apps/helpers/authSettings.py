"""Shared plumbing for the optional auth gate.

Settings → Security stores one JSON blob under `AUTH_SETTING_KEY`:
`{"require_auth": bool, "token": str, "password_enabled": bool, "password_hash": str}`.
Read here the same way `serverUrls.py` reads server-URL overrides — this module and
`image_tagging_server`'s Manager must agree on the key name and the JSON shape, which
is exactly the kind of thing that silently drifts when duplicated.

Two credentials, deliberately asymmetric in how they are stored:

* `token` is a machine credential that has to be presented verbatim by Home Assistant
  and by `<video>`/`<img>` URLs, and re-displayed in the UI for copying, so it is
  stored as-is. It is 256 CSPRNG bits — there is nothing to guess.
* `password_hash` is a human credential, so it is never stored in the clear; only an
  `scrypt` digest with a per-password salt is kept, and nothing can recover the
  plaintext from it.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import sys
from dataclasses import dataclass

import httpx

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)

AUTH_SETTING_KEY = "auth_config"

# scrypt work factors. 128 * N * r = 16 MiB of memory per hash, which is under
# OpenSSL's default 32 MiB maxmem, so no maxmem override is needed. Hashing costs
# ~100ms, which is why both callers run it off the event loop.
_SCRYPT_N = 1 << 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SCRYPT_PREFIX = "scrypt"


@dataclass(frozen=True)
class AuthConfig:
    """The Security panel's stored state.

    Attributes:
        require_auth (bool): Whether the API token is required. Named for its JSON
            key, which predates the password; it is one of two independent switches,
            not a master toggle — `Manager.require_auth` is what says whether the
            gate enforces.
        token (str): API token, presented as a Bearer header or `?token=`.
        password_enabled (bool): Whether interactive login wants a password rather
            than the API token.
        password_hash (str): scrypt digest of the login password, never the plaintext.
    """

    require_auth: bool = False
    token: str = ""
    password_enabled: bool = False
    password_hash: str = ""

    def to_json(self) -> str:
        """Serialize to the stored settings blob.

        Returns:
            str: JSON in the shape `read_auth_config` parses.
        """
        return json.dumps(
            {
                "require_auth": self.require_auth,
                "token": self.token,
                "password_enabled": self.password_enabled,
                "password_hash": self.password_hash,
            }
        )


def hash_password(plain: str) -> str:
    """Hash a login password for storage.

    Args:
        plain (str): The plaintext password.

    Returns:
        str: `scrypt$<n>$<r>$<p>$<salt_b64>$<digest_b64>`.
    """
    salt = os.urandom(16)
    digest = hashlib.scrypt(
        plain.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN
    )
    return "$".join(
        [
            _SCRYPT_PREFIX,
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(digest).decode("ascii"),
        ]
    )


def verify_password(plain: str, stored: str) -> bool:
    """Check a submitted password against a stored digest.

    The work factors come from the stored string rather than the constants above, so
    raising them later keeps every existing password verifiable.

    Args:
        plain (str): The submitted password.
        stored (str): A digest produced by `hash_password`.

    Returns:
        bool: True when the password matches.
    """
    if not plain or not stored:
        return False
    try:
        prefix, n, r, p, salt_b64, digest_b64 = stored.split("$")
        if prefix != _SCRYPT_PREFIX:
            return False
        expected = base64.b64decode(digest_b64)
        candidate = hashlib.scrypt(
            plain.encode("utf-8"),
            salt=base64.b64decode(salt_b64),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
    except Exception as e:
        logger.warning(f"Malformed stored password hash: {e}")
        return False
    return hmac.compare_digest(candidate, expected)


async def read_auth_config(client: httpx.AsyncClient, db_server_url: str) -> AuthConfig:
    """Read the Security settings blob from db_server's settings table.

    Returns the all-off default when the setting is absent, malformed, or unreadable —
    all of which mean "auth off", matching the feature's documented default. An
    unreachable db_server is deliberately not an error here, for the same boot-race
    reason `read_url_override` treats it that way: the caller re-reads on a later pass.

    Args:
        client (httpx.AsyncClient): Client to issue the request with.
        db_server_url (str): Base URL of db_server.

    Returns:
        AuthConfig: The stored configuration.
    """
    try:
        res = await client.get(f"{db_server_url.rstrip('/')}/api/settings/{AUTH_SETTING_KEY}")
        if res.status_code != 200:
            return AuthConfig()
        value = res.json().get("value")
        if not value:
            return AuthConfig()
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            return AuthConfig()
        return AuthConfig(
            require_auth=bool(parsed.get("require_auth")),
            token=str(parsed.get("token") or ""),
            password_enabled=bool(parsed.get("password_enabled")),
            password_hash=str(parsed.get("password_hash") or ""),
        )
    except Exception as e:
        logger.debug(f"Could not read auth settings: {e}")
        return AuthConfig()
