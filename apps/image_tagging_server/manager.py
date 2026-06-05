"""Manager for Image Tagging Server"""

import asyncio
import logging
import sys
from typing import Any, Dict, Optional

import httpx
import db_client
import inference_client
import tasks_client

from apps import APPS
from apps.helpers.reverseProxy import ReverseProxy

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

INFERENCE_URL_SETTING_KEY = "inference_server_url"
TRAINING_URL_SETTING_KEY = "training_server_url"


class Manager:
    """Manages image tagging frontend"""

    def __init__(
        self,
        inference_api_client: inference_client.ApiClient,
        tasks_api_client: tasks_client.ApiClient,
        db_api_client: db_client.ApiClient,
        default_training_url: str,
    ):
        """Initialize the Manager"""
        self._inference_api_client: inference_client.ApiClient = inference_api_client
        self._tasks_api_client: tasks_client.ApiClient = tasks_api_client
        self._db_api_client: db_client.ApiClient = db_api_client

        self._default_inference_url: str = inference_api_client.configuration.host
        self._default_training_url: str = default_training_url
        self._db_server_url: str = db_api_client.configuration.host

        self._inference_url_override: Optional[str] = None
        self._training_url_override: Optional[str] = None

        # Build initial proxies from defaults; reloaded at startup if overrides exist in DB
        self._inference_proxy = ReverseProxy(base_url=self._default_inference_url, path="/api/models")
        self._training_proxy = ReverseProxy(base_url=self._default_training_url, path="/api/training")

        self._test_client = httpx.AsyncClient(timeout=5.0)

    async def start(self) -> None:
        """Load URL overrides from DB settings on startup."""
        try:
            await self._reload_url_overrides()
        except Exception as e:
            logger.warning(f"Could not load server URL overrides from DB: {e}")

    async def _read_db_setting(self, key: str) -> Optional[str]:
        """Read a setting from db_server via HTTP. Returns None on error or empty value."""
        try:
            res = await self._test_client.get(f"{self._db_server_url}/api/settings/{key}")
            if res.status_code == 200:
                return res.json().get("value") or None
            return None
        except Exception as e:
            logger.debug(f"Could not read setting '{key}': {e}")
            return None

    async def _reload_url_overrides(self) -> None:
        """Re-read URL overrides from DB and rebuild proxies."""
        inference_url, training_url = await asyncio.gather(
            self._read_db_setting(INFERENCE_URL_SETTING_KEY),
            self._read_db_setting(TRAINING_URL_SETTING_KEY),
        )

        self._inference_url_override = inference_url
        self._training_url_override = training_url

        effective_inference = self._inference_url_override or self._default_inference_url
        effective_training = self._training_url_override or self._default_training_url

        await self._inference_proxy.client.aclose()
        await self._training_proxy.client.aclose()
        self._inference_proxy = ReverseProxy(base_url=effective_inference, path="/api/models")
        self._training_proxy = ReverseProxy(base_url=effective_training, path="/api/training")

        logger.info(f"Server proxies: inference={effective_inference}, training={effective_training}")

    async def reload_url_overrides(self) -> None:
        """Public method called after settings are saved to apply new proxy URLs."""
        await self._reload_url_overrides()

    @property
    def inference_proxy(self) -> ReverseProxy:
        return self._inference_proxy

    @property
    def training_proxy(self) -> ReverseProxy:
        return self._training_proxy

    async def test_server(self, url: str, expected_service: str) -> Dict[str, Any]:
        """Test connectivity to a server, verify it is the expected service type."""
        url = url.rstrip("/")
        try:
            res = await self._test_client.get(f"{url}/server-config")
        except httpx.ConnectError:
            return {"success": False, "message": "Connection refused — is the server running?"}
        except httpx.TimeoutException:
            return {"success": False, "message": "Connection timed out"}
        except Exception as e:
            return {"success": False, "message": str(e)}

        if res.status_code != 200:
            return {"success": False, "message": f"Server returned HTTP {res.status_code}"}

        try:
            data = res.json()
        except Exception:
            return {"success": False, "message": "Server response was not valid JSON"}

        config = data.get("config", {})
        actual_service = config.get("service")

        if actual_service is None:
            return {"success": False, "message": "Server did not report a service type"}
        if actual_service != expected_service:
            return {
                "success": False,
                "message": f"Wrong server type — expected '{expected_service}', got '{actual_service}'",
            }

        # Build a human-readable info string from the config
        info_parts = [actual_service]
        if "models_path" in config:
            info_parts.append(f"models path: {config['models_path']}")
        return {"success": True, "message": f"Connected — {' · '.join(info_parts)}", "config": config}

    async def get_server_config(self) -> Dict[str, Any]:
        return {
            "inference_server": self._default_inference_url,
            "tasks_server": self._tasks_api_client.configuration.host,
            "db_server": self._db_server_url,
            "training_server": self._default_training_url,
            "inference_server_url_override": self._inference_url_override,
            "training_server_url_override": self._training_url_override,
        }
