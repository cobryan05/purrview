"""Manager for Image Tagging Server"""

import asyncio
import logging
import sys
import time
from typing import Any, Dict, List, Optional

import db_client
import httpx
import inference_client
import tasks_client

from apps import APPS, APPS_CONFIG
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

        # Cache for aggregated service notices; refreshed at most every 5 seconds.
        self._notices_cache_ts: float = 0.0
        self._notices_cache: list[dict] = []
        self._NOTICES_CACHE_TTL: float = 5.0

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

    async def get_all_service_status(self) -> List[dict]:
        """Probe each backend service's /status endpoint and return aggregated results."""
        relay_cfg = APPS_CONFIG[APPS.RELAY_BUFFER_SERVER]
        services = [
            ("db_server", self._db_server_url),
            ("inference_server", self._default_inference_url),
            ("tasks_server", self._tasks_api_client.configuration.host),
            ("training_server", self._default_training_url),
            ("relay_buffer_server", f"http://{relay_cfg.host}:{relay_cfg.port}"),
        ]

        async def _probe(name: str, base_url: str) -> dict:
            try:
                resp = await self._test_client.get(f"{base_url}/status", timeout=3.0)
                if resp.status_code == 200:
                    data = resp.json()
                    data.setdefault("service", name)
                    return data
                return {"service": name, "status": "error", "message": f"HTTP {resp.status_code}"}
            except Exception as e:
                return {"service": name, "status": "unreachable", "message": str(e)}

        return list(await asyncio.gather(*[_probe(name, url) for name, url in services]))

    async def get_notices(self) -> list[dict]:
        """Return aggregated service notices from all backends, with a 5-second cache."""
        now = time.monotonic()
        if now - self._notices_cache_ts < self._NOTICES_CACHE_TTL:
            return self._notices_cache

        statuses = await self.get_all_service_status()
        notices: list[dict] = []
        for svc in statuses:
            notices.extend(svc.get("notices", []))

        self._notices_cache = notices
        self._notices_cache_ts = now
        return notices

    async def get_raw_notices(self) -> list[dict]:
        """Fetch notices from all backends without caching (used by SSE stream)."""
        statuses = await self.get_all_service_status()
        return [n for svc in statuses for n in svc.get("notices", [])]

    async def get_tasks_for_stream(self) -> dict:
        """Fetch current task status from tasks_server for SSE streaming."""
        url = f"{self._tasks_api_client.configuration.host}/api/tasks/instances/get"
        try:
            r = await self._test_client.post(url, json={}, timeout=5.0)
            r.raise_for_status()
            return r.json()
        except Exception:
            return {}
