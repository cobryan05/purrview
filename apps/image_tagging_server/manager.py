"""Manager for Image Tagging Server"""

import asyncio
import hashlib
import hmac
import logging
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import db_client
import httpx
import inference_client
import tasks_client

from apps import APPS, APPS_CONFIG
from apps.helpers import serviceStatus
from apps.helpers.authSettings import (
    AUTH_SETTING_KEY,
    AuthConfig,
    hash_password,
    read_auth_config,
    verify_password,
)
from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.reverseProxy import ReverseProxy
from apps.helpers.serverUrls import (
    INFERENCE_URL_SETTING_KEY,
    TRAINING_URL_SETTING_KEY,
    read_url_overrides,
)

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)

__all__ = ["Manager", "INFERENCE_URL_SETTING_KEY", "TRAINING_URL_SETTING_KEY"]

# Service /status probes: a service that's simply busy (e.g. mid-inference or
# mid-training-step) can take a while to answer its own health check. Rather
# than flagging it "not responding" the moment the old short timeout elapses,
# wait up to PROBE_HARD_TIMEOUT before giving up — but surface a non-alarming
# "still waiting" notice once PROBE_SOFT_TIMEOUT passes, so the UI doesn't
# look frozen while we wait.
PROBE_SOFT_TIMEOUT = 3.0
PROBE_HARD_TIMEOUT = 30.0
# How long a poll waits inline for a freshly-started probe before falling back
# to "checking"/"still waiting" — lets the common fast-success case resolve
# within the same poll instead of always waiting for the next one.
PROBE_INLINE_WAIT = 0.25

# How long the Settings "Test" button waits for a CONNACK. A broker that is firewalled
# never answers at all, so this is the wait a user sees before being told so; long
# enough for a slow LAN broker, short enough not to look hung.
MQTT_TEST_TIMEOUT = 5.0
MQTT_TEST_KEEPALIVE = 10


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

        self._test_client = httpx.AsyncClient(timeout=PROBE_HARD_TIMEOUT)

        # Cache for aggregated service notices; refreshed at most every 5 seconds.
        self._notices_cache_ts: float = 0.0
        self._notices_cache: list[dict] = []
        self._NOTICES_CACHE_TTL: float = 5.0

        # In-flight /status probes, keyed by service name, so a slow probe
        # doesn't need to be re-issued (or block) on every poll — see _probe().
        self._probe_tasks: Dict[str, asyncio.Task] = {}
        self._probe_started_at: Dict[str, float] = {}

        # Services that are expected to sometimes simply be off (a remote host
        # that isn't always on) get their "unreachable" notice surfaced once per
        # outage rather than on every poll — see _unreachable_notice() below.
        self._QUIET_UNREACHABLE_SERVICES = {"training_server"}
        self._unreachable_notified: Dict[str, bool] = {}

        # Off by default. See Settings → Security / AuthGate in web.py.
        self._auth: AuthConfig = AuthConfig()

    async def start(self) -> None:
        """Load URL overrides and auth settings from DB on startup."""
        try:
            await self._reload_url_overrides()
        except Exception as e:
            logger.warning(f"Could not load server URL overrides from DB: {e}")
        try:
            await self._reload_auth_config()
        except Exception as e:
            logger.warning(f"Could not load auth settings from DB: {e}")

    async def _reload_url_overrides(self) -> None:
        """Re-read URL overrides from DB and rebuild proxies."""
        inference_url, training_url = await read_url_overrides(self._test_client, self._db_server_url)

        # The Settings input accepts a trailing slash; the proxy concatenates
        # base_url + path, so leaving it on yields a double-slashed request path.
        self._inference_url_override = inference_url.rstrip("/") if inference_url else None
        self._training_url_override = training_url.rstrip("/") if training_url else None

        effective_inference = self._inference_url_override or self._default_inference_url
        effective_training = self._training_url_override or self._default_training_url

        await self._inference_proxy.client.aclose()
        await self._training_proxy.client.aclose()
        self._inference_proxy = ReverseProxy(base_url=effective_inference, path="/api/models")
        self._training_proxy = ReverseProxy(base_url=effective_training, path="/api/training")

        logger.info(f"Server proxies: inference={effective_inference}, training={effective_training}")

    async def reload_url_overrides(self) -> None:
        """Public method called after settings are saved to apply new proxy URLs.

        Also nudges tasks_server, which resolves the same two overrides for its
        background tasks. Without the nudge the UI would be proxying to the new
        host while every running task still used the old one until tasks_server's
        next worker pass — a split brain that is very hard to read from the UI.
        """
        await self._reload_url_overrides()
        await self._notify_tasks_server_reload()

    async def _notify_tasks_server_reload(self) -> None:
        """Best-effort: tasks_server re-reads the settings itself once a minute, so
        a failure here delays the change rather than losing it."""
        tasks_url = self._tasks_api_client.configuration.host
        try:
            res = await self._test_client.post(f"{tasks_url}/api/reload-server-config", timeout=PROBE_SOFT_TIMEOUT)
            if res.status_code != 200:
                logger.warning(f"tasks_server rejected server-config reload: HTTP {res.status_code}")
        except Exception as e:
            logger.warning(f"Could not notify tasks_server of server-config change: {e}")

    async def _reload_auth_config(self) -> None:
        """Re-read the Security setting from the DB.

        Only the startup path needs this: a save goes through `save_auth_config`,
        which updates the cache itself, and the gate lives entirely in this service's
        `AuthGate` so no other service has to be told (see `HA_INTEGRATION_PLAN.md`
        WP-3).
        """
        self._auth = await read_auth_config(self._test_client, self._db_server_url)

    @property
    def require_auth(self) -> bool:
        """Whether `AuthGate` should be enforcing.

        Either credential gates the app on its own: the two switches are independent,
        so a password with no API token enforces, and a token with no password
        enforces. Anything else would store "password required" while asking nobody
        for it.

        A toggle whose credential was never saved does not count, which deliberately
        fails open rather than locking the operator out of their own UI with no
        recovery path short of editing the SQLite settings row by hand.
        """
        return (self._auth.require_auth and bool(self._auth.token)) or self.password_enabled

    @property
    def password_enabled(self) -> bool:
        """Whether `/login` wants the password rather than the API token.

        Same fail-open reasoning as `require_auth`: a toggle with no hash behind it
        would make the login form unanswerable.
        """
        return self._auth.password_enabled and bool(self._auth.password_hash)

    @property
    def auth_config(self) -> AuthConfig:
        """The cached Security settings blob."""
        return self._auth

    def check_token(self, candidate: Optional[str]) -> bool:
        """Constant-time comparison against the configured API token.

        `hmac.compare_digest` over `==` because this is one of the two secrets in the
        app — everything else here has never needed to care about a timing
        side-channel.
        """
        if not candidate or not self._auth.token:
            return False
        return hmac.compare_digest(candidate, self._auth.token)

    def session_cookie_value(self) -> str:
        """The value `/login` writes into the auth cookie.

        Derived from both secrets rather than being either of them, so that (a) a
        stolen browser cookie cannot be replayed as a Bearer token against the API,
        and (b) rotating the token or changing the password invalidates every
        outstanding session for free, with no session store to keep.

        Returns:
            str: An opaque, stable-per-configuration session value.
        """
        key = f"{self._auth.token}\n{self._auth.password_hash}".encode("utf-8")
        return hmac.new(key, b"purrview-session", hashlib.sha256).hexdigest()

    def check_session(self, candidate: Optional[str]) -> bool:
        """Constant-time check of an auth cookie against `session_cookie_value`.

        Args:
            candidate (Optional[str]): The cookie value presented.

        Returns:
            bool: True when the cookie belongs to the current configuration.
        """
        if not candidate:
            return False
        return hmac.compare_digest(candidate, self.session_cookie_value())

    async def check_login(self, candidate: Optional[str]) -> bool:
        """Check a credential submitted to the interactive login form.

        With a password configured the API token is **not** accepted here: the point
        of the split is that the token is a machine credential, and a token that also
        opens a browser session is one a Home Assistant config entry can be read out
        of. That leaves no in-band way back from a forgotten password; the way back is
        `scripts/reset_login_password.py`, out of band on the host.

        Args:
            candidate (Optional[str]): The submitted secret.

        Returns:
            bool: True when the credential opens a session.
        """
        if not candidate:
            return False
        if not self.password_enabled:
            return self.check_token(candidate)
        # scrypt is deliberately slow and memory-hard; keep it off the event loop.
        return await asyncio.to_thread(verify_password, candidate, self._auth.password_hash)

    async def save_auth_config(
        self,
        require_auth: bool,
        token: str,
        password_enabled: bool,
        password: Optional[str],
    ) -> AuthConfig:
        """Persist the Security panel and apply it immediately.

        The plaintext password is hashed here and never stored or echoed back; an
        empty `password` means "leave the existing hash alone", which is what lets the
        panel re-save the other fields without re-typing it.

        Args:
            require_auth (bool): Whether the gate enforces.
            token (str): API token.
            password_enabled (bool): Whether login wants a password.
            password (Optional[str]): New plaintext password, or empty to keep the
                current one.

        Returns:
            AuthConfig: The configuration now in force.

        Raises:
            ValueError: If the request would lock the operator out — auth required
                with no token, or a password required with none ever set.
        """
        token = (token or "").strip()
        if require_auth and not token:
            raise ValueError("An API token is required before access control can be turned on.")

        password_hash = self._auth.password_hash
        if password:
            password_hash = await asyncio.to_thread(hash_password, password)
        if password_enabled and not password_hash:
            raise ValueError("Set a password before requiring one.")

        config = AuthConfig(
            require_auth=require_auth,
            token=token,
            password_enabled=password_enabled,
            password_hash=password_hash,
        )
        res = await self._test_client.put(
            f"{self._db_server_url.rstrip('/')}/api/settings/{AUTH_SETTING_KEY}",
            json={"value": config.to_json()},
        )
        res.raise_for_status()
        self._auth = config
        return config

    @property
    def effective_inference_url(self) -> str:
        return self._inference_url_override or self._default_inference_url

    @property
    def effective_training_url(self) -> str:
        return self._training_url_override or self._default_training_url

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
            return {
                "success": False,
                "message": "Connection refused — is the server running?",
            }
        except httpx.TimeoutException:
            return {"success": False, "message": "Connection timed out"}
        except Exception as e:
            return {"success": False, "message": str(e)}

        if res.status_code != 200:
            return {
                "success": False,
                "message": f"Server returned HTTP {res.status_code}",
            }

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

        info_parts = [actual_service]
        if "models_path" in config:
            info_parts.append(f"models path: {config['models_path']}")
        return {
            "success": True,
            "message": f"Connected — {' · '.join(info_parts)}",
            "config": config,
        }

    async def test_mqtt_broker(
        self,
        host: str,
        port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
        tls: bool = False,
        timeout: float = MQTT_TEST_TIMEOUT,
    ) -> Dict[str, Any]:
        """Connect to an MQTT broker, then disconnect, to validate settings.

        Runs in a worker thread: paho's connect is blocking, and a broker that is
        firewalled rather than refusing takes the full timeout to fail — which on the
        event loop would stall every other request in the app.
        """
        host = (host or "").strip()
        if not host:
            return {"success": False, "message": "Broker host is required"}

        def _probe() -> Dict[str, Any]:
            import paho.mqtt.client as mqtt

            connected = threading.Event()
            result_code: Dict[str, Any] = {}

            client = mqtt.Client(client_id=f"purrview-test-{int(time.time())}")
            if username:
                client.username_pw_set(username, password or None)
            if tls:
                client.tls_set()

            def _on_connect(_client, _userdata, _flags, rc):
                result_code["rc"] = rc
                connected.set()

            client.on_connect = _on_connect
            try:
                client.connect(host, port, keepalive=MQTT_TEST_KEEPALIVE)
                client.loop_start()
                if not connected.wait(timeout):
                    return {"success": False, "message": f"No CONNACK from broker within {timeout:g}s"}
                rc = result_code.get("rc", 0)
                if rc != 0:
                    # rc 4/5 are the ones a user actually hits, and "code 5" alone
                    # sends them to the paho source rather than to their credentials.
                    detail = {
                        1: "broker refused the protocol version",
                        2: "broker rejected the client id",
                        3: "broker unavailable",
                        4: "bad username or password",
                        5: "not authorised",
                    }.get(rc, f"connection refused (code {rc})")
                    # No ✗ prefix here: the caller decorates the message, and the
                    # server-test flow above sets that precedent.
                    return {"success": False, "message": detail}
                return {"success": True, "message": f"Connected to {host}:{port}"}
            except Exception as e:
                return {"success": False, "message": str(e)}
            finally:
                try:
                    client.loop_stop()
                    client.disconnect()
                except Exception:
                    logger.debug("Error closing MQTT test client", exc_info=True)

        return await asyncio.to_thread(_probe)

    async def get_server_config(self) -> Dict[str, Any]:
        return {
            "inference_server": self._default_inference_url,
            "tasks_server": self._tasks_api_client.configuration.host,
            "db_server": self._db_server_url,
            "training_server": self._default_training_url,
            "inference_server_url_override": self._inference_url_override,
            "training_server_url_override": self._training_url_override,
        }

    def get_own_status(self) -> dict:
        """This service's own ``/status``, built in-memory.

        image_tagging_server hosts the entire frontend and proxies every other
        service, yet it was the one service missing from its own Service Status
        panel — the panel could report "all services ok" while rendered by a
        process nobody was checking. It is not probed over HTTP like the others:
        issuing a request to ourselves to answer a request we are already
        serving costs a worker for no information the process doesn't already
        have.

        The detail rows are the *effective* upstream URLs, because those are
        overridable at runtime and a wrong override is otherwise invisible —
        "inference is fine" and "inference is fine on the box you are actually
        talking to" are different claims, and this panel is where you go to tell
        them apart. Kept to in-memory reads, per the /status contract.
        """
        relay_cfg = APPS_CONFIG[APPS.STREAM_SERVER]
        overrides = sum(1 for u in (self._inference_url_override, self._training_url_override) if u)
        rows = [
            serviceStatus.row("db_server", self._db_server_url),
            serviceStatus.row("tasks_server", self._tasks_api_client.configuration.host),
            serviceStatus.row("inference_server", self.effective_inference_url),
            serviceStatus.row("training_server", self.effective_training_url),
            serviceStatus.row("stream_server", f"http://{relay_cfg.host}:{relay_cfg.port}"),
        ]
        status: dict = {
            "service": "image_tagging_server",
            "status": "ok",
            "summary": f"frontend · proxying {len(rows)} services"
            + (f" · {overrides} URL override{'s' if overrides != 1 else ''}" if overrides else ""),
            "details": [serviceStatus.section("Upstreams (effective)", rows)],
        }
        return status

    async def get_all_service_status(self) -> List[dict]:
        """Probe each backend service's /status endpoint and return aggregated results."""
        relay_cfg = APPS_CONFIG[APPS.STREAM_SERVER]
        services = [
            ("db_server", self._db_server_url),
            # Effective, not default: probing localhost while the app proxies to a
            # remote host would report the health of a service nothing is using.
            ("inference_server", self.effective_inference_url),
            ("tasks_server", self._tasks_api_client.configuration.host),
            ("training_server", self.effective_training_url),
            ("stream_server", f"http://{relay_cfg.host}:{relay_cfg.port}"),
        ]

        def _unreachable_notice(name: str, message: str) -> Optional[dict]:
            # get_notices()/get_raw_notices() only surface a service's own
            # "notices" key when it *is* reachable, so a down service would
            # otherwise vanish from the notices bar instead of being flagged —
            # synthesize one here so it flows through that same pipeline.
            if name in self._QUIET_UNREACHABLE_SERVICES:
                # A remote service like training_server isn't always expected to
                # be up — flag it once per outage, then go quiet so the bar
                # doesn't nag while it stays down. Any action that actually needs
                # the service (e.g. starting a training run) surfaces its own
                # error independently of this ambient notice.
                if self._unreachable_notified.get(name):
                    return None
                self._unreachable_notified[name] = True
            return {"key": f"{name}_unreachable", "level": "warning", "message": message}

        def _waiting_notice(name: str) -> dict:
            return {
                "key": f"{name}_waiting",
                "level": "info",
                "message": f"Still waiting for {name}...",
            }

        async def _run_probe(name: str, base_url: str) -> dict:
            """Issue the actual /status request. Runs to completion (success,
            HTTP error, or PROBE_HARD_TIMEOUT) as a background task so a slow
            service can't block other services' probes or the SSE loop."""
            try:
                resp = await self._test_client.get(f"{base_url}/status", timeout=PROBE_HARD_TIMEOUT)
                if resp.status_code == 200:
                    data = resp.json()
                    data.setdefault("service", name)
                    # Recovered — the next outage should notify again.
                    self._unreachable_notified.pop(name, None)
                    return data
                message = f"HTTP {resp.status_code}"
                notice = _unreachable_notice(name, f"{name} is not responding ({message})")
                return {
                    "service": name,
                    "status": "error",
                    "message": message,
                    "notices": [notice] if notice else [],
                }
            except Exception as e:
                notice = _unreachable_notice(name, f"{name} is not responding")
                return {
                    "service": name,
                    "status": "unreachable",
                    "message": str(e),
                    "notices": [notice] if notice else [],
                }

        async def _probe(name: str, base_url: str) -> dict:
            task = self._probe_tasks.get(name)
            if task is not None and task.done():
                # Finished between polls (this poller runs every few seconds,
                # so a probe can complete without ever being awaited again) —
                # consume its result now rather than discarding it and
                # starting a fresh probe/timer from scratch.
                # pop (not del): concurrent pollers can both reach this cleanup
                # for the same task, so removal must be idempotent.
                self._probe_tasks.pop(name, None)
                return task.result()
            if task is None:
                task = asyncio.create_task(_run_probe(name, base_url))
                self._probe_tasks[name] = task
                self._probe_started_at[name] = time.monotonic()

            # Give a freshly-started probe a brief window to resolve inline,
            # so the common "everything is fine" case doesn't need to wait for
            # a second poll cycle just to report success.
            done, _pending = await asyncio.wait({task}, timeout=PROBE_INLINE_WAIT)
            if task in done:
                # pop (not del): another poller awaiting the same task may have
                # already consumed and removed it after it resolved.
                self._probe_tasks.pop(name, None)
                return task.result()

            elapsed = time.monotonic() - self._probe_started_at[name]
            if elapsed >= PROBE_SOFT_TIMEOUT:
                return {
                    "service": name,
                    "status": "waiting",
                    "notices": [_waiting_notice(name)],
                }
            # Still under the soft threshold — no notice yet, avoid flapping
            # the UI for probes that are merely a little slow.
            return {"service": name, "status": "checking", "notices": []}

        probed = list(await asyncio.gather(*[_probe(name, url) for name, url in services]))
        # Ourselves first: this is the process rendering the panel, so if it is
        # in the list at all it belongs at the top rather than sorted in among
        # the things it proxies to.
        return [self.get_own_status(), *probed]

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
