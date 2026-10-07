import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from apps import APPS, APPS_CONFIG
from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.reverseProxy import ReverseProxy

from .manager import Manager

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


IMAGE_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/images",
)
LABEL_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/labels",
)
SOURCES_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/sources",
)
TAG_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/tags",
)
TASKS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.TASKS_SERVER].host,
    port=APPS_CONFIG[APPS.TASKS_SERVER].port,
    path="/api/tasks",
)
# Answers to the clip-repair suggestion notice. On the tasks server because the watcher
# that found the damage is also what holds the finding in memory.
CLIP_REPAIR_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.TASKS_SERVER].host,
    port=APPS_CONFIG[APPS.TASKS_SERVER].port,
    path="/api/clip-repair",
)
DATASETS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/datasets",
)
MODEL_CONFIGS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/model-configs",
)
TRAINING_CONFIGS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/training-configs",
)
FILTER_CHAINS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/filter-chains",
)
TRAINING_COLLECTORS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/training-collectors",
)
PREVIEW_RUNS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/preview-runs",
)
MODEL_GROUPS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/model-groups",
)
WATCH_PROFILES_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/watch-profiles",
)
WATCH_EVENTS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/watch-events",
)
EVENT_BLIPS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/event-blips",
)
RECORDINGS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/recordings",
)
SETTINGS_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/settings",
)
STORAGE_PROXY = ReverseProxy(
    host=APPS_CONFIG[APPS.DB_SERVER].host,
    port=APPS_CONFIG[APPS.DB_SERVER].port,
    path="/api/storage",
)


class TestServerPayload(BaseModel):
    url: str
    server_type: str  # "inference_server" or "training_server"


class TestMqttPayload(BaseModel):
    """Broker settings to probe. Snake_case like TestServerPayload above — these two
    Settings-only payloads predate nothing and are hand-written, not `Payload`
    subclasses, so they don't carry the camelCase alias generator."""

    host: str
    port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    tls: bool = False


# Task SSE polling cadence. The stream is a poll loop behind a push-shaped API, so
# every connected browser costs a full image_tagging → tasks → db round trip per tick,
# forever. 2 Hz is only worth paying while something is actually moving: a task that
# is running or about to run changes its progress between ticks. Nothing live means
# nothing to animate, and the next change is user-initiated, so a slower tick just
# costs a little start-up latency on a card that was idle anyway.
TASKS_STREAM_ACTIVE_POLL_S = 0.5
TASKS_STREAM_IDLE_POLL_S = 3.0
_TASKS_STREAM_ACTIVE_STATUSES = frozenset({"new", "pending", "running"})


def _tasks_poll_interval(data: dict) -> float:
    """Poll fast while any task is live, slowly when the task list is at rest.

    Paused/completed/error instances are deliberately not "live" — they hold still
    until the user acts, and a user action re-renders from its own response.
    """
    try:
        tasks = (data or {}).get("tasks") or {}
        for entry in tasks.values():
            status = ((entry or {}).get("instance") or {}).get("status")
            if status in _TASKS_STREAM_ACTIVE_STATUSES:
                return TASKS_STREAM_ACTIVE_POLL_S
    except (AttributeError, TypeError):
        # A shape we don't recognise shouldn't silently pin the loop at 2 Hz.
        return TASKS_STREAM_IDLE_POLL_S
    return TASKS_STREAM_IDLE_POLL_S


class RevalidatedStaticFiles(StaticFiles):
    """StaticFiles that makes the browser revalidate every asset.

    The SPA is served as unbundled, unversioned ES modules: `main.js` statically
    imports `runs/main.js`, which imports named exports from `ui/tasks.js`, and
    every one of those is fetched by its own bare URL. Starlette sends etag and
    last-modified but no cache-control, which leaves the browser free to apply
    *heuristic* freshness — and it computes that per file, from each file's own
    last-modified. So the modules in one page load are not guaranteed to come
    from the same revision of the tree.

    That is not a theoretical problem. Adding an export to `ui/tasks.js` and a
    matching import in `runs/main.js` is enough: a browser that revalidates the
    importer but serves the imported module from cache raises

        The requested module '.../ui/tasks.js' does not provide an export
        named 'countAdvancingTasks'

    at module-evaluation time. `app/main.js` imports `runs/main.js` statically,
    so that kills the entire SPA — the tab shell is in index.html and still
    paints, which makes it look like one page rendered nothing rather than like
    a total failure to boot.

    `no-cache` does not mean "do not store": the copy is kept and the etag still
    turns the revalidation into a 304, so this costs a conditional request per
    asset, not a re-download. Correctness over a handful of 304s — the failure
    it prevents is invisible until someone opens the console.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


#: Name deliberately namespaced — this cookie only ever means "presented the shared
#: token once", not a real session, and the prefix keeps it from colliding with
#: anything a reverse-proxied backend might someday set.
AUTH_COOKIE_NAME = "pv_auth"

#: `/login` itself must stay reachable while logged out, or a wrong token becomes a
#: permanent lockout with no way to even see the form again.
_AUTH_EXEMPT_PATHS = frozenset({"/login"})


class AuthGate:
    """ASGI middleware gating every request behind Settings → Security's shared token.

    Deliberately **not** `starlette.middleware.base.BaseHTTPMiddleware`. That class
    reconstructs every response — including the `Range`/`206` responses `ReverseProxy`
    streams straight through from `db_server` for recording playback — as its own
    re-streamed `StreamingResponse`. That is exactly the class of bug
    `HA_INTEGRATION_PLAN.md` WP-4 warns about for clip scrubbing ("omit this and clips
    play but never scrub"), and there is no reason to risk it here: an authenticated
    request below only ever needs to be let through, never rewritten. A pure ASGI
    middleware does that by calling straight through to `self._app`, untouched.
    """

    def __init__(self, app, manager: Manager):
        self._app = app
        self._manager = manager

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not self._manager.require_auth:
            await self._app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        if request.url.path in _AUTH_EXEMPT_PATHS:
            await self._app(scope, receive, send)
            return

        if self._manager.check_token(self._extract_token(request)) or self._manager.check_session(
            request.cookies.get(AUTH_COOKIE_NAME)
        ):
            await self._app(scope, receive, send)
            return

        # A `<video>`/`<img>` tag and HA's proxy calls can't follow a redirect into an
        # HTML login form and would just render it as a broken image — only send one to
        # something that actually asked for a page.
        if "text/html" in request.headers.get("accept", ""):
            next_q = quote(request.url.path, safe="")
            response = RedirectResponse(url=f"/login?next={next_q}", status_code=303)
        else:
            response = JSONResponse(content={"detail": "Unauthorized"}, status_code=401)
        await response(scope, receive, send)

    @staticmethod
    def _extract_token(request: Request) -> Optional[str]:
        """Pull the API token out of a request.

        The cookie is deliberately not consulted here — it carries a session value,
        not the token, and the two are checked separately by the caller.
        """
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:]
        return request.query_params.get("token")


class WebApp:
    SUCCESS_KEY = "success"
    FAILURE_KEY = "failure"

    def __init__(self, app_name: str, manager: Manager):
        """Initialize the WebApp with the application name"""
        self._app_name: str = app_name
        self._app: FastAPI = FastAPI(lifespan=self._lifespan)
        self._templates: Jinja2Templates = Jinja2Templates(directory=os.path.join(SCRIPT_DIR, "templates"))
        self._manager: Manager = manager

        # Wraps every route and both static mounts below — see AuthGate's docstring
        # for why this is a plain ASGI middleware rather than BaseHTTPMiddleware.
        self._app.add_middleware(AuthGate, manager=self._manager)

        # Mount static files
        self._app.mount("/static", RevalidatedStaticFiles(directory="/app/static", check_dir=False), name="static")
        self._app.mount(
            "/app-static",
            RevalidatedStaticFiles(directory=os.path.join(SCRIPT_DIR, "static"), check_dir=False),
            name="app-static",
        )

        # Register routes
        self._register_routes()

    def app(self) -> FastAPI:
        """Return the FastAPI application instance"""
        return self._app

    @staticmethod
    def _safe_next(candidate: str) -> str:
        """Confine a post-login redirect target to this app.

        `//evil.example.com/x` is not caught by a bare `startswith("/")` check —
        browsers treat a leading `//` as protocol-relative and will happily follow it
        off-site. `next` only ever needs to survive round-tripping through our own
        `<form>` and query string, so anything else just falls back to `/`.
        """
        if candidate and candidate.startswith("/") and not candidate.startswith("//"):
            return candidate
        return "/"

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        logging.info("Application is starting up...")
        await self._manager.start()

        yield

        logging.info("Application is shutting down...")

    def _register_routes(self):
        """Register all routes for the application"""

        @self._app.get("/", response_class=HTMLResponse)
        async def get_home(request: Request):
            """Render the home page"""
            return self._templates.TemplateResponse(request=request, name="index.html")

        @self._app.get("/login", response_class=HTMLResponse)
        async def login_form(request: Request, next: str = "/", error: Optional[str] = None):
            """Render the login form.

            Reachable even with auth off — a direct visit just doesn't matter then,
            since AuthGate never redirects here in the first place.
            """
            return self._templates.TemplateResponse(
                request=request,
                name="login.html",
                context={
                    "next": self._safe_next(next),
                    "error": error,
                    "password_mode": self._manager.password_enabled,
                },
            )

        @self._app.post("/login")
        async def login_submit(request: Request):
            """Check the submitted credential and, if it matches, set the auth cookie.

            `Manager.check_login` decides which secret is wanted — the password when
            one is configured, the API token otherwise. The cookie never carries
            either: it holds the session value below, so a stolen cookie cannot be
            replayed as a Bearer token against the API.
            """
            form = await request.form()
            secret = str(form.get("token", ""))
            next_path = self._safe_next(str(form.get("next", "/")))
            client = request.client.host if request.client else ""

            # Checked before the credential, which is the point: refusing here is what
            # keeps a guesser from spending an scrypt hash per request on the one path
            # AuthGate exempts.
            retry_after = self._manager.login_retry_after(client)
            if retry_after:
                # Rendered rather than redirected. A 303 to the form would answer 200
                # and tell the browser nothing, and `Retry-After` is the only part of
                # this a non-browser caller can act on.
                response = self._templates.TemplateResponse(
                    request=request,
                    name="login.html",
                    context={
                        "next": next_path,
                        "error": "rate",
                        "password_mode": self._manager.password_enabled,
                    },
                    status_code=429,
                )
                response.headers["Retry-After"] = str(max(1, int(retry_after) + 1))
                return response

            ok = await self._manager.check_login(secret)
            self._manager.note_login_result(client, ok)
            if not ok:
                return RedirectResponse(url=f"/login?next={quote(next_path, safe='')}&error=1", status_code=303)

            response = RedirectResponse(url=next_path, status_code=303)
            response.set_cookie(
                AUTH_COOKIE_NAME,
                self._manager.session_cookie_value(),
                max_age=60 * 60 * 24 * 30,
                httponly=True,
                samesite="lax",
                secure=request.url.scheme == "https",
                path="/",
            )
            return response

        @self._app.get("/logout")
        async def logout(request: Request):
            """Clear the auth cookie and bounce back to the login form.

            GET rather than a form POST: the only thing it can do is discard the
            caller's own cookie, so the usual "state changes need POST" reasoning
            doesn't buy anything here, and a plain link is simpler for Settings to
            offer than a form.
            """
            response = RedirectResponse(url="/login", status_code=303)
            response.delete_cookie(AUTH_COOKIE_NAME, path="/")
            return response

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
                response_data = {"status": WebApp.SUCCESS_KEY, "config": server_config}
            except Exception as e:
                response_data = {"status": WebApp.FAILURE_KEY, "message": str(e)}
            return JSONResponse(content=response_data)

        @self._app.api_route("/api/images/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
        async def proxy_images(request: Request, path: str):
            return await IMAGE_PROXY.handle_request(path, request)

        @self._app.api_route("/api/labels/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
        async def proxy_labels(request: Request, path: str):
            return await LABEL_PROXY.handle_request(path, request)

        @self._app.api_route("/api/models/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
        async def proxy_models(request: Request, path: str):
            return await self._manager.inference_proxy.handle_request(path, request)

        @self._app.api_route(
            "/api/sources/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_sources(request: Request, path: str):
            return await SOURCES_PROXY.handle_request(path, request)

        @self._app.api_route("/api/tags/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
        async def proxy_tags(request: Request, path: str):
            return await TAG_PROXY.handle_request(path, request)

        _SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

        @self._app.get("/api/tasks/stream")
        async def tasks_stream(request: Request) -> StreamingResponse:
            """SSE stream — pushes a data frame whenever task state changes."""

            async def _gen():
                last_json: str | None = None
                last_hb = asyncio.get_running_loop().time()
                try:
                    while True:
                        try:
                            data = await self._manager.get_tasks_for_stream()
                            j = json.dumps(data)
                        except Exception:
                            data = {}
                            j = "{}"
                        now = asyncio.get_running_loop().time()
                        if j != last_json:
                            last_json = j
                            last_hb = now
                            yield f"data: {j}\n\n"
                        elif now - last_hb > 15:
                            last_hb = now
                            yield ":\n\n"  # keep-alive comment
                        await asyncio.sleep(_tasks_poll_interval(data))
                except asyncio.CancelledError:
                    pass

            return StreamingResponse(_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)

        @self._app.api_route("/api/tasks/{path:path}", methods=["GET", "POST", "PATCH", "DELETE"])
        async def proxy_tasks(request: Request, path: str):
            return await TASKS_PROXY.handle_request(path, request)

        @self._app.api_route("/api/clip-repair/{path:path}", methods=["POST"])
        async def proxy_clip_repair(request: Request, path: str):
            return await CLIP_REPAIR_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/datasets/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_datasets(request: Request, path: str):
            return await DATASETS_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/model-configs/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_model_configs(request: Request, path: str):
            return await MODEL_CONFIGS_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/training-configs/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_training_configs(request: Request, path: str):
            return await TRAINING_CONFIGS_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/filter-chains/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_filter_chains(request: Request, path: str):
            return await FILTER_CHAINS_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/training-collectors/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_training_collectors(request: Request, path: str):
            return await TRAINING_COLLECTORS_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/preview-runs/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_preview_runs(request: Request, path: str):
            return await PREVIEW_RUNS_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/model-groups/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_model_groups(request: Request, path: str):
            return await MODEL_GROUPS_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/watch-profiles/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_watch_profiles(request: Request, path: str):
            return await WATCH_PROFILES_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/watch-events/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_watch_events(request: Request, path: str):
            # Was GET-only, which made the review page's delete button a 405 in the
            # browser while db_server's endpoint sat there answering 200 to anything
            # that reached it directly. This route is the SPA's only way to db_server,
            # so the method list here is part of the API's surface, not a detail —
            # keep it in step with what `/api/watch-events*` actually exposes.
            return await WATCH_EVENTS_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/event-blips/{path:path}",
            methods=["GET", "POST"],
        )
        async def proxy_event_blips(request: Request, path: str):
            # GET lists them, POST covers `/replace` and `/states`. No DELETE, matching
            # db_server: a blip is never deleted on its own — it goes with its event, or
            # it expires when its clip is swept.
            return await EVENT_BLIPS_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/recordings/{path:path}",
            methods=["GET", "DELETE"],
        )
        async def proxy_recordings(request: Request, path: str):
            # Clip *files* go through here too, and must: the SPA talks only to this
            # port. That was once untrue — ReverseProxy buffered whole responses, so a
            # video was read into memory and lost range support — but it streams now and
            # forwards Range up and 206/Content-Range back, which is what makes the
            # event modal's scrubber work. See TestRangeRequests in
            # tests/helpers/test_reverse_proxy.py, which pins that.
            return await RECORDINGS_PROXY.handle_request(path, request)

        @self._app.api_route("/api/settings/{path:path}", methods=["GET", "PUT"])
        async def proxy_settings(request: Request, path: str):
            return await SETTINGS_PROXY.handle_request(path, request)

        @self._app.api_route("/api/storage/{path:path}", methods=["GET", "POST", "DELETE"])
        async def proxy_storage(request: Request, path: str):
            return await STORAGE_PROXY.handle_request(path, request)

        @self._app.api_route(
            "/api/training/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def proxy_training(request: Request, path: str):
            return await self._manager.training_proxy.handle_request(path, request)

        @self._app.get("/status", operation_id="image_tagging_server_status", response_class=JSONResponse)
        async def image_tagging_server_status() -> JSONResponse:
            """This service's own health, in the same shape every other service uses.

            Every other service answers /status; this one 404'd, so the process
            hosting the whole frontend was the one thing no health check could
            see. Anything pointed at the stack — a uptime monitor, a container
            healthcheck, or the Operations panel — got a 404 here and had to
            treat "the frontend is up" as unknowable.
            """
            return JSONResponse(content=self._manager.get_own_status())

        @self._app.get("/api/service-status", response_class=JSONResponse)
        async def service_status() -> JSONResponse:
            """Probe all backend services and return their status."""
            try:
                results = await self._manager.get_all_service_status()
                return JSONResponse(content={"status": WebApp.SUCCESS_KEY, "services": results})
            except Exception as e:
                logger.error(e, exc_info=True)
                return JSONResponse(content={"status": WebApp.FAILURE_KEY, "message": str(e)})

        @self._app.get("/api/notices", response_class=JSONResponse)
        async def get_notices() -> JSONResponse:
            """Return aggregated advisory notices from all backend services."""
            try:
                notices = await self._manager.get_notices()
                return JSONResponse(content={"notices": notices})
            except Exception as e:
                logger.error(e, exc_info=True)
                return JSONResponse(content={"notices": []})

        @self._app.get("/api/notices/stream")
        async def notices_stream(request: Request) -> StreamingResponse:
            """SSE stream — pushes a data frame whenever the aggregated notice list changes."""

            async def _gen():
                last_json: str | None = None
                last_hb = asyncio.get_running_loop().time()
                try:
                    while True:
                        try:
                            notices = await self._manager.get_raw_notices()
                            j = json.dumps(notices)
                        except Exception:
                            j = "[]"
                        now = asyncio.get_running_loop().time()
                        if j != last_json:
                            last_json = j
                            last_hb = now
                            yield f"data: {j}\n\n"
                        elif now - last_hb > 15:
                            last_hb = now
                            yield ":\n\n"  # keep-alive comment
                        await asyncio.sleep(2)
                except asyncio.CancelledError:
                    pass

            return StreamingResponse(_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)

        @self._app.post("/api/test-server", response_class=JSONResponse)
        async def test_server(payload: TestServerPayload) -> JSONResponse:
            """Test connectivity to an inference or training server URL."""
            result = await self._manager.test_server(payload.url, payload.server_type)
            return JSONResponse(content=result)

        @self._app.post("/api/test-mqtt", response_class=JSONResponse)
        async def test_mqtt(payload: TestMqttPayload) -> JSONResponse:
            """Test connectivity to an MQTT broker with the given credentials."""
            result = await self._manager.test_mqtt_broker(
                host=payload.host,
                port=payload.port,
                username=payload.username,
                password=payload.password,
                tls=payload.tls,
            )
            return JSONResponse(content=result)

        @self._app.post("/api/reload-server-config", response_class=JSONResponse)
        async def reload_server_config() -> JSONResponse:
            """Reload server URL overrides from DB settings and rebuild proxies."""
            try:
                await self._manager.reload_url_overrides()
                config = await self._manager.get_server_config()
                return JSONResponse(content={"status": WebApp.SUCCESS_KEY, "config": config})
            except Exception as e:
                logger.error(e, exc_info=True)
                return JSONResponse(content={"status": WebApp.FAILURE_KEY, "message": str(e)})

        @self._app.get("/api/auth-config", response_class=JSONResponse)
        async def get_auth_config() -> JSONResponse:
            """Return the Security panel's state for Settings → Security.

            The password hash never leaves the server — the panel only needs to know
            whether one has been set, so it can offer "leave blank to keep it".
            """
            config = self._manager.auth_config
            return JSONResponse(
                content={
                    "status": WebApp.SUCCESS_KEY,
                    "requireAuth": config.require_auth,
                    "token": config.token,
                    "passwordEnabled": config.password_enabled,
                    "passwordSet": bool(config.password_hash),
                }
            )

        @self._app.post("/api/auth-config", response_class=JSONResponse)
        async def save_auth_config(request: Request) -> JSONResponse:
            """Save Settings → Security and apply it without a restart.

            The password arrives as plaintext exactly once, here, and is hashed before
            it is written — which is why the panel posts to this route instead of
            PUTting the settings blob to db_server itself. `AuthGate` reads `Manager`
            state live on every request, so nothing else needs to change once this
            returns.
            """
            try:
                body = await request.json()
                config = await self._manager.save_auth_config(
                    require_auth=bool(body.get("requireAuth")),
                    token=str(body.get("token") or ""),
                    password_enabled=bool(body.get("passwordEnabled")),
                    password=str(body.get("password") or ""),
                )
                return JSONResponse(
                    content={
                        "status": WebApp.SUCCESS_KEY,
                        "requireAuth": config.require_auth,
                        "passwordEnabled": config.password_enabled,
                        "passwordSet": bool(config.password_hash),
                    }
                )
            except ValueError as e:
                return JSONResponse(content={"status": WebApp.FAILURE_KEY, "message": str(e)}, status_code=400)
            except Exception as e:
                logger.error(e, exc_info=True)
                return JSONResponse(content={"status": WebApp.FAILURE_KEY, "message": str(e)})
