"""Central log-level policy.

``LOG_LEVEL`` governs how verbose the application loggers are. It defaults to
INFO; set it to DEBUG for a debugging session::

    LOG_LEVEL=DEBUG docker compose up

Per-frame and per-request instrumentation — inference and preprocess timings,
query timings, per-image save lines — belongs at DEBUG. Those lines are what
you want when diagnosing a slow pipeline, and they are also enough volume to
fill a rotated log in hours, so they are opt-in rather than deleted.

A module picks up the level one of two ways. Either it names its logger after
the module and lets :func:`configure_app_loggers` set the level on the ``apps``
parent::

    logger = logging.getLogger(__name__)

or it sets its own::

    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger(__file__)
    logger.setLevel(LOG_LEVEL)

A ``getLogger(__name__)`` logger with no call to :func:`configure_app_loggers`
in the service's ``main.py`` inherits the root logger at WARNING and drops
every INFO and DEBUG line it writes.

This governs the *application* loggers only. Uvicorn's access log is a separate
stream with its own switch — see :data:`ACCESS_LOG_QUIET_PATHS` and
:data:`ACCESS_LOG_QUIET_FRAGMENTS`.
"""

import logging
import os
import sys
import time
from typing import Final, Optional, Protocol


class TaskLogFn(Protocol):
    """A task's `log` callable, as injected into helpers that report progress.

    `msg` always reaches the task's in-memory ring buffer, which is what the UI
    renders. `level` governs only whether the line is also written to the
    service's rotated log, so a per-frame caller can stay visible in the UI
    while passing DEBUG to keep itself out of the log file.
    """

    def __call__(self, msg: str, level: int = logging.INFO) -> None: ...


#: Application log level, from the ``LOG_LEVEL`` env var. Accepts any standard
#: level name (DEBUG/INFO/WARNING/ERROR/CRITICAL), case-insensitive. An
#: unrecognised value falls back to INFO rather than raising: a typo in an env
#: var should not stop the container from booting.
_level_name: Final[str] = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
LOG_LEVEL: Final[int] = getattr(logging, _level_name, None) if _level_name else None  # type: ignore[assignment]
if not isinstance(LOG_LEVEL, int):
    LOG_LEVEL = logging.INFO  # type: ignore[misc]


#: Request paths whose *successful* access-log lines are suppressed.
#:
#: Status polls and timer-driven calls. Each is hit several times a second per
#: open browser tab or per running watcher, and a 2xx only records that the poll
#: worked. Non-2xx responses are still logged, so a poll that starts failing
#: stays visible.
#:
#: Matching is by prefix so query strings and path parameters are covered.
ACCESS_LOG_QUIET_PATHS: Final[tuple[str, ...]] = (
    "/api/tasks/instances/get",
    "/api/tasks/instances/status-data",
    "/api/task-instances",
    "/api/task-configs",
    "/status",
    # Timer-driven, not user-driven: the lease renewer and the clip drain each
    # fire per camera every few seconds for as long as a watcher is up.
    "/api/streams/renew",
    "/api/streams/clip",
    # One call per analysed frame.
    "/api/models/detect-unit",
    # Polled by the Watchers tab and by every watcher task on startup.
    "/api/watch-profiles",
    "/api/images/dedup-check",
    "/api/storage/usage",
)

#: Path *fragments* whose successful access-log lines are suppressed.
#:
#: Separate from the prefix list because these endpoints carry an id before the
#: interesting segment (``/api/tasks/instances/{uuid}/live-preview``), so a
#: prefix wide enough to match them would also swallow unrelated routes under
#: the same parent. Same rationale otherwise: each is fetched in bulk by a UI
#: that is scrolling or streaming, and a 200 carries no information.
ACCESS_LOG_QUIET_FRAGMENTS: Final[tuple[str, ...]] = (
    "/live-preview",
    "/thumbnail",
    "/crop",
)


class _QuietPollAccessFilter(logging.Filter):
    """Drops uvicorn access-log records for successful polls of hot endpoints.

    Uvicorn passes the access line's parts as ``record.args`` —
    ``(client_addr, method, full_path, http_version, status_code)`` — so this
    reads the path and status from there rather than re-parsing the formatted
    message, which would be both slower and format-dependent.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 5:
            return True
        path, status = args[2], args[4]
        try:
            # Anything that is not a plain success stays in the log: a poll
            # endpoint that starts 404ing or 500ing is exactly the thing this
            # filter must not hide.
            if int(status) >= 400:
                return True
        except (TypeError, ValueError):
            return True
        if not isinstance(path, str):
            return True
        if path.startswith(ACCESS_LOG_QUIET_PATHS):
            return False
        return not any(fragment in path for fragment in ACCESS_LOG_QUIET_FRAGMENTS)


def configure_app_loggers() -> None:
    """Apply LOG_LEVEL to every logger in the ``apps`` package hierarchy.

    Modules that build their logger with ``getLogger(__name__)`` get a name under
    ``apps``, so setting the level on that one parent governs all of them. Without
    it they inherit the root logger, which sits at WARNING and silently discards
    every INFO and DEBUG line they write.

    Deliberately scoped to ``apps`` rather than the root logger: root would also
    unmute sqlalchemy, onnxruntime and ultralytics, which are verbose at INFO and
    are not ours to turn on.

    Modules using ``getLogger(__file__)`` are outside this hierarchy and set their
    own level; calling this does not affect them.
    """
    logging.getLogger("apps").setLevel(LOG_LEVEL)


#: Log line format. UTC with milliseconds, and the ``Z`` is load-bearing: every
#: timestamp the app *stores* — ``watch_events.started_at``, a recording's span, a task
#: heartbeat — is UTC, while the container's clock is not. A log stamped in local time
#: cannot be lined up against those rows without arithmetic, which is how a diagnosis
#: ends up guessing which layer went quiet.
LOG_FORMAT: Final[str] = "%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s: %(message)s"
LOG_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"

#: Uvicorn's loggers carry their own handlers, so its lines would otherwise be the one
#: unstamped thing in the file — and "did the service actually restart" is read off
#: exactly those.
_STAMPED_LOGGERS: Final[tuple[str, ...]] = ("uvicorn", "uvicorn.error")

#: Stamped by prefixing their own format string instead of replacing it. Uvicorn's access
#: formatter builds its line from record attributes (``client_addr``, ``request_line``)
#: rather than from ``%(message)s``, so a plain formatter here prints neither the request
#: nor an error — it prints nothing useful at all.
_PREFIXED_LOGGERS: Final[tuple[str, ...]] = ("uvicorn.access",)

#: The timestamp on its own, for prefixing a format string that already says the rest.
_STAMP: Final[str] = "%(asctime)s.%(msecs)03dZ"


def utc_formatter() -> logging.Formatter:
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    formatter.converter = time.gmtime
    return formatter


def _prefixed_formatter(existing: logging.Formatter) -> Optional[logging.Formatter]:
    """`existing` rebuilt with a UTC stamp in front, or None if it cannot be.

    Same class, so a subclass that computes its own fields keeps doing so. A formatter
    whose ``__init__`` does not take ``(fmt, datefmt)`` is left alone rather than
    replaced: an unstamped line is a smaller loss than a broken one.
    """
    fmt = getattr(existing, "_fmt", None)
    if not fmt or fmt.startswith(_STAMP):
        return None
    try:
        rebuilt = type(existing)(f"{_STAMP} {fmt}", datefmt=LOG_DATEFMT)
    except Exception:
        return None
    rebuilt.converter = time.gmtime
    return rebuilt


def configure_log_format() -> None:
    """Stamp every stream handler with a UTC timestamp.

    Set on the handlers rather than passed to ``basicConfig``: the first ``basicConfig``
    call in a process wins, and dozens of modules make a bare one at import time, so a
    format given here would survive or vanish depending on which module imported first.

    Safe to call more than once — a format already carrying the stamp is left as it is.
    """
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(logging.StreamHandler(sys.stdout))
    for logger in (root, *(logging.getLogger(name) for name in _STAMPED_LOGGERS)):
        for handler in logger.handlers:
            handler.setFormatter(utc_formatter())
    for logger in (logging.getLogger(name) for name in _PREFIXED_LOGGERS):
        for handler in logger.handlers:
            if handler.formatter is None:
                handler.setFormatter(utc_formatter())
                continue
            rebuilt = _prefixed_formatter(handler.formatter)
            if rebuilt is not None:
                handler.setFormatter(rebuilt)


def quiet_poll_access_logs() -> None:
    """Install the poll filter on uvicorn's access logger.

    Safe to call more than once (the filter is only added if absent) and safe
    to call when uvicorn is not the server — it just attaches a filter to a
    logger nobody writes to. Disable by setting ``ACCESS_LOG_ALL=1``.
    """
    if os.environ.get("ACCESS_LOG_ALL", "").strip() not in ("", "0", "false", "False"):
        return
    access_logger = logging.getLogger("uvicorn.access")
    if any(isinstance(f, _QuietPollAccessFilter) for f in access_logger.filters):
        return
    access_logger.addFilter(_QuietPollAccessFilter())
