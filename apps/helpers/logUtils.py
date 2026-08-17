"""Central log-level policy.

Every module in this repo builds its logger the same way::

    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger(__file__)
    logger.setLevel(LOG_LEVEL)

The level used to be hard-coded to ``logging.DEBUG`` in all ~38 of those call
sites, which meant the only way to turn verbosity down was to edit source. In
a running deployment that cost real disk: the per-frame timing lines in the
inference and task paths (``preprocess=..ms onnx=..ms``, ``Inference HTTP
round-trip model=..``) accounted for **75% of inference_server's log bytes and
55% of tasks_server's**, ~380 MB/day between them, on a box doing nothing
unusual.

Those lines are worth keeping — they are exactly what you want when
diagnosing a slow pipeline — so the fix is to make them opt-in rather than to
delete them. ``LOG_LEVEL`` defaults to INFO and can be set to DEBUG for a
debugging session::

    LOG_LEVEL=DEBUG docker compose up

Note this governs the *application* loggers only. Uvicorn's access log is a
separate stream with its own switch — see ``ACCESS_LOG_QUIET_PATHS``.
"""

import logging
import os
from typing import Final

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
#: These are the UI's status-poll endpoints. They are hit a few times a second
#: per open browser tab and each one logs a line, which is 100% of db_server's
#: log volume and 44% of tasks_server's — all of it recording that a poll
#: succeeded, which is the boring case. Non-2xx responses are still logged, so
#: a poll that starts failing remains visible.
#:
#: Matching is by prefix so query strings and path parameters are covered.
ACCESS_LOG_QUIET_PATHS: Final[tuple[str, ...]] = (
    "/api/tasks/instances/get",
    "/api/tasks/instances/status-data",
    "/api/task-instances",
    "/api/task-configs",
    "/status",
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
        return not path.startswith(ACCESS_LOG_QUIET_PATHS)


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
