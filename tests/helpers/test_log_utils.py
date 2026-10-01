"""Tests for the central log-level policy and the access-log poll filter.

Both exist to stop the services writing ~600 MB/day of log for an idle-ish
box. The properties worth pinning are the ones whose regression would be
invisible until a disk filled: that the default level is INFO (so the
per-frame timing lines stay off unless asked for), and that the access filter
drops *only* successful polls of hot endpoints — never an error, never an
ordinary request.
"""

import calendar
import importlib
import logging

import pytest

import apps.helpers.logUtils as logUtils


def _reload(monkeypatch, **env):
    """Re-import logUtils with a patched environment.

    LOG_LEVEL is read at import time (it is a module constant, not a lookup on
    every log call), so exercising it means reloading the module rather than
    just setting the variable.
    """
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    return importlib.reload(logUtils)


@pytest.fixture(autouse=True)
def _restore_module():
    """Leave the imported module in its default state for other tests."""
    yield
    importlib.reload(logUtils)


class TestLogLevel:
    def test_defaults_to_info(self, monkeypatch):
        mod = _reload(monkeypatch, LOG_LEVEL=None)

        assert mod.LOG_LEVEL == logging.INFO

    def test_debug_is_opt_in(self, monkeypatch):
        mod = _reload(monkeypatch, LOG_LEVEL="DEBUG")

        assert mod.LOG_LEVEL == logging.DEBUG

    def test_level_name_is_case_insensitive(self, monkeypatch):
        mod = _reload(monkeypatch, LOG_LEVEL="warning")

        assert mod.LOG_LEVEL == logging.WARNING

    def test_unknown_level_falls_back_to_info(self, monkeypatch):
        # A typo in an env var must not stop a container from booting.
        mod = _reload(monkeypatch, LOG_LEVEL="LOUD")

        assert mod.LOG_LEVEL == logging.INFO


def _access_record(path: str, status: int) -> logging.LogRecord:
    """An access-log record shaped the way uvicorn emits them."""
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:1234", "GET", path, "1.1", status),
        exc_info=None,
    )


class TestQuietPollAccessFilter:
    def setup_method(self):
        self.filter = logUtils._QuietPollAccessFilter()

    @pytest.mark.parametrize("path", list(logUtils.ACCESS_LOG_QUIET_PATHS))
    def test_drops_successful_polls(self, path):
        assert self.filter.filter(_access_record(path, 200)) is False

    def test_drops_poll_with_query_string(self):
        # Matching is by prefix precisely so query params don't defeat it.
        rec = _access_record("/api/task-configs?uuids=abc&uuids=def", 200)

        assert self.filter.filter(rec) is False

    def test_keeps_ordinary_requests(self):
        assert self.filter.filter(_access_record("/api/images/review", 200)) is True

    @pytest.mark.parametrize("status", [404, 422, 500, 503])
    def test_keeps_failing_polls(self, status):
        # The whole risk of this filter is hiding a poll endpoint that broke.
        rec = _access_record("/api/tasks/instances/get", status)

        assert self.filter.filter(rec) is True

    def test_keeps_records_that_are_not_access_lines(self):
        rec = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, "startup", None, None)

        assert self.filter.filter(rec) is True

    def test_keeps_record_with_unparseable_status(self):
        rec = _access_record("/api/tasks/instances/get", 200)
        rec.args = ("127.0.0.1", "GET", "/api/tasks/instances/get", "1.1", "not-a-status")

        assert self.filter.filter(rec) is True


class TestQuietPollAccessLogs:
    def test_installs_filter_once(self, monkeypatch):
        monkeypatch.delenv("ACCESS_LOG_ALL", raising=False)
        access_logger = logging.getLogger("uvicorn.access")
        for f in list(access_logger.filters):
            access_logger.removeFilter(f)

        logUtils.quiet_poll_access_logs()
        logUtils.quiet_poll_access_logs()

        installed = [f for f in access_logger.filters if isinstance(f, logUtils._QuietPollAccessFilter)]
        assert len(installed) == 1

        for f in list(access_logger.filters):
            access_logger.removeFilter(f)

    def test_access_log_all_opts_out(self, monkeypatch):
        monkeypatch.setenv("ACCESS_LOG_ALL", "1")
        access_logger = logging.getLogger("uvicorn.access")
        for f in list(access_logger.filters):
            access_logger.removeFilter(f)

        logUtils.quiet_poll_access_logs()

        assert not [f for f in access_logger.filters if isinstance(f, logUtils._QuietPollAccessFilter)]


class TestQuietPollAccessFragments:
    """The fragment list exists for endpoints with an id before the hot segment.

    A prefix wide enough to match `/api/tasks/instances/<uuid>/live-preview`
    would also swallow every other route under `/api/tasks/instances/`, so these
    match on a substring instead. The risk is the same as for the prefix list and
    so is the guarantee: a non-2xx is never dropped.
    """

    def setup_method(self):
        self.filter = logUtils._QuietPollAccessFilter()

    @pytest.mark.parametrize("fragment", list(logUtils.ACCESS_LOG_QUIET_FRAGMENTS))
    def test_drops_successful_fetch_with_fragment_mid_path(self, fragment):
        rec = _access_record(f"/api/tasks/instances/9f8e7d6c{fragment}", 200)

        assert self.filter.filter(rec) is False

    def test_drops_live_preview_with_id_and_query(self):
        rec = _access_record("/api/tasks/instances/5c1f/live-preview?source_uuid=abc&since_seq=91&quality=70", 200)

        assert self.filter.filter(rec) is False

    @pytest.mark.parametrize("status", [404, 500])
    def test_keeps_failing_fetch_with_fragment(self, status):
        rec = _access_record("/api/tasks/instances/5c1f/live-preview", status)

        assert self.filter.filter(rec) is True

    def test_sibling_route_under_the_same_parent_is_kept(self):
        # The reason these are fragments and not a `/api/tasks/instances/` prefix.
        rec = _access_record("/api/tasks/instances/5c1f/results", 200)

        assert self.filter.filter(rec) is True


class TestConfigureAppLoggers:
    """`getLogger(__name__)` modules depend on the `apps` parent carrying the level.

    Without it they inherit root, which sits at WARNING, and every INFO and DEBUG
    line they write is discarded with no indication that it happened.
    """

    @pytest.fixture(autouse=True)
    def _restore_levels(self):
        names = ("apps", "sqlalchemy.engine", "onnxruntime")
        saved = {n: logging.getLogger(n).level for n in names}
        yield
        for n, lvl in saved.items():
            logging.getLogger(n).setLevel(lvl)

    def test_module_logger_emits_info_only_after_configuring(self):
        logging.getLogger("apps").setLevel(logging.NOTSET)
        child = logging.getLogger("apps.helpers.recordings.clipDrain")
        child.setLevel(logging.NOTSET)
        assert child.isEnabledFor(logging.INFO) is False

        logUtils.configure_app_loggers()

        assert child.isEnabledFor(logging.INFO) is True

    def test_debug_stays_off_at_the_default_level(self):
        logUtils.configure_app_loggers()

        assert logging.getLogger("apps.db_server.manager").isEnabledFor(logging.DEBUG) is False

    def test_third_party_loggers_are_left_alone(self):
        # Scoping to `apps` rather than root is the whole point: these are verbose
        # at INFO and turning them on is not this function's business.
        for name in ("sqlalchemy.engine", "onnxruntime"):
            logging.getLogger(name).setLevel(logging.NOTSET)

        logUtils.configure_app_loggers()

        for name in ("sqlalchemy.engine", "onnxruntime"):
            assert logging.getLogger(name).isEnabledFor(logging.INFO) is False


class TestLogFormat:
    """The stamp has to be UTC, because nothing the app stores is in local time.

    `watch_events.started_at`, a recording's span and a task heartbeat are all written
    UTC while the container's clock is not, so a log stamped locally reads five hours off
    every row it would be compared against — silently, and only while diagnosing
    something else.
    """

    def _format(self, record_time: float, msecs: float) -> str:
        record = logging.LogRecord(
            name="apps.tasks_server.tasks.StreamWatcherTask",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Car Arrives",
            args=(),
            exc_info=None,
        )
        record.created = record_time
        record.msecs = msecs
        return logUtils.utc_formatter().format(record)

    def test_the_timestamp_is_utc_not_the_container_clock(self):
        # The arrival this format exists to line up with, as `watch_events` stored it:
        # 2026-10-01 13:27:19.399 UTC. `timegm` rather than `mktime` so the expectation
        # is not itself in local time.
        line = self._format(calendar.timegm((2026, 10, 1, 13, 27, 19, 0, 0, 0)), 399)

        assert line.startswith("2026-10-01 13:27:19.399Z "), line
        assert "INFO" in line and "Car Arrives" in line

    def test_a_handler_without_the_formatter_is_stamped(self):
        """`configure_log_format` has to reach handlers that already exist — every
        service's modules call a bare `basicConfig` at import time, long before it runs.
        """
        root = logging.getLogger()
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
        try:
            logUtils.configure_log_format()
            assert handler.formatter is not None
            assert handler.formatter._fmt == logUtils.LOG_FORMAT
        finally:
            root.removeHandler(handler)

    def test_the_access_logger_is_stamped_without_losing_its_own_fields(self):
        """Uvicorn's access formatter builds its line from record attributes, so the stamp
        goes in front of its format rather than replacing it."""

        class AccessLike(logging.Formatter):
            """Stands in for uvicorn's AccessFormatter: reads a field off the record."""

        access = logging.getLogger("uvicorn.access")
        handler = logging.StreamHandler()
        handler.setFormatter(AccessLike("%(client_addr)s - %(status_code)s"))
        access.addHandler(handler)
        try:
            logUtils.configure_log_format()
            fmt = handler.formatter._fmt
            assert "%(client_addr)s - %(status_code)s" in fmt, "the request line survives"
            assert fmt.startswith("%(asctime)s"), fmt
            assert isinstance(handler.formatter, AccessLike), "same class, so it still reads its own fields"
        finally:
            access.removeHandler(handler)

    def test_stamping_twice_does_not_stack_two_timestamps(self):
        """`configure_log_format` runs per service `main.py`, and a reload or a second
        call must not produce a line with the time on it twice."""
        access = logging.getLogger("uvicorn.access")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(client_addr)s"))
        access.addHandler(handler)
        try:
            logUtils.configure_log_format()
            once = handler.formatter._fmt
            logUtils.configure_log_format()
            assert handler.formatter._fmt == once
        finally:
            access.removeHandler(handler)

    def test_a_formatter_that_cannot_be_rebuilt_is_left_alone(self):
        """An unstamped line is a smaller loss than a broken one."""

        class Awkward(logging.Formatter):
            def __init__(self):  # no (fmt, datefmt) to rebuild from
                super().__init__("%(client_addr)s")

        access = logging.getLogger("uvicorn.access")
        handler = logging.StreamHandler()
        original = Awkward()
        handler.setFormatter(original)
        access.addHandler(handler)
        try:
            logUtils.configure_log_format()
            assert handler.formatter is original
        finally:
            access.removeHandler(handler)
