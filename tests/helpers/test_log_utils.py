"""Tests for the central log-level policy and the access-log poll filter.

Both exist to stop the services writing ~600 MB/day of log for an idle-ish
box. The properties worth pinning are the ones whose regression would be
invisible until a disk filled: that the default level is INFO (so the
per-frame timing lines stay off unless asked for), and that the access filter
drops *only* successful polls of hot endpoints — never an error, never an
ordinary request.
"""

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
