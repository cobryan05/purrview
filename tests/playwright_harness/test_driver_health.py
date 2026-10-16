"""Telling a dead Playwright driver apart from an ordinary closed page.

Pure logic, so it lives here rather than in the browser suite — nothing about this needs
a browser, and `playwright_tests/` is for what genuinely does.

The predicate behind the run-aborting guard is the thing worth protecting, and it is
worth protecting in **both** directions, because the two failure modes are opposite and
both are silent:

* Too narrow and the cascade comes back: one dead driver, 1100 identical fixture errors,
  twelve wasted minutes and a 332k-line log that names no cause (CI run 38898595092).
* Too *wide* and healthy runs abort. Closing a page is something this suite does after
  every single test, and the same exception type carries both messages — so a matcher
  that reached for `TargetClosedError`, or for the word "closed", would kill a green run
  at the first ordinary teardown.

The second is the worse failure, which is why the negative cases outnumber the positive
ones here.
"""

from playwright_tests.driver_health import DEAD_DRIVER_MARKERS, abort_banner, is_dead_driver, memory_snapshot


class _TargetClosedError(Exception):
    """Stands in for `playwright._impl._errors.TargetClosedError`.

    The real class is matched on its *message*, never its type — which is the whole
    point of the predicate — so the test does not need Playwright installed to state
    what it means.
    """


class TestADeadDriverIsRecognised:
    def test_the_message_ci_reported_1100_times(self):
        exc = _TargetClosedError("Playwright connection closed")
        assert is_dead_driver(exc)

    def test_the_transport_level_wording_too(self):
        assert is_dead_driver(_TargetClosedError("Connection closed while reading from the driver"))

    def test_the_marker_is_found_inside_a_longer_traceback_message(self):
        """Playwright wraps its errors with call logs and context, so the marker arrives
        embedded rather than as the whole string."""
        exc = _TargetClosedError("Page.close: Playwright connection closed\nCall log:\n  - waiting for locator('body')")
        assert is_dead_driver(exc)

    def test_every_declared_marker_is_actually_matched(self):
        """A marker added to the tuple but written so it cannot match — a typo, a
        regex-ism in a plain substring check — would look configured and do nothing."""
        for marker in DEAD_DRIVER_MARKERS:
            assert is_dead_driver(_TargetClosedError(marker)), marker


class TestAnOrdinaryClosureIsNot:
    """The direction that matters more: these happen on purpose, constantly."""

    def test_a_closed_page_is_not_a_dead_driver(self):
        """`p.close()` runs after every test in the suite, and a test that navigated
        away meets this message routinely."""
        exc = _TargetClosedError("Target page, context or browser has been closed")
        assert not is_dead_driver(exc)

    def test_a_bare_target_closed_is_not_a_dead_driver(self):
        assert not is_dead_driver(_TargetClosedError("Target closed"))

    def test_a_closed_context_is_not_a_dead_driver(self):
        """Module-scoped contexts close between files all run long."""
        assert not is_dead_driver(_TargetClosedError("BrowserContext.close: Target closed"))

    def test_an_ordinary_timeout_is_not_a_dead_driver(self):
        exc = _TargetClosedError("Timeout 60000ms exceeded waiting for locator('#thing')")
        assert not is_dead_driver(exc)

    def test_the_word_closed_alone_does_not_trip_it(self):
        """The cheap matcher someone would reach for, and the one that would abort a
        green run at its first teardown."""
        assert not is_dead_driver(_TargetClosedError("the modal was closed"))

    def test_an_unrelated_error_is_not_a_dead_driver(self):
        assert not is_dead_driver(ValueError("something else entirely"))


class TestTheDiagnosticsCannotMakeThingsWorse:
    """Both of these run on the way out of an already-failing run. A diagnostic that
    raises would replace the explanation with a second, unrelated failure."""

    def test_the_memory_snapshot_returns_something_and_does_not_raise(self):
        snapshot = memory_snapshot()
        assert isinstance(snapshot, str) and snapshot

    def test_the_memory_snapshot_survives_an_unreadable_proc(self, monkeypatch):
        """It is read from /proc by hand precisely because the box may be too short of
        memory to fork a `ps`; an unreadable /proc must still yield a line."""
        monkeypatch.setattr(
            "playwright_tests.driver_health.Path",
            lambda *a, **k: (_ for _ in ()).throw(OSError("no /proc here")),
        )
        assert isinstance(memory_snapshot(), str)

    def test_the_banner_names_where_it_happened_and_carries_the_error(self):
        """The banner is the one thing that has to survive in a log nobody reads
        twice, so it has to say both."""
        banner = abort_banner(
            "page setup (shared_context.new_page)", _TargetClosedError("Playwright connection closed")
        )
        assert "page setup (shared_context.new_page)" in banner
        assert "Playwright connection closed" in banner
        assert "ABORTING RUN" in banner
        assert "FLAKY_TESTS.md" in banner, "the reader needs somewhere to go"
