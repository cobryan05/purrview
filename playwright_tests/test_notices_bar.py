"""
The notices bar (`ui/noticesBar.js`) — the fixed strip at the bottom of the app.

Every backend service that fails a probe contributes its own "is not responding"
notice, so the outage case is precisely the case that produces the most rows.
The bar is fixed to the bottom above everything else, so unbounded growth there
eats the app rather than scrolling out of the way.

These feed the real `#notices-bar` (mounted by main.js at startup, with the real
CSS applied) through a stubbed EventSource.

Run with:
    pytest playwright_tests/test_notices_bar.py -p no:asyncio -v
"""

import json

from playwright.sync_api import Page, expect

# Records the notices stream so a test can push frames into the mounted bar.
FAKE_EVENT_SOURCE = """
  class FakeEventSource {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSED = 2;
    static notices = null;

    constructor(url) {
      this.url = url;
      this.readyState = FakeEventSource.OPEN;
      this.onmessage = null;
      this.onerror = null;
      if (String(url).includes('/api/notices/stream')) FakeEventSource.notices = this;
    }
    close() { this.readyState = FakeEventSource.CLOSED; }
    send(payload) { this.onmessage?.({ data: JSON.stringify(payload) }); }
  }
  window.FakeEventSource = FakeEventSource;
  window.EventSource = FakeEventSource;
"""

SERVICES = [
    "db_server",
    "inference_server",
    "tasks_server",
    "training_server",
    "relay_buffer_server",
]

# What a total backend outage actually produces: one warning per service.
OUTAGE = [
    {"key": f"{name}_unreachable", "level": "warning", "message": f"{name} is not responding"} for name in SERVICES
]


def _prepare(page: Page, base_url: str):
    page.add_init_script(FAKE_EVENT_SOURCE)
    page.goto(base_url)
    page.wait_for_load_state("networkidle")


def _push(page: Page, notices: list):
    page.evaluate("(n) => FakeEventSource.notices.send(n)", notices)


def _bar_height(page: Page) -> float:
    box = page.locator("#notices-bar").bounding_box()
    return box["height"] if box else 0.0


def _viewport_height(page: Page) -> float:
    return page.evaluate("() => window.innerHeight")


class TestNoticesBarBounds:
    def test_no_notices_hides_the_bar_entirely(self, page: Page, base_url: str):
        _prepare(page, base_url)
        _push(page, [])
        expect(page.locator("#notices-bar")).to_be_hidden()

    def test_an_outage_does_not_take_over_the_screen(self, page: Page, base_url: str):
        """One notice per unreachable service, each an unbounded row: this is the
        report — the bars 'end up taking up all of the screen'."""
        _prepare(page, base_url)
        _push(page, OUTAGE)
        expect(page.locator("#notices-bar")).to_be_visible()
        height = _bar_height(page)
        assert height < _viewport_height(page) * 0.4, f"the notices bar is {height}px tall"

    def test_a_long_list_scrolls_inside_the_bar(self, page: Page, base_url: str):
        """The rows have to go somewhere — capping the height without scrolling
        them would just hide the ones that matter."""
        _prepare(page, base_url)
        _push(page, OUTAGE * 6)
        height = _bar_height(page)
        assert height < _viewport_height(page) * 0.4, f"the notices bar is {height}px tall"
        overflow = page.locator("#notices-bar .notices-bar__list").evaluate("e => e.scrollHeight > e.clientHeight")
        assert overflow, "the list was capped but never became scrollable"

    def test_a_single_notice_stays_compact(self, page: Page, base_url: str):
        """The common case is one advisory line; it must not cost a header plus a
        list plus a scroller."""
        _prepare(page, base_url)
        _push(page, [{"key": "hash", "level": "info", "message": "hash index rebuilding"}])
        assert _bar_height(page) < 80, f"one notice draws {_bar_height(page)}px of bar"


class TestNoticesBarCollapse:
    def test_the_bar_can_be_collapsed(self, page: Page, base_url: str):
        _prepare(page, base_url)
        _push(page, OUTAGE)
        expanded = _bar_height(page)

        page.locator("#notices-bar .notices-bar__toggle").click()
        expect(page.locator("#notices-bar .notices-bar__list")).to_be_hidden()
        collapsed = _bar_height(page)
        assert collapsed < expanded, f"collapsing changed nothing: {collapsed} vs {expanded}"
        assert collapsed < 50, f"the collapsed bar is still {collapsed}px tall"

    def test_collapsing_survives_the_next_frame(self, page: Page, base_url: str):
        """render() rebuilds the bar from scratch on every frame, so collapse state
        held in the DOM would be undone by the next poll — and during an outage the
        next poll is always seconds away."""
        _prepare(page, base_url)
        _push(page, OUTAGE)
        page.locator("#notices-bar .notices-bar__toggle").click()

        _push(page, OUTAGE + [{"key": "x", "level": "error", "message": "another thing broke"}])
        expect(page.locator("#notices-bar .notices-bar__list")).to_be_hidden()
        assert _bar_height(page) < 50, "a new notice re-expanded a bar the user had collapsed"

    def test_collapsing_survives_a_reload(self, page: Page, base_url: str):
        """An outage outlives a page load; re-collapsing on every reload is the
        same annoyance as not being able to collapse at all."""
        _prepare(page, base_url)
        _push(page, OUTAGE)
        page.locator("#notices-bar .notices-bar__toggle").click()
        expect(page.locator("#notices-bar .notices-bar__list")).to_be_hidden()

        page.reload()
        page.wait_for_load_state("networkidle")
        _push(page, OUTAGE)
        expect(page.locator("#notices-bar .notices-bar__list")).to_be_hidden()

    def test_expanding_again_restores_the_list(self, page: Page, base_url: str):
        _prepare(page, base_url)
        _push(page, OUTAGE)
        toggle = page.locator("#notices-bar .notices-bar__toggle")
        toggle.click()
        expect(page.locator("#notices-bar .notices-bar__list")).to_be_hidden()
        toggle.click()
        expect(page.locator("#notices-bar .notices-bar__list")).to_be_visible()
        assert page.locator(".notices-bar__item").count() == len(OUTAGE)

    def test_the_collapsed_bar_still_says_what_is_wrong(self, page: Page, base_url: str):
        """Collapsed to one line, it has to remain a status bar — a bare chevron
        would just be a hidden outage."""
        _prepare(page, base_url)
        _push(page, OUTAGE)
        page.locator("#notices-bar .notices-bar__toggle").click()
        summary = page.locator("#notices-bar .notices-bar__summary")
        expect(summary).to_be_visible()
        text = summary.inner_text()
        assert "5" in text, f"the summary does not count the notices: {text!r}"
        assert "db_server is not responding" in text, f"the summary names nothing concrete: {text!r}"

    def test_the_worst_level_colours_the_summary(self, page: Page, base_url: str):
        """A collapsed bar reporting an error in info blue reads as 'all fine'."""
        _prepare(page, base_url)
        _push(
            page,
            [
                {"key": "a", "level": "info", "message": "hash index rebuilding"},
                {"key": "b", "level": "error", "message": "db_server is not responding"},
            ],
        )
        page.locator("#notices-bar .notices-bar__toggle").click()
        header = page.locator("#notices-bar .notices-bar__header")
        assert "notices-bar__item--error" in (header.get_attribute("class") or "")


class TestNoticesBarContent:
    def test_every_notice_is_listed_with_its_level(self, page: Page, base_url: str):
        _prepare(page, base_url)
        _push(
            page,
            [
                {"key": "a", "level": "info", "message": "hash index rebuilding"},
                {"key": "b", "level": "warning", "message": "tasks_server is not responding"},
                {"key": "c", "level": "error", "message": "db_server is not responding"},
            ],
        )
        items = page.locator("#notices-bar .notices-bar__item")
        expect(items).to_have_count(3)
        # Scoped to the list: the header carries the worst level's modifier too,
        # which is its own behaviour (see the collapsed-summary test).
        for level in ("info", "warning", "error"):
            expect(page.locator(f"#notices-bar .notices-bar__list .notices-bar__item--{level}")).to_have_count(1)

    def test_notices_clearing_hides_the_bar_again(self, page: Page, base_url: str):
        """Recovery has to put the screen back, collapsed or not."""
        _prepare(page, base_url)
        _push(page, OUTAGE)
        page.locator("#notices-bar .notices-bar__toggle").click()
        _push(page, [])
        expect(page.locator("#notices-bar")).to_be_hidden()


def test_the_bar_survives_a_malformed_frame(page: Page, base_url: str):
    """The stream is JSON over SSE; a truncated frame must not take the bar down."""
    _prepare(page, base_url)
    _push(page, OUTAGE)
    page.evaluate("() => FakeEventSource.notices.onmessage({ data: '{not json' })")
    expect(page.locator("#notices-bar")).to_be_visible()
    assert page.locator(".notices-bar__item").count() == len(OUTAGE)


def test_the_notices_payload_shape_matches_the_server(page: Page, base_url: str):
    """Guards the contract the fake frames above stand in for."""
    payload = json.loads(page.request.get(f"{base_url}/api/notices").text())
    assert "notices" in payload, payload
    for notice in payload["notices"]:
        assert {"key", "level", "message"} <= set(notice), notice
