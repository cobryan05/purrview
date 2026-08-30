"""
The shared task SSE stream (`shared/api/taskStream.js`).

One EventSource is fanned out to every subscriber, so its failure handling is
shared too: a stream left dead is dead for the Operations badge, every open
preview modal and every task runner at once.

These drive the module directly with a stubbed `window.EventSource`, because the
interesting states (a fully CLOSED connection, a server that stays down for
several retries) cannot be produced reliably against the real endpoint.

Run with:
    pytest playwright_tests/test_task_stream_sharing.py -p no:asyncio -v
"""

from playwright.sync_api import Page

# A stand-in for the browser's EventSource: records every instance, never talks to
# the network, and exposes `fail()` (a fatal, fully-CLOSED connection — what the
# browser gives you when the endpoint answers 502 or the wrong content type) and
# `drop()` (a transient error the real EventSource retries on its own).
FAKE_EVENT_SOURCE = """
  class FakeEventSource {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSED = 2;
    static instances = [];
    static autoFail = false;

    constructor(url) {
      this.url = url;
      this.readyState = FakeEventSource.OPEN;
      this.onmessage = null;
      this.onerror = null;
      this.closed = false;
      // Only the task stream is under test — the notices bar opens one of its own
      // at page load, and counting it would make every socket count off by one.
      if (!String(url).includes('/api/tasks/stream')) return;
      FakeEventSource.instances.push(this);
      if (FakeEventSource.autoFail) queueMicrotask(() => this.fail());
    }
    close() { this.closed = true; this.readyState = FakeEventSource.CLOSED; }
    send(payload) { this.onmessage?.({ data: JSON.stringify(payload) }); }
    fail() { this.readyState = FakeEventSource.CLOSED; this.onerror?.(new Event('error')); }
    drop() { this.readyState = FakeEventSource.CONNECTING; this.onerror?.(new Event('error')); }
  }
  window.FakeEventSource = FakeEventSource;
  window.EventSource = FakeEventSource;
"""


def _prepare(page: Page, base_url: str):
    """Load the app with EventSource stubbed before any module can grab it."""
    page.add_init_script(FAKE_EVENT_SOURCE)
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    # The landing tab (Watch) subscribes to the task stream itself, and drops the
    # subscription when the main tab changes. These tests drive taskStream.js's own
    # sharing and refcounting, so move to a pane that holds no subscriber and wait for
    # the refcount to actually reach zero — with a live app subscriber "closes when the
    # last subscriber leaves" is not merely untestable, it is false.
    page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()
    page.wait_for_function("() => FakeEventSource.instances.every(s => s.closed)")
    # Cleared so the tests' own socket is instances[0] again, in the same spirit as the
    # notices-bar exclusion in the stub above.
    page.evaluate("() => { FakeEventSource.instances.length = 0; }")


class TestSharing:
    def test_two_subscribers_share_one_connection(self, page: Page, base_url: str):
        _prepare(page, base_url)
        result = page.evaluate("""async () => {
            const { subscribeTaskStream } = await import('/app-static/js/shared/api/taskStream.js');
            const seen = [];
            subscribeTaskStream({ onMessage: (d) => seen.push(['a', d.n]) });
            subscribeTaskStream({ onMessage: (d) => seen.push(['b', d.n]) });
            FakeEventSource.instances.at(-1).send({ n: 1 });
            return { sockets: FakeEventSource.instances.length, seen };
        }""")
        assert result["sockets"] == 1, result
        assert result["seen"] == [["a", 1], ["b", 1]], result

    def test_the_connection_closes_when_the_last_subscriber_leaves(self, page: Page, base_url: str):
        _prepare(page, base_url)
        result = page.evaluate("""async () => {
            const { subscribeTaskStream } = await import('/app-static/js/shared/api/taskStream.js');
            const offA = subscribeTaskStream({ onMessage: () => {} });
            const offB = subscribeTaskStream({ onMessage: () => {} });
            offA();
            const afterFirst = FakeEventSource.instances[0].closed;
            offB();
            return { afterFirst, afterLast: FakeEventSource.instances[0].closed };
        }""")
        assert result["afterFirst"] is False, "the stream closed while a subscriber was still on it"
        assert result["afterLast"] is True, "the stream stayed open with no subscribers"

    def test_a_transient_drop_is_left_to_the_browser(self, page: Page, base_url: str):
        """readyState CONNECTING means EventSource is already retrying — opening a
        second socket over the top of that is how you end up with two."""
        _prepare(page, base_url)
        sockets = page.evaluate("""async () => {
            const { subscribeTaskStream } = await import('/app-static/js/shared/api/taskStream.js');
            subscribeTaskStream({ onMessage: () => {} });
            FakeEventSource.instances[0].drop();
            await new Promise(r => setTimeout(r, 1500));
            return FakeEventSource.instances.length;
        }""")
        assert sockets == 1, f"a transient drop opened {sockets} sockets"


class TestDeadStreamRecovery:
    """A fully CLOSED EventSource never reconnects itself, and the module used to
    keep the dead handle: `_open()` returns early while `_source` is non-null, so
    every later subscriber was handed a socket that could never deliver a frame.
    """

    def test_a_dead_stream_is_not_handed_to_the_next_subscriber(self, page: Page, base_url: str):
        """The Operations tab subscribes and never unsubscribes, so the subscriber
        set never empties and nothing ever cleared the dead handle."""
        _prepare(page, base_url)
        result = page.evaluate("""async () => {
            const { subscribeTaskStream } = await import('/app-static/js/shared/api/taskStream.js');
            // The long-lived subscriber (Operations badge): no onError, never leaves.
            subscribeTaskStream({ onMessage: () => {} });
            FakeEventSource.instances[0].fail();

            // A task runner starts up afterwards and must get a live stream.
            const seen = [];
            subscribeTaskStream({ onMessage: (d) => seen.push(d.n) });
            FakeEventSource.instances.at(-1).send({ n: 7 });
            return { sockets: FakeEventSource.instances.length, seen };
        }""")
        assert result["sockets"] > 1, "the next subscriber was handed the dead socket"
        assert result["seen"] == [7], f"a fresh subscriber received no frames: {result}"

    def test_the_stream_reconnects_on_its_own(self, page: Page, base_url: str):
        """Nothing re-subscribes while the user sits on Operations, so recovery has
        to come from the stream itself."""
        _prepare(page, base_url)
        result = page.evaluate("""async () => {
            const { subscribeTaskStream } = await import('/app-static/js/shared/api/taskStream.js');
            const seen = [];
            subscribeTaskStream({ onMessage: (d) => seen.push(d.n) });
            FakeEventSource.instances[0].fail();
            await new Promise(r => setTimeout(r, 2000));
            FakeEventSource.instances.at(-1).send({ n: 3 });
            return { sockets: FakeEventSource.instances.length, seen };
        }""")
        assert result["sockets"] == 2, f"the stream did not reconnect: {result}"
        assert result["seen"] == [3], f"the reconnected stream delivered nothing: {result}"

    def test_a_recovered_stream_forgets_its_backoff(self, page: Page, base_url: str):
        """Otherwise the second outage of a session starts at the previous one's
        longest delay, and a stream that flaps once takes minutes to come back."""
        _prepare(page, base_url)
        result = page.evaluate("""async () => {
            const { subscribeTaskStream } = await import('/app-static/js/shared/api/taskStream.js');
            subscribeTaskStream({ onMessage: () => {} });
            FakeEventSource.instances[0].fail();
            await new Promise(r => setTimeout(r, 2000));
            FakeEventSource.instances.at(-1).send({ n: 1 });   // proof of life
            FakeEventSource.instances.at(-1).fail();
            await new Promise(r => setTimeout(r, 2000));       // first-attempt delay again
            return FakeEventSource.instances.length;
        }""")
        assert result == 3, f"the retry delay did not reset after a working stream: {result} sockets"

    def test_reconnecting_stops_once_the_last_subscriber_leaves(self, page: Page, base_url: str):
        """A closed modal must not leave a socket reopening itself forever."""
        _prepare(page, base_url)
        sockets = page.evaluate("""async () => {
            const { subscribeTaskStream } = await import('/app-static/js/shared/api/taskStream.js');
            const off = subscribeTaskStream({ onMessage: () => {} });
            FakeEventSource.instances[0].fail();
            off();
            await new Promise(r => setTimeout(r, 2500));
            return FakeEventSource.instances.length;
        }""")
        assert sockets == 1, f"the stream kept reconnecting after everyone left: {sockets} sockets"

    def test_subscribers_are_told_once_the_outage_persists(self, page: Page, base_url: str):
        """The task runners await a promise that only an onError can reject — without
        it a modal spins forever against a server that is not coming back. It must
        not fire on the first failure, which is usually a restart mid-flight.
        """
        _prepare(page, base_url)
        result = page.evaluate("""async () => {
            const { subscribeTaskStream } = await import('/app-static/js/shared/api/taskStream.js');
            const errors = [];
            subscribeTaskStream({ onMessage: () => {}, onError: (e) => errors.push(e.message) });

            FakeEventSource.autoFail = true;   // every retry finds the server still down
            FakeEventSource.instances[0].fail();
            const afterFirst = errors.length;

            await new Promise(r => setTimeout(r, 9000));
            return { afterFirst, errors: errors.length };
        }""")
        assert result["afterFirst"] == 0, "a single failure was reported as fatal"
        assert result["errors"] >= 1, "a sustained outage was never reported to subscribers"
        assert result["errors"] == 1, f"subscribers were told {result['errors']} times for one outage"


class TestJoiningDuringAnOutage:
    """The outage flag is module state, and a subscriber arriving after it was
    raised has been told nothing at all. That is the exact shape of the bug the
    task runners hit: they await a promise only `onError` can settle, so a model
    import started while the stream is down waits for a frame that is not coming.
    """

    def test_a_subscriber_that_joins_mid_outage_is_told(self, page: Page, base_url: str):
        _prepare(page, base_url)
        result = page.evaluate("""async () => {
            const { subscribeTaskStream } = await import('/app-static/js/shared/api/taskStream.js');
            // The Runs badge: no onError, never leaves — so the subscriber set
            // never empties and the outage flag is never reset.
            subscribeTaskStream({ onMessage: () => {} });
            FakeEventSource.autoFail = true;
            FakeEventSource.instances.at(-1).fail();
            await new Promise(r => setTimeout(r, 9000));   // past NOTIFY_AFTER_ATTEMPTS

            // Now the task runner opens its modal, mid-outage.
            const errors = [];
            subscribeTaskStream({ onMessage: () => {}, onError: (e) => errors.push(e.message) });
            await new Promise(r => setTimeout(r, 200));
            FakeEventSource.autoFail = false;
            return errors.length;
        }""")
        assert result == 1, "a subscriber that joined during a reported outage was never told"

    def test_a_subscriber_that_joins_after_recovery_is_not_told(self, page: Page, base_url: str):
        """The flag clears on the first frame, so the next modal must open clean."""
        _prepare(page, base_url)
        result = page.evaluate("""async () => {
            const { subscribeTaskStream } = await import('/app-static/js/shared/api/taskStream.js');
            subscribeTaskStream({ onMessage: () => {} });
            FakeEventSource.autoFail = true;
            FakeEventSource.instances.at(-1).fail();
            await new Promise(r => setTimeout(r, 9000));
            FakeEventSource.autoFail = false;

            // The stream comes back: a frame is the proof, and clears the flag.
            await new Promise(r => setTimeout(r, 2000));
            FakeEventSource.instances.at(-1).send({ n: 1 });

            const errors = [];
            subscribeTaskStream({ onMessage: () => {}, onError: (e) => errors.push(e.message) });
            await new Promise(r => setTimeout(r, 200));
            return errors.length;
        }""")
        assert result == 0, "a healthy stream reported a stale outage to a new subscriber"

    def test_unsubscribing_immediately_cancels_the_notification(self, page: Page, base_url: str):
        """The report is deferred so the caller gets its handle first; a caller
        that leaves in between must not still be called back."""
        _prepare(page, base_url)
        result = page.evaluate("""async () => {
            const { subscribeTaskStream } = await import('/app-static/js/shared/api/taskStream.js');
            subscribeTaskStream({ onMessage: () => {} });
            FakeEventSource.autoFail = true;
            FakeEventSource.instances.at(-1).fail();
            await new Promise(r => setTimeout(r, 9000));

            const errors = [];
            const off = subscribeTaskStream({ onMessage: () => {}, onError: (e) => errors.push(e.message) });
            off();
            await new Promise(r => setTimeout(r, 200));
            FakeEventSource.autoFail = false;
            return errors.length;
        }""")
        assert result == 0, "a subscriber was notified after it had already left"
