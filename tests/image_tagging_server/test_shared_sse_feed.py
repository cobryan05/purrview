"""Tests for `SharedSseFeed` — the fan-out behind /api/tasks/stream and /api/notices/stream.

The shipped bug these pin: each SSE connection ran its own poll loop, so the cost of
these endpoints was O(open connections). The task payload is ~240 KB, so twenty
connections meant re-fetching and re-serialising a quarter of a megabyte forty times a
second and discarding all of it, because the payload had not changed.

Connection count is not something the server can hold down. A client reached through a
port forwarder leaves the upstream socket open when its tab goes away — nothing cancels
the request and `is_disconnected()` stays False, because the forwarder really is still
connected — so strays accumulate for the life of the process. Observed on a live box as
43 stranded streams holding ~160% CPU against an app whose idle cost is ~5%.

Two properties follow, and both are tested here: one poll serves every subscriber, and
a connection does not live forever.
"""

import asyncio

import pytest

from apps.image_tagging_server.web import SharedSseFeed


def _counting_feed(interval_s: float = 0.01, payload=None):
    """A feed whose upstream returns a constant, with a counter of real fetches."""
    calls = {"n": 0}

    async def fetch():
        calls["n"] += 1
        return {"stable": True} if payload is None else payload

    feed = SharedSseFeed(fetch, lambda _data, _settled: interval_s, empty={}, name="test")
    return feed, calls


async def _drain_for(feed: SharedSseFeed, seconds: float, subscribers: int) -> int:
    """Hold *subscribers* streams open for *seconds*. Returns frames seen."""
    frames = 0

    async def one():
        nonlocal frames
        agen = feed.stream(max_lifetime_s=60.0)
        try:
            while True:
                await anext(agen)
                frames += 1
        except (StopAsyncIteration, asyncio.CancelledError):
            pass
        finally:
            await agen.aclose()

    tasks = [asyncio.create_task(one()) for _ in range(subscribers)]
    await asyncio.sleep(seconds)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    return frames


class TestOnePollServesEverySubscriber:
    """The whole point: upstream cost must not scale with connection count."""

    async def test_upstream_fetches_do_not_scale_with_subscribers(self):
        """Compared as a ratio rather than an absolute count, so the assertion does
        not depend on how fast the machine ran the window."""
        window = 0.4

        solo_feed, solo = _counting_feed()
        await _drain_for(solo_feed, window, subscribers=1)

        many_feed, many = _counting_feed()
        await _drain_for(many_feed, window, subscribers=6)

        assert solo["n"] > 2, "the feed did not poll at all — the test window is too short to measure"
        # Per-connection polling would make this ~6x. Allowing 2x covers scheduling
        # noise while still failing loudly if the loop is per-connection again.
        assert many["n"] < solo["n"] * 2, (
            f"6 subscribers cost {many['n']} fetches against {solo['n']} for one — " "the poll is not being shared"
        )

    async def test_every_subscriber_still_receives_the_payload(self):
        """Sharing must not mean only the first subscriber gets frames."""
        feed, _ = _counting_feed()
        seen = await _drain_for(feed, 0.3, subscribers=4)
        assert seen >= 4, f"only {seen} frames across 4 subscribers — some got nothing"

    async def test_the_poller_stops_when_the_last_subscriber_leaves(self):
        """Otherwise the feed polls forever on behalf of nobody."""
        feed, calls = _counting_feed()
        await _drain_for(feed, 0.2, subscribers=2)
        assert feed.subscriber_count == 0

        await asyncio.sleep(0.15)
        settled = calls["n"]
        await asyncio.sleep(0.15)

        assert calls["n"] == settled, "the feed kept polling after every subscriber had gone"


class TestAStreamDoesNotLiveForever:
    """A stranded connection has to be reaped on a timer, because it cannot be identified.

    `is_disconnected()` cannot see this case: the forwarder holding the socket open is a
    genuinely connected client as far as this server can tell. A bounded lifetime is what
    makes a stray self-limiting — a live browser's EventSource reconnects and re-reads
    whole state from the next frame, a stray simply never comes back.
    """

    async def test_the_stream_ends_once_its_lifetime_is_up(self):
        feed, _ = _counting_feed()
        agen = feed.stream(max_lifetime_s=0.15)

        # First frame arrives promptly; the generator must then finish on its own.
        await anext(agen)
        with pytest.raises(StopAsyncIteration):
            for _ in range(500):
                await anext(agen)

        assert feed.subscriber_count == 0, "a reaped stream must not stay registered as a subscriber"

    async def test_a_long_lifetime_does_not_end_the_stream_early(self):
        """The reaper must not be so eager that it closes a stream nobody asked to end."""
        feed, _ = _counting_feed()
        agen = feed.stream(max_lifetime_s=60.0)
        await anext(agen)
        await asyncio.sleep(0.2)
        # Still live: cancelling is the only thing that should stop it here.
        assert feed.subscriber_count == 1
        await agen.aclose()


class TestAFailingUpstreamDoesNotCloseTheStream:
    async def test_a_raising_fetch_is_reported_as_empty_rather_than_ending_the_feed(self):
        """A brief upstream outage must not disconnect every open browser."""
        calls = {"n": 0}

        async def fetch():
            calls["n"] += 1
            raise RuntimeError("tasks_server is down")

        feed = SharedSseFeed(fetch, lambda _d, _s: 0.01, empty={}, name="test")
        agen = feed.stream(max_lifetime_s=60.0)
        frame = await anext(agen)

        assert "data:" in frame
        assert calls["n"] >= 1
        await agen.aclose()
