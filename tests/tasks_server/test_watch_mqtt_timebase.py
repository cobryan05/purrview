"""The clock Home Assistant sees on a watch event's `lastSeen`.

`WatchEventSignal.at` is on the *monotonic* clock — its docstring says "wallclock",
which is exactly what made this easy to get wrong. Passing it straight to
`datetime.fromtimestamp` raises nothing and produces a valid ISO 8601 string; it just
dates every event to January 1970. That reached Home Assistant as a
`device_class: timestamp` sensor, so "last seen" read 1970 for every rule and every
subject while the counts beside it were correct.

The assertion below is deliberately about the *era* of the result rather than an exact
value: anything that drops the offset lands in 1970, and nothing that keeps it can.
"""

import time
from datetime import datetime, timezone

from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask


def _task():
    task = StreamWatcherTask.__new__(StreamWatcherTask)
    task._wallclock_offset = time.time() - time.monotonic()
    return task


class TestWallclockConversion:
    def test_a_monotonic_signal_time_becomes_a_present_day_timestamp(self):
        task = _task()

        stamped = datetime.fromisoformat(task._wall_iso(time.monotonic()))

        assert abs((stamped - datetime.now(timezone.utc)).total_seconds()) < 60

    def test_the_offset_is_applied_rather_than_the_raw_monotonic_clock(self):
        """The raw clock is uptime-since-boot, which formats as 1970 — the exact
        symptom seen in HA. This fails if the conversion is dropped."""
        task = _task()
        at = time.monotonic()

        stamped = datetime.fromisoformat(task._wall_iso(at))
        naive = datetime.fromtimestamp(at, tz=timezone.utc)

        assert stamped.year > 2020
        assert naive.year == 1970  # guards the premise: `at` really is monotonic
