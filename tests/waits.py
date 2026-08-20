"""Polling helpers for the unit suite — the async twin of `playwright_tests/waits.py`.

The pattern this replaces is a wall-clock sleep used as a barrier:

    task.start()
    await asyncio.sleep(0.05)
    assert state["renewed"], "the provider must tell the relay it is still there"

That is a bet that 50ms is always enough for a background task to reach its first
beat. It holds on an idle box and loses on a loaded one — and this suite runs
xdist-parallel on a host that also runs the user's live services, so "loaded" is the
normal case, not the exception. When it loses, the failure names the feature under
test and looks like a real regression, which is the expensive kind of flake.

`await_condition` polls instead: it returns the moment the condition holds (so the
fast path is *faster* than the sleep it replaces) and only fails after a deadline
long enough that a real machine cannot miss it.

A sleep is only a legitimate barrier when the assertion is about something *not*
happening ("no beats may land after stop()"). You cannot poll for a non-event, so
those stay as they are — see the module docstrings where they occur.
"""

import asyncio
from typing import Awaitable, Callable, TypeVar, Union

T = TypeVar("T")

#: Generous on purpose. It is never *spent* unless the condition genuinely never
#: holds, so a large value costs nothing on a green run and buys immunity to a
#: loaded box.
DEFAULT_TIMEOUT_S = 5.0
POLL_INTERVAL_S = 0.005


async def await_condition(
    condition: Callable[[], Union[T, Awaitable[T]]],
    message: Union[str, Callable[[], str]] = "",
    timeout: float = DEFAULT_TIMEOUT_S,
) -> T:
    """Poll `condition` until it returns something truthy, then return it.

    `condition` may be sync or async. Raises AssertionError with `message` if the
    deadline passes; pass a callable to describe the *final* state, since an
    f-string would be interpolated before the first poll.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        value = condition()
        if asyncio.iscoroutine(value):
            value = await value
        if value:
            return value
        if loop.time() >= deadline:
            text = message() if callable(message) else message
            raise AssertionError(text or f"Condition never became true within {timeout}s")
        await asyncio.sleep(POLL_INTERVAL_S)
