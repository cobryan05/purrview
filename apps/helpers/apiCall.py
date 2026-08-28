"""Give every generated OpenAPI client a default request timeout.

Every cross-service call in this app is a synchronous client method run off the event
loop with ``asyncio.to_thread``. Two properties of that combination make an unbounded
one dangerous out of proportion to how ordinary it looks:

* the generated clients default to ``_request_timeout=None``, which urllib3 reads as
  "wait forever"; and
* ``asyncio.to_thread`` is **not cancellable**. A thread blocked in a socket read
  cannot be interrupted, so the coroutine awaiting it cannot be cancelled either. The
  task owning that coroutine stops responding to its own cancel flag and can no longer
  be stopped at all.

It escalates from there. The default executor holds ``min(32, cpu + 4)`` threads and is
shared by every ``to_thread`` in the process, so leaked calls accumulate until nothing
can reach a thread — heartbeats included, which is how a wedged service goes on looking
healthy. The two ``asyncio.wait_for(asyncio.to_thread(...))`` inference call sites leak
one thread per timeout for exactly this reason: ``wait_for`` abandons the future, and
the thread underneath keeps waiting on a socket nobody is going to answer.

So the timeout is not a tuning detail; it is what keeps a call site cancellable.

**Why this binds to the client rather than being passed per call.** ``Configuration``
has no timeout attribute (only ``retries`` and ``connection_pool_maxsize``), and the
generated ``rest.py`` always forwards ``timeout=`` explicitly — passing ``None`` when
unset — so a urllib3 pool-level default is overridden back to "forever". That leaves
two options: pass ``_request_timeout`` at all ~128 call sites, or wrap the one method
they all funnel through. Wrapping wins on the property that matters: a bound that has
to be remembered at every call site is one an unbounded call can slip past, and the
call-site version also drags the timeout constant into ~60 unrelated test assertions
that have no business knowing it.

The seam — ``api_client.rest_client.request(..., _request_timeout=...)`` — is generated
code, so ``generate_openapi_modules.sh`` could in principle move it. That is what
``bind_default_timeout`` raising and ``tests/helpers/test_api_call.py`` guarding it are
for: a regeneration that changes the shape fails loudly at startup and in the suite,
rather than silently restoring unbounded calls.
"""

import functools
import logging
from typing import Any, Tuple, TypeVar, Union

logger = logging.getLogger(__name__)

Timeout = Union[float, Tuple[float, float]]
ApiClientT = TypeVar("ApiClientT")

#: (connect, read) seconds for an ordinary request.
#:
#: Connect is short because every service in the default deployment is a process in the
#: same container: failing to connect means it is down, and waiting does not fix that.
#: Read is generous because it bounds the gap between bytes rather than the whole
#: request, and db_server under a heavy query legitimately thinks for a while. Both are
#: far longer than any healthy call and far shorter than forever — the point is to turn
#: "wedged until the process restarts" into "raises, and the caller's existing error
#: handling runs".
DEFAULT_TIMEOUT: Timeout = (5.0, 60.0)

#: For individual requests whose server side does real work — training submission,
#: model conversion, arming a relay path, extracting a clip. Pass explicitly as
#: ``_request_timeout=LONG_TIMEOUT`` at those call sites; still bounded, because a job
#: that has genuinely stopped answering must not hold a task hostage forever.
LONG_TIMEOUT: Timeout = (5.0, 600.0)

_BOUND_MARKER = "_encounterwatch_default_timeout"


def bind_default_timeout(api_client: ApiClientT, timeout: Timeout = DEFAULT_TIMEOUT) -> ApiClientT:
    """Make *api_client* apply *timeout* to any request that doesn't set its own.

    Call once on every generated ``ApiClient`` at construction. Returns the client so it
    can wrap the constructor call directly.

    A client whose work is *all* slow (the training server) is bound to LONG_TIMEOUT
    here rather than at each of its call sites — see the note on LONG_TIMEOUT.
    Idempotent: binding twice does not stack wrappers.

    Warns and returns unchanged if the object has no seam to bind. That is deliberately
    not an exception: this runs inside constructors that tests routinely hand a double,
    and a fake client has no business modelling urllib3 internals just to be
    constructible. The guarantee that the *real* generated clients still have the seam
    is asserted in tests/helpers/test_api_call.py, which is where a regeneration that
    moved it should fail — at build time, not by refusing to start a service.
    """
    rest_client = getattr(api_client, "rest_client", None)
    if rest_client is None or not callable(getattr(rest_client, "request", None)):
        logger.warning(
            "%s has no rest_client.request to bind a timeout to; its requests are "
            "unbounded. Expected for a test double; for a real client it means the "
            "generated shape has changed — see apps/helpers/apiCall.py.",
            type(api_client).__name__,
        )
        return api_client

    original = rest_client.request
    if getattr(original, _BOUND_MARKER, None) is not None:
        return api_client

    @functools.wraps(original)
    def request_with_default_timeout(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("_request_timeout") is None:
            kwargs["_request_timeout"] = timeout
        return original(*args, **kwargs)

    setattr(request_with_default_timeout, _BOUND_MARKER, timeout)
    rest_client.request = request_with_default_timeout
    return api_client


def default_timeout_of(api_client: Any) -> Any:
    """The bound default for *api_client*, or None if it was never bound.

    Only for assertions — nothing in the app should branch on this.
    """
    request = getattr(getattr(api_client, "rest_client", None), "request", None)
    return getattr(request, _BOUND_MARKER, None)
