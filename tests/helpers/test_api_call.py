"""Every generated API client carries a request timeout, and keeps carrying one.

The failure this prevents is not "a slow request". It is a task that can no longer be
stopped. The generated clients default to `_request_timeout=None` — urllib3 reads that
as wait forever — and `asyncio.to_thread` cannot be cancelled, so a thread parked in a
socket read makes the coroutine awaiting it uncancellable too. The task stops
responding to its own cancel flag and ignores every Stop the user presses.

It escalates: the default executor is `min(32, cpu + 4)` threads shared by every
`to_thread` in the process, so leaked calls accumulate until nothing can reach a thread
at all — heartbeats included, which is how a wedged service goes on looking healthy.

The bound is bound to the client rather than passed per call, so the guard here is
mostly about the *seam* holding: `rest_client.request` is generated code, and
`generate_openapi_modules.sh` rewrites it from scratch on every build.
"""

from unittest.mock import MagicMock

import pytest

from apps.helpers.apiCall import (
    DEFAULT_TIMEOUT,
    LONG_TIMEOUT,
    bind_default_timeout,
    default_timeout_of,
)


class _FakeRestClient:
    def __init__(self):
        self.calls = []

    def request(self, method, url, headers=None, body=None, post_params=None, _request_timeout=None):
        self.calls.append(_request_timeout)
        return "response"


class _FakeApiClient:
    def __init__(self):
        self.rest_client = _FakeRestClient()


class TestBinding:
    def test_an_unbounded_request_gets_the_default(self):
        client = bind_default_timeout(_FakeApiClient())

        client.rest_client.request("GET", "/x")

        assert client.rest_client.calls == [DEFAULT_TIMEOUT]

    def test_a_per_call_timeout_still_wins(self):
        """A genuinely long request opts out; the default must not clobber it."""
        client = bind_default_timeout(_FakeApiClient())

        client.rest_client.request("POST", "/train", _request_timeout=LONG_TIMEOUT)

        assert client.rest_client.calls == [LONG_TIMEOUT]

    def test_binding_returns_the_client_so_it_can_wrap_the_constructor(self):
        client = _FakeApiClient()
        assert bind_default_timeout(client) is client

    def test_binding_twice_does_not_stack_wrappers(self):
        client = bind_default_timeout(_FakeApiClient())
        first = client.rest_client.request
        bind_default_timeout(client)

        assert client.rest_client.request is first

    def test_a_custom_default_is_honoured(self):
        client = bind_default_timeout(_FakeApiClient(), timeout=(1.0, 2.0))

        client.rest_client.request("GET", "/x")

        assert client.rest_client.calls == [(1.0, 2.0)]

    def test_the_response_is_passed_through(self):
        client = bind_default_timeout(_FakeApiClient())
        assert client.rest_client.request("GET", "/x") == "response"

    def test_no_bound_is_ever_none(self):
        """None is the value that means "forever" — the whole bug in one argument."""
        for bound in (DEFAULT_TIMEOUT, LONG_TIMEOUT):
            assert bound is not None
            for part in bound if isinstance(bound, tuple) else (bound,):
                assert part is not None and part > 0


class TestTheSeamIsGuarded:
    """`rest_client.request` is generated code. If a regeneration moves it, this must
    fail loudly rather than silently leaving every call unbounded again."""

    def test_a_client_without_the_seam_is_returned_unchanged(self):
        """Not an exception: this runs inside constructors that tests hand a double,
        and a fake client has no business modelling urllib3 internals to be
        constructible. The real guarantee is the two tests below, which run against
        the actually-installed client."""
        double = MagicMock(spec=[])

        assert bind_default_timeout(double) is double
        assert default_timeout_of(double) is None

    def test_the_real_generated_client_still_has_the_seam(self):
        """Run against the actually-installed client, not a fake."""
        import db_client

        client = db_client.ApiClient(db_client.Configuration("http://localhost:1"))
        bind_default_timeout(client)
        assert default_timeout_of(client) == DEFAULT_TIMEOUT

    def test_the_real_generated_method_still_accepts_request_timeout(self):
        """The per-call override is part of the generated public signature; the
        LONG_TIMEOUT call sites depend on it."""
        import inspect

        from db_client.api.tasks_api import TasksApi

        sig = inspect.signature(TasksApi.list_task_instances)
        assert "_request_timeout" in sig.parameters


class TestEveryClientIsBound:
    """A guard on the pattern, not on any one construction site.

    The natural thing to type is a bare `SomeClient.ApiClient(conf)`, it looks correct,
    and nothing fails until a service stops answering in production.
    """

    def _unbound_construction_sites(self):
        import ast
        import os

        from apps.helpers import apiCall

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(apiCall.__file__))))
        apps_dir = os.path.join(repo_root, "apps")
        sites = []
        for dirpath, _, files in os.walk(apps_dir):
            if "__pycache__" in dirpath:
                continue
            for f in sorted(files):
                if not f.endswith(".py"):
                    continue
                path = os.path.join(dirpath, f)
                try:
                    tree = ast.parse(open(path).read())
                except SyntaxError:
                    continue
                # Every ApiClient(...) must sit inside a bind_default_timeout(...) call.
                bound = set()
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "bind_default_timeout"
                    ):
                        for inner in ast.walk(node):
                            if inner is not node and isinstance(inner, ast.Call):
                                bound.add(id(inner))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    fn = node.func
                    if not (isinstance(fn, ast.Attribute) and fn.attr == "ApiClient"):
                        continue
                    if id(node) in bound:
                        continue
                    rel = os.path.relpath(path, repo_root)
                    sites.append(f"{rel}:{node.lineno}  {ast.unparse(node)[:70]}")
        return sites

    def test_no_api_client_is_constructed_without_a_timeout(self):
        sites = self._unbound_construction_sites()
        assert not sites, (
            "These construct a generated ApiClient without binding a default timeout, so "
            "every request through them waits forever and cannot be cancelled. Wrap in "
            "bind_default_timeout() from apps.helpers.apiCall:\n  " + "\n  ".join(sites)
        )


class TestTheTimeoutStaysOutOfBusinessCode:
    """`_request_timeout` is the client library's parameter, not our vocabulary.

    A slow client is bound slow where it is built, so no task, provider or manager has
    to name a generated library's private-looking argument to say "this takes a while".
    """

    def test_request_timeout_appears_nowhere_outside_the_helper(self):
        import os

        from apps.helpers import apiCall

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(apiCall.__file__))))
        helper = os.path.abspath(apiCall.__file__)
        offenders = []
        for dirpath, _, files in os.walk(os.path.join(repo_root, "apps")):
            if "__pycache__" in dirpath:
                continue
            for f in sorted(files):
                if not f.endswith(".py"):
                    continue
                path = os.path.join(dirpath, f)
                if os.path.abspath(path) == helper:
                    continue
                for i, line in enumerate(open(path), start=1):
                    if "_request_timeout" in line:
                        offenders.append(f"{os.path.relpath(path, repo_root)}:{i}  {line.strip()[:60]}")
        assert not offenders, (
            "_request_timeout belongs to the generated client, not to app code. Bind the "
            "client with bind_default_timeout(..., timeout=LONG_TIMEOUT) where it is "
            "constructed instead:\n  " + "\n  ".join(offenders)
        )
