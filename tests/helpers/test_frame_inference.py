"""Tests for `apps.helpers.frameInference`.

Phase 3 of COLLECTOR_UNIFICATION_PLAN.md (repo root). What's worth pinning
here is the union math — the part that only starts mattering once a second
consumer shares a request in phase 4, and would fail silently (a model
dropped, a floor too tight, a retry that costs a stalled watcher a whole
frame budget) rather than loudly if it regressed.
"""

import asyncio
from types import SimpleNamespace

from apps.helpers.frameInference import (
    FrameInferenceRequest,
    FrameInferenceResult,
    run_frame_inference,
)


class _FakeResponse:
    def __init__(self, results=None, errors=None):
        self.results = results or {}
        self.errors = errors or {}


class _FakeInferenceApi:
    """Records every payload it was sent; scripted responses/exceptions per call."""

    def __init__(self, responses=None, exceptions=None):
        self.payloads: list = []
        self._responses = list(responses or [])
        self._exceptions = list(exceptions or [])
        self.calls = 0

    def detect_unit(self, payload):
        self.payloads.append(payload)
        self.calls += 1
        if self._exceptions:
            exc = self._exceptions.pop(0)
            if exc is not None:
                raise exc
        if self._responses:
            return self._responses.pop(0)
        return _FakeResponse()


class TestUnionMath:
    async def test_union_of_model_names_across_requests(self):
        api = _FakeInferenceApi()
        r1 = FrameInferenceRequest(model_names=["a"], conf_floor=0.5)
        r2 = FrameInferenceRequest(model_names=["b", "a"], conf_floor=0.5)

        await run_frame_inference(api, "img", [r1, r2], timeout_s=1, max_retries=1)

        assert set(api.payloads[0].model_names) == {"a", "b"}

    async def test_conf_floor_is_the_loosest_across_requests(self):
        """min(), not the watcher's own — a tighter collector floor must never
        silently degrade what the watcher's tracker receives."""
        api = _FakeInferenceApi()
        r1 = FrameInferenceRequest(model_names=["a"], conf_floor=0.5)
        r2 = FrameInferenceRequest(model_names=["b"], conf_floor=0.1)

        await run_frame_inference(api, "img", [r1, r2], timeout_s=1, max_retries=1)

        assert api.payloads[0].conf_thresh == 0.1

    async def test_top_k_is_the_per_model_max_across_requests(self):
        api = _FakeInferenceApi()
        r1 = FrameInferenceRequest(model_names=["a", "b"], conf_floor=0.5, top_k={"a": 2})
        r2 = FrameInferenceRequest(model_names=["b"], conf_floor=0.5, top_k={"b": 5})

        await run_frame_inference(api, "img", [r1, r2], timeout_s=1, max_retries=1)

        assert api.payloads[0].top_k.actual_instance == {"a": 2, "b": 5}

    async def test_a_model_no_request_asked_top_k_for_is_absent_not_zeroed(self):
        api = _FakeInferenceApi()
        req = FrameInferenceRequest(model_names=["a", "b"], conf_floor=0.5, top_k={"a": 3})

        await run_frame_inference(api, "img", [req], timeout_s=1, max_retries=1)

        assert api.payloads[0].top_k.actual_instance == {"a": 3}

    async def test_pin_ids_merge_across_requests(self):
        api = _FakeInferenceApi()
        r1 = FrameInferenceRequest(model_names=["a"], conf_floor=0.5, pin_ids={"a": "pin-a"})
        r2 = FrameInferenceRequest(model_names=["b"], conf_floor=0.5, pin_ids={"b": "pin-b"})

        await run_frame_inference(api, "img", [r1, r2], timeout_s=1, max_retries=1)

        assert api.payloads[0].pin_ids == {"a": "pin-a", "b": "pin-b"}

    async def test_no_models_returns_empty_without_calling_the_api(self):
        api = _FakeInferenceApi()

        result = await run_frame_inference(
            api, "img", [FrameInferenceRequest(model_names=[], conf_floor=0.5)], timeout_s=1, max_retries=1
        )

        assert result.results == {} and result.errors == {}
        assert api.calls == 0


class TestForModels:
    def test_orders_per_caller_not_the_union(self):
        """FIRST_WINS merge is order-dependent; one consumer's stack priority
        must never leak into another's just because they shared a request."""
        result = FrameInferenceResult(
            results={
                "a": SimpleNamespace(detections=["det-a"]),
                "b": SimpleNamespace(detections=["det-b"]),
            },
            errors={},
        )

        assert result.for_models(["b", "a"]) == [["det-b"], ["det-a"]]

    def test_a_model_missing_from_results_contributes_an_empty_list(self):
        """Missing = failed, or not requested by this particular caller —
        either way the caller's position list must stay the same length."""
        result = FrameInferenceResult(results={"a": SimpleNamespace(detections=["det-a"])}, errors={})

        assert result.for_models(["a", "missing"]) == [["det-a"], []]


class TestRetries:
    async def test_retries_on_timeout_then_succeeds(self):
        api = _FakeInferenceApi(
            responses=[_FakeResponse(results={"a": "ok"})],
            exceptions=[asyncio.TimeoutError(), None],
        )
        req = FrameInferenceRequest(model_names=["a"], conf_floor=0.5)

        result = await run_frame_inference(api, "img", [req], timeout_s=1, max_retries=2)

        assert result.results == {"a": "ok"}
        assert api.calls == 2

    async def test_gives_up_after_max_retries_and_logs(self):
        api = _FakeInferenceApi(exceptions=[asyncio.TimeoutError(), asyncio.TimeoutError()])
        logged = []
        req = FrameInferenceRequest(model_names=["a"], conf_floor=0.5)

        result = await run_frame_inference(api, "img", [req], timeout_s=1, max_retries=2, log=logged.append)

        assert result.results == {} and result.errors == {}
        assert api.calls == 2
        assert logged, "a max-retries timeout must be surfaced to the caller's log, not just logger.warning"

    async def test_non_timeout_exception_does_not_retry(self):
        """A batch-level programming/decode error is not the transient
        condition retries exist for — retrying it just delays the same failure."""
        api = _FakeInferenceApi(exceptions=[RuntimeError("boom")])
        req = FrameInferenceRequest(model_names=["a"], conf_floor=0.5)

        result = await run_frame_inference(api, "img", [req], timeout_s=1, max_retries=3)

        assert api.calls == 1
        assert result.results == {}
