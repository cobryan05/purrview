"""Running a model stack over one frame does the shared work once.

One request per model meant the server base64-decoded and letterboxed the
identical image once per model — measured at 3.60 ms of decode and 0.95 ms of
letterbox each, for a byte-identical result, so a four-model stack burned
~13.7 ms of CPU per frame recomputing what it already had and uploaded the
image four times.

The two properties worth pinning are the ones whose regression is silent:

* the shared preprocessing is only shared between models that would have
  produced the same tensor — feeding a model a tensor letterboxed for a
  different input size still returns plausible-looking boxes, just decoded
  against the wrong geometry;
* the results come back in the order the models were requested, because every
  caller merges overlapping detections first-model-wins.
"""

import asyncio

import numpy as np
import pytest

from apps.helpers.inferenceProviders.inferenceProvider import InferenceProvider
from apps.helpers.inferenceProviders.onnxYoloInferenceProvider import OnnxYoloInferenceProvider
from apps.inference_server.manager import _SharedPreprocess


class _FakeProvider(InferenceProvider):
    """A provider that records how often its preprocessing actually ran."""

    def __init__(self, key, calls):
        self._key = key
        self._calls = calls

    def preprocess_key(self):
        return self._key

    def preprocess_shared(self, image):
        self._calls.append(self._key)
        return f"tensor-for-{self._key}"

    async def processImage(self, image, **kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError


class TestSharedPreprocess:
    async def test_models_with_the_same_key_preprocess_once(self):
        calls: list = []
        shared = _SharedPreprocess(np.zeros((4, 4, 3), dtype=np.uint8))
        providers = [_FakeProvider("k", calls) for _ in range(4)]

        results = [await shared.get(p) for p in providers]

        assert calls == ["k"], "preprocessing ran more than once for one recipe"
        assert results == ["tensor-for-k"] * 4

    async def test_models_with_different_keys_each_preprocess(self):
        # A different input size or letterbox anchor is a different tensor;
        # sharing across them would silently mis-decode every box.
        calls: list = []
        shared = _SharedPreprocess(np.zeros((4, 4, 3), dtype=np.uint8))

        a = await shared.get(_FakeProvider("640", calls))
        b = await shared.get(_FakeProvider("416", calls))

        assert sorted(calls) == ["416", "640"]
        assert a != b

    async def test_concurrent_callers_still_preprocess_once(self):
        """The watcher fires its whole stack at once, so the common case is N
        callers arriving before the first has finished."""
        calls: list = []
        shared = _SharedPreprocess(np.zeros((4, 4, 3), dtype=np.uint8))
        providers = [_FakeProvider("k", calls) for _ in range(8)]

        results = await asyncio.gather(*(shared.get(p) for p in providers))

        assert calls == ["k"]
        assert results == ["tensor-for-k"] * 8

    async def test_a_provider_opting_out_gets_nothing_shared(self):
        """None means "do not share" — the safe default for any provider whose
        preprocessing isn't fully described by a key."""
        calls: list = []
        shared = _SharedPreprocess(np.zeros((4, 4, 3), dtype=np.uint8))

        assert await shared.get(_FakeProvider(None, calls)) is None
        assert calls == []

    async def test_a_provider_that_raises_falls_back_to_its_own(self):
        class _Broken(_FakeProvider):
            def preprocess_key(self):
                raise RuntimeError("no idea")

        shared = _SharedPreprocess(np.zeros((4, 4, 3), dtype=np.uint8))

        # Not being able to describe its preprocessing is not a reason to fail
        # the inference; it just does its own, as it always did.
        assert await shared.get(_Broken("x", [])) is None


class TestBaseProviderDefaults:
    def test_the_base_class_opts_out_by_default(self):
        """A new provider must not accidentally inherit sharing it cannot honour."""

        class _New(InferenceProvider):
            async def processImage(self, image, **kwargs):  # pragma: no cover
                raise NotImplementedError

        assert _New().preprocess_key() is None


class _Plain(OnnxYoloInferenceProvider):
    """Concrete stand-in for the base YOLO provider (which is abstract)."""

    def _extract(self, preds):  # pragma: no cover - never run
        raise NotImplementedError


class TestYoloPreprocessKey:
    """The real key, on the real provider class."""

    def _provider(self, shape=(640, 640), cls=None):
        """A provider instance without loading an ONNX session.

        ``__init__`` builds a real ort.InferenceSession; the key depends only on
        the class, input_shape and anchor, so the object is constructed
        directly and those are set by hand.
        """
        p = object.__new__(cls or _Plain)
        p.input_shape = list(shape)
        return p

    def test_same_shape_and_anchor_share(self):
        a, b = self._provider(), self._provider()

        assert a.preprocess_key() == b.preprocess_key()

    def test_a_different_input_shape_does_not_share(self):
        a, b = self._provider(shape=(640, 640)), self._provider(shape=(416, 416))

        assert a.preprocess_key() != b.preprocess_key()

    def test_a_different_letterbox_anchor_does_not_share(self):
        class _TopLeft(OnnxYoloInferenceProvider):
            LETTERBOX_ANCHOR = "top_left"

            def _extract(self, preds):  # pragma: no cover - abstract stub
                raise NotImplementedError

        a = self._provider()
        b = self._provider(cls=_TopLeft)

        assert a.preprocess_key() != b.preprocess_key()

    def test_a_subclass_that_overrides_preprocess_does_not_share(self):
        """YOLOX must not normalize; sharing a normalized tensor with it would
        return detections that look fine and are wrong."""

        class _Different(OnnxYoloInferenceProvider):
            def _preprocess(self, image):  # different implementation entirely
                raise NotImplementedError

            def _extract(self, preds):  # pragma: no cover - abstract stub
                raise NotImplementedError

        a = self._provider()
        b = self._provider(cls=_Different)

        assert a.preprocess_key() != b.preprocess_key()

    def test_a_subclass_that_only_changes_attributes_still_shares(self):
        """Sharing must key on behaviour, not on class identity, or every
        subclass would needlessly redo identical work."""

        class _SameBehaviour(OnnxYoloInferenceProvider):
            SKIP_NMS = True  # irrelevant to preprocessing

            def _extract(self, preds):  # pragma: no cover - abstract stub
                raise NotImplementedError

        a = self._provider()
        b = self._provider(cls=_SameBehaviour)

        assert a.preprocess_key() == b.preprocess_key()
