"""Unit tests for FilterChainRunner — pure step-matching logic, no I/O.

Uses a fake InferenceBackend so these tests exercise only the runner's
ordering/threshold/IoU/guard logic, independent of how a real backend talks
to inference_server.
"""

from apps.tasks_server.tasks._filter_chain_runner import (
    Detection,
    FilterChainRunner,
    StepMatch,
)


class FakeBackend:
    def __init__(self, kinds: dict, classify_results: dict = None, detections: dict = None):
        self._kinds = kinds
        self._classify_results = classify_results or {}
        self._detections = detections or {}
        self.detect_calls = []

    async def get_model_kind(self, model_name: str) -> str:
        return self._kinds.get(model_name, "detector")

    async def classify_whole_image(self, model_name: str, image_bytes: bytes) -> float:
        return self._classify_results.get(model_name, 0.0)

    async def detect(self, model_name: str, image_bytes: bytes, conf_thresh: float):
        self.detect_calls.append((model_name, conf_thresh))
        return self._detections.get(model_name, [])


def _step(
    order=0, model_name="m1", action="tag_image", confidence=0.5, iou_threshold=0.5, action_params=None, enabled=True
):
    return {
        "order": order,
        "model_name": model_name,
        "action": action,
        "confidence": confidence,
        "iou_threshold": iou_threshold,
        "action_params": action_params or {},
        "enabled": enabled,
    }


class TestClassifierSteps:
    async def test_confidence_above_threshold_matches(self):
        backend = FakeBackend(kinds={"corrupt": "image_classifier"}, classify_results={"corrupt": 0.9})
        runner = FilterChainRunner([_step(model_name="corrupt", action="discard_image", confidence=0.8)], backend)
        matches = await runner.evaluate_image(b"img", [])
        assert len(matches) == 1
        assert matches[0].image_match is True
        assert matches[0].bbox_matches == {}

    async def test_confidence_below_threshold_no_match(self):
        backend = FakeBackend(kinds={"corrupt": "image_classifier"}, classify_results={"corrupt": 0.5})
        runner = FilterChainRunner([_step(model_name="corrupt", action="discard_image", confidence=0.8)], backend)
        matches = await runner.evaluate_image(b"img", [])
        assert matches == []

    async def test_confidence_exactly_at_threshold_matches(self):
        backend = FakeBackend(kinds={"corrupt": "image_classifier"}, classify_results={"corrupt": 0.8})
        runner = FilterChainRunner([_step(model_name="corrupt", confidence=0.8)], backend)
        matches = await runner.evaluate_image(b"img", [])
        assert len(matches) == 1

    async def test_multi_label_classifier_kind_treated_as_image_level(self):
        backend = FakeBackend(kinds={"m": "multi_label_classifier"}, classify_results={"m": 0.95})
        runner = FilterChainRunner([_step(model_name="m", confidence=0.7)], backend)
        matches = await runner.evaluate_image(b"img", [])
        assert matches[0].image_match is True


class TestDetectorOverlapSteps:
    async def test_overlapping_detection_matches_bbox(self):
        existing = [{"uuid": "bbox-1", "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2, "label_uuid": "cat"}]
        det = Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9, label_uuid="water_valve")
        backend = FakeBackend(kinds={"valves": "detector"}, detections={"valves": [det]})
        runner = FilterChainRunner([_step(model_name="valves", action="relabel_bbox", iou_threshold=0.5)], backend)
        matches = await runner.evaluate_image(b"img", existing)
        assert len(matches) == 1
        assert matches[0].bbox_matches == {"bbox-1": det}

    async def test_non_overlapping_detection_no_match(self):
        existing = [{"uuid": "bbox-1", "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2, "label_uuid": "cat"}]
        det = Detection(x=0.7, y=0.7, width=0.1, height=0.1, confidence=0.9)
        backend = FakeBackend(kinds={"valves": "detector"}, detections={"valves": [det]})
        runner = FilterChainRunner([_step(model_name="valves", iou_threshold=0.5)], backend)
        matches = await runner.evaluate_image(b"img", existing)
        assert matches == []

    async def test_only_if_label_uuid_guard_filters_non_matching_label(self):
        existing = [{"uuid": "bbox-1", "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2, "label_uuid": "dog"}]
        det = Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9)
        backend = FakeBackend(kinds={"valves": "detector"}, detections={"valves": [det]})
        runner = FilterChainRunner([_step(model_name="valves", action_params={"only_if_label_uuid": "cat"})], backend)
        matches = await runner.evaluate_image(b"img", existing)
        assert matches == []

    async def test_only_if_label_uuid_guard_allows_matching_label(self):
        existing = [{"uuid": "bbox-1", "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2, "label_uuid": "cat"}]
        det = Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9)
        backend = FakeBackend(kinds={"valves": "detector"}, detections={"valves": [det]})
        runner = FilterChainRunner([_step(model_name="valves", action_params={"only_if_label_uuid": "cat"})], backend)
        matches = await runner.evaluate_image(b"img", existing)
        assert len(matches) == 1

    async def test_no_existing_bboxes_short_circuits_without_detect_call(self):
        backend = FakeBackend(kinds={"valves": "detector"}, detections={"valves": [Detection(0, 0, 1, 1, 0.9)]})
        runner = FilterChainRunner([_step(model_name="valves")], backend)
        matches = await runner.evaluate_image(b"img", [])
        assert matches == []

    async def test_each_new_detection_claims_at_most_one_bbox(self):
        """Two overlapping existing bboxes, one detection: only the first (in
        list order) is claimed — a detection can't match twice."""
        existing = [
            {"uuid": "bbox-1", "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2, "label_uuid": "cat"},
            {"uuid": "bbox-2", "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2, "label_uuid": "cat"},
        ]
        det = Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9)
        backend = FakeBackend(kinds={"valves": "detector"}, detections={"valves": [det]})
        runner = FilterChainRunner([_step(model_name="valves")], backend)
        matches = await runner.evaluate_image(b"img", existing)
        assert len(matches) == 1
        assert list(matches[0].bbox_matches.keys()) == ["bbox-1"]


class TestStepOrderingAndEnabled:
    async def test_steps_run_in_order_field_order_not_list_order(self):
        backend = FakeBackend(
            kinds={"a": "image_classifier", "b": "image_classifier"},
            classify_results={"a": 0.9, "b": 0.9},
        )
        # Passed out of order; runner must sort by `order`.
        steps = [_step(order=1, model_name="b", confidence=0.5), _step(order=0, model_name="a", confidence=0.5)]
        runner = FilterChainRunner(steps, backend)
        matches = await runner.evaluate_image(b"img", [])
        assert [m.step["model_name"] for m in matches] == ["a", "b"]

    async def test_disabled_step_is_skipped(self):
        backend = FakeBackend(kinds={"a": "image_classifier"}, classify_results={"a": 0.99})
        runner = FilterChainRunner([_step(model_name="a", confidence=0.5, enabled=False)], backend)
        matches = await runner.evaluate_image(b"img", [])
        assert matches == []

    async def test_detect_called_with_step_confidence_as_threshold(self):
        backend = FakeBackend(kinds={"valves": "detector"}, detections={"valves": []})
        runner = FilterChainRunner([_step(model_name="valves", confidence=0.65)], backend)
        await runner.evaluate_image(b"img", [{"uuid": "x", "x": 0, "y": 0, "width": 1, "height": 1}])
        assert backend.detect_calls == [("valves", 0.65)]
