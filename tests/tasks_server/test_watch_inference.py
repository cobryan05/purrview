"""The StreamWatcher's inference call — the seam between the task and the models.

This exists because the watcher shipped unable to detect anything at all: the payload
it built omitted a required field, and the resulting ValidationError was swallowed by
the same `except Exception` that guards genuine inference failures, so every frame
silently produced zero detections. The pure engine tests all passed throughout, because
they feed synthetic `Detection`s and never exercise this call.

The lesson those tests encode: assert on what comes *back* from `_run_inference`, not
just on what the engine does with it.
"""

from types import SimpleNamespace
from typing import Any, List

import pytest

from apps.helpers.cascade import CropSubModel
from apps.helpers.tracking.tracker import Detection
from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask

CAT = "label-cat"


def _bbox(x1: float, y1: float, x2: float, y2: float, label_uuid: str = CAT) -> SimpleNamespace:
    """A detection bbox as the inference client really returns it.

    ⚠️ Corner attributes are deliberately ABSENT. An earlier version of this helper
    invented `x1/y1/x2/y2`, which the real `BoundingBoxMetadataModel` has never had —
    so every test here passed while the watcher died with
    `'BoundingBoxMetadataModel' object has no attribute 'x1'` on its first frame.
    A fake that is more generous than the thing it stands in for tests nothing.

    Corners are taken as arguments only because they read more naturally at the call
    sites; they are stored the way the client stores them, as relative x/y/width/height.
    See TestRealClientBboxShape, which uses the generated model itself.

    ⚠️ `label_name` is absent for the same reason. This fake used to carry one, and the
    watcher read it — so every detection, track and live-view caption in the real app was
    labelled `None` (the live view showed bare uuids) while these tests read "Cat".
    The real model has `class_str`, and the *name* has to be resolved from the DB.
    """
    return SimpleNamespace(
        x=x1,
        y=y1,
        width=x2 - x1,
        height=y2 - y1,
        label_uuid=label_uuid,
        class_str="cat",
    )


class _FakeInferenceApi:
    """Records the payloads it is handed, so the conf floor can be asserted."""

    def __init__(self, detections: List[Any]) -> None:
        self._detections = detections
        self.payloads: List[Any] = []

    def detect_batch(self, payload: Any) -> SimpleNamespace:
        # One request carries the whole model stack now, so the fake answers in
        # that shape: results keyed by model name, plus errors for any that
        # failed. The task rebuilds its per-model list from its own model_names,
        # so this does not need to model response ordering.
        self.payloads.append(payload)
        return SimpleNamespace(
            results={name: SimpleNamespace(detections=list(self._detections)) for name in payload.model_names},
            errors={},
        )


def _task() -> StreamWatcherTask:
    """A task with only the attributes `_run_inference` touches.

    Built without `_init` so the test needs no manager, DB or watch profile — the call
    under test is deliberately narrow.
    """
    task = StreamWatcherTask.__new__(StreamWatcherTask)
    task._pin_ids = {}
    task._label_ancestry = {}
    task._label_meta = {CAT: {"text": "Cat", "color": "#ff8800"}}
    task._inference_ms = []
    return task


class TestRunInference:
    async def test_returns_detections(self):
        """The regression: this returned [] for every frame, for every model."""
        api = _FakeInferenceApi([SimpleNamespace(bbox=_bbox(0, 0, 10, 10), confidence=0.9)])

        detections = await _task()._run_inference(api, "base64-image", ["cat-model"], frame_size=(100, 100))

        assert len(detections) == 1, "watcher produced no detections — the payload is invalid again"
        assert isinstance(detections[0], Detection)
        assert detections[0].label_uuid == CAT
        assert detections[0].confidence == pytest.approx(0.9)

    async def test_requests_a_low_confidence_floor(self):
        """Thresholding belongs to the tracker, not to this call.

        ByteTrack's second association pass exists to recover boxes a conventional
        threshold discards, so filtering here would silently reduce it to plain SORT.
        """
        api = _FakeInferenceApi([])

        await _task()._run_inference(api, "base64-image", ["cat-model"], frame_size=(100, 100))

        assert len(api.payloads) == 1
        assert api.payloads[0].conf_thresh == pytest.approx(StreamWatcherTask._INFERENCE_CONF_FLOOR)
        assert StreamWatcherTask._INFERENCE_CONF_FLOOR <= 0.2, "floor too high for two-stage association"

    async def test_no_models_is_not_a_call(self):
        api = _FakeInferenceApi([SimpleNamespace(bbox=_bbox(0, 0, 10, 10), confidence=0.9)])

        assert await _task()._run_inference(api, "base64-image", [], frame_size=(100, 100)) == []
        assert api.payloads == []

    async def test_inference_failure_is_contained(self):
        """A failing model must yield no detections, not kill the frame loop."""

        class _Boom:
            def detect_batch(self, payload: Any):
                raise RuntimeError("model exploded")

        assert await _task()._run_inference(_Boom(), "base64-image", ["cat-model"], frame_size=(100, 100)) == []


class TestSecondPass:
    """The cascade: a crop classifier refines the detector's boxes on the same frame.

    `_run_inference` must return the REFINED label, because everything downstream —
    the tracker, the rules engine, notifications — reads only what comes back here.
    """

    WILLOW = "label-willow"

    class _FakeCascadeApi(_FakeInferenceApi):
        """Adds the crop endpoint on top of the detect fake."""

        def __init__(self, detections, per_box_predictions):
            super().__init__(detections)
            self._per_box = per_box_predictions
            self.crop_payloads = []

        def classify_crops(self, model_name, payload):
            self.crop_payloads.append((model_name, payload))
            return SimpleNamespace(
                status="success",
                results=[
                    SimpleNamespace(
                        predictions=[SimpleNamespace(label_uuid=label, confidence=conf) for label, conf in per_box]
                    )
                    for per_box in self._per_box[: len(payload.boxes)]
                ],
            )

    def _task_with_ancestry(self):
        task = _task()
        task._label_ancestry = {self.WILLOW: {CAT}, CAT: set()}
        task._label_meta = {**task._label_meta, self.WILLOW: {"text": "Willow", "color": "#00ff00"}}
        return task

    async def test_detection_comes_back_with_the_refined_label(self):
        api = self._FakeCascadeApi(
            [SimpleNamespace(bbox=_bbox(0, 0, 10, 10), confidence=0.8)],
            [[(self.WILLOW, 0.9), (CAT, 0.1)]],
        )
        sub = CropSubModel(model_name="cat_namer", parent_label_uuid=CAT)

        detections = await self._task_with_ancestry()._run_inference(
            api, "base64-image", ["cat-model"], frame_size=(100, 100), sub_models=[sub]
        )

        assert len(detections) == 1
        assert detections[0].label_uuid == self.WILLOW
        # Joint confidence: 0.8 that it's a cat, times 0.9 that it's this cat.
        assert detections[0].confidence == pytest.approx(0.8 * 0.9)

    async def test_crop_boxes_are_sent_in_relative_coordinates(self):
        """The cascade runs before the relative→pixel conversion. Sending pixels to
        an endpoint documented in relative coords would crop a sliver of the corner
        of every frame, and still return a confident answer about it."""
        api = self._FakeCascadeApi(
            [SimpleNamespace(bbox=_bbox(0.2, 0.3, 0.6, 0.7), confidence=0.8)],
            [[(self.WILLOW, 0.95)]],
        )
        sub = CropSubModel(model_name="cat_namer", parent_label_uuid=CAT)

        await self._task_with_ancestry()._run_inference(
            api, "base64-image", ["cat-model"], frame_size=(1920, 1080), sub_models=[sub]
        )

        _model, payload = api.crop_payloads[0]
        assert len(payload.boxes) == 1
        assert payload.boxes[0].x == pytest.approx(0.2)
        assert payload.boxes[0].width == pytest.approx(0.4)

    async def test_unsure_second_pass_keeps_the_coarse_detection(self):
        """Every class below the bar: the box keeps its label AND its confidence."""
        api = self._FakeCascadeApi(
            [SimpleNamespace(bbox=_bbox(0, 0, 10, 10), confidence=0.8)],
            [[(self.WILLOW, 0.4), ("label-maple", 0.35)]],
        )
        sub = CropSubModel(model_name="cat_namer", parent_label_uuid=CAT)

        detections = await self._task_with_ancestry()._run_inference(
            api, "base64-image", ["cat-model"], frame_size=(100, 100), sub_models=[sub]
        )

        assert detections[0].label_uuid == CAT
        assert detections[0].confidence == pytest.approx(0.8)

    async def test_second_pass_failure_does_not_lose_the_frame(self):
        """This runs on every frame of a live camera. A broken second pass must cost
        the refinement, not the detection."""

        class _BrokenCropApi(_FakeInferenceApi):
            def classify_crops(self, model_name, payload):
                raise RuntimeError("inference server unreachable")

        api = _BrokenCropApi([SimpleNamespace(bbox=_bbox(0, 0, 10, 10), confidence=0.8)])
        sub = CropSubModel(model_name="cat_namer", parent_label_uuid=CAT)

        detections = await self._task_with_ancestry()._run_inference(
            api, "base64-image", ["cat-model"], frame_size=(100, 100), sub_models=[sub]
        )

        assert len(detections) == 1
        assert detections[0].label_uuid == CAT


class TestNestedSecondPass:
    """Three levels deep, in one frame: animal -> cat -> willow.

    Each sub-model reads the label the previous one wrote, so nesting depth is
    bounded only by how many sub-models are configured. Asserted through
    `_run_inference` rather than the helpers, because the chain depends on the
    passes sharing one mutable detection list — a refactor that copied it would
    pass the helper tests and stop the chain dead here.
    """

    ANIMAL, CAT_L, WILLOW = "label-animal", "label-cat", "label-willow"

    class _PerModelCropApi(_FakeInferenceApi):
        """Answers classify_crops differently per model, like real sub-models."""

        def __init__(self, detections, by_model):
            super().__init__(detections)
            self._by_model = by_model
            self.calls = []

        def classify_crops(self, model_name, payload):
            self.calls.append((model_name, [b.x for b in payload.boxes]))
            label, conf = self._by_model[model_name]
            return SimpleNamespace(
                status="success",
                results=[
                    SimpleNamespace(predictions=[SimpleNamespace(label_uuid=label, confidence=conf)])
                    for _ in payload.boxes
                ],
            )

    def _task(self):
        task = _task()
        task._label_ancestry = {
            self.CAT_L: {self.ANIMAL},
            self.WILLOW: {self.CAT_L, self.ANIMAL},
            self.ANIMAL: set(),
        }
        task._label_meta = {}
        return task

    async def test_three_levels_resolve_in_one_frame(self):
        api = self._PerModelCropApi(
            [SimpleNamespace(bbox=_bbox(0, 0, 10, 10, label_uuid=self.ANIMAL), confidence=1.0)],
            {"animal_sub": (self.CAT_L, 1.0), "cat_sub": (self.WILLOW, 1.0)},
        )
        # Deliberately configured INNERMOST first — the ordering fix is what makes
        # this resolve. Before it, the cat sub-model ran while the box still said
        # "animal", matched nothing, and the frame stopped at "cat".
        subs = [
            CropSubModel(model_name="cat_sub", parent_label_uuid=self.CAT_L),
            CropSubModel(model_name="animal_sub", parent_label_uuid=self.ANIMAL),
        ]

        detections = await self._task()._run_inference(
            api, "base64-image", ["det"], frame_size=(100, 100), sub_models=subs
        )

        assert [c[0] for c in api.calls] == ["animal_sub", "cat_sub"], "passes ran inner-first"
        assert detections[0].label_uuid == self.WILLOW

    async def test_a_stalled_level_stops_the_chain_without_breaking_it(self):
        """If the middle pass declines, the deepest one has nothing matching its own
        boundary and the detection rests at the last confident level."""
        api = self._PerModelCropApi(
            [SimpleNamespace(bbox=_bbox(0, 0, 10, 10, label_uuid=self.ANIMAL), confidence=1.0)],
            {"animal_sub": (self.CAT_L, 0.1), "cat_sub": (self.WILLOW, 1.0)},
        )
        subs = [
            CropSubModel(model_name="animal_sub", parent_label_uuid=self.ANIMAL),
            CropSubModel(model_name="cat_sub", parent_label_uuid=self.CAT_L),
        ]

        detections = await self._task()._run_inference(
            api, "base64-image", ["det"], frame_size=(100, 100), sub_models=subs
        )

        assert detections[0].label_uuid == self.ANIMAL
        assert api.calls == [("animal_sub", [0.0])], "the deeper pass had nothing to refine"


class TestSplitModelStacks:
    """Startup wiring: crop classifiers must be moved out of the detector list."""

    def _task_with_pipeline(self, model_names):
        task = _task()
        task._label_meta = {}
        pipeline = SimpleNamespace(source_name="cam", model_names=list(model_names), sub_models=[])
        task._pipelines = [pipeline]
        task.log = lambda *a, **kw: None
        return task, pipeline

    def test_crop_classifier_is_moved_out_of_the_detector_list(self):
        """Left in `model_names` it would be sent to detect-batch every frame — which
        answers rather than errors, so the damage would be silent."""
        task, pipeline = self._task_with_pipeline(["yolo", "namer"])
        metas = {
            "yolo": SimpleNamespace(model_kind="object_detection", parent_label_uuid=None),
            "namer": SimpleNamespace(model_kind="crop_classifier", parent_label_uuid=CAT),
        }

        task._split_model_stacks(metas)

        assert pipeline.model_names == ["yolo"]
        assert pipeline.sub_models == [CropSubModel(model_name="namer", parent_label_uuid=CAT)]

    def test_a_stack_with_no_sub_models_is_unchanged(self):
        """The overwhelmingly common case: the cascade must cost nothing when unused."""
        task, pipeline = self._task_with_pipeline(["yolo_a", "yolo_b"])
        metas = {
            "yolo_a": SimpleNamespace(model_kind="object_detection", parent_label_uuid=None),
            "yolo_b": SimpleNamespace(model_kind="detector", parent_label_uuid=None),
        }

        task._split_model_stacks(metas)

        assert pipeline.model_names == ["yolo_a", "yolo_b"]
        assert pipeline.sub_models == []

    def test_models_with_unreadable_meta_stay_detectors(self):
        """Meta reads are best-effort (a remote inference server may not answer). The
        fallback must keep detecting, not silently empty the stack."""
        task, pipeline = self._task_with_pipeline(["yolo"])

        task._split_model_stacks({})

        assert pipeline.model_names == ["yolo"]


class TestLabelNames:
    """Inference returns a label *uuid*; humans need a *name*.

    The watcher read `bbox.label_name`, which the client model has never had, so every
    detection carried `label_name=None`. That propagates: the tracker copies the name
    onto the track, so the status table and the live view both fell back to raw uuids —
    "#1 3f2a9c1e-…" is the symptom this class exists to prevent.
    """

    async def test_label_name_comes_from_the_db_label_map(self):
        api = _FakeInferenceApi([SimpleNamespace(bbox=_bbox(0, 0, 10, 10), confidence=0.9)])

        detections = await _task()._run_inference(api, "img", ["cat-model"], frame_size=(100, 100))

        assert detections[0].label_name == "Cat"

    async def test_unmapped_class_falls_back_to_the_models_own_word(self):
        """A model class with no label mapping is a configuration problem — showing the
        model's class name makes that legible, where a uuid just looks broken."""
        task = _task()
        task._label_meta = {}
        api = _FakeInferenceApi([SimpleNamespace(bbox=_bbox(0, 0, 10, 10, label_uuid=None), confidence=0.9)])

        detections = await task._run_inference(api, "img", ["cat-model"], frame_size=(100, 100))

        assert detections[0].label_name == "cat"

    async def test_a_real_client_bbox_has_no_label_name_to_read(self):
        """Pins the shape that caused the bug, so the fake above cannot drift back."""
        from inference_client.models.bounding_box_metadata_model import (
            BoundingBoxMetadataModel,
        )

        assert "label_name" not in BoundingBoxMetadataModel.model_fields


class TestRealClientBboxShape:
    """The watcher crashed with `'BoundingBoxMetadataModel' object has no attribute
    'x1'` on every frame, while every test here passed.

    The fake above invented `x1/y1/x2/y2`. The real generated client has only
    `x, y, width, height`, and they are RELATIVE (0..1) — so the task both read
    fields that do not exist and, once they did, would have handed the tracker
    fractions where it documents pixels. Zones are scaled to the analysis frame's
    pixel size (`WatchPipeline.prepare_frame` → `engine.set_frame_size`), so a
    relative box silently collapses every rule into the top-left corner.

    These tests build the ACTUAL client model, so no fake can drift from it again.
    """

    def _real_bbox(self, x, y, w, h, label_uuid=CAT):
        from inference_client.models.bounding_box_metadata_model import (
            BoundingBoxMetadataModel,
        )

        return BoundingBoxMetadataModel(uuid="bbox-1", x=x, y=y, width=w, height=h, label_uuid=label_uuid)

    async def test_real_client_bbox_does_not_have_corner_attributes(self):
        """Pins the shape the task must code against."""
        bbox = self._real_bbox(0.1, 0.2, 0.3, 0.4)
        assert not hasattr(bbox, "x1")
        assert (bbox.x, bbox.y, bbox.width, bbox.height) == (0.1, 0.2, 0.3, 0.4)

    async def test_detections_are_converted_to_analysis_frame_pixels(self):
        """A relative box becomes x1,y1,x2,y2 in the pixel space of the frame that
        was actually sent to inference — the same space the zones are scaled to."""
        api = _FakeInferenceApi([SimpleNamespace(bbox=self._real_bbox(0.1, 0.2, 0.3, 0.4), confidence=0.9)])

        detections = await _task()._run_inference(api, "base64-image", ["cat-model"], frame_size=(640, 480))

        assert len(detections) == 1
        x1, y1, x2, y2 = detections[0].bbox
        assert (x1, y1) == pytest.approx((64.0, 96.0))
        assert (x2, y2) == pytest.approx((256.0, 288.0))

    async def test_a_full_frame_box_spans_the_whole_frame(self):
        api = _FakeInferenceApi([SimpleNamespace(bbox=self._real_bbox(0.0, 0.0, 1.0, 1.0), confidence=0.5)])

        detections = await _task()._run_inference(api, "base64-image", ["cat-model"], frame_size=(640, 480))

        assert detections[0].bbox == pytest.approx((0.0, 0.0, 640.0, 480.0))
