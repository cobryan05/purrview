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

    def detect_unit(self, payload: Any) -> SimpleNamespace:
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
    task._label_groups = {}
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
            def detect_unit(self, payload: Any):
                raise RuntimeError("model exploded")

        assert await _task()._run_inference(_Boom(), "base64-image", ["cat-model"], frame_size=(100, 100)) == []


class TestSubModelsRideAlongInTheRequest:
    """What the watcher still owns of the cascade: the request it builds.

    The cascade itself moved into inference_server's detect-unit — one implementation,
    shared with the canvas and the review preview — and is tested there
    (tests/inference_server/test_detect_unit.py). Duplicating the refinement assertions
    here would cover one rule at two layers, which CLAUDE.md rules out.

    These pin the wiring whose loss is silent: send the sub-models with the detectors, or
    no cascade happens at all; send the confidence floor, or the endpoint's default
    overwrites a true "a cat" with a hesitant "probably Willow".
    """

    CAT_L, WILLOW = "label-cat", "label-willow"

    def _task_with_ancestry(self):
        task = _task()
        task._label_ancestry = {self.WILLOW: {self.CAT_L}, self.CAT_L: set()}
        return task

    async def test_sub_models_are_sent_with_the_detectors(self):
        """One request, one frame decode — and the refinement happens before the answer
        comes back, so there is no coarse intermediate for this task to see."""
        api = _FakeInferenceApi([SimpleNamespace(bbox=_bbox(0, 0, 10, 10), confidence=0.8)])
        sub = CropSubModel(model_name="cat_namer", parent_label_uuid=self.CAT_L)

        await self._task_with_ancestry()._run_inference(
            api, "base64-image", ["cat-model"], frame_size=(100, 100), sub_models=[sub]
        )

        assert len(api.payloads) == 1, "the second pass must not cost a second round trip"
        assert api.payloads[0].model_names == ["cat-model", "cat_namer"]

    async def test_the_watchers_confidence_floor_is_sent_not_defaulted(self):
        """Now that the refinement runs server-side, this floor only applies if it
        travels with the request. Left off, the endpoint takes every classifier's best
        guess and a 0.2-confidence "Willow" silently replaces a true "Cat"."""
        api = _FakeInferenceApi([])
        sub = CropSubModel(model_name="cat_namer", parent_label_uuid=self.CAT_L)
        task = self._task_with_ancestry()

        await task._run_inference(api, "base64-image", ["cat-model"], frame_size=(100, 100), sub_models=[sub])

        assert api.payloads[0].min_sub_confidence == task._SUB_MODEL_MIN_CONFIDENCE

    async def test_the_label_tree_is_sent_so_foreign_detections_qualify(self):
        """A unit describes the labels it can emit, but not one emitted by another model
        in the same stack. That case exists only if the real ancestry is sent."""
        api = _FakeInferenceApi([])
        sub = CropSubModel(model_name="cat_namer", parent_label_uuid=self.CAT_L)

        await self._task_with_ancestry()._run_inference(
            api, "base64-image", ["cat-model"], frame_size=(100, 100), sub_models=[sub]
        )

        assert api.payloads[0].label_ancestry[self.WILLOW] == [self.CAT_L]


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


PERSON = "label-person"
ALICE = "label-alice"
PACKAGE = "label-package"

#: As the live tree has it: `Person` is an exclusive-variants root, `Alice` one of its
#: variants, `Package` unrelated.
PERSON_GROUP = {PERSON: PERSON, ALICE: PERSON, PACKAGE: PACKAGE, CAT: CAT}


def _grouped_task() -> StreamWatcherTask:
    task = _task()
    task._label_groups = dict(PERSON_GROUP)
    task._label_meta = {PERSON: {"text": "Person", "color": "#ffffff"}}
    return task


def _alt(label_uuid: str, confidence: float) -> SimpleNamespace:
    return SimpleNamespace(label_uuid=label_uuid, confidence=confidence, class_str=None, class_id=0)


class TestRunnerUpScores:
    """The watcher must ask for the runner-up classes and keep the ones it can vote on.

    Without them the label vote sees only per-frame argmax winners, which is the weaker
    estimator: two classes that genuinely alternate as winner for seconds can still tip
    a winner-count, but cannot tip a sum over both their scores.
    """

    @staticmethod
    def _requested_top_k(payload: Any, model_name: str) -> int:
        """The n-best ask for one model, through the generated anyOf wrapper.

        `top_k` is unioned per model (see `frameInference._union_requests`), so the
        wire value is a dict even when one consumer asked with a plain int.
        """
        value = getattr(payload.top_k, "actual_instance", payload.top_k)
        return value.get(model_name, 0) if isinstance(value, dict) else int(value or 0)

    async def test_top_k_covers_the_largest_group(self):
        api = _FakeInferenceApi([SimpleNamespace(bbox=_bbox(0, 0, 10, 10, PERSON), confidence=0.9)])
        task = _grouped_task()
        # Three cats plus the root is the widest set any vote sums over.
        task._label_groups.update({"label-luna": CAT, "label-willow": CAT, "label-maple": CAT})

        await task._run_inference(api, "img", ["m"], frame_size=(100, 100))

        asked = self._requested_top_k(api.payloads[0], "m")
        assert asked >= 4, "a vote cannot see a candidate inference was never asked for"

    async def test_no_groups_means_no_runner_ups_requested(self):
        """Nothing to vote over is the old behaviour, and must cost nothing."""
        api = _FakeInferenceApi([SimpleNamespace(bbox=_bbox(0, 0, 10, 10), confidence=0.9)])

        await _task()._run_inference(api, "img", ["m"], frame_size=(100, 100))

        assert self._requested_top_k(api.payloads[0], "m") == 0

    async def test_in_group_runner_ups_reach_the_detection(self):
        api = _FakeInferenceApi(
            [
                SimpleNamespace(
                    bbox=_bbox(0, 0, 10, 10, PERSON),
                    confidence=0.72,
                    alternates=[_alt(ALICE, 0.68), _alt(PACKAGE, 0.51)],
                )
            ]
        )

        detections = await _grouped_task()._run_inference(api, "img", ["m"], frame_size=(100, 100))

        # `Package` is dropped: it is not an alternate guess about this object, so no
        # vote can use it and carrying it would only be noise downstream.
        assert detections[0].alternates == [(ALICE, pytest.approx(0.68))]

    async def test_an_ungrouped_winner_carries_no_alternates(self):
        api = _FakeInferenceApi(
            [SimpleNamespace(bbox=_bbox(0, 0, 10, 10, PACKAGE), confidence=0.9, alternates=[_alt(PERSON, 0.4)])]
        )

        detections = await _grouped_task()._run_inference(api, "img", ["m"], frame_size=(100, 100))

        assert detections[0].alternates == []

    async def test_a_model_that_cannot_produce_them_still_works(self):
        """NMS-free exports bake the argmax in. The vote degrades, nothing breaks."""
        api = _FakeInferenceApi([SimpleNamespace(bbox=_bbox(0, 0, 10, 10, PERSON), confidence=0.9)])

        detections = await _grouped_task()._run_inference(api, "img", ["m"], frame_size=(100, 100))

        assert len(detections) == 1
        assert detections[0].alternates == []
