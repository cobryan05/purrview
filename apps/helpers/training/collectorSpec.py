"""The configuration half of training-image collection, independent of who runs it.

`CollectTrainingImagesTask` owned all of this inline, which was fine while the task
was the only thing that collected. It is about to stop being that: a watcher will
drive the same collection logic over frames it is already pulling (see
`COLLECTOR_UNIFICATION_PLAN.md`), and a watcher has no `source_uuids`, no
`snapshot_interval` and no provider rotation.

So the split here is deliberately drawn at **"what does collection need to decide
what to keep"** versus **"who supplies the frames"**:

* `CollectorSpec` — models, confidence bands, tags, filter chain, ambiguity, dedup,
  output directory, stop conditions. Everything that answers *should I save this
  frame, and how should it be annotated*.
* Left with the driver — `source_uuids`, `snapshot_interval`, `stream_interval`,
  the providers, the backoff, the resume bookkeeping.

`collector_params_schema()` exists so those knobs have exactly one definition.
`CollectTrainingImagesTask.params_schema()` splices it in beside its three loop
params, and the collector editor renders it directly. Adding a knob in one place
is the whole point — the alternative is two copies that drift.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from apps.helpers.detectionMerger import MergePolicy
from apps.helpers.logUtils import LOG_LEVEL

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)


# Two detections from different models are treated as the same physical object when
# their boxes overlap at or above this IoU; the higher-priority (earlier-configured)
# model's detection is kept.
SAME_BBOX_IOU_THRESHOLD = 0.65


# Ambiguity metrics over the top-2 class scores (p1 >= p2) of one detection.
#
# RATIO is the default because YOLO v8+ class scores are independent sigmoids,
# not a softmax: they don't sum to 1 and their absolute scale drifts with a
# label's training-set frequency, so a fixed margin means different things at
# 0.9/0.75 than at 0.3/0.15. The ratio (0.83 vs 0.5) separates those correctly.
# MARGIN is kept for softmax classifiers, where the classes genuinely compete.
AMBIGUITY_METRIC_RATIO = "ratio"
AMBIGUITY_METRIC_MARGIN = "margin"

AMBIGUITY_KIND_SAME_MODEL = "same_model"
AMBIGUITY_KIND_CROSS_MODEL = "cross_model"


@dataclass
class ConfidenceBand:
    """A `[min, max]` confidence range for a target label (Phase 5B of
    LIVE_COLLECTION_PLAN.md). Both bounds are INCLUSIVE:
    `min_confidence <= confidence <= max_confidence` is "in band".

    The wire format's bare scalar (`labelValues: {labelUuid: 0.25}`, every
    already-saved task config) means `{min: 0.25, max: 1.0}` — today's
    floor-only behaviour, unchanged. See `parse_confidence_band`.

    `review_disabled` is the mirror image of "min <= 0 disables the label":
    an explicit `max` of 0 means "still tag this label, but never let it be
    the reason a frame gets saved for review" — set when the UI's 'Review
    up to' field is 0. `contains()` always returns False in that state,
    regardless of `max_confidence` (which is otherwise unused/meaningless).
    """

    min_confidence: float
    max_confidence: float = 1.0
    review_disabled: bool = False

    def contains(self, confidence: float) -> bool:
        if self.review_disabled:
            return False
        return self.min_confidence <= confidence <= self.max_confidence


@dataclass
class TargetLabel:
    label_uuid: str
    band: ConfidenceBand


@dataclass
class AmbiguityInfo:
    """Why one detection was judged confusable, and by which model.

    Attribution is the whole point (AMBIGUITY_REVIEW_PLAN.md §1): the question
    being answered is "is *this* model confused, and with what?", so `model` and
    `rival_*` are not optional colour — they are the payload.

    Two kinds, and the distinction matters when reading the results:
      - SAME_MODEL: one model scored two classes almost equally on one box. This
        is a statement about that model's own decision boundary.
      - CROSS_MODEL: two different models put incompatible labels on the same
        object. A statement about the two models disagreeing, not about either
        one's internal uncertainty — `rival_model` names the other.
    """

    metric: str
    value: float
    model: str
    rival_label_uuid: Optional[str]
    rival_class_str: Optional[str]
    rival_confidence: float
    kind: str = AMBIGUITY_KIND_SAME_MODEL
    rival_model: Optional[str] = None

    def as_metadata(self) -> dict[str, str]:
        """Flat, string-valued metadata_json keys.

        ⚠️ Flat and stringly-typed on purpose, not for lack of ambition:
        `metadata_json` is typed `RawJsonDict = Dict[str, str]` (helpers/types.py)
        end to end — server model, generated client, and every existing reader.
        A nested dict is rejected outright by the client's validation, and
        widening the type would change the contract for every bbox-metadata
        consumer in the app to serve one feature. So this follows the convention
        the existing keys already use (`confidence`, `source_model`,
        `source_confidence`): flat keys, string values.
        """
        meta = {
            "ambiguity_kind": self.kind,
            "ambiguity_metric": self.metric,
            "ambiguity_value": str(round(self.value, 4)),
            "ambiguity_model": self.model,
            "ambiguity_rival_label_uuid": self.rival_label_uuid or "",
            "ambiguity_rival_class_str": self.rival_class_str or "",
            "ambiguity_rival_confidence": str(round(self.rival_confidence, 4)),
        }
        if self.rival_model:
            meta["ambiguity_rival_model"] = self.rival_model
        return meta


def ambiguity_value(metric: str, top: float, rival: float) -> float:
    """The chosen metric's value for one (winner, runner-up) pair.

    Guards `top == 0`: a zero-confidence winner can't be meaningfully "close to"
    anything, and dividing by it would produce inf/NaN that then compares True
    against every threshold.
    """
    if metric == AMBIGUITY_METRIC_MARGIN:
        return top - rival
    if top <= 0:
        return 0.0
    return rival / top


def ambiguity_fires(metric: str, value: float, threshold: float) -> bool:
    """Threshold direction differs by metric — a HIGH ratio and a LOW margin both
    mean "the runner-up is close". Both bounds inclusive, matching
    ConfidenceBand.contains."""
    if metric == AMBIGUITY_METRIC_MARGIN:
        return value <= threshold
    return value >= threshold


def parse_confidence_band(raw: Any) -> Optional[ConfidenceBand]:
    """Parses one `labelValues` entry into a `ConfidenceBand`.

    Params are unvalidated raw JSON (CLAUDE.md) — this must not raise on garbage
    input. Accepts either:
      - the legacy scalar floor, e.g. `0.25`               -> {min: 0.25, max: 1.0}
      - the Phase 5B band dict, e.g. `{"min": 0.2, "max": 0.6}`

    Returns None (label disabled — matches the pre-Phase-5B "skip
    zero-confidence" convention, where the UI writes 0 to mean "not
    collecting this label") when the parsed floor is <= 0 or the value can't
    be parsed as a number at all.

    An explicit `max` of 0 in the band dict means "review disabled" — the
    label still gets tagged (min still applies in `_annotatable`), but it
    can never itself trigger a frame save (see `ConfidenceBand.contains`).
    """
    if isinstance(raw, dict):
        try:
            min_f = float(raw.get("min", 0.0))
        except (TypeError, ValueError):
            return None
        try:
            max_f = float(raw.get("max", 1.0))
        except (TypeError, ValueError):
            max_f = 1.0
    else:
        try:
            min_f = float(raw)
        except (TypeError, ValueError):
            return None
        max_f = 1.0

    if min_f <= 0:
        return None

    min_f = max(0.0, min(1.0, min_f))

    review_disabled = max_f <= 0
    if review_disabled:
        # max_confidence is meaningless while review is disabled (contains()
        # short-circuits on the flag); keep it valid for the merge-union math
        # in CollectorSpec.from_params.
        max_f = 1.0
    else:
        # Defensive clamps: keep both bounds in [0, 1] and max >= min, so a
        # malformed/inverted band (e.g. {"min": 0.6, "max": 0.2}) degrades to
        # a single-point band rather than silently never matching or
        # matching everything.
        max_f = max(0.0, min(1.0, max_f))
        if max_f < min_f:
            max_f = min_f
    return ConfidenceBand(min_confidence=min_f, max_confidence=max_f, review_disabled=review_disabled)


@dataclass
class CollectorSpec:
    """Everything collection needs to decide what to keep and how to annotate it.

    Built by `from_params` from the same raw JSON a task config carries, so a
    standalone task and a watcher-attached collector are configured by one code
    path and cannot drift.

    Deliberately holds no API clients, no manager and no provider — it is inert
    configuration. The runtime that acts on it owns all of those.
    """

    # ── models and per-label bands ────────────────────────────────────────────
    #: Config order IS model priority: earlier-configured models win when two
    #: models produce overlapping (same-bbox) detections.
    model_names: List[str] = field(default_factory=list)
    target_labels: List[TargetLabel] = field(default_factory=list)
    #: No models configured at all → collect-everything mode: no inference runs and
    #: every offered frame is saved unannotated, after the filter chain and dedup
    #: have still had their say. Deliberately keyed on "no models", not "no target
    #: labels": a model whose labels are all at zero confidence is an explicitly
    #: disabled model, and keeps its existing behaviour of collecting nothing.
    collect_all: bool = True
    #: Network-level floor sent as the inference call's conf_thresh — the loosest
    #: (lowest) min across all configured bands, so no in-band detection for ANY
    #: target label is ever filtered out before it even reaches the collector. A
    #: coarse pre-merge floor only, never a substitute for the per-label band
    #: check, which happens post-merge.
    #:
    #: ⚠️ Being the loosest floor has a cost that bites when inference is SHARED
    #: with a watcher (whose own floor is 0.1): the raw detection list is then
    #: sized by that floor, and anything O(n²) over it — the hierarchy merge, the
    #: cross-model ambiguity scan — is priced accordingly. Measured in NOTES.md at
    #: 63 ms vs 18 ms. A shared-inference driver must prefilter to each
    #: detection's own label floor before those stages.
    min_all_conf: float = 0.5
    overlap_iou_threshold: float = SAME_BBOX_IOU_THRESHOLD
    merge_policy: MergePolicy = MergePolicy.FIRST_WINS

    # ── output ────────────────────────────────────────────────────────────────
    #: Relative to the dataset root. Validated against traversal in `from_params`.
    output_dir: str = ""

    # ── tags ──────────────────────────────────────────────────────────────────
    apply_tag_uuids: List[str] = field(default_factory=list)
    generate_task_tag: bool = False
    task_tag_name_override: str = ""

    # ── filter chain ──────────────────────────────────────────────────────────
    filter_chain_uuid: Optional[str] = None

    # ── stop conditions (both None = unlimited) ───────────────────────────────
    max_images: Optional[int] = None
    max_duration_s: Optional[float] = None

    # ── ambiguity ("confusable detections") ───────────────────────────────────
    enable_ambiguity: bool = False
    ambiguity_metric: str = AMBIGUITY_METRIC_RATIO
    ambiguity_threshold: float = 0.7
    ambiguity_top_k: int = 3
    ambiguity_scope: str = "configured"
    ambiguity_cross_model: bool = True
    ambiguity_min_confidence: float = 0.0

    # ── deduplication ─────────────────────────────────────────────────────────
    enable_dedup: bool = False
    image_dup_threshold: int = 5
    bbox_dup_threshold: int = 8
    duplicate_action: str = "discard"

    # ── detector-miss collection (plan §4A, watcher-only) ─────────────────────
    #: Off by default: only meaningful for a watcher-attached collector (needs a
    #: tracker), and even there it changes what gets saved, so it's opt-in
    #: rather than a silent behaviour change for existing configs.
    enable_detector_miss: bool = False

    # ── per-track novelty sampling (plan §4B, watcher-only) ───────────────────
    #: Safety valve, NOT the rule: appearance novelty decides what to keep, and
    #: this only bounds the pathological case of an animal in constant motion for
    #: an hour. The plan is explicit that it belongs around 20–30, not 5 — a low
    #: cap throws away the pose variation that is the whole point of the feature.
    max_frames_per_track: int = 25

    @property
    def target_label_bands(self) -> Dict[str, ConfidenceBand]:
        return {t.label_uuid: t.band for t in self.target_labels}

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "CollectorSpec":
        """Parse raw, untrusted param JSON into a validated spec.

        Raises only for `output_dir`, which is a security boundary (see below).
        Everything else degrades to a documented default rather than raising:
        these params come from stored JSON that predates several of the fields,
        and a task that refuses to start because one knob is malformed is worse
        than one that runs with the default for it.
        """
        output_dir = str(params.get("output_dir") or "")
        # output_dir is joined onto files_root when writing collected images.
        # Reject absolute paths and '..' traversal so collection can't write
        # outside the dataset root (raw params are untrusted — validate here per
        # CLAUDE.md). This one DOES raise: silently rewriting a traversal to
        # something safe would hide an attempt worth seeing.
        _out = Path(output_dir)
        if _out.is_absolute() or ".." in _out.parts:
            raise ValueError(f"Invalid output_dir {output_dir!r}: must be a relative path within the dataset root")

        # model_label_config: [{modelName, labelValues: {labelUuid: confidence | {min, max}}}]
        # (Phase 5B: labelValues entries are either the legacy scalar floor or a
        # {min, max} confidence band — see parse_confidence_band.)
        model_label_config: list[dict] = params.get("model_label_config", []) or []
        ordered_model_names: list[str] = []
        # Flatten to label_uuid → ConfidenceBand; skip disabled (parsed as None,
        # e.g. zero-confidence, matching the pre-band "disabled in UI" convention).
        label_bands: dict[str, ConfidenceBand] = {}
        for item in model_label_config:
            name = item.get("modelName")
            if name and name not in ordered_model_names:
                ordered_model_names.append(name)
            for label_uuid, raw_band in (item.get("labelValues", {}) or {}).items():
                band = parse_confidence_band(raw_band)
                if band is None:
                    continue
                existing = label_bands.get(label_uuid)
                if existing is not None:
                    # Two model configs target the same label: widen to the union
                    # (loosest min, loosest max) so a detection in-band for either
                    # config's range still counts. For the all-scalar case (max
                    # always 1.0) this reduces to the pre-Phase-5B behaviour of
                    # taking the min of the two floors. Review is only disabled
                    # for the merged label if BOTH configs disabled it — one
                    # config still wanting review for this label should win.
                    band = ConfidenceBand(
                        min_confidence=min(existing.min_confidence, band.min_confidence),
                        max_confidence=max(existing.max_confidence, band.max_confidence),
                        review_disabled=existing.review_disabled and band.review_disabled,
                    )
                label_bands[label_uuid] = band

        target_labels = [TargetLabel(label_uuid=u, band=b) for u, b in label_bands.items()]

        try:
            merge_policy = MergePolicy(params.get("merge_policy", MergePolicy.FIRST_WINS.value))
        except ValueError:
            merge_policy = MergePolicy.FIRST_WINS

        _max_images_raw = params.get("max_images")
        _max_duration_raw = params.get("max_duration")

        _metric_raw = str(params.get("ambiguity_metric", AMBIGUITY_METRIC_RATIO) or AMBIGUITY_METRIC_RATIO)
        _scope_raw = str(params.get("ambiguity_scope", "configured") or "configured")

        return cls(
            model_names=ordered_model_names,
            target_labels=target_labels,
            collect_all=not ordered_model_names,
            min_all_conf=min((b.min_confidence for b in label_bands.values()), default=0.5),
            overlap_iou_threshold=_as_float(params.get("overlap_iou_threshold"), SAME_BBOX_IOU_THRESHOLD),
            merge_policy=merge_policy,
            output_dir=output_dir,
            apply_tag_uuids=list(params.get("apply_tags", []) or []),
            generate_task_tag=bool(params.get("generate_task_tag", False)),
            task_tag_name_override=params.get("task_tag_name", "") or "",
            filter_chain_uuid=params.get("filter_chain_uuid") or None,
            max_images=int(_max_images_raw) if _max_images_raw else None,
            max_duration_s=float(_max_duration_raw) if _max_duration_raw else None,
            enable_ambiguity=bool(params.get("enable_ambiguity", False)),
            ambiguity_metric=(
                _metric_raw
                if _metric_raw in (AMBIGUITY_METRIC_RATIO, AMBIGUITY_METRIC_MARGIN)
                else AMBIGUITY_METRIC_RATIO
            ),
            ambiguity_threshold=_as_float(params.get("ambiguity_threshold"), 0.7),
            # top_k counts the winner, so k=1 would leave no runner-up at all;
            # floor at 2 whenever the feature is on.
            ambiguity_top_k=max(2, _as_int(params.get("ambiguity_top_k"), 3)),
            ambiguity_scope=_scope_raw if _scope_raw in ("configured", "any") else "configured",
            # Cross-model disagreement needs no runner-up scores, so it is the only
            # confusion signal available for NMS-free/E2E models. Defaults on: with a
            # single model configured it is inert anyway.
            ambiguity_cross_model=bool(params.get("ambiguity_cross_model", True)),
            ambiguity_min_confidence=_as_float(params.get("ambiguity_min_confidence"), 0.0),
            enable_dedup=bool(params.get("enable_dedup", False)),
            image_dup_threshold=_as_int(params.get("image_dup_threshold"), 5),
            bbox_dup_threshold=_as_int(params.get("bbox_dup_threshold"), 8),
            duplicate_action=params.get("duplicate_action", "discard") or "discard",
            enable_detector_miss=bool(params.get("enable_detector_miss", False)),
            # Floored at 1: a 0 would make every track's kept-hash deque
            # zero-length, so nothing would ever compare as non-novel and the
            # prefilter would silently become a no-op.
            max_frames_per_track=max(1, _as_int(params.get("max_frames_per_track"), 25)),
        )


def collector_params_schema() -> dict[str, dict[str, Any]]:
    """The form schema for everything `CollectorSpec` reads — and nothing else.

    Deliberately excludes the three params that belong to whoever drives the loop
    (`source_uuids`, `snapshot_interval`, `stream_interval`):
    `CollectTrainingImagesTask.params_schema()` splices those back in, and the
    collector editor renders this as-is. One definition, two renderers — without
    that, every future knob has to be added in two places and they drift.

    `meta.order` is the collector's own ordering; the task inserts its loop params
    into a copy of it rather than mutating this one.
    """
    return {
        "meta": {
            "order": [
                "output_dir",
                "model_label_config",
                "overlap_iou_threshold",
                "merge_policy",
                "apply_tags",
                "generate_task_tag",
                "task_tag_name",
                "filter_chain_uuid",
                "max_images",
                "max_duration",
                "ambiguity_group",
                "dedup_group",
                "enable_detector_miss",
                "max_frames_per_track",
            ]
        },
        "output_dir": {
            "type": "directory",
            "label": "Output Directory",
            "required": True,
            "description": "Directory relative to image dataset root where images will be saved",
        },
        "model_label_config": {
            "type": "array",
            "required": False,
            "label": "Model/Label Configuration",
            "description": (
                "Models to run on each frame, and the per-label confidence range that decides what "
                "gets tagged and what gets collected. The two bounds do different jobs:\n\n"
                "• Tag from — detections at or above this get an unverified box on any frame that "
                "is saved, no matter how confident they are. Set to 0 to disable the label.\n"
                "• Review up to — a frame is only collected when some detection falls at or below "
                "this. Detections above it are still boxed on frames saved for something else; "
                "they just aren't a reason to save a frame on their own. Set to 0 to tag the label "
                "without ever letting it trigger a review on its own — it'll still be boxed on any "
                "frame saved for another reason.\n\n"
                "That split is the point: a detection your model is already certain about "
                "(say a cat at 0.95) teaches you nothing at review time, while the 0.4 guesses are "
                "where your labelling actually adds signal. Lower 'Review up to' to collect only "
                "the uncertain cases; leave it at 1 to collect everything above the floor.\n\n"
                "Leave the whole list empty to collect every frame unannotated (the filter chain "
                "and deduplication settings below still apply)."
            ),
            "allow_group_insert": True,
            "items": {
                "type": "model_label",
                "label": "Model/Label selection",
                "required": True,
                "description": "Labels to collect",
                "min_label": "Tag from",
                "max_label": "Review up to",
                "band_help": (
                    "Select a model, then set each label's range: tag anything at or above "
                    "'Tag from' (0 disables the label), but only save the frame for review when a "
                    "detection is also at or below 'Review up to' (0 disables review for this label "
                    "without disabling tagging)."
                ),
            },
        },
        "overlap_iou_threshold": {
            "type": "float",
            "label": "Multi-Model Overlap IoU Threshold",
            "description": (
                "When multiple models detect the same region, boxes from later-configured models are discarded "
                "if they overlap a higher-priority model's box at or above this IoU value (0–1). "
                "Lower values discard more aggressively; higher values allow more overlap before deduplication."
            ),
            "default": SAME_BBOX_IOU_THRESHOLD,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
        },
        "merge_policy": {
            "type": "select",
            "label": "Multi-Model Merge Policy",
            "description": (
                "When two models produce a same-label, overlapping detection, which one wins: "
                "the higher-priority (earlier-configured) model, or the lower-priority (later-configured) one. "
                "Hierarchy supersede (a more specific label beats a coarser ancestor label) applies either way."
            ),
            "default": MergePolicy.FIRST_WINS.value,
            "options": [
                {"value": MergePolicy.FIRST_WINS.value, "label": "First model wins (priority order)"},
                {"value": MergePolicy.LAST_WINS.value, "label": "Last model wins (reverse priority order)"},
            ],
        },
        "apply_tags": {
            "type": "tags",
            "label": "Apply Tags",
            "description": "Tags to apply to every bounding box in each saved image.",
            "default": [],
            "allow_add_tag": True,
        },
        "generate_task_tag": {
            "type": "boolean",
            "label": "Tag images with a tag unique to this task",
            "description": "Creates a dedicated tag for this task config and applies it to all saved images. Useful for reviewing just this task's output.",
            "default": False,
        },
        "task_tag_name": {
            "type": "string",
            "label": "Generated tag name (optional)",
            "description": "Custom name for the per-task generated tag. Defaults to the config name if left blank.",
            "default": "",
        },
        "filter_chain_uuid": {
            "type": "filter_chain_uuid",
            "label": "Filter Chain (optional)",
            "required": False,
            "description": (
                "Run this chain against each frame's detections before saving — "
                "discard corrupt frames, relabel known false-positives, or strip/tag "
                "matches before they ever land in the dataset."
            ),
        },
        "max_images": {
            "type": "int",
            "label": "Max Images to Save",
            "description": (
                "Stop the task once this many images have been saved. "
                "Leave blank / 0 for unlimited (the default — live sources are normally user-stopped)."
            ),
            "default": 0,
        },
        "max_duration": {
            "type": "float",
            "label": "Max Duration (s)",
            "description": (
                "Stop the task after this many seconds of run time have elapsed. "
                "Leave blank / 0 for unlimited (the default — live sources are normally user-stopped)."
            ),
            "default": 0,
        },
        "ambiguity_group": {
            "type": "group",
            "label": "Confusable Detections",
            "collapsed": True,
            "fields": {
                "enable_ambiguity": {
                    "type": "boolean",
                    "label": "Collect confusable detections",
                    "description": (
                        "Also save a frame when a model is confident but can't decide *what* it saw — "
                        "its runner-up class scored almost as high as the winner (e.g. cat 0.91, "
                        "dog 0.88).\n\n"
                        "This is a separate trigger from the confidence range above, and it catches "
                        "the opposite failure. The range finds detections the model is unsure "
                        "about; this finds detections it is sure about but confused by. A confident "
                        "confusion usually means the model learned a feature that fires for two "
                        "classes at once, so one correction here fixes a systematic error rather "
                        "than a single hard frame.\n\n"
                        "Boxes that trigger it are tagged 'Confusable' — filter on that tag in "
                        "Review, or use the task's 'Browse confusable' button."
                    ),
                    "default": False,
                },
                "ambiguity_metric": {
                    "type": "select",
                    "label": "Closeness Measure",
                    "description": (
                        "How 'close' the runner-up has to be. Ratio compares it as a fraction of the "
                        "winner (0.88/0.91 = 0.97); margin is the absolute gap (0.03).\n\n"
                        "Prefer ratio for detection models (YOLO): their class scores are independent "
                        "and don't sum to 1, so a fixed margin means very different things at high and "
                        "low confidence. Margin suits classifiers, where the classes compete directly."
                    ),
                    "options": [
                        {"value": AMBIGUITY_METRIC_RATIO, "label": "Ratio — runner-up ÷ winner (recommended)"},
                        {"value": AMBIGUITY_METRIC_MARGIN, "label": "Margin — winner − runner-up"},
                    ],
                    "default": AMBIGUITY_METRIC_RATIO,
                },
                "ambiguity_threshold": {
                    "type": "float",
                    "label": "Closeness Threshold",
                    "description": (
                        "Flag the detection when the ratio is at or ABOVE this (0.7 = runner-up scored "
                        "at least 70% of the winner), or when the margin is at or BELOW it.\n\n"
                        "Not sure what to use? Turn this on with a high ratio (0.9) for a short run and "
                        "watch 'Confusable' in the task's live status — it reports how many detections "
                        "were measured, how many were flagged, and the median closeness actually seen, "
                        "so you can set this from your own footage instead of guessing."
                    ),
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                },
                "ambiguity_scope": {
                    "type": "select",
                    "label": "Which Runner-up Classes Count",
                    "description": (
                        "Whether the runner-up has to be a label you configured above. Restricting it "
                        "keeps a long run focused on confusions you can act on; 'any class' also "
                        "reports being confused with something you aren't collecting, which is a real "
                        "finding but much noisier."
                    ),
                    "options": [
                        {"value": "configured", "label": "Only labels configured above (recommended)"},
                        {"value": "any", "label": "Any class the model knows"},
                    ],
                    "default": "configured",
                },
                "ambiguity_min_confidence": {
                    "type": "float",
                    "label": "Minimum Winner Confidence",
                    "description": (
                        "Ignore confusion below this confidence. Weak detections are already collected "
                        "by the confidence range above, so this stops the same frames being flagged "
                        "twice for two different reasons. 0 = no floor."
                    ),
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                },
                "ambiguity_cross_model": {
                    "type": "boolean",
                    "label": "Also flag disagreements between models",
                    "description": (
                        "When two of the models above put incompatible labels on the same object "
                        "(one says cat, the other says dog), flag it — two independently trained "
                        "models disagreeing is a strong signal, and both boxes get marked with the "
                        "other model named.\n\n"
                        "A parent/child pair (Animal vs Cat) is not a disagreement and is never "
                        "flagged. Has no effect with only one model configured, and unlike the "
                        "check above it works with every model type — including end-to-end exports "
                        "that cannot report runner-up classes at all."
                    ),
                    "default": True,
                },
                "ambiguity_top_k": {
                    "type": "int",
                    "label": "Classes to Compare",
                    "description": (
                        "How many of the model's best guesses to look at per detection, including the "
                        "winner. 3 is plenty — only the best in-scope runner-up is ever used. Costs "
                        "nothing extra to compute; the model already scores every class."
                    ),
                    "default": 3,
                },
            },
        },
        "dedup_group": {
            "type": "group",
            "label": "Deduplication",
            "collapsed": True,
            "fields": {
                "enable_dedup": {
                    "type": "boolean",
                    "label": "Enable Deduplication",
                    "description": "Check each frame against the perceptual hash index and handle duplicate detections.",
                    "default": False,
                },
                "duplicate_action": {
                    "type": "select",
                    "label": "Action on Duplicate",
                    "description": "What to do when a detection matches an existing entry in the perceptual hash index.",
                    "options": [
                        {
                            "value": "discard",
                            "label": "Keep if any unique — discard the frame only when all detections are duplicates",
                        },
                        {
                            "value": "discard_image",
                            "label": "Discard if image matches — skip the frame when the whole scene is a perceptual duplicate",
                        },
                        {
                            "value": "ignore",
                            "label": "Ignore — save normally without any duplicate marking",
                        },
                        {
                            "value": "measure",
                            "label": "Measure only (calibration) — keep every frame, but measure how duplicate it was",
                        },
                    ],
                    "default": "discard",
                    "help": (
                        "Not sure what to set the thresholds below to? Run the task once with "
                        "'Measure only (calibration)'. It discards nothing — every frame is saved as if "
                        "deduplication were off — but it measures how close each frame was to what you "
                        "already have, and the task's live status panel then shows a distance histogram "
                        "plus a sweep table telling you how many frames each candidate threshold would "
                        "have discarded. Let it run over a representative stretch of footage, read the "
                        "threshold off the table, set it here, then switch back to a discard action.\n\n"
                        "Calibrating this way matters because once frames are being discarded the "
                        "measurement is biased: discarded frames never enter the index, so the remaining "
                        "frames look more unique than they are. Calibration mode keeps the sample honest."
                    ),
                },
                "image_dup_threshold": {
                    "type": "int",
                    "label": "Image Duplicate Threshold",
                    "description": (
                        "Max Hamming distance for two whole-image phashes to be considered the same scene. "
                        "0 = only near-identical frames; larger = more aggressive discarding. "
                        "Run in 'Measure only (calibration)' mode to see what value suits your cameras."
                    ),
                    "default": 5,
                },
                "bbox_dup_threshold": {
                    "type": "int",
                    "label": "BBox Duplicate Threshold",
                    "description": (
                        "Max Hamming distance for two bbox crop phashes to be considered the same object "
                        "appearance. Crop hashes are wider than image hashes, so this scale runs higher — "
                        "don't copy the image threshold across. Calibrate it the same way."
                    ),
                    "default": 8,
                },
            },
        },
        "enable_detector_miss": {
            "type": "boolean",
            "label": "Collect Detector Misses",
            "description": (
                "Watcher-attached collectors only: when a confirmed track has no supporting "
                "detection for a frame (a short gap), save the tracker's interpolated box as an "
                "unverified 'Detector Miss' box if the track re-acquires within a few frames. "
                "No effect on the standalone task, which has no tracker."
            ),
            "default": False,
        },
        "max_frames_per_track": {
            "type": "int",
            "label": "Max Frames Per Track",
            "description": (
                "Watcher-attached collectors only: hard cap on how many distinct crops one track "
                "can contribute during a single visit. A safety valve for an animal that never "
                "stops moving — appearance novelty, not this number, is what normally decides. "
                "Keep it around 20-30; a low cap discards the pose variation the detector needs."
            ),
            "default": 25,
        },
    }


def _as_float(raw: Any, default: float) -> float:
    try:
        return float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _as_int(raw: Any, default: int) -> int:
    try:
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default
