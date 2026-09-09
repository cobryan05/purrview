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
import uuid as uuid_module
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from apps.helpers.db.db_client import CONFUSABLE_TAG_UUID, DETECTOR_MISS_TAG_UUID
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


# Review-queue eviction strategies — which already-saved, still-unverified image
# a cap-exceeding collector deletes first. See `reviewQueueEviction.py` for the
# selection logic itself; this module only owns the identifiers and the schema
# that lets a config name one.
EVICTION_FIFO = "fifo"
EVICTION_LEAST_UNIQUE = "least_unique"
EVICTION_LABEL_BALANCE = "label_balance"
EVICTION_STRATEGIES: tuple[str, ...] = (EVICTION_FIFO, EVICTION_LEAST_UNIQUE, EVICTION_LABEL_BALANCE)


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

    # ── review queue limits ────────────────────────────────────────────────────
    #: Distinct from `max_images`/`max_duration_s` above: those stop the
    #: collector; these prune what it has already saved. A collector left
    #: running unattended has no other way to stay bounded, and an unbounded
    #: unverified pile is the thing that actually breaks review — not disk
    #: usage, but a human giving up on a queue too long to face.
    max_unverified_images: Optional[int] = None
    #: Bytes, not MB — see `parse_size_mb` for the params-JSON side of that unit
    #: choice. Independent of `max_unverified_images`: either can be set alone.
    max_unverified_bytes: Optional[int] = None
    eviction_strategy: str = EVICTION_FIFO

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
            # `> 0` and not truthiness: the caps drive a *permanent delete*, and a
            # negative number (someone reaching for the usual "-1 means unlimited")
            # would make `total > cap` true forever and evict the whole queue.
            # 0 and anything below it mean "no cap", exactly as the field help says.
            max_unverified_images=(
                _max_unverified_images_raw
                if (_max_unverified_images_raw := _as_int(params.get("max_unverified_images"), 0)) > 0
                else None
            ),
            # Stored as MB in params JSON (a human picks a number of megabytes,
            # not bytes) and converted once here, so every reader downstream
            # compares against a plain byte count without redoing the unit math.
            max_unverified_bytes=(
                int(_max_unverified_mb_raw * 1024 * 1024)
                if (_max_unverified_mb_raw := _as_float(params.get("max_unverified_mb"), 0.0)) > 0
                else None
            ),
            eviction_strategy=(
                params.get("eviction_strategy")
                if params.get("eviction_strategy") in EVICTION_STRATEGIES
                else EVICTION_FIFO
            ),
        )


def task_tag_candidates(collector_uuid: str, collector_name: str, spec: "CollectorSpec") -> tuple[str, str, str]:
    """The tag identity a run of this collector would resolve to, computed with
    no I/O — mirrors `TrainingCollector._resolve_tags`'s naming/hashing exactly
    (see there for why it's name-keyed, not scope-alone) but stops short of
    reading or creating anything, so a **stopped** collector's tag can still be
    named for a "browse what this collector has produced" button (plan: review
    integration without a running task).

    Returns `(legacy_tag_uuid, task_tag_uuid, tag_name)`. The caller decides
    which of the two uuids is the one actually in use — that requires a DB
    lookup this function deliberately doesn't do.
    """
    tag_name = spec.task_tag_name_override or f"[Task] {collector_name}"
    legacy_tag_uuid = str(uuid_module.uuid5(uuid_module.NAMESPACE_OID, collector_uuid))
    task_tag_uuid = str(uuid_module.uuid5(uuid_module.NAMESPACE_OID, f"{collector_uuid}:{tag_name}"))
    return legacy_tag_uuid, task_tag_uuid, tag_name


def build_review_actions(task_tag_uuid: Optional[str], spec: "CollectorSpec") -> list[dict]:
    """Ways into Review over one collector's images: everything, the part still
    needing a human, and (when enabled) the confusable / detector-miss subsets.

    Pure function of the resolved tag and the spec — no host, no live collector
    instance — so it can build the exact same button set whether the collector
    is currently running (`TrainingCollector.review_actions()`) or has never run
    at all (a "browse this collector's output" button on its config page).

    The collector auto-annotates with its models and tags what it draws
    UNVERIFIED, so after a long run "what did I collect" and "what do I still
    have to look at" are different, and usually very different, questions. One
    button could only answer the first — and "Browse unverified" alone already
    answers "what might be worth training on": every category this collector can
    produce (a band match, a confusable flag, a detector-miss seed, or a bare
    collect-all frame) starts life unverified, so it's a strict superset of the
    narrower buttons below, not an alternative reading of the same data.
    """
    if not task_tag_uuid:
        return []

    # image_tag_uuids is what actually finds every collected frame: the tag is
    # written to the image record, so it matches whether or not the frame ended
    # up with bboxes. tag_uuids (a bbox-tag filter) is kept for annotated runs,
    # where narrowing to the tagged bboxes is more useful.
    scope = {
        "tag_uuids": [] if spec.collect_all else [task_tag_uuid],
        "image_tag_uuids": [task_tag_uuid],
    }
    actions = [
        {
            "type": "open_review",
            "label": "Browse all results",
            # Sent explicitly rather than omitted so this also *clears* an
            # 'unverified' filter left over from the button next to it.
            "payload": {**scope, "bbox_verification": "any"},
        },
        {
            "type": "open_review",
            "label": "Browse unverified",
            "payload": {**scope, "bbox_verification": "unverified"},
        },
    ]
    if spec.enable_ambiguity:
        # The confusable set is usually a small fraction of a run and is the
        # part with the highest review value per box, so it gets its own way in
        # rather than being something to reconstruct by hand in the filter bar.
        # Narrowed by bbox tag, which is what the tag is for.
        actions.append(
            {
                "type": "open_review",
                "label": "Browse confusable",
                "payload": {
                    **scope,
                    "tag_uuids": [*(scope["tag_uuids"] or []), CONFUSABLE_TAG_UUID],
                    "bbox_verification": "any",
                },
            }
        )
    if spec.enable_detector_miss:
        # Always unverified — a detector-miss box is a fresh seed, never a
        # confirmed one (plan §5: "unverified": narrower than confusable's
        # "any", since a verified detector-miss box has already had its one
        # human look).
        actions.append(
            {
                "type": "open_review",
                "label": "Browse detector misses",
                "payload": {
                    **scope,
                    "tag_uuids": [*(scope["tag_uuids"] or []), DETECTOR_MISS_TAG_UUID],
                    "bbox_verification": "unverified",
                },
            }
        )
    return actions


def collector_status_schema() -> list[dict[str, Any]]:
    """The status widgets covering everything `TrainingCollector.status_stats()`
    and `notices()` can report — one definition, mirroring
    `collector_params_schema()`'s own reasoning, shared by:

    * `CollectTrainingImagesTask.status_schema()`, which splices its own
      task-owned fields (`processed`, `captures_per_min`, `uptime` — properties
      of the driver loop, not of collection) into this schema's `stats` widget,
      alongside its own `last_capture` and `sources` widgets this function
      knows nothing about;
    * the per-collector diagnostics view a watcher-attached collector renders
      on its own, unmodified — a watcher has no "processed" or single
      "uptime" of its own to add (it drives many collectors and sources at
      once), so there is nothing for it to splice in.

    Confusable-detection fields are listed here even though the standalone
    task's card never showed them before this was written — `status_stats()`
    always computed them, they were simply missing from the schema that
    decides what a `stat_grid` renders. Both surfaces get them by fixing it
    once, here.
    """
    return [
        {
            "id": "stats",
            "type": "stat_grid",
            "source": "status",
            "label": "Collection stats",
            # Superset of every field `status_stats()` can ever populate — a
            # feature that isn't enabled for this collector simply never sends
            # its key, which the stat_grid widget renders as "—" rather than a
            # misleading 0.
            "fields": {
                "offered": "Offered",
                "saved": "Saved",
                "saved_band": "Saved (band match)",
                "saved_detector_miss": "Saved (detector miss)",
                "saved_track_worst": "Saved (track worst)",
                "track_novelty_kept": "Novel poses kept",
                "track_novelty_dropped": "Redundant poses dropped",
                "track_worst_committed": "Hardest-pose commits",
                "detector_miss_offered": "Detector-miss candidates",
                "dropped_behind": "Dropped (collector behind)",
                "evicted_total": "Evicted (review queue cap)",
                "dedup_checked": "Dedup checked",
                "dedup_image_hits": "Dedup image hits",
                "dedup_bbox_hits": "Dedup bbox hits",
                "dedup_discarded": "Dedup discarded",
                "dedup_image_dist_median": "Image dist (median)",
                "dedup_image_dist_p10": "Image dist (p10)",
                "dedup_bbox_dist_median": "BBox dist (median)",
                "dedup_sample_size": "Calibration sample",
                "confusable_flagged": "Confusable flagged",
                "confusable_measured": "Confusable measured",
                "confusable_median_closeness": "Confusable median closeness",
                "confusable_most_extreme": "Confusable most extreme seen",
                "confusable_top_pairs": "Confusable top pairs",
                "confusable_unsupported_models": "Confusable-unsupported models",
                "filter_discarded": "Filter discarded",
                "filter_relabeled": "Filter relabeled",
                "filter_tagged": "Filter tagged",
                "filter_stripped": "Filter stripped",
            },
        },
        # ── Dedup threshold calibration ──────────────────────────────────
        # Only populated when dedup is enabled; the data source omits these
        # keys otherwise and the widgets render their own empty state.
        #
        # Collapsed by default, and nested one level deeper: four charts and
        # tables is the right amount of detail while you are actually reading
        # a threshold off them, and far too much to scroll past every other
        # time.
        {
            "id": "dedup_calibration",
            "type": "group",
            "label": "Duplicate threshold calibration",
            "collapsed": True,
            "widgets": [
                # Nested one level deeper than the two distance groups below,
                # collapsed by default: `_dedup_hint_text()` is a genuinely
                # useful multi-sentence explanation of how to read the
                # histograms and sweep tables, but it is the same paragraph
                # every time, and reading it once is not a reason to scroll
                # past it on every later visit.
                {
                    "id": "dedup_hint_group",
                    "type": "group",
                    "label": "How to read this",
                    "collapsed": True,
                    "widgets": [
                        {
                            "id": "dedup_hint",
                            "type": "text",
                            "source": "status",
                        },
                    ],
                },
                {
                    "id": "dedup_image_group",
                    "type": "group",
                    "label": "Image distance",
                    "widgets": [
                        {
                            "id": "dedup_image_hist",
                            "type": "histogram",
                            "source": "status",
                            "label": "Distance to nearest stored image",
                        },
                        {
                            "id": "dedup_image_sweep",
                            "type": "table",
                            "source": "status",
                            "label": "Image threshold sweep (what each value would have done)",
                            "columns": [
                                {"key": "threshold", "label": "Image threshold"},
                                {"key": "matched", "label": "Frames matching"},
                                {"key": "discarded", "label": "Would discard"},
                                {"key": "kept", "label": "Would keep"},
                            ],
                        },
                    ],
                },
                {
                    "id": "dedup_bbox_group",
                    "type": "group",
                    "label": "BBox distance",
                    "widgets": [
                        {
                            "id": "dedup_bbox_hist",
                            "type": "histogram",
                            "source": "status",
                            "label": "Distance to nearest same-label crop",
                        },
                        {
                            "id": "dedup_bbox_sweep",
                            "type": "table",
                            "source": "status",
                            "label": "BBox threshold sweep (what each value would have done)",
                            "columns": [
                                {"key": "threshold", "label": "BBox threshold"},
                                {"key": "matched", "label": "BBoxes matching"},
                                {"key": "discarded", "label": "Would discard"},
                                {"key": "kept", "label": "Would keep"},
                            ],
                        },
                    ],
                },
            ],
        },
    ]


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
            # Two answers everyone has to give, then six groups nobody has to open.
            # Flat, this was a run of a dozen stacked fields with no shape: "max
            # duration" sat between the filter chain and the confusable settings for
            # no reason beyond the order someone thought of them, and the two
            # watcher-only knobs ended up at the bottom, apart, with nothing saying
            # they were related or that neither does anything for a standalone job.
            #
            # A `group` is value-transparent — GroupField merges its fields' values
            # into the parent — so `from_params` still reads every key flat and
            # nothing about the stored shape changes.
            "order": [
                "output_dir",
                "model_label_config",
                "merge_group",
                "tagging_group",
                "limits_group",
                "ambiguity_group",
                "dedup_group",
                "detector_miss_group",
                "review_queue_group",
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
        "merge_group": {
            "type": "group",
            "label": "Multi-model merge",
            "description": (
                "Only does anything when more than one model is configured above: how to resolve two "
                "models boxing the same thing. Safe to ignore with a single model."
            ),
            # Inert with one model configured, which is the common case — so it starts
            # closed rather than showing two controls that cannot do anything yet.
            "collapsed": True,
            "fields": {
                "overlap_iou_threshold": {
                    "type": "float",
                    "label": "Multi-Model Overlap IoU Threshold",
                    "description": (
                        "When multiple models detect the same region, boxes from later-configured models are "
                        "discarded if they overlap a higher-priority model's box at or above this IoU value "
                        "(0–1). Lower values discard more aggressively; higher values allow more overlap "
                        "before deduplication."
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
                        "the higher-priority (earlier-configured) model, or the lower-priority "
                        "(later-configured) one. Hierarchy supersede (a more specific label beats a coarser "
                        "ancestor label) applies either way."
                    ),
                    "default": MergePolicy.FIRST_WINS.value,
                    "options": [
                        {"value": MergePolicy.FIRST_WINS.value, "label": "First model wins (priority order)"},
                        {"value": MergePolicy.LAST_WINS.value, "label": "Last model wins (reverse priority order)"},
                    ],
                },
            },
        },
        "tagging_group": {
            "type": "group",
            "label": "Tagging & filtering",
            "description": (
                "What gets stamped onto every saved frame, and an optional filter chain that can drop, "
                "relabel or tag detections before they reach the dataset."
            ),
            "collapsed": True,
            "fields": {
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
                    "description": (
                        "Creates a dedicated tag for this task config and applies it to all saved images. "
                        "Useful for reviewing just this task's output — and required for the 'Browse in "
                        "review' buttons on the status card, which have nothing to filter by without it."
                    ),
                    "default": False,
                },
                "task_tag_name": {
                    "type": "string",
                    "label": "Generated tag name (optional)",
                    "description": (
                        "Custom name for the per-task generated tag. Defaults to the config name if left blank."
                    ),
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
            },
        },
        "limits_group": {
            "type": "group",
            "label": "Run limits",
            "description": (
                "Optional stop conditions for a job running this collector. Both blank/0 (the default) "
                "means it runs until you stop it."
            ),
            "collapsed": True,
            "fields": {
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
            },
        },
        "ambiguity_group": {
            "type": "group",
            "label": "Confusable Detections",
            "description": (
                "Collect the frames where the model was sure something was there but unsure what — the "
                "cases where your labelling adds the most signal. Off by default."
            ),
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
                        "(one says car, the other says truck), flag it — two independently trained "
                        "models disagreeing is a strong signal, and both boxes get marked with the "
                        "other model named.\n\n"
                        "A parent/child pair (Vehicle vs Car) is not a disagreement and is never "
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
            "description": (
                "Skip frames that look like ones already collected, so the dataset does not fill with "
                "near-identical images. The thresholds want calibrating per camera — see the help "
                "under 'Action on Duplicate'."
            ),
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
        "detector_miss_group": {
            "type": "group",
            # Named for what the two settings have in common, which is the thing that
            # was invisible while they sat loose at the bottom of a flat form: neither
            # does anything at all for a standalone job, because both need a tracker.
            "label": "Watcher-only collection (needs a tracker)",
            "description": (
                "Ignored entirely by a standalone collect job: both settings need a tracker, which only "
                "a watcher has."
            ),
            "collapsed": True,
            "fields": {
                "enable_detector_miss": {
                    "type": "boolean",
                    "label": "Collect Detector Misses",
                    "description": (
                        "When a confirmed track has no supporting detection for a frame (a short gap), "
                        "save the tracker's interpolated box as an unverified 'Detector Miss' box — but "
                        "only if the track re-acquires within a few frames, so a box is never seeded onto "
                        "empty background. No effect on the standalone task, which has no tracker."
                    ),
                    "default": False,
                },
                "max_frames_per_track": {
                    "type": "int",
                    "label": "Max Frames Per Track",
                    "description": (
                        "Hard cap on how many distinct crops one track can contribute during a single "
                        "visit. A safety valve for a subject that never stops moving — appearance novelty, "
                        "not this number, is what normally decides. Keep it around 20-30; a low cap "
                        "discards the pose variation the detector needs."
                    ),
                    "default": 25,
                },
            },
        },
        "review_queue_group": {
            "type": "group",
            "label": "Review queue limits",
            "description": (
                "Caps on how much still-unverified data this collector may pile up. Over a cap, images "
                "are permanently deleted to make room — verified images never count against it."
            ),
            "collapsed": True,
            "fields": {
                "max_unverified_images": {
                    "type": "int",
                    "label": "Max Unverified Images",
                    "description": (
                        "Once this many of this collector's images are still unverified, the oldest "
                        "excess (or the one the eviction strategy below picks) is deleted to make room "
                        "for new ones. This bounds your review queue, not the collector's lifetime "
                        "output — a verified image no longer counts against it. Leave blank / 0 for "
                        "unlimited."
                    ),
                    "default": 0,
                },
                "max_unverified_mb": {
                    "type": "float",
                    "label": "Max Unverified Size (MB)",
                    "description": (
                        "Same idea as the image count above, but by disk space instead — evicts the "
                        "oldest/least-useful unverified images once their total size passes this many "
                        "megabytes. Independent of the count limit; set either, both, or neither. Leave "
                        "blank / 0 for unlimited."
                    ),
                    "default": 0,
                },
                "eviction_strategy": {
                    "type": "select",
                    "label": "Eviction Strategy",
                    "description": (
                        "Which still-unverified image to delete first when a limit above is exceeded.\n\n"
                        "• Oldest first (FIFO) — simplest and most predictable.\n"
                        "• Least unique — deletes whichever image is closest (by perceptual hash) to "
                        "another one already in the queue, so near-duplicates go before genuinely "
                        "distinct frames. Costs a re-hash of each candidate image, done only when a "
                        "limit is actually exceeded.\n"
                        "• Balance labels — deletes from whichever label currently has the most "
                        "unverified images, so one common subject can't crowd out the rarer ones you "
                        "actually need to see."
                    ),
                    "options": [
                        {"value": EVICTION_FIFO, "label": "Oldest first (FIFO)"},
                        {"value": EVICTION_LEAST_UNIQUE, "label": "Least unique first"},
                        {"value": EVICTION_LABEL_BALANCE, "label": "Balance labels"},
                    ],
                    "default": EVICTION_FIFO,
                },
            },
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
