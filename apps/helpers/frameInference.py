"""Shared per-frame inference call — one detect-unit, several consumers.

⚠️ The detections this returns are **final**, not first-pass. Any crop classifier in the
stack has already been applied server-side, so a consumer must never run a second pass of
its own: the cascade is part of the model, and a hierarchy has no intermediate output.
`StreamWatcherTask` used to refine *after* its own cross-model merge; that made the coarse
label observable and gave two consumers of the same frame different answers.

Phase 3 of COLLECTOR_UNIFICATION_PLAN.md (repo root). Both `StreamWatcherTask`
and `TrainingCollector` used to hand-roll their own `BatchInferencePayload` +
retry/timeout loop against the same endpoint. This extracts that into one
call so a future consumer sharing a frame with another (a training collector
attached to a watcher) gets a real union — one `detect_unit`, one
server-side decode+letterbox — instead of two round trips over the same JPEG.

Phase 4 landed that attachment but **deliberately kept the two calls
separate**: an attached collector still makes its own single-request
`run_frame_inference` call rather than joining the watcher's. That deferral
is intentional, not an oversight — see COLLECTOR_UNIFICATION_PLAN.md and
`.claude/claude_docs/HANDOFF-collector-unification.md` for why. So the
multi-request union path below is built and tested but not yet exercised in
production; do not "clean it up" as dead code.

`run_frame_inference` takes a *list* of requests and resolves them into one
payload:

    models  = union of every request's model_names
    conf    = min(r.conf_floor for r in requests)   — the loosest floor wins
    top_k   = per-model max across requests that named it
    pin_ids = merged (requests pinning the same model should agree)

Each caller then reads back only its own models, in its own order, via
`FrameInferenceResult.for_models()` — never a shared ordered list. FIRST_WINS
merge (`detectionMerger.py`) is order-dependent, so one consumer's stack
priority must never leak into another's.

⚠️ A union floor is necessarily looser than (or equal to) any single
consumer's own per-label floor, so a consumer must prefilter the detections
`for_models()` hands back to its OWN per-label floors before any O(n²) stage
(the hierarchy merge, a cross-model ambiguity scan) — never run that work
over the raw union-floor list. See NOTES.md §9: measured 63ms vs 18ms/7.6ms
for exactly this prefilter on a 3-model stack. Every call today still has
exactly one requester (see above), so this doesn't yet change any behaviour —
it matters as soon as a call carries two.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from inference_client.models.detect_unit_payload import DetectUnitPayload
from inference_client.models.topk import Topk

from apps.helpers.logUtils import TaskLogFn

logger = logging.getLogger(__file__)


@dataclass
class FrameInferenceRequest:
    """One consumer's ask for one frame, before union."""

    model_names: List[str]
    conf_floor: float
    top_k: Union[int, Dict[str, int]] = 0
    pin_ids: Dict[str, str] = field(default_factory=dict)
    #: {label_uuid: [ancestor label uuids]}, for the cascade the endpoint runs. Optional
    #: and additive: a hierarchy already describes the labels it can emit, and this only
    #: adds the case where a detection from a FOREIGN model in the same stack is a
    #: descendant of some sub-model's boundary. Unioned across requests, since a label's
    #: ancestors are a fact about the label tree and cannot differ per consumer.
    label_ancestry: Dict[str, list] = field(default_factory=dict)
    #: Confidence a sub-model's verdict must clear to replace the coarser label. Unlike
    #: `conf_floor` this is NOT recoverable downstream: a consumer handed a rewritten
    #: label cannot get the coarse one back, so the union takes the STRICTEST value
    #: rather than the loosest. Two consumers that genuinely disagree here therefore
    #: cannot share one call — see the module docstring on single-requester calls.
    sub_model_min_confidence: float = 0.0


@dataclass
class FrameInferenceResult:
    #: model name -> InferenceResultModel-like (only present for models that
    #: succeeded — a model absent here failed and its error is in `errors`).
    results: Dict[str, Any]
    #: model name -> error string, for models that failed.
    errors: Dict[str, str]

    def for_models(self, model_names: Sequence[str]) -> List[list]:
        """Detections for exactly these models, in exactly this order.

        Never returns the union's own ordering: the caller's model_names list
        IS its merge priority, and that must not depend on which other
        consumer shared this request or the order it happened to ask in. A
        model that failed (or that this caller didn't itself request)
        contributes an empty list, keeping positions of the rest intact.
        """
        return [list(getattr(self.results.get(n), "detections", None) or []) for n in model_names]


def _resolve_top_k(top_k: Union[int, Dict[str, int]], model_name: str) -> int:
    return top_k.get(model_name, 0) if isinstance(top_k, dict) else top_k


def _union_requests(requests: Sequence[FrameInferenceRequest]) -> Dict[str, Any]:
    model_names: List[str] = []
    seen = set()
    conf_floor = 1.0
    top_k: Dict[str, int] = {}
    pin_ids: Dict[str, str] = {}
    ancestry: Dict[str, set] = {}
    sub_min = 0.0

    for req in requests:
        conf_floor = min(conf_floor, req.conf_floor)
        for name in req.model_names:
            if name not in seen:
                seen.add(name)
                model_names.append(name)
            resolved = _resolve_top_k(req.top_k, name)
            if resolved > top_k.get(name, 0):
                top_k[name] = resolved
        pin_ids.update(req.pin_ids)
        for label, ancestors in (req.label_ancestry or {}).items():
            ancestry.setdefault(label, set()).update(ancestors)
        sub_min = max(sub_min, req.sub_model_min_confidence)

    return {
        "model_names": model_names,
        "conf_thresh": conf_floor,
        "top_k": top_k,
        "pin_ids": {n: p for n, p in pin_ids.items() if n in seen},
        "label_ancestry": {label: sorted(ancestors) for label, ancestors in ancestry.items()},
        "sub_model_min_confidence": sub_min,
    }


async def run_frame_inference(
    inference_api,
    base64_img: str,
    requests: Sequence[FrameInferenceRequest],
    *,
    timeout_s: float,
    max_retries: int,
    log: Optional[TaskLogFn] = None,
) -> FrameInferenceResult:
    """Union *requests* into one detect-unit call and run it, with retries.

    Retries are per-batch, not per-model: a timeout here means the inference
    server isn't keeping up with the frame, which every model in the stack
    shares, not a condition one recovers from while its siblings don't. A
    model that fails individually still comes back in `errors` and only costs
    its own detections — the batch itself is not retried for that.

    Returns an empty result (all models missing, no errors) if every retry
    times out or the model list is empty; callers already treat "no
    detections" identically to that, since a caller with zero model_names of
    its own reads back `[]` from `for_models()` regardless.
    """
    union = _union_requests(requests)
    if not union["model_names"]:
        return FrameInferenceResult(results={}, errors={})

    # detect_unit, not detect_batch: a crop classifier in the stack is applied here,
    # server-side, and the detections that come back are already as specific as the
    # models can make them. That is deliberate and load-bearing — the cascade IS the
    # model, so there is no intermediate coarse output for any consumer to see, merge or
    # re-refine, and no consumer reimplements a second pass.
    payload = DetectUnitPayload(
        image_base64=base64_img,
        model_names=union["model_names"],
        conf_thresh=union["conf_thresh"],
        # The generated client models top_k's Union[int, Dict[str, int]] as an anyOf
        # wrapper (`Topk`) rather than a plain field — passing a bare dict straight
        # through silently produces actual_instance=None (no error, no top_k on the wire)
        # instead of raising. Must be wrapped.
        top_k=Topk(union["top_k"]),
        pin_ids=union["pin_ids"],
        label_ancestry=union["label_ancestry"],
        min_sub_confidence=union["sub_model_min_confidence"],
    )

    for attempt in range(max_retries):
        try:
            resp = await asyncio.wait_for(
                asyncio.to_thread(inference_api.detect_unit, payload),
                timeout=timeout_s,
            )
            results = resp.results or {}
            errors = resp.errors or {}
            for name, message in errors.items():
                logger.error("Inference error on model %s: %s", name, message)
            for note in getattr(resp, "notes", None) or []:
                # A sub-model that could not run leaves detections at their parent label,
                # which is indistinguishable from a confident coarse answer. This is
                # per-frame state, so the operator reads it in the task log while the
                # rotated log only carries it at DEBUG.
                if log is not None:
                    log(f"Cascade: {note}", logging.DEBUG)
                else:
                    logger.debug("Cascade note: %s", note)
            return FrameInferenceResult(results=results, errors=errors)
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                logger.warning(
                    "Inference timeout on model batch (attempt %d/%d), retrying",
                    attempt + 1,
                    max_retries,
                )
            else:
                message = f"Inference timed out after {max_retries} attempts — skipping frame"
                logger.warning(message)
                if log is not None:
                    log(message)
        except Exception:
            logger.exception("Inference error on model batch %s", union["model_names"])
            break

    return FrameInferenceResult(results={}, errors={})
