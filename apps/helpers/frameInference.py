"""Shared per-frame inference call — one detect_batch, several consumers.

Phase 3 of COLLECTOR_UNIFICATION_PLAN.md (repo root). Both `StreamWatcherTask`
and `TrainingCollector` used to hand-roll their own `BatchInferencePayload` +
retry/timeout loop against the same endpoint. This extracts that into one
call so a future consumer sharing a frame with another (a training collector
attached to a watcher) gets a real union — one `detect_batch`, one
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from inference_client.models.batch_inference_payload import BatchInferencePayload
from inference_client.models.topk import Topk

logger = logging.getLogger(__file__)


@dataclass
class FrameInferenceRequest:
    """One consumer's ask for one frame, before union."""

    model_names: List[str]
    conf_floor: float
    top_k: Union[int, Dict[str, int]] = 0
    pin_ids: Dict[str, str] = field(default_factory=dict)


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

    return {
        "model_names": model_names,
        "conf_thresh": conf_floor,
        "top_k": top_k,
        "pin_ids": {n: p for n, p in pin_ids.items() if n in seen},
    }


async def run_frame_inference(
    inference_api,
    base64_img: str,
    requests: Sequence[FrameInferenceRequest],
    *,
    timeout_s: float,
    max_retries: int,
    log: Optional[Callable[[str], None]] = None,
) -> FrameInferenceResult:
    """Union *requests* into one BatchInferencePayload and run it, with retries.

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

    payload = BatchInferencePayload(
        image_base64=base64_img,
        model_names=union["model_names"],
        conf_thresh=union["conf_thresh"],
        pin_ids=union["pin_ids"],
        # The generated client models top_k's Union[int, Dict[str, int]] as an
        # anyOf wrapper (`Topk`) rather than a plain field — passing a bare
        # dict straight through silently produces actual_instance=None (no
        # error, no top_k on the wire) instead of raising. Must be wrapped.
        top_k=Topk(union["top_k"]),
    )

    for attempt in range(max_retries):
        try:
            resp = await asyncio.wait_for(
                asyncio.to_thread(inference_api.detect_batch, payload),
                timeout=timeout_s,
            )
            results = resp.results or {}
            errors = resp.errors or {}
            for name, message in errors.items():
                logger.error("Inference error on model %s: %s", name, message)
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
