"""Shared FilterChain evaluation logic.

Used by ApplyFilterChainTask (retroactive, DB-backed bboxes) and, per the
WORKFLOW_FILTERCHAIN_PLAN.md Phase 4 design, CollectTrainingImagesTask's
collect-time integration (in-memory detections). This module only decides
*which* steps match — it never touches the DB or in-memory detection list
itself; callers interpret the returned `StepMatch` list in whatever context
they run in (mirrors the `_training_job_runner.py` / `_classifier_job_builders.py`
shared-helper pattern already used in this package).

Match semantics (see plan section 2.1):
  - whole-image classifier steps ("image_classifier" / "multi_label_classifier"
    sidecar `model_kind`): a match is confidence >= step.confidence -> an
    image-level action (discard_image / tag_image / strip_unverified_bboxes).
  - detector steps (any other model_kind): run on the whole image, then each
    new detection is matched against existing bboxes by IoU >= iou_threshold
    (optionally gated by action_params.only_if_label_uuid) -> a bbox-level
    action (relabel_bbox / tag_bbox) on the overlapped existing bbox.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from apps.helpers.bboxUtils import iou_xywh

logger = logging.getLogger(__file__)

_CLASSIFIER_KINDS = {"image_classifier", "multi_label_classifier"}


@dataclass
class Detection:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    label_uuid: Optional[str] = None


class InferenceBackend(Protocol):
    """Executes one model against one image.

    Implementations decide *how* (HTTP round-trip to inference_server, an
    in-process ONNX session, or a test double) and *where* (pinned model,
    lazily-loaded session, per Phase 1's device preference) — the runner only
    needs the three calls below.
    """

    async def get_model_kind(self, model_name: str) -> str:
        """Return the model sidecar's `model_kind` ('detector' / 'image_classifier'
        / 'multi_label_classifier'), resolved fresh each call so a chain can't go
        stale if the model is retrained/reclassified (plan 2.1: "prefer resolve at
        run time from the sidecar so it can't go stale")."""
        ...

    async def classify_whole_image(self, model_name: str, image_bytes: bytes) -> float:
        """Return the max confidence across the classifier's predicted label(s)."""
        ...

    async def detect(self, model_name: str, image_bytes: bytes, conf_thresh: float) -> List[Detection]:
        """Return detections at/above conf_thresh."""
        ...


@dataclass
class StepMatch:
    step: Dict[str, Any]
    # True for a whole-image classifier match (image-level action).
    image_match: bool = False
    # existing-bbox uuid -> the detection that matched it (bbox-level action).
    bbox_matches: Dict[str, Detection] = field(default_factory=dict)


class FilterChainRunner:
    """Evaluates one FilterChain's enabled steps, in order, against one image."""

    def __init__(self, steps: List[Dict[str, Any]], backend: InferenceBackend):
        self._steps = sorted((s for s in steps if s.get("enabled", True)), key=lambda s: s["order"])
        self._backend = backend

    async def evaluate_image(
        self,
        image_bytes: bytes,
        existing_bboxes: List[Dict[str, Any]],
    ) -> List[StepMatch]:
        """Run every enabled step against one image + its existing bboxes.

        `existing_bboxes` items need `uuid`, `x`, `y`, `width`, `height`,
        `label_uuid` (relative coords, same coordinate space detections come
        back in). Pass `[]` for collect-time in-memory detections that have no
        "existing" bboxes yet to overlap against — such steps simply produce no
        bbox_matches (bbox-level actions are meaningless pre-save; the plan's
        Phase 4 integration only consumes image-level matches for `discard_image`
        directly on the in-memory detection list).
        """
        matches: List[StepMatch] = []
        for step in self._steps:
            model_kind = await self._backend.get_model_kind(step["model_name"])
            if model_kind in _CLASSIFIER_KINDS:
                confidence = await self._backend.classify_whole_image(step["model_name"], image_bytes)
                if confidence >= step["confidence"]:
                    matches.append(StepMatch(step=step, image_match=True))
                continue

            if not existing_bboxes:
                # A detector-overlap step has nothing to overlap against yet.
                continue
            detections = await self._backend.detect(step["model_name"], image_bytes, step["confidence"])
            if not detections:
                continue

            action_params = step.get("action_params") or {}
            only_if_label = action_params.get("only_if_label_uuid")
            iou_threshold = step.get("iou_threshold", 0.5)

            bbox_matches: Dict[str, Detection] = {}
            for det in detections:
                for bbox in existing_bboxes:
                    b_uuid = bbox.get("uuid")
                    if b_uuid is None or b_uuid in bbox_matches:
                        continue
                    if only_if_label and bbox.get("label_uuid") != only_if_label:
                        continue
                    if iou_xywh(det, bbox) >= iou_threshold:
                        bbox_matches[b_uuid] = det
                        break
            if bbox_matches:
                matches.append(StepMatch(step=step, bbox_matches=bbox_matches))
        return matches
