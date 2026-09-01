"""Second-pass resolution for hierarchical model configs.

A model config can put a *sub-model* on a label: the top detector finds "Cat",
and a crop classifier trained on that boundary says which cat it is. This module
holds the small amount of shared logic that decides, at inference time, which
detections a given sub-model should be handed and what the refined result means.

It lives in `helpers/` because both consumers need it and neither should import
the other: the live watcher (`StreamWatcherTask`, per frame) and the retroactive
preview path (`PreviewModelRelabelTask`, per stored image).

Deliberately pure and shape-agnostic — it takes label UUIDs and returns indices
and labels, never detection objects. The watcher's detections come off the
inference client, the preview task's are its own dataclass, and a shared helper
that tried to own both shapes would have to know about both packages.

## What is NOT here

Choosing *which* models run. That stays with the caller: a sub-model is applied
because the operator put it in the watch profile's model stack (or picked it in
the review UI), not because it exists in the models directory. An automatic
"config X implies model Y" lookup would silently start running models nobody
asked for on live camera frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

#: Sidecar `model_kind` of a second-pass crop classifier.
CROP_CLASSIFIER_KIND = "crop_classifier"


@dataclass(frozen=True)
class CropSubModel:
    """A loaded crop classifier and the boundary label whose boxes it refines."""

    model_name: str
    parent_label_uuid: str


def split_model_stack(
    model_names: Sequence[str],
    model_kinds: Mapping[str, Optional[str]],
    parent_labels: Mapping[str, Optional[str]],
) -> tuple[list[str], list[CropSubModel]]:
    """Split a configured model stack into first-pass detectors and second-pass sub-models.

    Returns ``(detector_names, sub_models)`` with detector order preserved — the
    watcher's merge is FIRST_WINS, so that order is its stack priority and must
    survive the split untouched.

    A crop classifier with no ``parent_label_uuid`` is dropped from both lists: it
    cannot be aimed at anything, and running it as a detector (which is what
    leaving it in the first list would do) sends a classifier through the
    detection endpoint, where it produces garbage rather than an error.
    """
    detectors: list[str] = []
    sub_models: list[CropSubModel] = []
    for name in model_names:
        if model_kinds.get(name) != CROP_CLASSIFIER_KIND:
            detectors.append(name)
            continue
        parent = parent_labels.get(name)
        if parent:
            sub_models.append(CropSubModel(model_name=name, parent_label_uuid=parent))
    return detectors, sub_models


def order_sub_models(
    sub_models: Sequence[CropSubModel],
    label_ancestry: Mapping[str, "set[str]"],
) -> list[CropSubModel]:
    """Order sub-models outermost boundary first, so a deep hierarchy resolves fully.

    Nesting works by each pass feeding the next: a detector emits "Animal", the
    Animal sub-model rewrites it to "Cat", and the Cat sub-model — seeing "Cat" —
    rewrites it to "Willow". That only happens if Animal's pass runs *before* Cat's.
    Run in the other order, Cat's pass finds nothing labelled Cat yet, and the
    detection stops one level short. Nothing errors; the label is simply less
    specific than it should be, which is invisible unless you know what to expect.

    The configured order is whatever order the operator added models to the stack,
    which carries no hierarchy information, so it is replaced here by depth —
    ancestor count of the boundary label. Ties (siblings, or labels whose ancestry
    is unknown) keep their configured relative order, since `sorted` is stable.
    """
    return sorted(sub_models, key=lambda s: len(label_ancestry.get(s.parent_label_uuid, ())))


def refinable_indices(
    label_uuids: Sequence[Optional[str]],
    parent_label_uuid: str,
    label_ancestry: Mapping[str, "set[str]"],
) -> list[int]:
    """Positions in *label_uuids* that this sub-model should be asked about.

    A detection qualifies when its label IS the boundary label or is a descendant
    of it. Descendants count because the top model may already emit a specific
    label — a stack where an older per-cat detector runs alongside the new
    collapsed one, say — and re-running the classifier on it is exactly the
    second opinion the cascade is for.

    `label_ancestry` maps a label to the set of its ancestors, the same map the
    detection merger uses, so "is a descendant of" is one lookup.
    """
    return [
        i
        for i, label in enumerate(label_uuids)
        if label is not None and (label == parent_label_uuid or parent_label_uuid in label_ancestry.get(label, ()))
    ]


def resolve_refinement(
    detection_confidence: float,
    predictions: Iterable[tuple[str, float]],
    min_confidence: float,
    current_label_uuid: Optional[str] = None,
) -> Optional[tuple[str, float]]:
    """Turn one box's class scores into a (label_uuid, confidence) replacement.

    Returns None for "leave this detection alone", which happens two ways:

    - The sub-model isn't confident enough. The detection keeps its coarse label —
      "a cat, not sure which" is a true statement and a useful one, so an unsure
      second pass must degrade to it rather than guess or drop the box.
    - The winning class IS the label the detection already has. Nothing to change,
      and saying so matters: applying it anyway would multiply the confidence (see
      below) for a no-op relabel, so a 0.8 detection would come back at 0.48 and
      could fall out of a watch rule's confidence band without its label ever
      having changed.

    The reported confidence is the **product** of the two passes: the detector's
    confidence that this is a cat, times the classifier's that it is this cat.
    Reporting the classifier's score alone would let a 0.3-confidence detection
    come back as a 0.95-confidence Willow, and every downstream confidence band
    (watch rules, collector floors) would be reading a number that no longer
    accounts for whether there was an animal there at all.
    """
    top = next(iter(sorted(predictions, key=lambda p: p[1], reverse=True)), None)
    if top is None:
        return None
    label_uuid, class_confidence = top
    if class_confidence < min_confidence:
        return None
    if current_label_uuid is not None and label_uuid == current_label_uuid:
        return None
    return label_uuid, detection_confidence * class_confidence
