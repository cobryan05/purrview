"""Resolve a trained model config into the set of artifacts that make up one unit.

A hierarchical model config is *used* as a single model: the operator picks "Test Nest",
not "test_nest plus test_nest__cat, in that order, and mind the ordering". This module
turns a `model_config_uuid` into the detectors and sub-models that implement it, reading
nothing but the sidecars sitting next to the ONNX files.

## Why this is sidecar-only

inference_server may be deployed on a different host from db_server (see the
split-container topology), so anything it needs at inference time has to be answerable
from its own MODELS_ROOT. It is: `TrainModelTask` stamps `model_config_uuid`, `role` and
`model_config_node_uuid` into every artifact it trains, and `crop_classifier_trainer`
adds `parent_label_uuid`. Membership in a unit is therefore *recorded by the training
run*, not looked up.

The one thing the DB knows and the sidecars don't is the full label tree. `label_ancestry`
is only consulted to (a) order the passes outermost-first and (b) decide which detections
a sub-model may refine — and within a unit both are derivable from the artifacts
themselves, because a sub-model's classes ARE its boundary label's assigned descendants.
See `intra_unit_ancestry`. A caller that has the real tree (the watcher does) can pass it
in and get the extra case: a detection produced by a *foreign* model in the same stack
that happens to be a descendant of one of this unit's boundaries.

## What this module will not do

Decide that a unit is incomplete. A boundary whose sub-model was never trained leaves no
artifact, so nothing here can see it is missing — only the config knows how many parts
there should be. `resolve_unit` therefore reports what it *found*, and the caller (which
has the config) diffs that against `derive_training_targets`. Guessing here would mean
silently reporting a half-trained hierarchy as healthy, which is the exact failure this
whole feature exists to stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from apps.helpers.cascade import (
    CROP_CLASSIFIER_KIND,
    CropSubModel,
    split_model_stack,
)

#: Sidecar `role` values written by TrainModelTask.
ROLE_TOP = "top"
ROLE_SUBMODEL = "submodel"


@dataclass(frozen=True)
class UnitPart:
    """One trained artifact belonging to a unit, with the node it was trained for."""

    model_name: str
    role: str
    node_uuid: Optional[str] = None
    label_uuid: Optional[str] = None
    trained_at: Optional[str] = None


@dataclass(frozen=True)
class ModelUnit:
    """Everything needed to run one model config as a single model."""

    model_config_uuid: str
    #: First-pass detectors, in a stable order (see `resolve_unit`).
    detector_names: list[str] = field(default_factory=list)
    #: Second-pass crop classifiers. NOT yet ordered by depth — `order_sub_models` does
    #: that, and it needs the ancestry map, which the caller may want to augment first.
    sub_models: list[CropSubModel] = field(default_factory=list)
    #: Every artifact found for this config, including ones that could not be used.
    parts: list[UnitPart] = field(default_factory=list)
    #: Human-readable reasons an artifact was found but not run.
    problems: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.detector_names and not self.sub_models


def _newest_first(parts: list[UnitPart]) -> list[UnitPart]:
    """Sort by `trained_at`, newest first; artifacts with no timestamp sort last.

    Only used to pick between two artifacts trained for the SAME node, which happens
    when a node is retrained under a new model name and the old file is left in place.
    Running both would put two disagreeing opinions of the same boundary in one stack,
    and which one won would depend on directory order.
    """
    return sorted(parts, key=lambda p: (p.trained_at is None, p.trained_at or ""), reverse=False)[::-1]


def resolve_unit(sidecars: Mapping[str, Mapping], model_config_uuid: str) -> ModelUnit:
    """Group `{model_name: sidecar_dict}` into the unit for one model config.

    Detector order is `role == "top"` artifacts sorted by name. There is normally
    exactly one; the sort only makes the multi-top case deterministic rather than
    dependent on filesystem order, since the merge downstream is FIRST_WINS and a
    non-deterministic stack order would silently change which label wins a box.

    A sub-model with no `parent_label_uuid` is reported as a problem and dropped: it
    cannot be aimed at anything, and running it as a detector sends a classifier through
    the detection endpoint, where it answers with garbage instead of failing.
    """
    tops: list[UnitPart] = []
    subs: dict[str, list[UnitPart]] = {}
    parts: list[UnitPart] = []
    problems: list[str] = []
    parent_by_name: dict[str, Optional[str]] = {}

    for name in sorted(sidecars):
        meta = sidecars[name] or {}
        if meta.get("model_config_uuid") != model_config_uuid:
            continue
        part = UnitPart(
            model_name=name,
            role=str(meta.get("role") or ""),
            node_uuid=meta.get("model_config_node_uuid"),
            label_uuid=meta.get("parent_label_uuid") or meta.get("label_uuid"),
            trained_at=meta.get("trained_at"),
        )
        parts.append(part)
        parent_by_name[name] = meta.get("parent_label_uuid") or None

        # `role` is the authority, but fall back to model_kind for artifacts trained
        # before role was stamped — those exist in the wild and are still runnable.
        is_crop = meta.get("model_kind") == CROP_CLASSIFIER_KIND
        if part.role == ROLE_SUBMODEL or (not part.role and is_crop):
            subs.setdefault(part.node_uuid or name, []).append(part)
        else:
            tops.append(part)

    detector_names = [p.model_name for p in tops]

    sub_models: list[CropSubModel] = []
    for node_uuid, candidates in sorted(subs.items()):
        chosen = _newest_first(candidates)[0]
        for skipped in _newest_first(candidates)[1:]:
            problems.append(
                f"'{skipped.model_name}' and '{chosen.model_name}' were both trained for the same "
                f"sub-model; using the newer '{chosen.model_name}'."
            )
        parent = parent_by_name.get(chosen.model_name)
        if not parent:
            problems.append(
                f"Sub-model '{chosen.model_name}' has no parent label in its sidecar, so there is "
                "nothing to aim it at — skipping it. Retrain it from its model config node."
            )
            continue
        sub_models.append(CropSubModel(model_name=chosen.model_name, parent_label_uuid=parent))

    return ModelUnit(
        model_config_uuid=model_config_uuid,
        detector_names=detector_names,
        sub_models=sub_models,
        parts=parts,
        problems=problems,
    )


def resolve_stack(sidecars: Mapping[str, Mapping], model_names: Sequence[str]) -> ModelUnit:
    """Treat an explicitly listed stack of models as a unit.

    The hand-assembled counterpart to `resolve_unit`: the operator named these models, so
    any crop classifier among them is aimed at its own sidecar's boundary label. Same
    return type on purpose — the cascade must behave identically whether the models came
    from a config or from a list, or the two drift and only one of them gets fixed.

    Detector order is the caller's order, untouched. It is the stack priority a FIRST_WINS
    merge reads, and reordering it here would silently change which model's label wins an
    overlap.
    """
    model_kinds = {name: (sidecars.get(name) or {}).get("model_kind") for name in model_names}
    parent_labels = {name: (sidecars.get(name) or {}).get("parent_label_uuid") for name in model_names}
    detectors, sub_models = split_model_stack(model_names, model_kinds, parent_labels)

    problems = [
        f"Sub-model '{name}' has no parent label in its sidecar, so there is nothing to aim it at — "
        "skipping it. Retrain it from its model config node."
        for name in model_names
        if model_kinds.get(name) == CROP_CLASSIFIER_KIND and not parent_labels.get(name)
    ]

    parts = [
        UnitPart(
            model_name=name,
            role=ROLE_SUBMODEL if model_kinds.get(name) == CROP_CLASSIFIER_KIND else ROLE_TOP,
            node_uuid=(sidecars.get(name) or {}).get("model_config_node_uuid"),
            label_uuid=parent_labels.get(name),
            trained_at=(sidecars.get(name) or {}).get("trained_at"),
        )
        for name in model_names
    ]

    return ModelUnit(
        # An explicit stack belongs to no single config, and saying it belongs to one
        # would let a caller diff it against that config's targets and "find" parts
        # missing that were never asked for.
        model_config_uuid="",
        detector_names=detectors,
        sub_models=sub_models,
        parts=parts,
        problems=problems,
    )


def intra_unit_ancestry(
    sub_model_classes: Mapping[str, list[str]],
    parent_of_sub_model: Mapping[str, str],
    base: Optional[Mapping[str, set]] = None,
) -> dict[str, set]:
    """Build `{label_uuid: {ancestor label uuids}}` from the unit's own artifacts.

    `sub_model_classes` maps a sub-model's name to the label UUIDs it predicts;
    `parent_of_sub_model` maps the same names to their boundary label. Because a
    sub-model's classes are exactly its boundary's assigned descendants, every class
    gets that boundary — and the boundary's own ancestors — as ancestors. Applied over
    all sub-models this yields the full chain for a nested unit
    (willow → {cat, animal}) without ever reading the label tree.

    *base* is merged in first, so a caller holding the real ancestry (the watcher) keeps
    the cases this cannot see: a label that is a descendant of a boundary but is not one
    of that sub-model's classes, which only arises when a foreign model in the same stack
    emits it.

    Deliberately tolerant of a cycle or a self-parent in the input — a corrupt pair of
    sidecars must degrade to a shallower hierarchy, not spin.
    """
    direct: dict[str, str] = {}
    for name, classes in sub_model_classes.items():
        parent = parent_of_sub_model.get(name)
        if not parent:
            continue
        for cls in classes or []:
            if cls and cls != parent:
                direct[cls] = parent

    ancestry: dict[str, set] = {k: set(v) for k, v in (base or {}).items()}
    for label in direct:
        seen: set = set()
        current = direct.get(label)
        while current is not None and current not in seen:
            seen.add(current)
            current = direct.get(current)
        ancestry[label] = ancestry.get(label, set()) | seen

    # A boundary that is nobody's class still needs an entry, or order_sub_models reads
    # it as depth 0 and may run an inner pass before the one that feeds it.
    for parent in parent_of_sub_model.values():
        ancestry.setdefault(parent, set())
    return ancestry
