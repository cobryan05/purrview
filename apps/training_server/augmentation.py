"""Shared image-augmentation presets.

One place for "how aggressively do we distort training images", used by the
three timm classifier trainers (via ``dataset.py``) and, where the underlying
library actually exposes real augmentation knobs, ``ultralytics_object_detection``.

Why a preset table instead of per-field hyperparams: cats look extremely
similar to each other (fur pattern, eye shape, pose), so a classifier with too
little augmentation memorises specific crops instead of learning the visual
difference between two cats, and overfits in 10-15 epochs. "Standard" below is
the augmentation this app already shipped (RandomResizedCrop 0.8-1.0, flip,
mild colour jitter); "heavy" is for exactly that failure mode.

``object_detection`` (LibreYOLO9) is deliberately not wired to this: its
``train()`` takes an undocumented ``**kwargs``, and there is no confirmed way
to know which augmentation knobs (if any) it forwards — guessing would be
silently wrong in the same way ``_backbone.py``'s hand-rolled head once was.
"""

from __future__ import annotations

from typing import Any

#: Preset name -> classifier-dataset augmentation parameters.
#: ``scale_min`` feeds ``RandomResizedCrop(scale=(scale_min, 1.0))``.
AUGMENTATION_PRESETS: dict[str, dict[str, float]] = {
    "off": {
        "horizontal_flip": 0.0,
        "rotate_limit": 0,
        "color_jitter": 0.0,
        "scale_min": 1.0,
    },
    "light": {
        "horizontal_flip": 0.5,
        "rotate_limit": 5,
        "color_jitter": 0.1,
        "scale_min": 0.9,
    },
    # What this app already shipped before augmentation became configurable.
    "standard": {
        "horizontal_flip": 0.5,
        "rotate_limit": 0,
        "color_jitter": 0.2,
        "scale_min": 0.8,
    },
    "heavy": {
        "horizontal_flip": 0.5,
        "rotate_limit": 20,
        "color_jitter": 0.35,
        "scale_min": 0.6,
    },
}

DEFAULT_AUGMENTATION_PRESET = "standard"


def resolve_augmentation_preset(name: str | None) -> dict[str, float]:
    """A classifier-dataset preset dict for `name`, falling back to "standard"
    for `None` or an unrecognised name (never raises — a typo'd hyperparam
    should not fail a training run)."""
    return AUGMENTATION_PRESETS.get(
        name or DEFAULT_AUGMENTATION_PRESET, AUGMENTATION_PRESETS[DEFAULT_AUGMENTATION_PRESET]
    )


def build_classifier_train_transform(size: tuple[int, int], preset: dict[str, float]) -> Any:
    """The train-split ``A.Compose`` for a classifier dataset, from a resolved preset.

    Kept as one function so the three classifier datasets (``ImageLabelDataset``,
    ``MultiLabelDataset``, ``CropDataset``) build augmentation identically instead
    of drifting — they used to each hand-roll the same ``A.Compose`` block.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    h, w = size
    steps: list[Any] = [A.RandomResizedCrop(size=(h, w), scale=(preset["scale_min"], 1.0))]
    if preset["horizontal_flip"] > 0:
        steps.append(A.HorizontalFlip(p=preset["horizontal_flip"]))
    if preset["rotate_limit"] > 0:
        steps.append(A.Rotate(limit=preset["rotate_limit"], p=0.5, border_mode=0))
    if preset["color_jitter"] > 0:
        cj = preset["color_jitter"]
        steps.append(A.ColorJitter(brightness=cj, contrast=cj, saturation=cj, hue=min(cj / 4, 0.1), p=0.5))
    steps.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    steps.append(ToTensorV2())
    return A.Compose(steps)


#: Preset name -> kwargs forwarded straight into ``ultralytics`` ``model.train()``.
#: "standard" is deliberately empty — omitting these keys lets ultralytics apply
#: its own (well-tuned) defaults rather than us reproducing them and risking drift.
YOLO_AUGMENTATION_KWARGS: dict[str, dict[str, float]] = {
    "off": dict(
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
    ),
    "light": dict(
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
    ),
    "standard": {},
    "heavy": dict(
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=20.0,
        translate=0.15,
        scale=0.7,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
    ),
}


def resolve_yolo_augmentation_kwargs(name: str | None) -> dict[str, float]:
    """kwargs to splat into ``YOLO(...).train(...)`` for `name` — ``{}`` (no
    override) for "standard" or an unrecognised name."""
    return dict(YOLO_AUGMENTATION_KWARGS.get(name or DEFAULT_AUGMENTATION_PRESET, {}))
