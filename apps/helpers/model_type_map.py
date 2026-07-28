"""Maps DB model_type values to trainer registry keys.

Used by both tasks_server (to derive trainer_type when submitting a job) and
training_server/manager.py (as a fallback when trainer_type is absent from config).
"""

MODEL_TYPE_TO_TRAINER_TYPE: dict[str, str] = {
    "image_classifier": "binary_image_classifier",  # leaf node; root node uses multi_label_classifier
    "crop_classifier": "crop_classifier",
    "object_detection": "object_detection",
}

# Architecture name prefixes/sets that route object_detection to a trainer
# OTHER than the default LibreYOLO9 ("object_detection") trainer. Checked in
# order; the first match wins. Extend this when adding another arch-dependent
# family under object_detection.
_OBJECT_DETECTION_ARCH_OVERRIDES: dict[str, str] = {
    # Ultralytics' own architectures (v8, its own GELAN v9, v11, v12) — distinct
    # from LibreYOLO9's "yolo_v9_*" names, trained by a different trainer.
    "yolov8n": "ultralytics_object_detection",
    "yolov8s": "ultralytics_object_detection",
    "yolov8m": "ultralytics_object_detection",
    "yolov8l": "ultralytics_object_detection",
    "yolov8x": "ultralytics_object_detection",
    "yolov9t": "ultralytics_object_detection",
    "yolov9s": "ultralytics_object_detection",
    "yolov9m": "ultralytics_object_detection",
    "yolov9c": "ultralytics_object_detection",
    "yolov9e": "ultralytics_object_detection",
    "yolo11n": "ultralytics_object_detection",
    "yolo11s": "ultralytics_object_detection",
    "yolo11m": "ultralytics_object_detection",
    "yolo11l": "ultralytics_object_detection",
    "yolo11x": "ultralytics_object_detection",
    "yolo12n": "ultralytics_object_detection",
    "yolo12s": "ultralytics_object_detection",
    "yolo12m": "ultralytics_object_detection",
    "yolo12l": "ultralytics_object_detection",
    "yolo12x": "ultralytics_object_detection",
    # YOLOv10 / YOLO26 — NMS-free ("end-to-end") exports, still trained by the
    # same UltralyticsDetectionTrainer (only the decode/model_family differs;
    # see ultralytics_detection_trainer.py's _MODEL_FAMILY_MAP).
    "yolov10n": "ultralytics_object_detection",
    "yolov10s": "ultralytics_object_detection",
    "yolov10m": "ultralytics_object_detection",
    "yolov10b": "ultralytics_object_detection",
    "yolov10l": "ultralytics_object_detection",
    "yolov10x": "ultralytics_object_detection",
    "yolo26n": "ultralytics_object_detection",
    "yolo26s": "ultralytics_object_detection",
    "yolo26m": "ultralytics_object_detection",
    "yolo26l": "ultralytics_object_detection",
    "yolo26x": "ultralytics_object_detection",
}


def resolve_trainer_type(model_type: str, architecture: str) -> str:
    """Resolve the trainer registry key for a given ``model_type``/``architecture`` pair.

    For most model types this is a simple 1:1 lookup via
    ``MODEL_TYPE_TO_TRAINER_TYPE`` (arch-independent). ``object_detection`` is
    arch-aware: LibreYOLO9's own ``yolo_v9_*`` architectures keep training via
    the ``"object_detection"`` trainer unchanged, while the newer Ultralytics
    architectures (``yolov8*``, Ultralytics' own ``yolov9*``, ``yolo11*``,
    ``yolo12*``) route to ``"ultralytics_object_detection"``.

    Raises:
        ValueError: if ``model_type`` is unrecognized, or if it's
            ``object_detection`` with an architecture that isn't mapped to
            either trainer family (fail loudly rather than silently falling
            back to a wrong trainer).
    """
    if model_type == "object_detection":
        if architecture.startswith("yolo_v9_"):
            return "object_detection"
        override = _OBJECT_DETECTION_ARCH_OVERRIDES.get(architecture)
        if override is not None:
            return override
        raise ValueError(
            f"Unrecognized object_detection architecture {architecture!r} — cannot resolve a trainer. "
            f"Known LibreYOLO9 prefix: 'yolo_v9_*'. Known Ultralytics architectures: "
            f"{sorted(_OBJECT_DETECTION_ARCH_OVERRIDES)}."
        )

    trainer_type = MODEL_TYPE_TO_TRAINER_TYPE.get(model_type)
    if trainer_type is None:
        raise ValueError(
            f"Unrecognized model_type {model_type!r} — cannot resolve a trainer. "
            f"Known model_types: {sorted(MODEL_TYPE_TO_TRAINER_TYPE)}."
        )
    return trainer_type
