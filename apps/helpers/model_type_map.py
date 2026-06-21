"""Maps DB model_type values to trainer registry keys.

Used by both tasks_server (to derive trainer_type when submitting a job) and
training_server/manager.py (as a fallback when trainer_type is absent from config).
"""

MODEL_TYPE_TO_TRAINER_TYPE: dict[str, str] = {
    "image_classifier": "binary_image_classifier",  # leaf node; root node uses multi_label_classifier
    "crop_classifier": "crop_classifier",
    "object_detection": "object_detection",
}
