"""Shared dataset assembly for classifier training jobs.

TrainModelTask (driven by a TrainingConfig's node lookup) assembles the
(model_name, manifest, job_config) triples for binary and multi-label classifier
jobs here. This was shared with the since-removed TrainBundleTask; it stays its
own module because the dataset-scoping and manifest semantics are worth keeping
in one place, independent of whichever task drives them.

Callers run these in a thread (synchronous generated db_client APIs).
"""

from __future__ import annotations

import logging

from db_client.models.image_item_status import ImageItemStatus

logger = logging.getLogger(__file__)


def _status_value(status) -> str:
    return status.value if hasattr(status, "value") else str(status)


def build_binary_classifier_job(
    img_api,
    *,
    label_uuid: str,
    trainer_type: str,
    model_name: str,
    architecture: str,
    role: str,
    model_config_uuid: str,
    model_config_node_uuid: str,
    hyperparams: dict,
    dataset_uuids: list[str],
    dataset_uuids_set: set[str],
    record_image_manifest: bool = True,
) -> tuple[str, list[dict], dict]:
    """Assemble a binary-classifier job from one label's present/absent rows."""
    all_rows_resp = img_api.list_images_for_label(
        label_uuid=label_uuid,
        status=[ImageItemStatus.PRESENT, ImageItemStatus.ABSENT],
    )
    all_rows = all_rows_resp.images if hasattr(all_rows_resp, "images") else []
    raw_count = len(all_rows)

    if dataset_uuids:
        all_rows = [r for r in all_rows if r.image_uuid in dataset_uuids_set]
        logger.debug(
            "Dataset filter: %d → %d rows (dataset_set size=%d)",
            raw_count,
            len(all_rows),
            len(dataset_uuids_set),
        )

    if not all_rows:
        raise RuntimeError(
            f"No labelled images found for label_uuid={label_uuid!r} "
            f"with {len(dataset_uuids)} dataset(s). "
            f"Found {raw_count} labeled rows before dataset filter "
            f"(dataset_set size={len(dataset_uuids_set)})."
        )

    manifest = [
        {
            "image_uuid": r.image_uuid,
            "filename": r.filename,
            "status": _status_value(r.status),
        }
        for r in all_rows
    ]

    job_config = {
        "trainer_type": trainer_type,
        "label_uuid": label_uuid,
        "model_name": model_name,
        "architecture": architecture,
        "class_names": ["absent", "present"],
        "role": role,
        "model_config_uuid": model_config_uuid,
        "model_config_node_uuid": model_config_node_uuid,
        **hyperparams,
        # After the splat: threaded explicitly, so a stale copy left in
        # hyperparams_json can never win over the caller's resolved value.
        "record_image_manifest": record_image_manifest,
    }
    return model_name, manifest, job_config


def build_multi_label_classifier_job(
    img_api,
    *,
    label_uuids: list[str],
    model_name: str,
    architecture: str,
    role: str,
    model_config_uuid: str,
    model_config_node_uuid: str,
    hyperparams: dict,
    dataset_uuids: list[str],
    dataset_uuids_set: set[str],
    record_image_manifest: bool = True,
) -> tuple[str, list[dict], dict]:
    """Assemble a multi-label classifier job from several labels' rows.

    Manifest holds unique images (its status field is unused by the
    multi_label_classifier trainer); per-image per-label targets ride in
    job_config["label_targets"].
    """
    uuid_to_filename: dict[str, str] = {}
    label_targets: dict[str, dict[str, int]] = {}

    for luuid in label_uuids:
        rows_resp = img_api.list_images_for_label(
            label_uuid=luuid,
            status=[ImageItemStatus.PRESENT, ImageItemStatus.ABSENT],
        )
        rows = rows_resp.images if hasattr(rows_resp, "images") else []
        raw_count = len(rows)
        if dataset_uuids:
            rows = [r for r in rows if r.image_uuid in dataset_uuids_set]
        logger.debug("Label %s: %d labeled rows, %d after dataset filter", luuid, raw_count, len(rows))

        for r in rows:
            uuid_to_filename[r.image_uuid] = r.filename
            label_targets.setdefault(r.image_uuid, {})[luuid] = (
                1 if _status_value(r.status) == ImageItemStatus.PRESENT.value else 0
            )

    if not uuid_to_filename:
        raise RuntimeError(
            f"No labeled images found across {len(label_uuids)} label(s) "
            f"with {len(dataset_uuids)} dataset(s). "
            f"dataset_set size={len(dataset_uuids_set)}. "
            f"Check that labeled images are included in the selected dataset(s)."
        )

    manifest = [
        {"image_uuid": uuid, "filename": fname, "status": "present"} for uuid, fname in uuid_to_filename.items()
    ]

    job_config = {
        "trainer_type": "multi_label_classifier",
        "model_name": model_name,
        "architecture": architecture,
        "label_uuids": list(label_uuids),
        "label_targets": label_targets,
        "role": role,
        "model_config_uuid": model_config_uuid,
        "model_config_node_uuid": model_config_node_uuid,
        **hyperparams,
        "record_image_manifest": record_image_manifest,
    }
    return model_name, manifest, job_config
