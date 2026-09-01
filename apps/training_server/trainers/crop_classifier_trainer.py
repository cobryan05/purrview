"""Crop classifier trainer.

Trains an image classifier on bounding-box crops — the second pass of a
hierarchical model config. The top detector finds the parent concept ("Cat"),
this model says which child it is ("Willow" / "Maple" / "Luna") from the crop
alone.

Single-label: a crop is exactly one class, so this is softmax/cross-entropy,
unlike ``multi_label_classifier`` (independent per-label sigmoids over a whole
image).

Expected job config keys:
  trainer_type      "crop_classifier"
  label_uuids       Ordered list of label UUIDs (defines output unit order)
  class_names       Label names aligned with label_uuids (sidecar/logging only)
  crop_targets      {image_uuid: [{class_idx, x, y, width, height}]} — relative coords
  parent_label_uuid The boundary label these classes sit under (e.g. "Cat"), so a
                    caller knows which detections to feed this model
  crop_padding      Context added around each box, as a fraction of its size
  model_name        Output filename stem
  architecture      DB arch name (see _ARCH_MAP)
  max_epochs        Maximum training epochs (default 50)
  batch_size        DataLoader batch size (default 32)
  early_stopping_patience  Epochs without val improvement before stopping (default 10)
  manifest          [{image_uuid, filename, status}] — unique images; status ignored
  cache_dir         Path to uuid-keyed image cache
  artifact_dir      Path where .onnx and .json should be written
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from apps.helpers.classifier_utils import DEFAULT_CROP_PADDING

from . import register_trainer
from ._backbone import create_classifier_backbone
from .base import STOP_CANCELLED, STOP_COMPLETED, STOP_EARLY, Trainer, image_manifest_fields, stop_summary
from .onnx_validation import export_and_validate_onnx

logger = logging.getLogger(__name__)

_ARCH_MAP: dict[str, str] = {
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b4": "efficientnet_b4",
    "mobilenet_v3_small": "mobilenetv3_small_100",
    "mobilenet_v3_large": "mobilenetv3_large_100",
    "resnet_18": "resnet18",
    "resnet_50": "resnet50",
}
_DEFAULT_TIMM_MODEL = "resnet18"


@register_trainer("crop_classifier")
class CropClassifierTrainer(Trainer):
    """Crop-level single-label classifier.

    Architecture: pretrained timm backbone with an N-way head.

    Training:
        - Loss:  CrossEntropyLoss with inverse-frequency class weights
        - Optim: AdamW (lr=1e-3, weight_decay=1e-4)
        - LR:    CosineAnnealingLR
        - Early stopping: patience on val loss
    """

    HYPERPARAM_DEFAULTS: dict[str, int] = {
        "max_epochs": 50,
        "batch_size": 32,
        "early_stopping_patience": 10,
    }

    async def train(
        self,
        config: dict[str, Any],
        progress_cb: Callable[[dict], None],
        cancel_event: asyncio.Event,
    ) -> dict[str, Any]:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        from apps.training_server.dataset import build_crop_dataset_from_manifest

        d = self.HYPERPARAM_DEFAULTS
        label_uuids: list[str] = config.get("label_uuids") or []
        class_names: list[str] = config.get("class_names") or list(label_uuids)
        crop_targets: dict[str, list[dict]] = config.get("crop_targets") or {}
        parent_label_uuid: str = config.get("parent_label_uuid", "")
        crop_padding: float = float(config.get("crop_padding", DEFAULT_CROP_PADDING))
        model_name: str = config.get("model_name", "model")
        role: str = config.get("role", "")
        model_config_uuid: str = config.get("model_config_uuid", "")
        model_config_node_uuid: str = config.get("model_config_node_uuid", "")
        manifest: list[dict] = config.get("manifest") or []
        cache_dir: str = config["cache_dir"]
        artifact_dir_str: str = config["artifact_dir"]
        max_epochs: int = int(config.get("max_epochs", d["max_epochs"]))
        batch_size: int = int(config.get("batch_size", d["batch_size"]))
        patience: int = int(config.get("early_stopping_patience", d["early_stopping_patience"]))
        arch_name: str = config.get("architecture", "")
        timm_model = _ARCH_MAP.get(arch_name, _DEFAULT_TIMM_MODEL)

        N = len(label_uuids)
        if N < 2:
            # One class can't be discriminated from anything: the model would learn
            # the constant function and report 100% accuracy. Fail loudly instead —
            # this is what an unfinished sub-model (a boundary with one child, or
            # with none) looks like from here.
            raise RuntimeError(
                f"A crop classifier needs at least 2 classes to discriminate between, got {N}. "
                "Assign more child labels under this sub-model's boundary label."
            )
        if not manifest:
            raise RuntimeError("No manifest provided — images must be pushed before training")
        if not crop_targets:
            raise RuntimeError("No crop_targets provided — there are no verified bboxes in this sub-model's scope")
        if arch_name and arch_name not in _ARCH_MAP:
            logger.warning("Unknown architecture %r — falling back to %s", arch_name, _DEFAULT_TIMM_MODEL)

        total_crops = sum(len(v) for v in crop_targets.values())
        logger.info(
            "Training crop classifier %r: %d classes, %d crops across %d images (arch=%s → timm=%s)",
            model_name,
            N,
            total_crops,
            len(manifest),
            arch_name or "(default)",
            timm_model,
        )
        self.log_effective_hyperparams(
            logger,
            progress_cb,
            architecture=arch_name or timm_model,
            epochs=max_epochs,
            batch=batch_size,
            patience=patience,
            crop_padding=crop_padding,
        )

        # ── Build datasets ──────────────────────────────────────────────────
        try:
            ds_train = await asyncio.to_thread(
                build_crop_dataset_from_manifest,
                manifest,
                crop_targets,
                N,
                cache_dir,
                "train",
                0.2,
                crop_padding,
            )
            ds_val = await asyncio.to_thread(
                build_crop_dataset_from_manifest,
                manifest,
                crop_targets,
                N,
                cache_dir,
                "val",
                0.2,
                crop_padding,
            )
        except ValueError as e:
            raise RuntimeError(f"Dataset build failed: {e}") from e

        counts_msg = ", ".join(f"{name}={count}" for name, count in zip(class_names, ds_train.class_counts()))
        logger.info("Crop counts: train=%d val=%d (train per class: %s)", len(ds_train), len(ds_val), counts_msg)
        progress_cb({"log": f"Training on {len(ds_train)} crops ({counts_msg}), validating on {len(ds_val)}"})

        train_image_uuids = ds_train.image_uuids
        val_image_uuids = ds_val.image_uuids

        loader_train = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        loader_val = DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # ── Model ────────────────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training on device: %s (timm model: %s)", device, timm_model)

        # See _backbone.py — one head construction, shared with the two whole-image
        # classifier trainers, because the hand-rolled alternative is wrong on four
        # of the seven architectures offered here.
        model = create_classifier_backbone(timm_model, N).to(device)

        try:
            class_weights = ds_train.class_weights().to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

            best_val_loss = float("inf")
            best_val_acc = 0.0
            patience_counter = 0
            best_state: Optional[dict] = None
            stop_reason = STOP_COMPLETED
            stop_detail = ""
            epochs_run = 0
            best_epoch: Optional[int] = None

            artifact_dir = Path(artifact_dir_str)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            onnx_path = artifact_dir / f"{model_name}.onnx"
            json_path = artifact_dir / f"{model_name}.json"
            checkpoint_path = artifact_dir / "checkpoint.pt"

            start_epoch = 1
            if checkpoint_path.exists():
                try:
                    ckpt = await asyncio.to_thread(torch.load, str(checkpoint_path), map_location=device)
                    model.load_state_dict(ckpt["model_state_dict"])
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    best_val_loss = ckpt["best_val_loss"]
                    best_val_acc = ckpt["best_val_acc"]
                    patience_counter = ckpt["patience_counter"]
                    if ckpt.get("best_state"):
                        best_state = {k: v.cpu().clone() for k, v in ckpt["best_state"].items()}
                    start_epoch = ckpt["epoch"] + 1
                    logger.info("Resuming from checkpoint at epoch %d/%d", ckpt["epoch"], max_epochs)
                except Exception:
                    logger.exception("Failed to load checkpoint; starting from scratch")

            for epoch in range(start_epoch, max_epochs + 1):
                if cancel_event.is_set():
                    stop_reason = STOP_CANCELLED
                    stop_detail = f"cancelled by user before epoch {epoch}"
                    break
                epochs_run = epoch

                # ── Train ─────────────────────────────────────────────────
                model.train()
                train_loss_sum = 0.0
                for xb, yb in loader_train:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    logits = model(xb)  # [B, N]
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    train_loss_sum += loss.item() * xb.size(0)
                train_loss = train_loss_sum / len(ds_train)

                # ── Validate ──────────────────────────────────────────────
                model.eval()
                val_loss_sum = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for xb, yb in loader_val:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)
                        val_loss_sum += criterion(logits, yb).item() * xb.size(0)
                        correct += (logits.argmax(dim=1) == yb).sum().item()
                        total += yb.numel()
                val_loss = val_loss_sum / len(ds_val)
                val_acc = correct / total if total > 0 else 0.0

                scheduler.step()

                log_msg = (
                    f"Epoch {epoch}/{max_epochs}  "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                )
                logger.info(log_msg)
                progress_cb(
                    {
                        "progress": epoch / max_epochs,
                        "epoch": epoch,
                        "total_epochs": max_epochs,
                        "val_accuracy": round(val_acc, 4),
                        "train_loss": round(train_loss, 4),
                        "val_loss": round(val_loss, 4),
                        "log": log_msg,
                    }
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        stop_reason = STOP_EARLY
                        stop_detail = (
                            f"early stopping at epoch {epoch} of {max_epochs}: validation loss did not "
                            f"improve for {patience} epoch(s) (patience={patience}); "
                            f"best was {best_val_loss:.4f}"
                            + (f" at epoch {best_epoch}" if best_epoch is not None else "")
                        )
                        logger.info(stop_detail)
                        progress_cb({"log": stop_detail})
                        break

                await asyncio.to_thread(
                    torch.save,
                    {
                        "epoch": epoch,
                        "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "best_val_acc": best_val_acc,
                        "patience_counter": patience_counter,
                        "best_state": best_state,
                    },
                    str(checkpoint_path),
                )
                await asyncio.sleep(0)

            # ── Export best weights ──────────────────────────────────────────
            if best_state is not None:
                model.load_state_dict(best_state)

            export_and_validate_onnx(model, onnx_path, timm_model)
            checkpoint_path.unlink(missing_ok=True)
        finally:
            self.shutdown_dataloaders(loader_train, loader_val)
            del model
            self.release_gpu_memory()

        model_config = {
            "model_kind": "crop_classifier",
            "trainer": "crop_classifier",
            "architecture": arch_name or "resnet_18",
            "timm_model": timm_model,
            "input_size": [224, 224],
            "num_labels": N,
            "label_uuids": label_uuids,
            "class_names": class_names,
            # The boundary label this sub-model refines. A cascade reads this to know
            # which of the top model's detections to hand it.
            "parent_label_uuid": parent_label_uuid,
            # Serve-time crop geometry must match train-time exactly; see crop_region.
            "crop_padding": crop_padding,
            "model_name": model_name,
            "val_accuracy": round(best_val_acc, 4),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            **image_manifest_fields(config, train_image_uuids, val_image_uuids),
            "role": role,
            "model_config_uuid": model_config_uuid,
            "model_config_node_uuid": model_config_node_uuid,
        }
        json_path.write_text(json.dumps(model_config, indent=2))
        logger.info("Model config written to %s", json_path)

        if stop_reason == STOP_COMPLETED and not stop_detail:
            stop_detail = f"ran the full {max_epochs} epoch(s)" + (
                f"; best validation loss at epoch {best_epoch}" if best_epoch is not None else ""
            )

        return {
            "model_path": str(onnx_path),
            "val_accuracy": round(best_val_acc, 4),
            "model_name": model_name,
            **stop_summary(
                reason=stop_reason,
                detail=stop_detail,
                epochs_run=epochs_run,
                max_epochs=max_epochs,
                best_epoch=best_epoch,
            ),
        }
