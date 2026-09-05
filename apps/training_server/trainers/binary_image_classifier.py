"""Binary image-level classifier trainer.

Trains a timm backbone (default: MobileNetV3-Small) to predict presence/absence
of a label in a whole image.  Exports to ONNX opset 17.

Expected job config keys (all provided by TrainModelTask via the push lifecycle):
  trainer_type      "binary_image_classifier"
  label_uuid        UUID of the label trained on (written to companion JSON)
  model_name        Output filename stem (e.g. "corrupt_detector")
  architecture      DB arch name → timm model string (see _ARCH_MAP below)
  class_names       ["absent", "present"]  (default)
  max_epochs        Maximum training epochs (default 50)
  batch_size        DataLoader batch size (default 32)
  early_stopping_patience  Epochs without val improvement before stopping (default 10)
  manifest          [{image_uuid, filename, status}] — the full dataset description
  cache_dir         Path to uuid-keyed image cache populated by ingest API
  artifact_dir      Path where .onnx and .json should be written
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from . import register_trainer
from ._backbone import create_classifier_backbone
from ._two_phase import build_phase_optimizer_and_scheduler, phase_for_epoch
from .base import STOP_CANCELLED, STOP_COMPLETED, STOP_EARLY, Trainer, image_manifest_fields, stop_summary
from .onnx_validation import export_and_validate_onnx

logger = logging.getLogger(__name__)

# Map DB architecture names (from MODEL_TYPE_ARCHITECTURES) to timm model strings.
_ARCH_MAP: dict[str, str] = {
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b4": "efficientnet_b4",
    "mobilenet_v3_small": "mobilenetv3_small_100",
    "mobilenet_v3_large": "mobilenetv3_large_100",
    "resnet_18": "resnet18",
    "resnet_50": "resnet50",
}
_DEFAULT_TIMM_MODEL = "mobilenetv3_small_100"


@register_trainer("binary_image_classifier")
class BinaryImageClassifier(Trainer):
    """Architecture-aware binary image classifier.

    Architecture: pretrained timm backbone with the classifier head replaced by
    a 2-class linear layer.  Input: 224×224 RGB.

    Training — two phases, see ``_two_phase.py`` for why:
        - Phase 1 (``freeze_epochs`` epochs): backbone frozen, only the head
          trains, AdamW lr=1e-3.
        - Phase 2 (remaining epochs): everything trains, AdamW
          lr=``finetune_lr``, CosineAnnealingLR over the epochs left.
        - Loss:  weighted CrossEntropyLoss (handles class imbalance)
        - Early stopping: patience on val loss
    """

    HYPERPARAM_DEFAULTS: dict[str, Any] = {
        "max_epochs": 50,
        "batch_size": 32,
        "early_stopping_patience": 10,
        "freeze_epochs": 3,
        "finetune_lr": 3e-5,
        "augmentation": "standard",
    }

    async def train(
        self,
        config: dict[str, Any],
        progress_cb: Callable[[dict], None],
        cancel_event: asyncio.Event,
    ) -> dict[str, Any]:
        # Defer heavy imports — torch / timm are only in the training container.
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        from apps.training_server.dataset import build_dataset_from_manifest

        d = self.HYPERPARAM_DEFAULTS
        label_uuid: str = config.get("label_uuid", "")
        model_name: str = config.get("model_name", "model")
        role: str = config.get("role", "")
        model_config_uuid: str = config.get("model_config_uuid", "")
        model_config_node_uuid: str = config.get("model_config_node_uuid", "")
        max_epochs: int = int(config.get("max_epochs", d["max_epochs"]))
        batch_size: int = int(config.get("batch_size", d["batch_size"]))
        patience: int = int(config.get("early_stopping_patience", d["early_stopping_patience"]))
        freeze_epochs: int = int(config.get("freeze_epochs", d["freeze_epochs"]))
        finetune_lr: float = float(config.get("finetune_lr", d["finetune_lr"]))
        head_lr: float = 1e-3
        augmentation: str = config.get("augmentation", d["augmentation"])
        class_names: list[str] = config.get("class_names") or ["absent", "present"]
        manifest: list[dict] = config.get("manifest") or []
        cache_dir: str = config["cache_dir"]
        artifact_dir_str: str = config["artifact_dir"]

        # Architecture: map DB name → timm model string.
        arch_name: str = config.get("architecture", "")
        timm_model = _ARCH_MAP.get(arch_name, _DEFAULT_TIMM_MODEL)
        if arch_name and arch_name not in _ARCH_MAP:
            logger.warning("Unknown architecture %r — falling back to %s", arch_name, _DEFAULT_TIMM_MODEL)

        if not manifest:
            raise RuntimeError("No manifest provided — image data must be pushed before training")

        logger.info(
            "Training %s on %d images (arch=%s → timm=%s)",
            model_name,
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
            freeze_epochs=freeze_epochs,
            finetune_lr=finetune_lr,
            augmentation=augmentation,
        )

        # ── Build datasets ──────────────────────────────────────────────────
        try:
            ds_train = await asyncio.to_thread(
                build_dataset_from_manifest, manifest, cache_dir, "train", 0.2, augmentation
            )
            ds_val = await asyncio.to_thread(build_dataset_from_manifest, manifest, cache_dir, "val", 0.2, augmentation)
        except ValueError as e:
            raise RuntimeError(f"Dataset build failed: {e}") from e

        logger.info("Dataset sizes: train=%d val=%d", len(ds_train), len(ds_val))

        train_image_uuids = [r["image_uuid"] for r in ds_train._rows]
        val_image_uuids = [r["image_uuid"] for r in ds_val._rows]

        loader_train = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # ── Model ────────────────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training on device: %s (timm model: %s)", device, timm_model)
        self.warn_if_cpu(logger, progress_cb)

        # See _backbone.py for why the head is built there and not inline — the
        # hand-rolled version was silently wrong on resnet and raised on mobilenet.
        model = create_classifier_backbone(timm_model, len(class_names)).to(device)

        try:
            class_weights = ds_train.class_weights().to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            # ── Training loop ────────────────────────────────────────────────
            best_val_loss = float("inf")
            best_val_acc = 0.0
            patience_counter = 0
            # Early stopping compares against this, reset at every phase
            # transition — NOT the same as best_val_loss above. Comparing an
            # unfrozen epoch's val_loss against a threshold the frozen (head-only)
            # phase already reached lets a few slow-improving fine-tuning epochs
            # exhaust the whole patience budget right after unfreezing, exporting
            # a model whose backbone was never actually fine-tuned. best_val_loss
            # itself is never reset — it's what actually gets exported, and must
            # reflect the true best epoch across the whole run regardless of phase.
            phase_best_val_loss = float("inf")
            best_state: Optional[dict] = None
            # Why the loop below ends, decided at each of its three exits. Seeded
            # with the "ran the whole budget" answer, the only one reached without
            # a `break`.
            stop_reason = STOP_COMPLETED
            stop_detail = ""
            epochs_run = 0
            best_epoch: Optional[int] = None

            artifact_dir = Path(artifact_dir_str)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            onnx_path = artifact_dir / f"{model_name}.onnx"
            json_path = artifact_dir / f"{model_name}.json"
            checkpoint_path = artifact_dir / "checkpoint.pt"

            # Resume from checkpoint if one exists (e.g. after server restart).
            start_epoch = 1
            resumed_phase: Optional[str] = None
            resumed_optimizer_state: Optional[dict] = None
            resumed_scheduler_state: Optional[dict] = None
            if checkpoint_path.exists():
                try:
                    ckpt = await asyncio.to_thread(torch.load, str(checkpoint_path), map_location=device)
                    model.load_state_dict(ckpt["model_state_dict"])
                    best_val_loss = ckpt["best_val_loss"]
                    best_val_acc = ckpt["best_val_acc"]
                    patience_counter = ckpt["patience_counter"]
                    if ckpt.get("best_state"):
                        best_state = {k: v.cpu().clone() for k, v in ckpt["best_state"].items()}
                    phase_best_val_loss = ckpt.get("phase_best_val_loss", float("inf"))
                    start_epoch = ckpt["epoch"] + 1
                    resumed_phase = ckpt.get("phase")
                    resumed_optimizer_state = ckpt.get("optimizer_state_dict")
                    resumed_scheduler_state = ckpt.get("scheduler_state_dict")
                    logger.info("Resuming from checkpoint at epoch %d/%d", ckpt["epoch"], max_epochs)
                except Exception:
                    logger.exception("Failed to load checkpoint; starting training from scratch")

            current_phase: Optional[str] = None
            optimizer = None
            scheduler = None

            for epoch in range(start_epoch, max_epochs + 1):
                if cancel_event.is_set():
                    stop_reason = STOP_CANCELLED
                    stop_detail = f"cancelled by user before epoch {epoch}"
                    break
                epochs_run = epoch

                phase = phase_for_epoch(epoch, freeze_epochs)
                if phase != current_phase:
                    if current_phase is not None:
                        progress_cb({"log": f"Epoch {epoch}: unfreezing backbone, dropping lr to {finetune_lr:g}"})
                    if current_phase is not None or phase != resumed_phase:
                        # A genuine phase transition (mid-run boundary, or resuming
                        # into a different phase than the checkpoint's) — not the
                        # first loop iteration of a resume back into the same
                        # phase, which must keep its loaded patience state.
                        patience_counter = 0
                        phase_best_val_loss = float("inf")
                    optimizer, scheduler = build_phase_optimizer_and_scheduler(
                        model,
                        phase=phase,
                        max_epochs=max_epochs,
                        freeze_epochs=freeze_epochs,
                        head_lr=head_lr,
                        finetune_lr=finetune_lr,
                    )
                    if phase == resumed_phase and resumed_optimizer_state is not None:
                        try:
                            optimizer.load_state_dict(resumed_optimizer_state)
                            if scheduler is not None and resumed_scheduler_state is not None:
                                scheduler.load_state_dict(resumed_scheduler_state)
                        except Exception:
                            logger.warning(
                                "Could not restore optimizer/scheduler state for phase %r; starting phase fresh", phase
                            )
                        resumed_optimizer_state = None
                        resumed_scheduler_state = None
                    current_phase = phase

                # Reported before the epoch runs, not just after: on slow (e.g.
                # CPU) hardware, an epoch can take minutes, and with no event
                # between "job started" and "epoch 1 finished" the job looks
                # stuck rather than merely slow. `progress` mirrors the last
                # completed epoch's fraction so the bar doesn't jump backward.
                progress_cb(
                    {
                        "progress": (epoch - 1) / max_epochs,
                        "epoch": epoch,
                        "total_epochs": max_epochs,
                        "log": f"Epoch {epoch}/{max_epochs} [{phase}] starting...",
                    }
                )

                # Train
                model.train()
                train_loss = 0.0
                for xb, yb in loader_train:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * xb.size(0)
                train_loss /= len(ds_train)

                # Validate
                model.eval()
                val_loss = 0.0
                correct = 0
                with torch.no_grad():
                    for xb, yb in loader_val:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)
                        val_loss += criterion(logits, yb).item() * xb.size(0)
                        correct += (logits.argmax(dim=1) == yb).sum().item()
                val_loss /= len(ds_val)
                val_acc = correct / len(ds_val)

                if scheduler is not None:
                    scheduler.step()

                log_msg = (
                    f"Epoch {epoch}/{max_epochs} [{phase}]  "
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

                if val_loss < phase_best_val_loss:
                    phase_best_val_loss = val_loss
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
                        # The frontend log is the one the user reads; logger.info
                        # reaches only this server's process log.
                        progress_cb({"log": stop_detail})
                        break

                # Save checkpoint so training can resume after a server restart.
                await asyncio.to_thread(
                    torch.save,
                    {
                        "epoch": epoch,
                        "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                        "phase": phase,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                        "best_val_loss": best_val_loss,
                        "best_val_acc": best_val_acc,
                        "patience_counter": patience_counter,
                        "phase_best_val_loss": phase_best_val_loss,
                        "best_state": best_state,
                    },
                    str(checkpoint_path),
                )

                # Yield to the event loop so cancel_event can be checked.
                await asyncio.sleep(0)

            # ── Export best checkpoint ───────────────────────────────────────
            if best_state is not None:
                model.load_state_dict(best_state)

            export_and_validate_onnx(model, onnx_path, timm_model)
            if checkpoint_path.exists():
                checkpoint_path.unlink(missing_ok=True)
        finally:
            # These use persistent_workers=False, so torch keeps no reference to
            # their iterators and the loop's own local is freed on exit — they do
            # not leak the way ultralytics' InfiniteDataLoader does. Passed in
            # anyway so the gc pass inside reaps one still pinned by an exception
            # traceback, which is exactly the path a failed run takes to get here.
            self.shutdown_dataloaders(loader_train, loader_val)
            del model
            self.release_gpu_memory()

        # Write companion JSON config.
        model_config = {
            "model_kind": "image_classifier",
            "trainer": "binary_image_classifier",
            "architecture": arch_name or "mobilenet_v3_small",
            "timm_model": timm_model,
            "input_size": [224, 224],
            "num_classes": len(class_names),
            "class_names": class_names,
            "label_uuid": label_uuid,
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
