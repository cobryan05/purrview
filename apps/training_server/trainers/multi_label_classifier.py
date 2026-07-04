"""Multi-label image-level classifier trainer.

Trains a shared timm backbone to predict presence/absence of N labels simultaneously
in a whole image, exporting a single ONNX model with N sigmoid outputs.

Expected job config keys:
  trainer_type      "multi_label_classifier"
  label_uuids       Ordered list of label UUIDs (defines output vector order)
  label_targets     {image_uuid: {label_uuid: 0_or_1}} — per-image per-label status.
                    Missing entries are masked from loss (-1 in the dataset).
  model_name        Output filename stem
  architecture      DB arch name (see _ARCH_MAP)
  max_epochs        Maximum training epochs (default 50)
  batch_size        DataLoader batch size (default 32)
  early_stopping_patience  Epochs without val improvement before stopping (default 10)
  manifest          [{image_uuid, filename, status}] — unique images; status is ignored
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

import numpy as np

from . import register_trainer
from .base import Trainer

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
_DEFAULT_TIMM_MODEL = "mobilenetv3_small_100"


@register_trainer("multi_label_classifier")
class MultiLabelClassifier(Trainer):
    """Shared-backbone multi-label image classifier.

    Architecture: pretrained timm backbone with an N-output linear head.
    Each output is an independent binary decision (sigmoid, not softmax).

    Training:
        - Loss:  masked BCEWithLogitsLoss with per-label positive-class weighting
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
        import timm
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        from apps.training_server.dataset import build_multilabel_dataset_from_manifest

        d = self.HYPERPARAM_DEFAULTS
        label_uuids: list[str] = config.get("label_uuids") or []
        label_targets: dict[str, dict[str, int]] = config.get("label_targets") or {}
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
        if N == 0:
            raise RuntimeError("No label_uuids provided in config")
        if not manifest:
            raise RuntimeError("No manifest provided — images must be pushed before training")
        if arch_name and arch_name not in _ARCH_MAP:
            logger.warning("Unknown architecture %r — falling back to %s", arch_name, _DEFAULT_TIMM_MODEL)

        logger.info(
            "Training multi-label model %r: %d labels, %d images (arch=%s → timm=%s)",
            model_name,
            N,
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
        )

        # ── Build datasets ──────────────────────────────────────────────────
        try:
            ds_train = await asyncio.to_thread(
                build_multilabel_dataset_from_manifest,
                manifest,
                label_uuids,
                label_targets,
                cache_dir,
                "train",
            )
            ds_val = await asyncio.to_thread(
                build_multilabel_dataset_from_manifest,
                manifest,
                label_uuids,
                label_targets,
                cache_dir,
                "val",
            )
        except ValueError as e:
            raise RuntimeError(f"Dataset build failed: {e}") from e

        logger.info("Dataset sizes: train=%d val=%d", len(ds_train), len(ds_val))

        train_image_uuids = list(ds_train._image_uuids)
        val_image_uuids = list(ds_val._image_uuids)

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

        model = timm.create_model(timm_model, pretrained=True, num_classes=0)
        in_features = model.num_features
        model.classifier = nn.Linear(in_features, N)
        model = model.to(device)

        pos_weight = ds_train.pos_weights().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        # ── Training loop ────────────────────────────────────────────────────
        best_val_loss = float("inf")
        best_val_acc = 0.0
        patience_counter = 0
        best_state: Optional[dict] = None

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
                break

            # ── Train ─────────────────────────────────────────────────────
            model.train()
            train_loss_sum = 0.0
            for xb, yb in loader_train:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)  # [B, N]
                mask = yb >= 0  # [B, N] — known labels only
                loss_elem = criterion(logits, yb.clamp(min=0))  # [B, N]
                loss = (loss_elem * mask).sum() / mask.sum().clamp(min=1)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * xb.size(0)
            train_loss = train_loss_sum / len(ds_train)

            # ── Validate ──────────────────────────────────────────────────
            model.eval()
            val_loss_sum = 0.0
            correct = 0
            total_known = 0
            with torch.no_grad():
                for xb, yb in loader_val:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    mask = yb >= 0
                    loss_elem = criterion(logits, yb.clamp(min=0))
                    val_loss_sum += ((loss_elem * mask).sum() / mask.sum().clamp(min=1)).item() * xb.size(0)
                    preds = (logits.sigmoid() > 0.5).float()
                    correct += ((preds == yb.clamp(min=0)) & mask).sum().item()
                    total_known += mask.sum().item()
            val_loss = val_loss_sum / len(ds_val)
            val_acc = correct / total_known if total_known > 0 else 0.0

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
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
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

        # ── Export best weights ──────────────────────────────────────────────
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        model = model.cpu()

        dummy = torch.zeros(1, 3, 224, 224)
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        import onnxruntime as ort

        ort_session = ort.InferenceSession(str(onnx_path))
        with torch.no_grad():
            pt_out = model(dummy).numpy()
        ort_out = ort_session.run(None, {"input": dummy.numpy()})[0]
        max_diff = float(np.abs(pt_out - ort_out).max())
        logger.info("ONNX validation: max abs diff = %.6f", max_diff)
        if not np.allclose(pt_out, ort_out, atol=1e-2):
            raise RuntimeError(
                f"ONNX validation failed: PyTorch and OnnxRuntime outputs differ (max diff {max_diff:.6f})"
            )
        logger.info("ONNX validation passed, model saved to %s", onnx_path)
        checkpoint_path.unlink(missing_ok=True)

        model_config = {
            "model_kind": "multi_label_classifier",
            "trainer": "multi_label_classifier",
            "architecture": arch_name or "mobilenet_v3_small",
            "timm_model": timm_model,
            "input_size": [224, 224],
            "num_labels": N,
            "label_uuids": label_uuids,
            "model_name": model_name,
            "val_accuracy": round(best_val_acc, 4),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "train_image_uuids": train_image_uuids,
            "val_image_uuids": val_image_uuids,
            "role": role,
            "model_config_uuid": model_config_uuid,
            "model_config_node_uuid": model_config_node_uuid,
        }
        json_path.write_text(json.dumps(model_config, indent=2))
        logger.info("Model config written to %s", json_path)

        return {
            "model_path": str(onnx_path),
            "val_accuracy": round(best_val_acc, 4),
            "model_name": model_name,
        }
