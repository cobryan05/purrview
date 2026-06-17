"""Binary image-level classifier trainer.

Trains a MobileNetV3-Small (via timm) to predict presence/absence of a label
in a whole image.  Exports to ONNX opset 17.

Expected job config keys (from SubmitJobPayload):
  trainer_type   "binary_image_classifier"
  label_uuid     UUID of the label to train on
  model_name     Output filename stem (e.g. "corrupt_detector")
  max_epochs     Maximum training epochs (default 50)
  image_source   "filesystem" | "http" (default "filesystem")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import register_trainer
from .base import Trainer

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))
_DB_BASE_URL_DEFAULT = "http://localhost:8003"
_FS_ROOT_DEFAULT = "/app/image_datasets"


@register_trainer("binary_image_classifier")
class BinaryImageClassifier(Trainer):
    """MobileNetV3-Small binary image classifier.

    Architecture: pretrained MobileNetV3-Small (timm) with the classifier head
    replaced by a 2-class linear layer.  Input: 224×224 RGB.

    Training:
        - Loss:  weighted CrossEntropyLoss (auto-handles class imbalance)
        - Optim: AdamW (lr=1e-3, weight_decay=1e-4)
        - LR:    CosineAnnealingLR
        - Early stopping: patience=10 epochs on val loss
    """

    async def train(
        self,
        config: dict[str, Any],
        progress_cb: Callable[[dict], None],
        cancel_event: asyncio.Event,
    ) -> dict[str, Any]:
        # Defer heavy imports — torch / timm are only in the training container.
        import timm
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        from apps.training_server.dataset import (
            FilesystemImageFetcher,
            HttpImageFetcher,
            _build_dataset,
        )

        label_uuid: str = config["label_uuid"]
        model_name: str = config.get("model_name", "model")
        max_epochs: int = int(config.get("max_epochs", 50))
        image_source: str = config.get("image_source", "filesystem")

        db_base_url = self._db_base_url()
        fs_root = os.environ.get("TRAINING_FS_ROOT", _FS_ROOT_DEFAULT)

        # ── Build datasets ──────────────────────────────────────────────────
        if image_source == "filesystem":
            fetcher_train = FilesystemImageFetcher(fs_root=fs_root)
            fetcher_val = FilesystemImageFetcher(fs_root=fs_root)
        else:
            fetcher_train = HttpImageFetcher(db_base_url=db_base_url)
            fetcher_val = HttpImageFetcher(db_base_url=db_base_url)

        try:
            logger.info("Building datasets for label_uuid=%s", label_uuid)
            ds_train = await asyncio.to_thread(_build_dataset, label_uuid, db_base_url, fetcher_train, "train")
            ds_val = await asyncio.to_thread(_build_dataset, label_uuid, db_base_url, fetcher_val, "val")
        except ValueError as e:
            raise RuntimeError(f"Dataset build failed: {e}") from e

        logger.info("Dataset sizes: train=%d val=%d", len(ds_train), len(ds_val))

        batch_size = int(config.get("batch_size", 32))
        loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # ── Model ────────────────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training on device: %s", device)

        model = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=0)
        in_features = model.num_features
        model.classifier = nn.Linear(in_features, 2)
        model = model.to(device)

        class_weights = ds_train.class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        # ── Training loop ────────────────────────────────────────────────────
        best_val_loss = float("inf")
        best_val_acc = 0.0
        patience = int(config.get("early_stopping_patience", 10))
        patience_counter = 0
        best_state: dict | None = None

        _models_dir = _MODELS_DIR
        _models_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = _models_dir / f"{model_name}.onnx"
        json_path = _models_dir / f"{model_name}.json"

        for epoch in range(1, max_epochs + 1):
            if cancel_event.is_set():
                break

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

            scheduler.step()

            log_msg = (
                f"Epoch {epoch}/{max_epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )
            logger.info(log_msg)
            progress_cb({
                "progress": epoch / max_epochs,
                "epoch": epoch,
                "total_epochs": max_epochs,
                "val_accuracy": round(val_acc, 4),
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "log": log_msg,
            })

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

            # Yield to the event loop between epochs so cancel_event can be polled.
            await asyncio.sleep(0)

        # ── Export best checkpoint ───────────────────────────────────────────
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

        # Validate ONNX output matches PyTorch output.
        import onnxruntime as ort
        ort_session = ort.InferenceSession(str(onnx_path))
        with torch.no_grad():
            pt_out = model(dummy).numpy()
        ort_out = ort_session.run(None, {"input": dummy.numpy()})[0]
        if not np.allclose(pt_out, ort_out, atol=1e-4):
            raise RuntimeError("ONNX validation failed: PyTorch and OnnxRuntime outputs differ")
        logger.info("ONNX validation passed, model saved to %s", onnx_path)

        # Write companion JSON config.
        model_config = {
            "model_kind": "image_classifier",
            "trainer": "binary_image_classifier",
            "input_size": [224, 224],
            "num_classes": 2,
            "class_names": ["absent", "present"],
            "label_uuid": label_uuid,
            "model_name": model_name,
            "val_accuracy": round(best_val_acc, 4),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        json_path.write_text(json.dumps(model_config, indent=2))
        logger.info("Model config written to %s", json_path)

        return {
            "model_path": str(onnx_path),
            "val_accuracy": round(best_val_acc, 4),
            "model_name": model_name,
        }

    @staticmethod
    def _db_base_url() -> str:
        host = os.environ.get("DB_SERVER_HOST", "localhost")
        port = os.environ.get("DB_SERVER_PORT", "8003")
        return f"http://{host}:{port}"
