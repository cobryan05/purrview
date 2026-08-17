"""RunClassifierTask — apply a trained ONNX image classifier to images.

Loads the ONNX model written by BinaryImageClassifier and tags images in the DB
based on confidence.
"""

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import db_client
import numpy as np
from db_client.api.images_api import ImagesApi
from db_client.api.sources_api import SourcesApi
from db_client.models.image_item_status import ImageItemStatus
from db_client.models.set_image_label_payload import SetImageLabelPayload

from apps import FILES_ROOT
from apps.helpers.classifier_utils import preprocess_image_bytes as _preprocess_image
from apps.helpers.classifier_utils import softmax as _softmax
from apps.helpers.fileUtils import get_safe_path
from apps.helpers.logUtils import LOG_LEVEL

from ._classifier_runtime import ClassifierRuntime
from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)

_MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", "/models"))


@register_task()
class RunClassifierTask(Task):
    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        self._model_name: str = str(params["model_name"])
        # model_name is joined onto _MODELS_ROOT to build .onnx/.json paths.
        # It must be a bare model name (single path component) so it can't be used
        # to read files outside the models root via separators or '..' traversal.
        if (
            not self._model_name
            or "/" in self._model_name
            or "\\" in self._model_name
            or self._model_name in (".", "..")
            or Path(self._model_name).is_absolute()
        ):
            raise ValueError(
                f"Invalid model_name {self._model_name!r}: must be a bare model name (no path separators or '..')"
            )
        self._source_uuids: list[str] = list(params.get("source_uuids", []))
        self._confidence_threshold: float = float(params.get("confidence_threshold", 0.7))
        self._batch_size: int = int(params.get("batch_size", 32))
        self._image_source: str = str(params.get("image_source", "filesystem"))

        model_path = _MODELS_ROOT / f"{self._model_name}.onnx"
        config_path = _MODELS_ROOT / f"{self._model_name}.json"

        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")

        with open(config_path) as f:
            self._model_config: dict = json.load(f)

        model_kind = self._model_config.get("model_kind", "image_classifier")
        if model_kind != "image_classifier":
            raise ValueError(
                f"Model {self._model_name!r} has kind {model_kind!r}; RunClassifierTask only supports 'image_classifier'"
            )

        self._label_uuid: Optional[str] = self._model_config.get("label_uuid") or None
        if not self._label_uuid:
            raise ValueError(f"Model config for {self._model_name!r} is missing 'label_uuid'")

        self._input_size: tuple[int, int] = tuple(self._model_config.get("input_size", [224, 224]))  # type: ignore
        self._model_path: Path = model_path

        self._images_processed: int = resume_data.get("images_processed", 0) if resume_data else 0
        self._images_labeled: int = resume_data.get("images_labeled", 0) if resume_data else 0
        self._runtime: Optional[ClassifierRuntime] = None

    async def _run(self) -> dict[str, Any]:
        self.log(f"Loading model {self._model_name} from {self._model_path}")
        self._runtime = ClassifierRuntime(self._model_path, self._model_config)
        await self._runtime.load()
        self.log("Model loaded")

        db_api_client = self._manager.get_db_api_client()
        images_api = ImagesApi(db_api_client)

        # Build list of (uuid, filename) to classify.
        stubs = await self._collect_stubs(db_api_client, images_api)
        # Skip images already processed in a previous run.
        if self._images_processed > 0 and self._images_processed < len(stubs):
            stubs = stubs[self._images_processed :]
            self.log(f"Resuming from image {self._images_processed}")
        total = len(stubs)
        self.log(f"Classifying {total} images in batches of {self._batch_size}")

        await self.heartbeat(force=True)

        # Process in batches.
        for batch_start in range(0, total, self._batch_size):
            if self._cancel_flag.is_set():
                break

            batch = stubs[batch_start : batch_start + self._batch_size]
            tensors, valid_stubs = await self._load_batch(batch, images_api)

            if not tensors:
                self._images_processed += len(batch)
                continue

            # Run inference in a batch (CPU/GPU-bound); image_classifier models
            # always yield exactly one (label_uuid, confidence) pair per image.
            predictions = await self._runtime.predict_batch(tensors)

            # Write image_labels (PRESENT / ABSENT) for each classified image.
            for stub, preds in zip(valid_stubs, predictions):
                confidence = preds[0][1]
                is_present = confidence >= self._confidence_threshold
                status = ImageItemStatus.PRESENT if is_present else ImageItemStatus.ABSENT
                try:
                    await asyncio.to_thread(
                        images_api.set_image_label_status,
                        SetImageLabelPayload(
                            image_path=stub.filename,
                            label_uuid=self._label_uuid,
                            status=status,
                        ),
                    )
                    self._images_labeled += 1
                except Exception as exc:
                    logger.warning("Failed to set image label for %s: %s", stub.filename, exc)

            self._images_processed += len(batch)

            # Update progress and log.
            done_count = self._images_processed
            progress = min(100.0, done_count / max(total, 1) * 100.0)
            self._update_progress(progress)
            if done_count % (self._batch_size * 5) == 0 or done_count >= total:
                self.log(f"Processed {done_count}/{total}, labeled {self._images_labeled}")

            # Persist progress for resume.
            if self._data_req_flag.is_set():
                self._data_req_flag.clear()
                self._resume_data = {
                    "images_processed": self._images_processed,
                    "images_labeled": self._images_labeled,
                }
                self._data_ready_flag.set()

            await self.heartbeat()
            await asyncio.sleep(0)

        self.log(f"Complete: {self._images_processed} processed, {self._images_labeled} labeled")
        return {
            "images_processed": self._images_processed,
            "images_labeled": self._images_labeled,
            "model_name": self._model_name,
        }

    async def _collect_stubs(self, db_api_client: db_client.ApiClient, images_api: ImagesApi) -> list:
        """Return a sorted list of ImageStub objects to classify."""
        self.log("Fetching image list from DB")
        resp = await asyncio.to_thread(images_api.list_image_stubs)
        all_stubs = resp.stubs or []

        if not self._source_uuids:
            return all_stubs

        # Filter to images under the given sources.
        sources_resp = await asyncio.to_thread(SourcesApi(db_api_client).get_sources)
        source_by_uuid = {s.uuid: s for s in (sources_resp.sources or [])}
        path_prefixes: list[str] = []
        for uuid in self._source_uuids:
            src = source_by_uuid.get(uuid)
            if src and src.params.get("path"):
                path_prefixes.append(src.params["path"])

        if not path_prefixes:
            self.log(f"Warning: no valid paths found for source_uuids {self._source_uuids}")
            return all_stubs

        filtered = [s for s in all_stubs if any(s.filename.startswith(p) for p in path_prefixes)]
        self.log(f"Filtered to {len(filtered)} images from {len(path_prefixes)} source(s)")
        return filtered

    async def _load_batch(self, stubs: list, images_api: ImagesApi) -> tuple[list[np.ndarray], list]:
        """Load and preprocess images for *stubs*.

        Returns (tensors, valid_stubs) where invalid images are skipped.
        """
        tensors: list[np.ndarray] = []
        valid_stubs: list = []

        for stub in stubs:
            try:
                img_bytes = await self._load_image_bytes(stub.filename, images_api)
            except Exception as exc:
                logger.warning("Failed to load image %s: %s", stub.filename, exc)
                continue

            tensor = await asyncio.to_thread(_preprocess_image, img_bytes, self._input_size)
            if tensor is None:
                logger.warning("Could not decode image: %s", stub.filename)
                continue

            tensors.append(tensor)
            valid_stubs.append(stub)

        return tensors, valid_stubs

    async def _load_image_bytes(self, filename: str, images_api: ImagesApi) -> bytes:
        # `filename` as stored/returned by the DB is FILES_ROOT-relative (or, in
        # some cases, an absolute path under FILES_ROOT) — it is never valid to
        # open() directly against the task-server process's cwd. Resolve it the
        # same way db_server/manager.py does (get_safe_path handles both forms).
        if self._image_source == "filesystem":
            safe_path = get_safe_path(FILES_ROOT, Path(filename))
            if safe_path is None:
                raise FileNotFoundError(f"Refusing to read path outside FILES_ROOT: {filename!r}")
            return await asyncio.to_thread(_read_file, str(safe_path))
        else:
            resp = await asyncio.to_thread(images_api.get_image_file, filename, True)  # include_binary=True
            if not resp.image_base64:
                raise ValueError(f"No image data returned for {filename}")
            return base64.b64decode(resp.image_base64)

    async def _deinit(self) -> None:
        if self._runtime is not None:
            await self._runtime.close()

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        return {
            "model_name": {
                "type": "str",
                "required": True,
                "help": "Name of the trained model (without extension) in /models/",
            },
            "source_uuids": {
                "type": "list",
                "required": False,
                "default": [],
                "help": "Restrict to images from these source UUIDs (empty = all images)",
            },
            "confidence_threshold": {
                "type": "float",
                "required": False,
                "default": 0.7,
                "help": "Minimum confidence to classify as 'present'",
            },
            "batch_size": {
                "type": "int",
                "required": False,
                "default": 32,
                "help": "Number of images per inference batch",
            },
            "image_source": {
                "type": "str",
                "required": False,
                "default": "filesystem",
                "options": ["filesystem", "http"],
                "help": "How to load images (filesystem = direct file read, http = via db_server)",
            },
        }


def _read_file(filename: str) -> bytes:
    with open(filename, "rb") as f:
        return f.read()
