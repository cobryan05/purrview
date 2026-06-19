"""RunClassifierTask — apply a trained ONNX image classifier to images.

Loads the ONNX model written by BinaryImageClassifier and tags images in the DB
based on confidence. Uses the same FilesystemImageFetcher / HttpImageFetcher
abstraction as the training pipeline.
"""

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import cv2
import db_client
import numpy as np
from db_client.api.images_api import ImagesApi
from db_client.api.sources_api import SourcesApi
from db_client.models.set_image_tag_payload import SetImageTagPayload

from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", "/models"))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def _preprocess_image(img_bytes: bytes, input_size: tuple[int, int]) -> Optional[np.ndarray]:
    """Decode *img_bytes*, resize to *input_size*, and return a (3, H, W) float32 tensor.

    Returns None if the image cannot be decoded.
    """
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size[1], input_size[0]))  # cv2 takes (W, H)
    img = img.astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return img.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)


@register_task()
class RunClassifierTask(Task):
    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        self._model_name: str = str(params["model_name"])
        self._source_uuids: list[str] = list(params.get("source_uuids", []))
        self._present_tag_uuid: Optional[str] = params.get("present_tag_uuid")
        self._absent_tag_uuid: Optional[str] = params.get("absent_tag_uuid")
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

        self._input_size: tuple[int, int] = tuple(self._model_config.get("input_size", [224, 224]))  # type: ignore
        self._model_path: Path = model_path

        self._images_processed: int = resume_data.get("images_processed", 0) if resume_data else 0
        self._images_tagged: int = resume_data.get("images_tagged", 0) if resume_data else 0

    async def _run(self) -> dict[str, Any]:
        import onnxruntime as ort

        self.log(f"Loading model {self._model_name} from {self._model_path}")
        session = await asyncio.to_thread(
            ort.InferenceSession,
            str(self._model_path),
            providers=["CPUExecutionProvider"],
        )
        input_name: str = session.get_inputs()[0].name
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

            # Run inference in a thread (CPU-bound).
            batch_arr = np.stack(tensors)  # (N, 3, H, W)
            raw_outputs = await asyncio.to_thread(session.run, None, {input_name: batch_arr})
            logits: np.ndarray = raw_outputs[0]  # (N, num_classes)
            probs = _softmax(logits, axis=1)

            # "present" class is assumed to be index 1 (matches BinaryImageClassifier: absent=0, present=1).
            present_class_idx = (
                self._model_config.get("class_names", ["absent", "present"]).index("present")
                if "present" in self._model_config.get("class_names", [])
                else 1
            )
            present_probs = probs[:, present_class_idx]

            # Apply tags.
            for i, (stub, confidence) in enumerate(zip(valid_stubs, present_probs)):
                is_present = float(confidence) >= self._confidence_threshold
                tag_uuid = self._present_tag_uuid if is_present else self._absent_tag_uuid
                if tag_uuid:
                    try:
                        await asyncio.to_thread(
                            images_api.set_image_tag,
                            SetImageTagPayload(
                                image_path=stub.filename,
                                tag_uuid=tag_uuid,
                                action="add",
                            ),
                        )
                        self._images_tagged += 1
                    except Exception as exc:
                        logger.warning("Failed to tag %s: %s", stub.filename, exc)

            self._images_processed += len(batch)

            # Update progress and log.
            done_count = self._images_processed
            progress = min(100.0, done_count / max(total, 1) * 100.0)
            self._update_progress(progress)
            if done_count % (self._batch_size * 5) == 0 or done_count >= total:
                self.log(f"Processed {done_count}/{total}, tagged {self._images_tagged}")

            # Persist progress for resume.
            if self._data_req_flag.is_set():
                self._data_req_flag.clear()
                self._resume_data = {
                    "images_processed": self._images_processed,
                    "images_tagged": self._images_tagged,
                }
                self._data_ready_flag.set()

            await self.heartbeat()
            await asyncio.sleep(0)

        self.log(f"Complete: {self._images_processed} processed, {self._images_tagged} tagged")
        result: dict[str, Any] = {
            "images_processed": self._images_processed,
            "images_tagged": self._images_tagged,
            "model_name": self._model_name,
        }
        if self._present_tag_uuid:
            result["actions"] = [
                {
                    "type": "open_review",
                    "label": "See classified images",
                    "payload": {"tag_uuids": [self._present_tag_uuid]},
                }
            ]
        return result

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
        if self._image_source == "filesystem":
            return await asyncio.to_thread(_read_file, filename)
        else:
            resp = await asyncio.to_thread(images_api.get_image_file, filename, True)  # include_binary=True
            if not resp.image_base64:
                raise ValueError(f"No image data returned for {filename}")
            return base64.b64decode(resp.image_base64)

    async def _deinit(self) -> None:
        pass

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
            "present_tag_uuid": {
                "type": "str",
                "required": False,
                "help": "Tag UUID to apply to images classified as 'present'",
            },
            "absent_tag_uuid": {
                "type": "str",
                "required": False,
                "help": "Tag UUID to apply to images classified as 'absent'",
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
