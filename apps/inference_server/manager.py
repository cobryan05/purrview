"""Manager models on Inference Server"""

import asyncio
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from apps import APPS, CACHE_DIR
from apps.helpers.imageUtils import annotate_image
from apps.helpers.inferenceProviders.inferenceProvider import (
    InferenceProvider,
    InferenceResult,
)
from apps.helpers.metadataUtils import (
    VALID_MODEL_DEVICES,
    get_model_metadata,
    save_model_json_metadata,
    validate_bare_model_name,
    write_onnx_with_sidecar,
)
from apps.helpers.modelFormat import (
    ModelFormat,
    detect_model_format,
    is_conversion_candidate,
)
from apps.helpers.modelStaging import STAGING_SUBDIR
from apps.helpers.serviceStatus import fmt_duration, row, section
from apps.helpers.yoloUtils import load_yolo_onnx

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

# Subdirectory (under CACHE_DIR) where raw uploaded model files are staged pending
# an asynchronous conversion task (e.g. pytorch -> onnx) run by the tasks_server.
# The subdirectory name + staging-id validation are centralized in
# apps.helpers.modelStaging (also used directly by ConvertModelTask); this module
# keeps resolving the directory from its OWN module-level CACHE_DIR (rather than
# calling into the helper's staging_dir()) so tests that monkeypatch
# apps.inference_server.manager.CACHE_DIR keep working unmodified.
_STAGING_SUBDIR = STAGING_SUBDIR


@dataclass
class Pin:
    """Represents a pin for a model with its unique ID and expiration time."""

    pin_id: str
    expiry: float
    timeout: float


@dataclass
class InferenceModel:
    """Represents a model that is pinned in memory."""

    model_name: str
    path: str
    inference: Optional[InferenceProvider] = None
    pins: Dict[str, Pin] = field(default_factory=dict)  # {pin_id: Pin}
    # mtime_ns of the .onnx this session was built from, so pin_model can tell a
    # warm session apart from a stale one after the file is replaced underneath
    # it. None only if the stat failed at load time.
    weights_mtime_ns: Optional[int] = None


class Manager:
    """Manages inference models, including loading, pinning, and unpinning."""

    POLLING_INTERVAL: float = 5.0  # Interval in seconds for periodic tasks
    MODEL_PIN_TIME: float = 10.0 * 60  # Default timeout in seconds before unloading model from GPU

    # Pin id used by recognize() when the caller holds no pin of its own (e.g. the
    # /classify endpoint, which has no pin/unpin handshake). A single shared id per
    # model — rather than a fresh uuid per call — means back-to-back calls refresh
    # one pin instead of accumulating thousands of them, and the model stays loaded
    # between calls. Without this, every pinless recognize() built and immediately
    # destroyed an ort.InferenceSession; under CUDA that churns a GPU context plus
    # host-side arenas per call and RSS grows monotonically (observed: 1500+ session
    # loads in 33 minutes and ~700 MB of unreclaimed heap). The model still unloads
    # on the normal expiry path once MODEL_PIN_TIME passes with no further calls.
    IMPLICIT_PIN_ID: str = "__implicit_keepwarm__"

    def __init__(self, models_path: str):
        """
        Initialize the Manager with a models directory path.

        Args:
            models_path (str): Path to the directory containing model files.
        """
        self._task: Optional[asyncio.Task] = None  # Background task for periodic operations
        self._models_path: str = models_path  # Store the models directory path
        self._avail_models: Dict[str, str] = {}  # {basename (no .onnx): full_path}
        self._pinned_models: Dict[str, InferenceModel] = {}  # {model_name: PinnedModel}
        self._loading_models: Dict[str, asyncio.Future] = {}  # in-progress loads; awaited by concurrent pin requests
        self._model_locks: Dict[str, asyncio.Lock] = {}  # serializes pin_model/delete_model per model_name
        # model_path -> (sidecar mtime_ns, class name -> label_uuid). See
        # get_models_classes(); keyed on mtime so it self-invalidates on edit.
        self._class_map_cache: Dict[str, Tuple[int, Dict[str, Optional[str]]]] = {}
        self._config = {"service": APPS.INFERENCE_SERVER, "models_path": models_path}

    async def get_server_config(self) -> Dict[str, Any]:
        return self._config.copy()

    def get_models_path(self) -> str:
        return self._models_path

    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Read the companion JSON for *model_name*; returns None if missing."""
        import json
        from pathlib import Path

        config_path = Path(self._models_path) / f"{model_name}.json"
        if not config_path.exists():
            return None
        with open(config_path) as f:
            return json.load(f)

    async def list_models(self, pattern: Optional[str] = "*.onnx") -> List[str]:
        """
        List all files in the models directory, optionally filtered by a glob pattern.
        Updates the available models cache.

        Args:
            pattern (Optional[str]): A glob pattern to filter files (default: "*.onnx").

        Returns:
            List[str]: A list of filenames in the models directory.
        """
        if not self._models_path:
            logger.warning("Models path is not set.")
            return []

        try:
            search_path = os.path.join(self._models_path, pattern)
            model_files = [f for f in glob(search_path) if os.path.isfile(f)]

            # Update the available models cache
            self._avail_models = {os.path.splitext(os.path.basename(f))[0]: f for f in model_files}
            return list(self._avail_models.keys())
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []

    async def get_models_classes(self, model_names: List[str]) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Get the classes for a specific model.

        Args:
            model_name (str): The name of the model to get classes for.

        Returns:
            Dict[str, str]: A dictionary mapping class names to class IDs.
        """

        model_paths = {name: self._avail_models[name] for name in model_names if name in self._avail_models}

        # Create metadata but don't entirely fail on get_model_metadata failrule just leave that one out
        ret = {}
        for model_name, model_path in model_paths.items():
            try:
                # recognize() calls this for every image that produced a detection,
                # so an uncached call means a sidecar stat+read+json.parse per image
                # on the event loop during a bulk run. Cache on the sidecar's
                # mtime_ns so an edit (from this process or any other) invalidates
                # itself without callers having to remember to flush.
                cached = self._class_map_cache.get(model_path)
                sidecar_mtime = await self._sidecar_mtime_ns(model_path)
                if cached is not None and sidecar_mtime is not None and cached[0] == sidecar_mtime:
                    ret[model_name] = cached[1]
                    continue

                metadata = get_model_metadata(model_path)
                model_classes = metadata.get("classes", [])
                class_map = metadata.get("class_map", {})
                # Models trained in-app already know their class -> label
                # mapping (written positionally into the sidecar by the
                # trainer); use it to pre-fill classes that don't yet have
                # an explicit class_map entry, instead of making the user
                # re-enter an assignment we already have. Evaluated per
                # class so an explicit clear on one class (which persists
                # as class_map[name] = None) doesn't suppress the
                # positional fallback for every other class.
                label_uuids = metadata.get("label_uuids") or []
                positional_mappings = (
                    dict(zip(model_classes, label_uuids)) if len(label_uuids) == len(model_classes) else {}
                )
                class_uuids: Dict[str, Optional[str]] = {
                    name: class_map[name] if name in class_map else positional_mappings.get(name)
                    for name in model_classes
                }
                ret[model_name] = class_uuids
                if sidecar_mtime is not None:
                    self._class_map_cache[model_path] = (sidecar_mtime, class_uuids)
            except FileNotFoundError:
                continue
        return ret

    @staticmethod
    async def _sidecar_mtime_ns(model_path: str) -> Optional[int]:
        """mtime of a model's JSON sidecar, or None if it isn't there yet.

        None disables caching for that model rather than caching a miss: with no
        sidecar, get_model_metadata() derives metadata from the ONNX file and
        *writes* the sidecar, so the next call has a real mtime to key on.

        Runs os.stat() in a thread — get_models_classes() awaits this once per
        detection during a bulk inference run, so a blocking stat() here would
        stall the event loop on every other in-flight request.
        """
        try:
            return await asyncio.to_thread(lambda: os.stat(Path(model_path).with_suffix(".json")).st_mtime_ns)
        except OSError:
            return None

    @staticmethod
    async def _weights_mtime_ns(model_path: str) -> Optional[int]:
        """mtime of the model's .onnx itself, or None if it can't be stat'ed.

        None means "can't tell", and pin_model deliberately treats that as "keep
        serving what's loaded": a file that has gone missing (unlinked, or a
        redeploy mid-write) is no reason to tear down a working session.

        Threaded for the same reason as _sidecar_mtime_ns — pin_model is on the
        path of every pinless recognize().
        """
        try:
            return await asyncio.to_thread(lambda: os.stat(model_path).st_mtime_ns)
        except OSError:
            return None

    async def set_model_class_label_uuid(self, model_name: str, model_class: str, label_uuid: Optional[str]) -> bool:
        """
        Associate a model's class label with a label uuid

        Args:
            model_name (str): The name of the model to set the association on
            model_class (str): The class label to associate with the UUID
            label_uuid (Optional[str]): The UUID of the label to associate, or to clear if None

        Returns:
            bool: True if the association was successful, False otherwise.
        """
        if model_name in self._avail_models:
            model_path = self._avail_models[model_name]
        else:
            raise FileNotFoundError(f"Model '{model_name}' not found in available models")

        try:
            metadata = get_model_metadata(model_path)
        except FileNotFoundError:
            metadata = {}
        if model_class not in metadata.get("classes", {}):
            raise ValueError(f"Class '{model_class}' not found in metadata for model '{model_name}'")
        class_map = metadata.get("class_map", {})
        if label_uuid is None and model_class in class_map:
            del class_map[model_class]
        else:
            class_map[model_class] = label_uuid
        metadata["class_map"] = class_map
        save_model_json_metadata(model_path, metadata=metadata)
        return True

    async def set_model_device(self, model_name: str, device: str) -> bool:
        """
        Set a model's device preference (auto/gpu/cpu), persisted to its sidecar JSON.

        Args:
            model_name (str): The name of the model to set the preference on
            device (str): One of "auto", "gpu", "cpu"

        Returns:
            bool: True if the preference was persisted.
        """
        if device not in VALID_MODEL_DEVICES:
            raise ValueError(f"Invalid device {device!r}; must be one of {VALID_MODEL_DEVICES}")

        if model_name in self._avail_models:
            model_path = self._avail_models[model_name]
        else:
            raise FileNotFoundError(f"Model '{model_name}' not found in available models")

        try:
            metadata = get_model_metadata(model_path)
        except FileNotFoundError:
            metadata = {}
        metadata["device"] = device
        save_model_json_metadata(model_path, metadata=metadata)
        return True

    async def import_model_upload(self, model_name: str, onnx_bytes: bytes, overwrite: bool = False) -> Dict[str, Any]:
        """Write an uploaded ONNX file to the models dir and auto-generate its sidecar
        (classes parsed from the ONNX itself).

        Args:
            model_name (str): Bare model name (no extension, no path separators).
            onnx_bytes (bytes): Raw ONNX file content.
            overwrite (bool): If True, replace an existing model + sidecar. If False and
                the model already exists, raise FileExistsError.

        Returns:
            Dict[str, Any]: {"model_name": ..., "classes": [...]}
        """
        validate_bare_model_name(model_name)

        model_path = Path(self._models_path) / f"{model_name}.onnx"

        if model_path.exists() and not overwrite:
            raise FileExistsError(f"Model '{model_name}' already exists; pass overwrite=True to replace it")

        metadata = await asyncio.to_thread(write_onnx_with_sidecar, str(model_path), onnx_bytes, overwrite=overwrite)

        await self.list_models()

        return {"model_name": model_name, "classes": metadata.get("classes", [])}

    def _get_model_lock(self, model_name: str) -> asyncio.Lock:
        """Lock serializing pin_model/delete_model for a given model_name, so a
        delete can't run while a load is in flight (and vice versa) and see the
        model as neither pinned nor fully absent. Created lazily per model_name."""
        lock = self._model_locks.get(model_name)
        if lock is None:
            lock = asyncio.Lock()
            self._model_locks[model_name] = lock
        return lock

    async def delete_model(self, model_name: str) -> None:
        """Delete a model's ONNX weights and sidecar JSON from the models dir.

        Args:
            model_name (str): Bare model name (no extension) to delete.

        Raises:
            FileNotFoundError: If the model is not in the available-models list.
            ValueError: If the model is currently pinned (in use) and can't be
                safely removed.
        """
        if model_name not in self._avail_models:
            raise FileNotFoundError(f"Model '{model_name}' not found in available models")

        # Shares a lock with pin_model so this can't interleave with a concurrent
        # load: without it, delete_model could see the model as "not yet pinned"
        # mid-load, delete its files, and then have the in-flight pin_model call
        # finish anyway and re-insert a ghost entry into _pinned_models.
        async with self._get_model_lock(model_name):
            if model_name not in self._avail_models:
                raise FileNotFoundError(f"Model '{model_name}' not found in available models")

            pinned = self._pinned_models.get(model_name)
            if pinned is not None:
                # The implicit keep-warm pin is not a caller reservation — it only means the
                # model was used recently and hasn't aged out yet. Drop it and continue;
                # only a real caller-held pin blocks deletion.
                if set(pinned.pins) - {self.IMPLICIT_PIN_ID}:
                    raise ValueError(f"Model '{model_name}' is currently pinned/in use and cannot be deleted")
                del self._pinned_models[model_name]
                logger.info(f"Model '{model_name}' unloaded from memory to allow deletion.")

            model_path = Path(self._avail_models[model_name])
            sidecar_path = model_path.with_suffix(".json")

            await asyncio.to_thread(model_path.unlink, True)
            await asyncio.to_thread(sidecar_path.unlink, True)

            del self._avail_models[model_name]

        self._model_locks.pop(model_name, None)

    def detect_uploaded_model(self, filename: str, header: bytes) -> Dict[str, Any]:
        """Detect the format of an about-to-be-uploaded model file from its filename and
        leading bytes (torch-free — see apps/helpers/modelFormat.py). Pure/sync, no I/O.

        Args:
            filename (str): The original filename of the uploaded file.
            header (bytes): The first chunk of the file's bytes (used for magic-byte sniffing).

        Returns:
            Dict[str, Any]: {"format": ..., "is_onnx": ..., "conversion_supported": ...}
        """
        fmt = detect_model_format(filename, header)
        return {
            "format": fmt,
            "is_onnx": fmt == ModelFormat.ONNX,
            "conversion_supported": is_conversion_candidate(fmt),
        }

    def _staging_dir(self) -> Path:
        """Path to the shared staging directory (under CACHE_DIR) used to hand raw
        uploaded model bytes off to a later conversion task run by another service
        (same container/filesystem). Computed only — no filesystem side effects."""
        return CACHE_DIR / _STAGING_SUBDIR

    async def stage_source_model(self, data: bytes) -> str:
        """Write raw uploaded model bytes to the shared staging dir under a fresh id.

        Args:
            data (bytes): Raw file content to stage.

        Returns:
            str: The generated staging id (bare hex token, no extension); the caller
                (a conversion task) can resolve it back to a path via `staged_model_path`.
        """
        if not data:
            raise ValueError("Cannot stage empty model data")

        staging_id = uuid.uuid4().hex
        staging_dir = self._staging_dir()
        staging_path = staging_dir / f"{staging_id}.bin"

        await asyncio.to_thread(staging_dir.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(staging_path.write_bytes, data)

        return staging_id

    def staged_model_path(self, staging_id: str) -> Path:
        """Resolve a staging id (as returned by `stage_source_model`) to its file path.

        Args:
            staging_id (str): Bare staging id — must not contain path separators or '..'.

        Returns:
            Path: The path to the staged file (may not exist).
        """
        validate_bare_model_name(staging_id)
        return self._staging_dir() / f"{staging_id}.bin"

    def delete_staged_model(self, staging_id: str) -> None:
        """Best-effort removal of a staged model file. The conversion task owns cleanup
        after a successful convert; this is provided for defensive/manual cleanup."""
        path = self.staged_model_path(staging_id)
        path.unlink(missing_ok=True)

    async def _reload_if_weights_changed(self, model_name: str, model_path: str) -> None:
        """Rebuild a loaded session if its .onnx has been rewritten under it.

        Retraining lands new weights over $MODELS_ROOT/<name>.onnx from a different
        process entirely (tasks_server's `_training_job_runner._save_artifact`), so
        nothing here is told the file changed. Without this check a warm session
        outlives the file it was built from and keeps answering from the previous
        model's weights until its pin happens to idle out — a retrained detector
        scored 0.0006 on an image its new weights scored 0.94 on. Compared on
        mtime_ns, matching how _class_map_cache invalidates sidecars.

        No-ops if the model isn't loaded. Callers must hold the model's lock, so
        this can't interleave with a concurrent load or delete.
        """
        existing = self._pinned_models.get(model_name)
        if existing is None:
            return

        # Stat before loading, never after: a write landing *during* the load
        # would otherwise be recorded as already-loaded and stay stale. Re-reading
        # once too often is harmless; missing a write is the bug this prevents.
        current_mtime = await self._weights_mtime_ns(model_path)
        if current_mtime is None or current_mtime == existing.weights_mtime_ns:
            return

        logger.info(
            f"Model '{model_name}' changed on disk (mtime {existing.weights_mtime_ns} -> "
            f"{current_mtime}); reloading the in-memory session."
        )
        try:
            # Mutated in place rather than replaced so `pins` survives: dropping the
            # entry would invalidate every caller's pin id and make their later
            # unpin_model raise.
            existing.inference = await asyncio.to_thread(load_yolo_onnx, model_path)
            existing.path = model_path
        except Exception:
            # _save_artifact lands the new weights with a plain write_bytes(), so
            # there is a window where the file is a truncated ONNX. Keep answering
            # from the session we already have rather than failing the request
            # outright — degraded (stale) beats down. The mtime is still recorded so
            # a permanently broken file doesn't re-attempt a ~30MB load on every
            # call; the completing write bumps mtime again, which is what makes the
            # torn-write case self-heal on a later call.
            logger.warning(
                f"Model '{model_name}' changed on disk but the new file failed to load; "
                f"continuing with the previously loaded session.",
                exc_info=True,
            )
        existing.weights_mtime_ns = current_mtime

    async def pin_model(self, model_name: str, timeout: float = MODEL_PIN_TIME, pin_id: Optional[str] = None) -> str:
        """
        Pin a model into memory, keeping it loaded until unpinned.

        Args:
            model_name (str): Name of the model to pin.
            timeout (float): Timeout in seconds for the pin.
            pin_id (Optional[str]): Reuse a specific pin id instead of generating a
                fresh one. Passing an id that is already pinned refreshes its expiry
                rather than adding a second pin — used for IMPLICIT_PIN_ID so pinless
                callers share one long-lived pin per model.

        Returns:
            str: A unique pin ID that can be used to unpin the model.
        """
        # Determine the model path
        if model_name in self._avail_models:
            model_path = self._avail_models[model_name]
        else:
            raise FileNotFoundError(f"Model '{model_name}' not found in available models")

        # Shares a lock with delete_model — see _get_model_lock — so a concurrent
        # delete can't run mid-load and leave a ghost entry once this finishes.
        async with self._get_model_lock(model_name):
            # If a concurrent pin_model call is already loading this model, wait for it.
            if model_name in self._loading_models:
                await self._loading_models[model_name]

            if model_name not in self._avail_models:
                raise FileNotFoundError(f"Model '{model_name}' not found in available models")

            await self._reload_if_weights_changed(model_name, model_path)

            if model_name not in self._pinned_models:
                load_future: asyncio.Future = asyncio.get_running_loop().create_future()
                self._loading_models[model_name] = load_future
                try:
                    mtime_at_load = await self._weights_mtime_ns(model_path)
                    new_model = InferenceModel(model_name=model_name, path=model_path)
                    new_model.inference = await asyncio.to_thread(load_yolo_onnx, model_path)
                    new_model.weights_mtime_ns = mtime_at_load
                    self._pinned_models[model_name] = new_model
                    logger.info(f"Model '{model_name}' loaded into memory.")
                    load_future.set_result(None)
                except Exception as exc:
                    load_future.set_exception(exc)
                    raise
                finally:
                    self._loading_models.pop(model_name, None)

            pin_id = pin_id or str(uuid.uuid4())
            expiry = time.time() + timeout
            self._pinned_models[model_name].pins[pin_id] = Pin(pin_id=pin_id, expiry=expiry, timeout=timeout)
            logger.debug(f"Model '{model_name}' pinned with ID '{pin_id}' and timeout {timeout}s.")

            return pin_id

    async def unpin_model(self, model_name: str, pin_id: str) -> None:
        """
        Unpin a model using the provided pin ID.

        Args:
            model_name (str): Name of the model to unpin.
            pin_id (str): The unique pin ID returned by `pin_model`.

        Raises:
            ValueError: If the model or pin ID is not found.
        """
        if model_name not in self._pinned_models or pin_id not in self._pinned_models[model_name].pins:
            raise ValueError(f"Pin ID '{pin_id}' for model '{model_name}' not found.")

        # Remove the pin
        del self._pinned_models[model_name].pins[pin_id]
        logger.debug(f"Model '{model_name}' unpinned with ID '{pin_id}'.")

        # If no more pins exist for the model, unload it from memory
        if not self._pinned_models[model_name].pins:
            del self._pinned_models[model_name]
            logger.info(f"Model '{model_name}' unloaded from memory.")

    async def recognize(
        self,
        model_name: str,
        image: np.ndarray,
        conf_thresh: float,
        return_annotated_img: bool = False,
        pin_id: Optional[str] = None,
        **kwargs,
    ) -> InferenceResult:
        """
        Recognize objects in an image using a specified model.

        Args:
            model_name (str): Name of the model to use for recognition.
            image (np.ndarray): The input image in BGR format.
            conf_thresh (float): Confidence threshold for detections.
            pin_id (str): pin_id to refresh if using a pinned model
            return_annotated_img (bool): Whether to return the annotated image.
            **kwargs: Additional parameters to pass to the inference backend.

        Returns:
            Tuple[InferenceResult, Optional[np.ndarray]]:
                - An InferenceResult with information from the recognition
        """
        if model_name not in self._avail_models:
            raise ValueError(f"Model '{model_name}' is not available. Please check the model name.")

        # A caller holding its own valid pin just gets its expiry refreshed, so the pin
        # can't lapse mid-inference. Everyone else falls back to the shared implicit
        # pin, which keeps the session loaded across calls.
        pinned = self._pinned_models.get(model_name)
        caller_pin_valid = pin_id is not None and pinned is not None and pin_id in pinned.pins
        if caller_pin_valid:
            pinned.pins[pin_id].expiry = time.time() + pinned.pins[pin_id].timeout
            # A caller holding its own pin never re-enters pin_model, so the staleness
            # check there would never fire for it: a long-running detection task pins
            # once and then loops on recognize(), and used to keep serving pre-retrain
            # weights until it was suspended and resumed (dropping the pin). Same
            # per-call stat() the pinless branch already pays via pin_model.
            async with self._get_model_lock(model_name):
                # Re-check under the lock, as pin_model and delete_model both do.
                # The guard above ran before the lock was taken, and the wait for
                # it is exactly when a concurrent delete_model — or a list_models
                # scan, which replaces _avail_models wholesale — can drop the key.
                # Indexing it blind turns that into a KeyError out of the middle
                # of inference instead of the caller-facing error below.
                model_path = self._avail_models.get(model_name)
                if model_path is None:
                    raise ValueError(f"Model '{model_name}' is not available. Please check the model name.")
                await self._reload_if_weights_changed(model_name, model_path)
        else:
            # Pinless caller: load once and keep the model warm under the shared
            # implicit pin (refreshing its expiry), rather than tearing the session
            # down in a finally: and rebuilding it on the very next call. See
            # IMPLICIT_PIN_ID. _cleanup_expired_pins still unloads it once the model
            # has been idle for MODEL_PIN_TIME.
            await self.pin_model(model_name, pin_id=self.IMPLICIT_PIN_ID)

        model = self._pinned_models.get(model_name)
        if not model or not model.inference:
            raise ValueError(f"Model '{model_name}' is not properly loaded or initialized.")

        t0 = time.perf_counter()
        inference_result: InferenceResult = await model.inference.processImage(image, conf_thresh=conf_thresh, **kwargs)
        t_infer = time.perf_counter()

        if return_annotated_img:
            inference_result.annotated_image = annotate_image(image, inference_result.detections)

        if len(inference_result.detections) > 0:
            t_cls = time.perf_counter()
            class_uuid_maps = await self.get_models_classes([model_name])
            class_uuid_map = class_uuid_maps.get(model_name, {})
            for det in inference_result.detections:
                # Runner-up classes resolve through the SAME sidecar map as the
                # winner, so a caller comparing "cat vs dog" is comparing label
                # UUIDs, not a model's private class ordering. Done here rather
                # than in the provider because the provider has no DB/sidecar
                # access — it only knows class names.
                for alt in det.alternates or ():
                    if alt.class_str:
                        alt.label_uuid = class_uuid_map.get(alt.class_str, alt.label_uuid)
                if not det.bbox.class_str:
                    continue
                det.bbox.label_uuid = class_uuid_map.get(det.bbox.class_str, det.bbox.label_uuid)
            logger.debug(
                f"recognize '{model_name}': inference={1000*(t_infer-t0):.1f}ms "
                f"class_lookup={1000*(time.perf_counter()-t_cls):.1f}ms "
                f"({len(inference_result.detections)} dets)"
            )
        else:
            logger.debug(f"recognize '{model_name}': inference={1000*(t_infer-t0):.1f}ms (no detections)")

        return inference_result

    def _cleanup_expired_pins(self):
        """
        Periodically clean up expired pins.
        """
        current_time = time.time()
        for model_name, pinned_model in list(self._pinned_models.items()):
            for pin_id, pin in list(pinned_model.pins.items()):
                if pin.expiry < current_time:
                    logger.info(f"Pin ID '{pin_id}' for model '{model_name}' expired.")
                    del pinned_model.pins[pin_id]

            # If no more pins exist for the model, unload it
            if not pinned_model.pins:
                del self._pinned_models[model_name]
                logger.info(f"Model '{model_name}' unloaded from memory due to expired pins.")

    def start(self):
        """Start the periodic worker task, should be called from the event loop to run on"""
        if self._task and not self._task.done():
            self._task.cancel()  # Cancel the existing task if it's still running
        self._task = asyncio.get_running_loop().create_task(self._worker_task())  # Schedule a new task

    async def _init(self):
        """Initialization that should run on event loop"""
        await self.list_models()

    async def _worker_task(self):
        """
        Periodic worker task that runs at regular intervals.
        """
        try:
            await self._init()
            while True:
                self._cleanup_expired_pins()
                await asyncio.sleep(Manager.POLLING_INTERVAL)  # Wait for the polling interval
        except asyncio.CancelledError:
            logger.info("Periodic task was cancelled.")  # Handle task cancellation
        finally:
            logger.info("Periodic task cleanup.")  # Perform cleanup when the task is stopped

    def get_status(self) -> dict:
        """Return service health and diagnostic info.

        ``details`` lists the models actually resident in memory with their pin
        counts — the question this panel gets asked is "what is holding the GPU
        right now", which the loaded/available counters alone can't answer.
        """
        loaded = len(self._pinned_models)
        available = len(self._avail_models)
        total_pins = sum(len(m.pins) for m in self._pinned_models.values())

        summary = f"loaded: {loaded}/{available}"
        if total_pins:
            summary += f"  ·  {total_pins} pin{'' if total_pins == 1 else 's'}"

        now = time.time()
        rows = []
        for name, model in sorted(self._pinned_models.items()):
            pins = model.pins
            explicit = [p for pid, p in pins.items() if pid != Manager.IMPLICIT_PIN_ID]
            parts = [f"{len(pins)} pin{'' if len(pins) == 1 else 's'}"]
            if len(explicit) != len(pins):
                # An implicit pin means nobody is holding this model — it is only
                # kept warm between pinless recognize() calls and will drop on
                # expiry. Worth distinguishing from a held model that won't.
                parts.append(f"{len(explicit)} held + keep-warm")
            if pins:
                # Soonest expiry: the model unloads when its *last* pin goes, but
                # the next thing to happen is the earliest one expiring.
                parts.append(f"next expiry in {fmt_duration(min(p.expiry for p in pins.values()) - now)}")
            if model.inference is None:
                parts.append("session not built")
            rows.append(row(name, "  ·  ".join(parts)))

        return {
            "service": "inference_server",
            "status": "ok",
            "models": {
                "loaded": loaded,
                "available": available,
                "pins": total_pins,
            },
            "summary": summary,
            "details": [
                section(
                    f"Loaded models ({loaded})",
                    rows,
                    empty="No models loaded — nothing is holding the GPU",
                ),
            ],
        }
