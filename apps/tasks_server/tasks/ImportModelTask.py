"""ImportModelTask — download a curated pretrained ONNX base model in the
background and land it (+ auto-generated sidecar) in the models directory.

Curated download is a network side-effect, unlike a plain file upload (which
is synchronous — see inference_server.manager.import_model_upload), so it runs
as a background task mirroring the download-and-save pattern already used by
`_training_job_runner.py::_save_artifact` (httpx.AsyncClient GET, self.log(...)
progress messages, write bytes to MODELS_ROOT).
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from apps.helpers.curatedModels import get_curated_model
from apps.helpers.metadataUtils import (
    DEFAULT_MODEL_DEVICE,
    get_model_metadata,
    save_model_json_metadata,
    validate_bare_model_name,
    validate_onnx_bytes,
)

from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

_MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", "/models"))
_DOWNLOAD_TIMEOUT = 300.0  # seconds; matches _training_job_runner._save_artifact's pattern


@register_task()
class ImportModelTask(Task):
    async def _init(self, params: Dict[str, Any], resume_data: Optional[Dict[str, Any]]) -> None:
        # Task _init params are raw, unvalidated JSON — validate everything here.
        curated_key = str(params.get("curated_key") or "")
        try:
            self._curated_entry = get_curated_model(curated_key)
        except KeyError as exc:
            raise ValueError(f"Unknown curated_key {curated_key!r}: {exc}") from exc

        self._model_name: str = str(params.get("model_name") or "")
        validate_bare_model_name(self._model_name)

        self._overwrite: bool = bool(params.get("overwrite", False))
        self._onnx_url: str = self._curated_entry["onnx_url"]
        self._model_family: str = self._curated_entry["model_family"]
        self._display_name: str = self._curated_entry["display_name"]

        self._model_path: Path = _MODELS_ROOT / f"{self._model_name}.onnx"
        self._sidecar_path: Path = self._model_path.with_suffix(".json")

        # Fail fast (before downloading anything) if the target already exists.
        if self._model_path.exists() and not self._overwrite:
            raise FileExistsError(f"Model '{self._model_name}' already exists; pass overwrite=True to replace it")

    async def _run(self) -> Dict[str, Any]:
        if self._cancel_flag.is_set():
            return {"cancelled": True}

        self.log(f"Downloading {self._display_name} from {self._onnx_url}...")

        # A single GET is not chunked, so there's no mid-download cancellation
        # checkpoint — same trade-off _training_job_runner._save_artifact accepts.
        # Abandoning an in-flight request on cancel (rather than streaming+checking
        # _cancel_flag per-chunk) is acceptable here: this is a one-shot cold-start
        # import, not a long-running batch loop.
        async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(self._onnx_url)
            resp.raise_for_status()
            onnx_bytes = resp.content

        self.log(f"Downloaded {len(onnx_bytes)} bytes; validating...")
        # Validate before touching disk: a corrupted/truncated download must not leave
        # an unparseable orphan file, and (on overwrite) must not clobber a previously
        # working model.
        await asyncio.to_thread(validate_onnx_bytes, onnx_bytes)

        self.log(f"Writing to {self._model_path}")
        self._model_path.parent.mkdir(parents=True, exist_ok=True)

        if self._overwrite and self._sidecar_path.exists():
            # Drop the stale sidecar first so get_model_metadata regenerates it fresh
            # from the new ONNX's own `names` metadata instead of merging in the old
            # model's classes/class_map.
            self._sidecar_path.unlink()

        await asyncio.to_thread(self._model_path.write_bytes, onnx_bytes)

        metadata = await asyncio.to_thread(get_model_metadata, str(self._model_path))
        metadata["model_family"] = self._model_family
        metadata.setdefault("device", DEFAULT_MODEL_DEVICE)
        metadata.setdefault("class_map", {})
        await asyncio.to_thread(save_model_json_metadata, str(self._model_path), metadata)

        self.log("Import complete")
        return {"model_name": self._model_name, "classes": metadata.get("classes", [])}

    async def _deinit(self) -> None:
        # Nothing to clean up — no pinned models, no open sessions.
        pass

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        from apps.helpers.curatedModels import CURATED_BASE_MODELS

        return {
            "curated_key": {
                "type": "select",
                "required": True,
                "label": "Curated Model",
                "options": [{"value": e["key"], "label": e["display_name"]} for e in CURATED_BASE_MODELS],
                "help": "Which curated pretrained base model to download",
            },
            "model_name": {
                "type": "str",
                "required": True,
                "label": "Model Name",
                "help": "Target filename to save as (without extension) in /models/",
            },
            "overwrite": {
                "type": "boolean",
                "required": False,
                "default": False,
                "label": "Overwrite",
                "help": "Replace an existing model + sidecar of the same name",
            },
        }
