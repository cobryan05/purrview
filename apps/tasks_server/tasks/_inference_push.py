"""Push a locally-landed model's files to inference_server over HTTP.

Every task that lands a model under $MODELS_ROOT for tasks_server's own use — training
completion (`_training_job_runner._save_artifact`), a curated direct-ONNX download
(`ImportModelTask`), a curated download+convert (`ImportCuratedModelTask`) — also needs
inference_server to learn about the same files, since it may not share a filesystem
with this host. This is the one place that builds an artifact tar and posts it to
inference_server's `/api/models/artifact/upload`, so all three do it identically.
"""

from __future__ import annotations

import asyncio
import base64
import io
import tarfile
from typing import Dict, List

from inference_client import ApiClient
from inference_client.api.models_api import ModelsApi
from inference_client.models.import_model_artifact_payload import ImportModelArtifactPayload


def build_artifact_tar(files: Dict[str, bytes]) -> bytes:
    """Pack {filename: bytes} into a tar in memory, ready for the artifact endpoint."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


async def push_artifact_tar_to_inference_server(inference_api_client: ApiClient, tar_bytes: bytes) -> List[str]:
    """POST an artifact tar to inference_server; raises if it's rejected.

    A silently-swallowed failure here would leave inference_server serving stale (or
    no) weights indefinitely with no other signal that anything is wrong, so a
    rejection propagates as a task failure rather than a logged warning.
    """
    models_api = ModelsApi(inference_api_client)
    payload = ImportModelArtifactPayload(artifact_tar_base64=base64.b64encode(tar_bytes).decode("ascii"))
    response = await asyncio.to_thread(models_api.import_model_artifact, payload)
    if response.status != "success":
        raise RuntimeError(f"inference server rejected model artifact: {response.message}")
    return response.files
