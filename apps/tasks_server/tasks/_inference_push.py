"""Push a locally-landed model's files to inference_server over HTTP.

Every task that lands a model under $MODELS_ROOT for tasks_server's own use — training
completion (`_training_job_runner._save_artifact`), a curated direct-ONNX download
(`ImportModelTask`), a curated download+convert (`ImportCuratedModelTask`) — also needs
inference_server to learn about the same files, since it may not share a filesystem
with this host. This is the one place that builds an artifact tar and posts it to
inference_server's `/api/models/artifact/upload`, so all three do it identically.

It is also the one place that decides whether that push is needed at all — see
`inference_server_is_colocated`.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import tarfile
from typing import Dict, List, Optional

from inference_client import ApiClient
from inference_client.api.models_api import ModelsApi
from inference_client.models.import_model_artifact_payload import ImportModelArtifactPayload

from apps.helpers.serverUrls import url_is_colocated

logger = logging.getLogger(__name__)


def inference_server_is_colocated(inference_api_client: ApiClient) -> bool:
    """Is inference_server the same process's host as ours, so the local write is
    already its copy?

    In the default `docker-compose.yaml` deployment every service is one container:
    tasks_server and inference_server run side by side under the same supervisord with
    the same `MODELS_ROOT` bind mount. There, `_save_artifact` writing the artifact to
    $MODELS_ROOT *is* inference_server's copy, and posting the same bytes back over
    HTTP re-writes the files it already has — a base64-inflated round trip of a
    multi-hundred-MB model held whole in memory on both ends, whose only other effect
    is that a transient HTTP hiccup can fail a completed multi-hour training run.

    The gate is deliberately narrow: only a loopback name or this host's own hostname
    counts. Anything else — a sibling container on the compose network, a remote GPU
    box, a Settings → Servers override — pushes, because a shared filesystem cannot be
    assumed and a missed push leaves inference_server on stale weights.

    The one case this gets wrong is inference_server running as a *separate* container
    on host networking, where the loopback name reaches a different filesystem. That
    is not a topology this deployment supports (see the docstring on
    `Manager.get_relay_api_client` for the split-container rules); a remote
    inference_server is addressed by its own host.
    """
    configured = getattr(getattr(inference_api_client, "configuration", None), "host", None)
    return url_is_colocated(configured)


async def refresh_inference_server_models(inference_api_client: ApiClient) -> None:
    """Make a co-located inference_server notice files written under its own models dir.

    Its available-models map is only rebuilt by `list_models()`, which the artifact
    upload used to call as its last step. When the push is skipped because the two
    services share a filesystem, nothing rescans — and a model whose *name* is new
    (a config's first training run) stays unresolvable, so the first inference on it
    fails with "Model 'X' is not available" until some unrelated caller happens to
    list. One cheap GET closes that gap.

    Best-effort: a failure here costs a rescan that the next models listing will do
    anyway, and must not fail a completed training run.
    """
    try:
        models_api = ModelsApi(inference_api_client)
        await asyncio.to_thread(models_api.list_models_api_api_models_get)
    except Exception:
        logger.warning("Could not refresh the co-located inference server's model list", exc_info=True)


def build_artifact_tar(files: Dict[str, bytes]) -> bytes:
    """Pack {filename: bytes} into a tar in memory, ready for the artifact endpoint."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


async def push_artifact_tar_to_inference_server(
    inference_api_client: ApiClient, tar_bytes: bytes, *, versions_keep: Optional[int] = None
) -> List[str]:
    """POST an artifact tar to inference_server; raises if it's rejected.

    A silently-swallowed failure here would leave inference_server serving stale (or
    no) weights indefinitely with no other signal that anything is wrong, so a
    rejection propagates as a task failure rather than a logged warning.

    ``versions_keep`` travels with the artifact because a remote inference server
    cannot read db_server's settings for itself; sending it is what keeps the two
    copies of a model on the same retention policy.
    """
    models_api = ModelsApi(inference_api_client)
    payload = ImportModelArtifactPayload(
        artifact_tar_base64=base64.b64encode(tar_bytes).decode("ascii"),
        versions_keep=versions_keep,
    )
    response = await asyncio.to_thread(models_api.import_model_artifact, payload)
    if response.status != "success":
        raise RuntimeError(f"inference server rejected model artifact: {response.message}")
    return response.files
