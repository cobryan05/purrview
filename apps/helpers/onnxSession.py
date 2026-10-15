"""Session construction shared by every inference provider."""

import logging
import sys
from typing import Any, List, Optional, Tuple

from apps.helpers.logUtils import LOG_LEVEL

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)

# Substrings of the CUDA/cuBLAS/cuDNN initialization failures that mean "the GPU
# had no room left", as onnxruntime reports them. Matched case-insensitively
# against the whole exception text, which nests the originating cuda_call.cc
# message inside an ORT FAIL.
#
# The list is deliberately about exhaustion only. A genuinely broken CUDA setup
# (no driver, a model the EP cannot compile) must keep raising: running it on the
# CPU instead would turn a loud misconfiguration into a permanent 20x slowdown
# nobody is told about.
_OOM_MARKERS = (
    "cublas failure 3",  # CUBLAS_STATUS_ALLOC_FAILED
    "cudnn_status_alloc_failed",
    "out of memory",
    "cudaerrormemoryallocation",
    "failed to allocate memory",
    "the resource allocation failed",
)


def is_gpu_exhaustion(exc: BaseException) -> bool:
    """Whether *exc* is an ONNX session failure caused by a full GPU."""
    text = str(exc).lower()
    return any(marker in text for marker in _OOM_MARKERS)


def _names(providers: List[Any]) -> List[str]:
    """Provider names, whether each entry is a bare name or a ``(name, options)`` pair.

    onnxruntime accepts both forms in one list, and a card-pinned CUDA entry is always
    the tuple form (see metadataUtils.providers_for_device), so no check on this list
    may assume strings.
    """
    return [p[0] if isinstance(p, (tuple, list)) else p for p in providers]


def create_session(model_path: str, providers: Optional[List[Any]] = None) -> Tuple["object", Optional[str]]:
    """Build an ``ort.InferenceSession``, falling back to CPU if the GPU is full.

    Returns ``(session, fallback_reason)`` — the reason is None on the normal
    path and a one-line, user-facing explanation when the session ended up on
    the CPU after CUDA refused to initialize.

    onnxruntime's provider list is a *preference order for placing operators*,
    not a retry chain: if the CUDA provider cannot be constructed at all —
    ``cublasCreate`` needing a few hundred MB of handle and workspace on a card
    another process has filled — the whole session raises and the
    CPUExecutionProvider sitting behind it in the list is never reached. So the
    fallback has to be a second, explicit attempt.

    Only exhaustion is retried (see ``_OOM_MARKERS``). A model already holding a
    live session is unaffected by a GPU filling up afterwards, which is why a
    big model keeps working while a small one fails to load: the difference is
    load-time versus already-loaded, not size.
    """
    import onnxruntime as ort

    requested = list(providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
    try:
        return ort.InferenceSession(model_path, providers=requested), None
    except Exception as exc:
        if "CUDAExecutionProvider" not in _names(requested) or not is_gpu_exhaustion(exc):
            raise
        logger.warning(
            "CUDA could not initialize for %s (%s); retrying on CPU",
            model_path,
            exc,
            exc_info=True,
        )
        try:
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        except Exception:
            # The CPU attempt failing says the model itself is the problem, so
            # the caller must see the original GPU error rather than a second
            # one that merely confirms it.
            raise exc from None
        reason = (
            "GPU memory was exhausted, so this model loaded on the CPU and will run "
            "much slower. Something else is using the GPU — often a training job."
        )
        logger.warning("%s loaded on CPU: %s", model_path, reason)
        return session, reason
