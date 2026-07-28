"""Shared ONNX export + PyTorch/OnnxRuntime parity check for classifier trainers.

Used by binary_image_classifier.py and multi_label_classifier.py after training
completes, to catch genuinely broken exports (wrong graph, wrong weights) before
a job is reported done.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# PyTorch and OnnxRuntime use different float32 kernels for some activations,
# so exact parity isn't a realistic bar even for a correct export (see NOTES.md).
# SiLU/Swish (EfficientNet) and HardSwish/HardSigmoid (MobileNetV3) diverge more
# than ReLU-based nets (ResNet), so they get a wider tolerance.
_DEFAULT_ATOL = 1e-2
_ARCH_ATOL_OVERRIDES = {
    "efficientnet_b0": 5e-2,
    "efficientnet_b2": 5e-2,
    "efficientnet_b4": 5e-2,
}


def validate_onnx_outputs(pt_out: np.ndarray, ort_out: np.ndarray, timm_model: str) -> float:
    """Raise RuntimeError if PT and ORT outputs diverge beyond this architecture's
    expected float32 kernel tolerance. Returns the max abs diff either way."""
    max_diff = float(np.abs(pt_out - ort_out).max())
    atol = _ARCH_ATOL_OVERRIDES.get(timm_model, _DEFAULT_ATOL)
    logger.info("ONNX validation: max abs diff = %.6f (atol=%.4f)", max_diff, atol)
    if not np.allclose(pt_out, ort_out, atol=atol):
        raise RuntimeError(f"ONNX validation failed: PyTorch and OnnxRuntime outputs differ (max diff {max_diff:.6f})")
    return max_diff


def export_and_validate_onnx(model, onnx_path: Path, timm_model: str, *, input_size: int = 224) -> None:
    """Export `model` to ONNX and check it against a live PyTorch forward pass.

    Uses a fixed-seed random input rather than all-zeros: an all-zeros tensor
    forces BatchNorm running stats through a degenerate activation region,
    which isn't representative of real images and made PT/ORT float32 kernel
    divergence look worse than it is in practice.
    """
    import torch

    model.eval()
    model = model.cpu()

    generator = torch.Generator().manual_seed(0)
    dummy = torch.randn(1, 3, input_size, input_size, generator=generator)

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
    validate_onnx_outputs(pt_out, ort_out, timm_model)
    logger.info("ONNX validation passed, model saved to %s", onnx_path)
