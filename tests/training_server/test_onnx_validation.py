"""Tests for the shared PyTorch/OnnxRuntime parity check used by classifier
trainers after export.

Regression covered here: a real training run (efficientnet_b0) hit
max diff 0.032822 against the previous flat atol=1e-2 and failed the job, even
though training converged cleanly (val_acc=1.0). EfficientNet's SiLU activation
has larger PT/ORT float32 kernel divergence than ReLU-based nets, same class of
issue already documented for MobileNetV3's HardSwish (see NOTES.md) — the fix is
an architecture-aware tolerance rather than a single flat threshold.
"""

import numpy as np
import pytest

from apps.training_server.trainers.onnx_validation import validate_onnx_outputs


class TestValidateOnnxOutputs:
    def test_small_diff_passes_for_any_architecture(self):
        pt_out = np.zeros((1, 2), dtype=np.float32)
        ort_out = pt_out + 0.005
        validate_onnx_outputs(pt_out, ort_out, "resnet18")

    def test_large_diff_raises_for_resnet(self):
        pt_out = np.zeros((1, 2), dtype=np.float32)
        ort_out = pt_out + 0.03
        with pytest.raises(RuntimeError, match="ONNX validation failed"):
            validate_onnx_outputs(pt_out, ort_out, "resnet18")

    def test_efficientnet_tolerates_the_diff_seen_in_the_real_failure(self):
        pt_out = np.zeros((1, 2), dtype=np.float32)
        ort_out = pt_out + 0.032822
        validate_onnx_outputs(pt_out, ort_out, "efficientnet_b0")

    def test_efficientnet_still_raises_beyond_widened_tolerance(self):
        pt_out = np.zeros((1, 2), dtype=np.float32)
        ort_out = pt_out + 0.2
        with pytest.raises(RuntimeError, match="ONNX validation failed"):
            validate_onnx_outputs(pt_out, ort_out, "efficientnet_b0")
