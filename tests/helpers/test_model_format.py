"""Unit tests for torch-free model format detection — apps/helpers/modelFormat.py.

All tests are pure, in-memory byte/string checks: no I/O, no network, no database.
"""

import pytest

from apps.helpers.modelFormat import (
    CONVERTIBLE_FORMATS,
    ModelFormat,
    detect_model_format,
    is_conversion_candidate,
)

# ──────────────────────────────────────────────────────────────────────────────
# ONNX
# ──────────────────────────────────────────────────────────────────────────────


class TestDetectOnnx:
    def test_onnx_extension_with_empty_header(self):
        assert detect_model_format("model.onnx", b"") == ModelFormat.ONNX

    def test_onnx_extension_case_insensitive(self):
        assert detect_model_format("MODEL.ONNX", b"") == ModelFormat.ONNX

    def test_onnx_extension_wins_over_junk_content(self):
        junk = bytes(range(64))
        assert detect_model_format("weights.onnx", junk) == ModelFormat.ONNX


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch
# ──────────────────────────────────────────────────────────────────────────────


class TestDetectPytorch:
    def test_pt_extension(self):
        assert detect_model_format("model.pt", b"") == ModelFormat.PYTORCH

    def test_pth_extension(self):
        assert detect_model_format("model.pth", b"") == ModelFormat.PYTORCH

    def test_torchscript_extension(self):
        assert detect_model_format("model.torchscript", b"") == ModelFormat.PYTORCH

    def test_zip_magic_with_non_pt_name(self):
        header = b"PK\x03\x04" + b"\x00" * 60
        assert detect_model_format("weights.bin", header) == ModelFormat.PYTORCH

    def test_legacy_pickle_magic(self):
        header = b"\x80\x02" + b"\x00" * 60
        assert detect_model_format("weights.bin", header) == ModelFormat.PYTORCH

    def test_legacy_pickle_magic_proto5(self):
        header = b"\x80\x05" + b"\x00" * 60
        assert detect_model_format("weights.bin", header) == ModelFormat.PYTORCH


# ──────────────────────────────────────────────────────────────────────────────
# HDF5 / Keras
# ──────────────────────────────────────────────────────────────────────────────


class TestDetectHdf5:
    def test_hdf5_signature(self):
        header = b"\x89HDF\r\n\x1a\n" + b"\x00" * 56
        assert detect_model_format("weights.bin", header) == ModelFormat.HDF5

    def test_h5_extension(self):
        assert detect_model_format("model.h5", b"") == ModelFormat.HDF5

    def test_keras_extension(self):
        assert detect_model_format("model.keras", b"") == ModelFormat.HDF5


# ──────────────────────────────────────────────────────────────────────────────
# TFLite
# ──────────────────────────────────────────────────────────────────────────────


class TestDetectTflite:
    def test_tfl3_signature_at_offset_4(self):
        header = b"\x00\x00\x00\x00" + b"TFL3" + b"\x00" * 56
        assert detect_model_format("weights.bin", header) == ModelFormat.TFLITE

    def test_tflite_extension(self):
        assert detect_model_format("model.tflite", b"") == ModelFormat.TFLITE


# ──────────────────────────────────────────────────────────────────────────────
# Unknown / robustness
# ──────────────────────────────────────────────────────────────────────────────


class TestDetectUnknown:
    def test_junk_bytes_and_unknown_extension(self):
        header = bytes(range(64))
        assert detect_model_format("model.bin", header) == ModelFormat.UNKNOWN

    def test_empty_header_and_no_extension(self):
        assert detect_model_format("model", b"") == ModelFormat.UNKNOWN

    def test_none_header_does_not_raise(self):
        assert detect_model_format("model", None) == ModelFormat.UNKNOWN

    def test_short_header_does_not_raise(self):
        assert detect_model_format("model.bin", b"\x80") == ModelFormat.UNKNOWN

    def test_empty_filename_does_not_raise(self):
        assert detect_model_format("", b"\x00" * 64) == ModelFormat.UNKNOWN

    def test_none_filename_does_not_raise(self):
        assert detect_model_format(None, b"\x00" * 64) == ModelFormat.UNKNOWN


# ──────────────────────────────────────────────────────────────────────────────
# is_conversion_candidate
# ──────────────────────────────────────────────────────────────────────────────


class TestIsConversionCandidate:
    def test_pytorch_is_candidate(self):
        assert is_conversion_candidate(ModelFormat.PYTORCH) is True

    def test_onnx_is_not_candidate(self):
        assert is_conversion_candidate(ModelFormat.ONNX) is False

    def test_hdf5_is_not_candidate(self):
        assert is_conversion_candidate(ModelFormat.HDF5) is False

    def test_tflite_is_not_candidate(self):
        assert is_conversion_candidate(ModelFormat.TFLITE) is False

    def test_unknown_is_not_candidate(self):
        assert is_conversion_candidate(ModelFormat.UNKNOWN) is False

    def test_convertible_formats_contents(self):
        assert CONVERTIBLE_FORMATS == frozenset({ModelFormat.PYTORCH})
