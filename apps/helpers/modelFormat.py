"""Torch-free model file format detection.

This module runs on the **lean inference/web container**, which has `onnx` and
`onnxruntime-gpu` installed but deliberately does **not** ship `torch` (it would
bloat the image and pull in a heavy, mostly-unused dependency graph on a
container that only needs to run ONNX inference). Because of that constraint,
we cannot rely on any framework's own loader (e.g. `torch.load`, `onnx.load`,
`h5py.File`) to sniff a model file's format — those either require the
framework to be installed or are needlessly expensive (fully parsing a
protobuf/HDF5 file just to answer "what format is this?").

Instead, format detection here is purely extension- and magic-byte-based:
cheap, dependency-free, and safe to run on any container. It is a heuristic,
not a guarantee — but it's good enough to route a file to the right importer
or converter without paying for a full parse.
"""

from __future__ import annotations

# Magic byte signatures.
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_TFLITE_SIGNATURE = b"TFL3"
_ZIP_SIGNATURE = b"PK\x03\x04"
_PICKLE_PROTO_MAGICS = (b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05")

# Extension groupings (lower-cased, including the leading dot).
_HDF5_EXTENSIONS = (".h5", ".hdf5", ".keras")
_TFLITE_EXTENSIONS = (".tflite",)
_PYTORCH_EXTENSIONS = (".pt", ".pth", ".torchscript")
_ONNX_EXTENSIONS = (".onnx",)


class ModelFormat:
    """Enumeration (as plain string constants) of recognized model formats."""

    ONNX = "onnx"
    PYTORCH = "pytorch"
    HDF5 = "hdf5"
    TFLITE = "tflite"
    UNKNOWN = "unknown"


# Formats for which the training server has a converter (e.g. pytorch -> onnx).
CONVERTIBLE_FORMATS = frozenset({ModelFormat.PYTORCH})


def _extension_of(filename: str) -> str:
    """Return the lower-cased extension (with leading dot) of `filename`, or "" if none."""
    if not filename:
        return ""
    idx = filename.rfind(".")
    if idx < 0:
        return ""
    return filename[idx:].lower()


def detect_model_format(filename: str, header: bytes) -> str:
    """Detect a model file's format from its filename and leading bytes.

    `header` should be at least the first 64 bytes of the file, but this
    function is robust to `None`, empty, or shorter input — it simply treats
    any missing bytes as "not a match" for that signature.

    Detection priority:
      1. HDF5 / Keras — 8-byte HDF5 signature, or `.h5`/`.hdf5`/`.keras` extension.
      2. TFLite — `TFL3` at byte offset 4, or `.tflite` extension.
      3. ONNX — `.onnx` extension (checked with high priority so that an ONNX
         file lacking any other recognizable magic still resolves correctly;
         ONNX protobufs have no single reliable magic signature to sniff).
      4. PyTorch — `.pt`/`.pth`/`.torchscript` extension, or ZIP magic
         (modern `torch.save` writes a zip archive), or legacy pickle
         protocol 2-5 magic (older `torch.save` format).
      5. Otherwise — UNKNOWN.
    """
    header = header or b""
    ext = _extension_of(filename or "")

    # 1. HDF5 / Keras
    if header[:8] == _HDF5_SIGNATURE or ext in _HDF5_EXTENSIONS:
        return ModelFormat.HDF5

    # 2. TFLite
    if header[4:8] == _TFLITE_SIGNATURE or ext in _TFLITE_EXTENSIONS:
        return ModelFormat.TFLITE

    # 3. ONNX — extension is the primary (and highest-priority) signal since
    # there is no reliable single magic byte sequence for ONNX protobufs.
    if ext in _ONNX_EXTENSIONS:
        return ModelFormat.ONNX

    # 4. PyTorch
    if ext in _PYTORCH_EXTENSIONS or header[:4] == _ZIP_SIGNATURE or header[:2] in _PICKLE_PROTO_MAGICS:
        return ModelFormat.PYTORCH

    # 5. Unknown
    return ModelFormat.UNKNOWN


def is_conversion_candidate(fmt: str) -> bool:
    """Return True if `fmt` is a format the training server knows how to convert."""
    return fmt in CONVERTIBLE_FORMATS
