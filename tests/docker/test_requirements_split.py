"""Guard the onnxruntime requirements split introduced by the image-tier refactor.

The frontend+inference `app` image must never pull torch, and its onnxruntime
flavor is chosen per-tier:
  - requirements.txt        -> shared base, NO onnxruntime
  - requirements-cpu.txt    -> onnxruntime (slim/CPU app)
  - requirements-gpu.txt    -> onnxruntime-gpu (CUDA app / GPU inference)
  - requirements-training.txt -> heavy torch stack (+ onnxruntime-gpu)

If these drift (e.g. onnxruntime creeps back into the base file, or the CPU/GPU
pins diverge) the tier split silently breaks.
"""
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _reqs(name: str) -> list[str]:
    lines = []
    for raw in (REPO / name).read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def _has_pkg(reqs: list[str], pkg: str) -> list[str]:
    """Return the requirement lines that pin exactly `pkg` (not a prefix match)."""
    out = []
    for r in reqs:
        name = r.split("<")[0].split(">")[0].split("=")[0].split("~")[0].split("[")[0].strip()
        if name == pkg:
            out.append(r)
    return out


def test_base_requirements_have_no_onnxruntime():
    base = _reqs("requirements.txt")
    assert not _has_pkg(base, "onnxruntime"), "onnxruntime must not be in the base image reqs"
    assert not _has_pkg(base, "onnxruntime-gpu"), "onnxruntime-gpu must not be in the base image reqs"
    # onnx (the model format lib) is small and shared — it *should* stay in base.
    assert _has_pkg(base, "onnx"), "onnx (model format) should remain in the base reqs"


def test_base_requirements_have_no_torch():
    base = _reqs("requirements.txt")
    for heavy in ("torch", "torchvision", "ultralytics", "libreyolo", "timm"):
        assert not _has_pkg(base, heavy), f"{heavy} must not be in the slim base reqs"


def test_cpu_requirements_pin_plain_onnxruntime():
    cpu = _reqs("requirements-cpu.txt")
    hits = _has_pkg(cpu, "onnxruntime")
    assert hits, "requirements-cpu.txt must pin onnxruntime"
    assert not _has_pkg(cpu, "onnxruntime-gpu"), "CPU reqs must not use the gpu wheel"


def test_gpu_requirements_pin_onnxruntime_gpu():
    gpu = _reqs("requirements-gpu.txt")
    assert _has_pkg(gpu, "onnxruntime-gpu"), "requirements-gpu.txt must pin onnxruntime-gpu"


def test_cpu_gpu_and_training_onnxruntime_pins_are_aligned():
    """The version constraint should match across cpu/gpu/training so switching
    inference flavor never drifts the onnxruntime version."""

    def constraint(line: str) -> str:
        # strip the package name, keep the version spec (e.g. "<1.27")
        for pkg in ("onnxruntime-gpu", "onnxruntime"):
            if line.startswith(pkg):
                return line[len(pkg):].strip()
        return ""

    cpu = constraint(_has_pkg(_reqs("requirements-cpu.txt"), "onnxruntime")[0])
    gpu = constraint(_has_pkg(_reqs("requirements-gpu.txt"), "onnxruntime-gpu")[0])
    train_hits = _has_pkg(_reqs("requirements-training.txt"), "onnxruntime-gpu")
    assert cpu == gpu, f"cpu ({cpu!r}) and gpu ({gpu!r}) onnxruntime pins differ"
    if train_hits:
        train = constraint(train_hits[0])
        assert train == gpu, f"training ({train!r}) and gpu ({gpu!r}) onnxruntime pins differ"


@pytest.mark.parametrize("name", ["requirements-cpu.txt", "requirements-gpu.txt"])
def test_ort_files_are_single_purpose(name):
    """The cpu/gpu overlay files should only carry the onnxruntime pin, nothing else."""
    reqs = _reqs(name)
    assert len(reqs) == 1, f"{name} should contain exactly one requirement, got {reqs}"
