"""Drive ./setup.sh through every role/GPU permutation and assert the .env it
writes. GPU detection is made deterministic by shadowing `nvidia-smi` on PATH
(exit 0 = GPU present, exit 1 = absent), so the matrix is host-independent.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SETUP = REPO / "setup.sh"


def _make_fake_nvidia_smi(bindir: Path, present: bool) -> None:
    bindir.mkdir(parents=True, exist_ok=True)
    smi = bindir / "nvidia-smi"
    # exit 0 (with GPU line) when present; non-zero when absent
    smi.write_text("#!/usr/bin/env bash\n" + ("echo 'GPU 0: fake'\nexit 0\n" if present else "exit 1\n"))
    smi.chmod(0o755)


def run_setup(tmp_path: Path, gpu: bool, stdin: str, preexisting_env: str | None = None) -> dict | None:
    """Run setup.sh in an isolated dir; return the parsed .env dict (or None)."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    shutil.copy(SETUP, workdir / "setup.sh")
    if preexisting_env is not None:
        (workdir / ".env").write_text(preexisting_env)

    bindir = tmp_path / "bin"
    _make_fake_nvidia_smi(bindir, gpu)
    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"

    subprocess.run(
        ["bash", str(workdir / "setup.sh")],
        input=stdin,
        text=True,
        cwd=str(workdir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
    )

    env_file = workdir / ".env"
    if not env_file.exists():
        return None
    parsed = {}
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            parsed[k] = v
    return parsed


# (label, gpu, stdin, expected-subset, forbidden-keys)
CASES = [
    (
        "frontend_cpu",
        False,
        "2\n\n\n",
        {"COMPOSE_PROFILES": "app", "APP_BASE": "slim-runtime", "ORT_REQUIREMENTS": "requirements-cpu.txt"},
        {"COMPOSE_FILE", "TRAINING_SERVER_HOST"},
    ),
    (
        "frontend_gpu_inference",
        True,
        "2\ny\n\n\n",
        {
            "COMPOSE_PROFILES": "app",
            "APP_BASE": "cuda-runtime",
            "ORT_REQUIREMENTS": "requirements-gpu.txt",
            "COMPOSE_FILE": "docker-compose.yaml:docker-compose.gpu.yaml",
        },
        set(),
    ),
    (
        "frontend_gpu_but_cpu_inference",
        True,
        "2\nn\n\n\n",
        {"COMPOSE_PROFILES": "app", "APP_BASE": "slim-runtime", "ORT_REQUIREMENTS": "requirements-cpu.txt"},
        {"COMPOSE_FILE"},
    ),
    (
        "frontend_cpu_remote_trainer",
        False,
        "2\ny\ngpu-box.lan\n\n",
        {"COMPOSE_PROFILES": "app", "APP_BASE": "slim-runtime", "TRAINING_SERVER_HOST": "gpu-box.lan"},
        {"COMPOSE_FILE"},
    ),
    (
        "everything_mono_gpu_inference",
        True,
        "1\n1\ny\n\n",
        {"COMPOSE_PROFILES": "mono", "APP_BASE": "cuda-runtime", "ORT_REQUIREMENTS": "requirements-gpu.txt"},
        {"COMPOSE_FILE"},
    ),
    (
        "everything_mono_cpu_inference",
        True,
        "1\n1\nn\n\n",
        {"COMPOSE_PROFILES": "mono", "APP_BASE": "cuda-runtime", "ORT_REQUIREMENTS": "requirements-cpu.txt"},
        {"COMPOSE_FILE"},
    ),
    (
        "everything_split_gpu_inference",
        True,
        "1\n2\ny\n\n",
        {
            "COMPOSE_PROFILES": "app,training",
            "APP_BASE": "cuda-runtime",
            "ORT_REQUIREMENTS": "requirements-gpu.txt",
            "COMPOSE_FILE": "docker-compose.yaml:docker-compose.gpu.yaml",
        },
        set(),
    ),
    (
        # split with CPU inference: the frontend uses the slim base, no CUDA.
        "everything_split_cpu_inference",
        True,
        "1\n2\nn\n\n",
        {"COMPOSE_PROFILES": "app,training", "APP_BASE": "slim-runtime", "ORT_REQUIREMENTS": "requirements-cpu.txt"},
        {"COMPOSE_FILE"},
    ),
    (
        # no GPU but "everything" confirmed -> CUDA base for torch, CPU inference.
        "everything_mono_no_gpu_confirmed",
        False,
        "1\ny\n1\n\n",
        {"COMPOSE_PROFILES": "mono", "APP_BASE": "cuda-runtime", "ORT_REQUIREMENTS": "requirements-cpu.txt"},
        set(),
    ),
    (
        "dev_mono",
        True,
        "4\n1\ny\n\n",
        {"COMPOSE_PROFILES": "dev-mono", "APP_BASE": "cuda-runtime", "ORT_REQUIREMENTS": "requirements-gpu.txt"},
        {"COMPOSE_FILE"},
    ),
    (
        "dev_split_gpu_inference",
        True,
        "4\n2\ny\n\n",
        {"COMPOSE_PROFILES": "dev", "APP_BASE": "cuda-runtime", "ORT_REQUIREMENTS": "requirements-gpu.txt"},
        {"COMPOSE_FILE"},
    ),
    (
        "dev_split_cpu_inference",
        True,
        "4\n2\nn\n\n",
        {"COMPOSE_PROFILES": "dev", "APP_BASE": "slim-runtime", "ORT_REQUIREMENTS": "requirements-cpu.txt"},
        {"COMPOSE_FILE"},
    ),
    (
        "training_only_gpu",
        True,
        "3\n\n",
        {"COMPOSE_PROFILES": "training", "APP_BASE": "cuda-runtime", "ORT_REQUIREMENTS": "requirements-gpu.txt"},
        set(),
    ),
    (
        "training_only_no_gpu_confirmed",
        False,
        "3\ny\n\n",
        {"COMPOSE_PROFILES": "training"},
        set(),
    ),
]


@pytest.mark.parametrize("label,gpu,stdin,expected,forbidden", CASES, ids=[c[0] for c in CASES])
def test_setup_permutations(tmp_path, label, gpu, stdin, expected, forbidden):
    result = run_setup(tmp_path, gpu=gpu, stdin=stdin)
    assert result is not None, f"{label}: setup.sh wrote no .env"
    for k, v in expected.items():
        assert result.get(k) == v, f"{label}: expected {k}={v!r}, got {result.get(k)!r}"
    for k in forbidden:
        assert k not in result, f"{label}: {k} should not be set (got {result.get(k)!r})"
    # every deployment must always carry the uid/gid for bind-mount ownership
    assert "APPUSER_UID" in result and "APPUSER_GID" in result


def test_setup_respects_declined_overwrite(tmp_path):
    sentinel = "COMPOSE_PROFILES=SENTINEL\n"
    # role 2, CPU, no remote trainer, then decline the overwrite prompt.
    result = run_setup(tmp_path, gpu=False, stdin="2\n\nn\n", preexisting_env=sentinel)
    assert result == {"COMPOSE_PROFILES": "SENTINEL"}, "declining overwrite must leave .env untouched"


def test_setup_cpu_never_selects_cuda_or_overlay(tmp_path):
    """A CPU-only frontend must never end up on the CUDA base or with the GPU overlay."""
    result = run_setup(tmp_path, gpu=False, stdin="2\n\n\n")
    assert result["APP_BASE"] == "slim-runtime"
    assert "COMPOSE_FILE" not in result
