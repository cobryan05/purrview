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
        "2\nn\nn\n",
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
        "2\nn\ny\ngpu-box.lan\n\n",
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
        "1\ny\n1\nn\n",
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
        {
            "COMPOSE_PROFILES": "dev",
            "APP_BASE": "cuda-runtime",
            "ORT_REQUIREMENTS": "requirements-gpu.txt",
            "COMPOSE_FILE": "docker-compose.yaml:docker-compose.gpu.yaml",
        },
        set(),
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
    # role 2, decline GPU inference, no remote trainer, default ports, decline size
    # tuning, then decline the overwrite prompt.
    result = run_setup(tmp_path, gpu=False, stdin="2\nn\nn\n\n\n\nn\nn\n", preexisting_env=sentinel)
    assert result == {"COMPOSE_PROFILES": "SENTINEL"}, "declining overwrite must leave .env untouched"


def test_setup_cpu_never_selects_cuda_or_overlay(tmp_path):
    """A CPU-only frontend must never end up on the CUDA base or with the GPU overlay."""
    result = run_setup(tmp_path, gpu=False, stdin="2\nn\nn\n")
    assert result["APP_BASE"] == "slim-runtime"
    assert "COMPOSE_FILE" not in result


def test_gpu_inference_can_be_chosen_when_detection_finds_nothing(tmp_path):
    """The probe is advisory, not a verdict.

    Detection runs `nvidia-smi` on the *host*, which can be absent on a machine whose
    containers reach the GPU perfectly well. This question used to be skipped entirely
    on an undetected host and forced to "n", making it the only role-affecting choice in
    the script with no override — every other "no GPU found" branch offers a "continue
    anyway". The answer now defaults to no and is still asked.
    """
    result = run_setup(tmp_path, gpu=False, stdin="2\ny\nn\nn\n")
    assert result["APP_BASE"] == "cuda-runtime"
    assert result["ORT_REQUIREMENTS"] == "requirements-gpu.txt"
    assert result["COMPOSE_FILE"] == "docker-compose.yaml:docker-compose.gpu.yaml"


def test_declining_gpu_on_a_gpu_host_still_builds_cpu_only(tmp_path):
    """The override runs both ways: a detected GPU is a default, not a commitment."""
    result = run_setup(tmp_path, gpu=True, stdin="2\nn\nn\nn\n")
    assert result["APP_BASE"] == "slim-runtime"
    assert result["ORT_REQUIREMENTS"] == "requirements-cpu.txt"
    assert "COMPOSE_FILE" not in result


class TestSizingPrompts:
    """The sizing section (RAM buffer + disk caches).

    Two properties matter. First, it is opt-in: every knob has a working default in
    docker-compose.yaml, so declining must leave .env exactly as short as before —
    otherwise every existing deployment gains config it never asked for. Second, the
    prompts are gated on whether the deployment actually contains the consuming
    service, so a training-only node is never asked about a stream buffer it has no
    mediamtx to fill.
    """

    def test_declining_writes_no_sizing_keys(self, tmp_path):
        # role 2 (frontend), decline GPU inference, no remote trainer, decline tuning.
        result = run_setup(tmp_path, gpu=False, stdin="2\nn\nn\nn\n")
        for key in (
            "STREAM_BUFFER_SIZE",
            "THUMB_CACHE_MAX_MB",
            "TRAINING_SHM_SIZE",
            "TRAINING_CACHE_MAX_MB",
        ):
            assert key not in result, f"{key} must not be written when tuning is declined"

    def test_frontend_role_is_asked_only_about_app_group_sizes(self, tmp_path):
        # role 2 has mediamtx but no training container: buffer + thumb cache only.
        # n(gpu) n(remote) ''(app port) ''(supervisord port) ''(rtsp port) y(tune) 4(cams) 60(secs)
        # ''(buffer default) 250(thumb)
        result = run_setup(tmp_path, gpu=False, stdin="2\nn\nn\n\n\n\ny\n4\n60\n\n250\n")
        assert result["STREAM_BUFFER_SIZE"] == "240m"
        assert result["THUMB_CACHE_MAX_MB"] == "250"
        assert "TRAINING_SHM_SIZE" not in result
        assert "TRAINING_CACHE_MAX_MB" not in result

    def test_training_only_role_is_not_asked_about_the_stream_buffer(self, tmp_path):
        # role 3 runs no mediamtx, so a /buffer tmpfs would have nothing to fill it.
        result = run_setup(tmp_path, gpu=True, stdin="3\n\n\ny\n4gb\n20000\n14\n900\n")
        assert "STREAM_BUFFER_SIZE" not in result
        assert "THUMB_CACHE_MAX_MB" not in result
        assert result["TRAINING_SHM_SIZE"] == "4gb"
        assert result["TRAINING_CACHE_MAX_MB"] == "20000"
        assert result["TRAINING_CACHE_MAX_AGE_DAYS"] == "14"
        assert result["TRAINING_CACHE_SWEEP_INTERVAL_SEC"] == "900"

    def test_buffer_default_is_computed_from_cameras_and_rewind(self, tmp_path):
        # 8 cameras x 30 s at 4 Mbps -> 8*30/2*2 = 240 MB suggested.
        result = run_setup(tmp_path, gpu=False, stdin="2\nn\nn\n\n\n\ny\n8\n30\n\n100\n")
        assert result["STREAM_BUFFER_SIZE"] == "240m"

    def test_tiny_deployments_get_a_floor_not_a_useless_buffer(self, tmp_path):
        # 1 camera x 5 s would compute 2 MB, which cannot hold a single segment.
        result = run_setup(tmp_path, gpu=False, stdin="2\nn\nn\n\n\n\ny\n1\n5\n\n100\n")
        assert result["STREAM_BUFFER_SIZE"] == "128m"

    def test_size_fields_reject_a_bare_number_then_accept_a_valid_size(self, tmp_path):
        # A typo must be caught at the prompt, not passed through to compose where it
        # becomes an opaque runtime error far from its cause.
        result = run_setup(tmp_path, gpu=False, stdin="2\nn\nn\n\n\n\ny\n4\n60\nnonsense\n1g\n100\n")
        assert result["STREAM_BUFFER_SIZE"] == "1g"

    def test_cache_fields_reject_a_unit_suffix(self, tmp_path):
        # THUMB_CACHE_MAX_MB is a plain MB count; '100mb' must not be accepted, or the
        # value reaches the app as a string it cannot int().
        result = run_setup(tmp_path, gpu=False, stdin="2\nn\nn\n\n\n\ny\n4\n60\n\n100mb\n100\n")
        assert result["THUMB_CACHE_MAX_MB"] == "100"


class TestHostPortPrompts:
    """The host-port section.

    Host-side mappings only, so the questions follow what this host publishes: the app
    and its dashboard wherever the app runs, the training API wherever training runs,
    and a training dashboard only where training is its own container. Defaults are
    never written, so an untouched deployment's .env stays as short as it was.
    """

    PORT_KEYS = ("APP_PORT", "SUPERVISORD_PORT", "RTSP_PORT", "TRAINING_PORT", "TRAINING_SUPERVISORD_PORT")

    def test_defaults_write_nothing(self, tmp_path):
        result = run_setup(tmp_path, gpu=False, stdin="2\nn\nn\n\n\n\nn\n")
        for key in self.PORT_KEYS:
            assert key not in result, f"{key} must not be written at its default"

    def test_frontend_role_publishes_only_the_app_ports(self, tmp_path):
        # n(gpu) n(remote) 8080(app) 9101(supervisord) 8653(rtsp) n(tune). Had a training
        # port been asked, "n" would have been rejected as a port and tuning silently entered.
        result = run_setup(tmp_path, gpu=False, stdin="2\nn\nn\n8080\n9101\n8653\nn\n")
        assert result["APP_PORT"] == "8080"
        assert result["SUPERVISORD_PORT"] == "9101"
        assert result["RTSP_PORT"] == "8653"
        assert "TRAINING_PORT" not in result
        assert "TRAINING_SUPERVISORD_PORT" not in result
        assert "STREAM_BUFFER_SIZE" not in result

    def test_training_only_role_publishes_only_the_training_ports(self, tmp_path):
        result = run_setup(tmp_path, gpu=True, stdin="3\n8105\n9102\nn\n")
        assert result["TRAINING_PORT"] == "8105"
        assert result["TRAINING_SUPERVISORD_PORT"] == "9102"
        assert "APP_PORT" not in result
        assert "SUPERVISORD_PORT" not in result
        assert "RTSP_PORT" not in result

    def test_mono_has_no_separate_training_dashboard(self, tmp_path):
        # 1(everything) 1(mono) y(gpu inference) 7999 9001 8553 8105 n(tune): four port
        # questions, not five — training shares the app container's supervisord.
        result = run_setup(tmp_path, gpu=True, stdin="1\n1\ny\n\n\n\n8105\nn\n")
        assert result["TRAINING_PORT"] == "8105"
        assert "TRAINING_SUPERVISORD_PORT" not in result
        assert "STREAM_BUFFER_SIZE" not in result

    def test_split_asks_for_all_five(self, tmp_path):
        result = run_setup(tmp_path, gpu=True, stdin="1\n2\ny\n8080\n9101\n8653\n8105\n9102\nn\n")
        assert result["APP_PORT"] == "8080"
        assert result["SUPERVISORD_PORT"] == "9101"
        assert result["RTSP_PORT"] == "8653"
        assert result["TRAINING_PORT"] == "8105"
        assert result["TRAINING_SUPERVISORD_PORT"] == "9102"

    def test_a_port_clash_is_asked_again(self, tmp_path):
        # 8080 for both, rejected as a pair; then 8080/9101 accepted.
        result = run_setup(tmp_path, gpu=False, stdin="2\nn\nn\n8080\n8080\n\n8080\n9101\n\nn\n")
        assert result["APP_PORT"] == "8080"
        assert result["SUPERVISORD_PORT"] == "9101"

    def test_an_invalid_port_is_asked_again(self, tmp_path):
        result = run_setup(tmp_path, gpu=False, stdin="2\nn\nn\n70000\n8080\n\n\nn\n")
        assert result["APP_PORT"] == "8080"
        assert "SUPERVISORD_PORT" not in result

    def test_remote_trainer_takes_the_port_it_publishes(self, tmp_path):
        # y(remote) host 8105(port) then default app ports, decline tuning.
        result = run_setup(tmp_path, gpu=False, stdin="2\nn\ny\ngpu-box.lan\n8105\n\n\n\nn\n")
        assert result["TRAINING_SERVER_HOST"] == "gpu-box.lan"
        assert result["TRAINING_SERVER_PORT"] == "8105"

    def test_remote_trainer_on_the_default_port_writes_no_port(self, tmp_path):
        result = run_setup(tmp_path, gpu=False, stdin="2\nn\ny\ngpu-box.lan\n\n\n\n\nn\n")
        assert result["TRAINING_SERVER_HOST"] == "gpu-box.lan"
        assert "TRAINING_SERVER_PORT" not in result
