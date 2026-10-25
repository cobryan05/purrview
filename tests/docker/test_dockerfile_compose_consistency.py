"""Static consistency checks between the Dockerfile, docker-compose, the GPU
overlay, and .env.example.

These catch the classic drift bugs of a multi-stage / multi-profile setup:
a compose service targeting a Dockerfile stage that no longer exists, a stray
profile name, the parameterized build args getting dropped, or the GPU overlay
accidentally granting a GPU to the wrong service.
"""

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]

KNOWN_PROFILES = {"app", "training", "mono", "dev", "dev-mono"}


def _dockerfile_stages() -> set[str]:
    text = (REPO / "Dockerfile").read_text()
    # matches: FROM <image-or-${ARG}> AS <stage>
    return set(re.findall(r"^FROM\s+\S+\s+AS\s+(\S+)", text, re.MULTILINE))


def _load_compose(name: str) -> dict:
    return yaml.safe_load((REPO / name).read_text())


def _with_defaults(value: str) -> str:
    """Resolve compose `${VAR:-default}` interpolation as it reads with VAR unset."""
    return re.sub(r"\$\{\w+:-([^}]*)\}", r"\1", value)


def _assert_default_reservation_is_all_nvidia_gpus(where: str, devices: list[dict]) -> None:
    # An unset .env must keep the legacy behaviour: the nvidia runtime, every GPU.
    gpu = [d for d in devices if _with_defaults(d.get("driver", "")) == "nvidia"]
    assert gpu, f"{where} must reserve an nvidia GPU by default"
    device_ids = [_with_defaults(i) for i in gpu[0].get("device_ids", [])]
    assert device_ids == ["all"] or gpu[0].get("count") == "all", f"{where} must reserve every GPU by default"


def test_compose_targets_exist_in_dockerfile():
    stages = _dockerfile_stages()
    compose = _load_compose("docker-compose.yaml")
    seen_targets = False
    for svc_name, svc in compose["services"].items():
        target = svc.get("build", {}).get("target")
        if target is None:
            continue
        seen_targets = True
        assert target in stages, f"service {svc_name!r} targets missing stage {target!r} (have {sorted(stages)})"
    assert seen_targets, "expected at least one service with a build target"


def test_app_target_is_parameterized_by_base():
    """The `app` stage must be the ARG-selected base so CPU/GPU is a build-arg choice."""
    text = (REPO / "Dockerfile").read_text()
    assert re.search(r"^FROM\s+\$\{APP_BASE\}\s+AS\s+app\b", text, re.MULTILINE), "expected `FROM ${APP_BASE} AS app`"
    # both concrete runtime bases the arg can point at must be defined
    stages = _dockerfile_stages()
    assert {"slim-runtime", "cuda-runtime"} <= stages


def test_compose_profiles_are_known():
    compose = _load_compose("docker-compose.yaml")
    for svc_name, svc in compose["services"].items():
        for prof in svc.get("profiles", []):
            assert prof in KNOWN_PROFILES, f"service {svc_name!r} has unknown profile {prof!r}"


def test_app_and_mono_pass_base_build_args():
    compose = _load_compose("docker-compose.yaml")
    for svc_name in ("app", "mono"):
        args = compose["services"][svc_name]["build"]["args"]
        assert (
            "APP_BASE" in args and "ORT_REQUIREMENTS" in args
        ), f"{svc_name} must forward APP_BASE + ORT_REQUIREMENTS build args"


def test_app_service_has_no_hardcoded_gpu_reservation():
    """The `app`/`app-dev` services must start on CPU-only hosts; GPU is opt-in via the overlay."""
    compose = _load_compose("docker-compose.yaml")
    for svc_name in ("app", "app-dev"):
        svc = compose["services"][svc_name]
        assert "deploy" not in svc or "resources" not in svc.get(
            "deploy", {}
        ), f"base `{svc_name}` must not reserve a GPU (that would break CPU-only hosts)"


def test_gpu_services_reserve_a_gpu():
    compose = _load_compose("docker-compose.yaml")
    for svc_name in ("training", "mono", "training-dev", "mono-dev"):
        devices = compose["services"][svc_name]["deploy"]["resources"]["reservations"]["devices"]
        _assert_default_reservation_is_all_nvidia_gpus(svc_name, devices)


def test_gpu_overlay_only_augments_app():
    overlay = _load_compose("docker-compose.gpu.yaml")
    assert set(overlay["services"].keys()) == {
        "app",
        "app-dev",
    }, "overlay must only touch the CPU-buildable `app`/`app-dev` services"
    for svc_name in ("app", "app-dev"):
        devices = overlay["services"][svc_name]["deploy"]["resources"]["reservations"]["devices"]
        _assert_default_reservation_is_all_nvidia_gpus(f"overlay {svc_name}", devices)


def test_env_example_documents_core_keys():
    text = (REPO / ".env.example").read_text()
    for key in ("COMPOSE_PROFILES", "APP_BASE", "ORT_REQUIREMENTS", "APPUSER_UID", "APPUSER_GID"):
        assert re.search(rf"^\s*#?\s*{key}=", text, re.MULTILINE), f"{key} undocumented in .env.example"
