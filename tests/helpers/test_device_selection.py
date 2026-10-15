"""Device resolution for inference providers and training runs.

Covers the precedence both sides must obey — per-unit-of-work override beats the
deployment default beats "let the runtime choose" — and the rule that an unconfigured
deployment behaves exactly as it did before these settings existed.
"""

import pytest

from apps.helpers import metadataUtils
from apps.helpers.metadataUtils import (
    parse_model_device,
    providers_for_device,
    validate_model_device,
)
from apps.training_server.trainers.base import device_kwarg, resolve_gpu_ordinal


@pytest.fixture
def gpu_build(monkeypatch):
    monkeypatch.setattr(
        metadataUtils,
        "available_providers",
        lambda: ("CUDAExecutionProvider", "CPUExecutionProvider"),
    )


@pytest.fixture(autouse=True)
def no_device_env(monkeypatch):
    """Both settings unset unless a test sets them, so nothing leaks between tests."""
    monkeypatch.delenv("INFERENCE_GPU", raising=False)
    monkeypatch.delenv("TRAINING_GPU", raising=False)


class TestParseModelDevice:
    @pytest.mark.parametrize(
        "device,expected",
        [
            ("auto", ("auto", None)),
            ("gpu", ("gpu", None)),
            ("cpu", ("cpu", None)),
            ("gpu:0", ("gpu", 0)),
            ("gpu:3", ("gpu", 3)),
        ],
    )
    def test_known_forms(self, device, expected):
        assert parse_model_device(device) == expected

    def test_a_malformed_ordinal_degrades_to_a_bare_gpu_request(self):
        """A bad sidecar value must not stop the model loading — it loses the pin, not
        the GPU."""
        assert parse_model_device("gpu:left") == ("gpu", None)


class TestValidateModelDevice:
    @pytest.mark.parametrize("device", ["auto", "gpu", "cpu", "gpu:0", "gpu:2"])
    def test_accepts_every_supported_form(self, device):
        validate_model_device(device)

    @pytest.mark.parametrize("device", ["tpu", "gpu:", "gpu:x", "gpu:-1", ""])
    def test_rejects_what_the_api_must_not_store(self, device):
        with pytest.raises(ValueError):
            validate_model_device(device)


class TestProvidersForDevice:
    def test_an_unconfigured_deployment_names_no_device_at_all(self, gpu_build):
        """Bare strings, byte-identical to the list this returned before the setting
        existed — so enabling nothing cannot change how anything loads."""
        assert providers_for_device("auto") == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_the_env_default_pins_the_cuda_provider(self, gpu_build, monkeypatch):
        monkeypatch.setenv("INFERENCE_GPU", "1")
        assert providers_for_device("auto") == [
            ("CUDAExecutionProvider", {"device_id": 1}),
            "CPUExecutionProvider",
        ]

    def test_a_per_model_pin_overrides_the_env_default(self, gpu_build, monkeypatch):
        monkeypatch.setenv("INFERENCE_GPU", "1")
        providers = providers_for_device("gpu:0")
        assert providers[0] == ("CUDAExecutionProvider", {"device_id": 0})

    def test_cpu_ignores_the_env_default_entirely(self, gpu_build, monkeypatch):
        monkeypatch.setenv("INFERENCE_GPU", "1")
        assert providers_for_device("cpu") == ["CPUExecutionProvider"]

    @pytest.mark.parametrize("bad", ["", "nonsense", "-1"])
    def test_an_unusable_env_value_is_ignored_rather_than_breaking_loads(self, gpu_build, monkeypatch, bad):
        monkeypatch.setenv("INFERENCE_GPU", bad)
        assert providers_for_device("auto") == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_a_cpu_only_build_never_names_cuda_even_when_a_card_is_pinned(self, monkeypatch):
        monkeypatch.setattr(metadataUtils, "available_providers", lambda: ("CPUExecutionProvider",))
        monkeypatch.setenv("INFERENCE_GPU", "1")
        assert providers_for_device("gpu:1") == ["CPUExecutionProvider"]


class TestTrainingDeviceResolution:
    def test_nothing_configured_leaves_the_choice_to_the_runtime(self):
        assert resolve_gpu_ordinal({}) is None
        assert device_kwarg({}) == {}

    def test_the_env_default_applies_when_the_job_names_no_card(self, monkeypatch):
        monkeypatch.setenv("TRAINING_GPU", "1")
        assert resolve_gpu_ordinal({}) == 1
        assert device_kwarg({}) == {"device": "1"}

    def test_the_job_override_beats_the_env_default(self, monkeypatch):
        monkeypatch.setenv("TRAINING_GPU", "1")
        assert resolve_gpu_ordinal({"gpu_device": "0"}) == 0

    def test_an_empty_job_value_falls_through_to_the_env_rather_than_meaning_cpu(self, monkeypatch):
        """The UI sends "" for "deployment default", which must not read as a request
        for device 0 or for the CPU."""
        monkeypatch.setenv("TRAINING_GPU", "1")
        assert resolve_gpu_ordinal({"gpu_device": ""}) == 1

    @pytest.mark.parametrize("spelling", ["1", "cuda:1", "gpu:1", 1])
    def test_the_spellings_a_human_might_use_all_resolve_the_same(self, spelling):
        assert resolve_gpu_ordinal({"gpu_device": spelling}) == 1

    def test_cpu_is_passed_through_as_cpu(self):
        assert resolve_gpu_ordinal({"gpu_device": "cpu"}) == "cpu"
        assert device_kwarg({"gpu_device": "cpu"}) == {"device": "cpu"}

    @pytest.mark.parametrize("bad", ["nonsense", "-1"])
    def test_an_unusable_selection_runs_on_the_default_instead_of_failing(self, bad):
        assert resolve_gpu_ordinal({"gpu_device": bad}) is None
        assert device_kwarg({"gpu_device": bad}) == {}
