"""Tests for _inference_push — the shared mechanism every model-landing task uses
to keep inference_server's copy of a model in sync over HTTP (no shared filesystem
required). Exercised directly here; TrainModelTask/ImportModelTask/
ImportCuratedModelTask each only need to prove they call it with the right bytes.
"""

import base64
import io
import socket
import tarfile
from unittest.mock import MagicMock, patch

import pytest

from apps.tasks_server.tasks._inference_push import (
    build_artifact_tar,
    inference_server_is_colocated,
    push_artifact_tar_to_inference_server,
)


def _mock_models_api(status: str, message: str = None, files: list = None):
    response = MagicMock(status=status, message=message, files=files)
    instance = MagicMock()
    instance.import_model_artifact = MagicMock(return_value=response)
    return patch("apps.tasks_server.tasks._inference_push.ModelsApi", return_value=instance), instance


class TestBuildArtifactTar:
    def test_round_trips_file_contents(self):
        tar_bytes = build_artifact_tar({"m.onnx": b"weights", "m.json": b"{}"})
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tf:
            names = {m.name for m in tf.getmembers()}
            assert names == {"m.onnx", "m.json"}
            assert tf.extractfile("m.onnx").read() == b"weights"


class TestPushArtifactTarToInferenceServer:
    async def test_sends_base64_of_the_exact_tar(self):
        patcher, instance = _mock_models_api("success", files=["m.onnx"])
        tar_bytes = build_artifact_tar({"m.onnx": b"weights"})

        with patcher:
            result = await push_artifact_tar_to_inference_server(MagicMock(), tar_bytes)

        payload = instance.import_model_artifact.call_args.args[0]
        assert base64.b64decode(payload.artifact_tar_base64) == tar_bytes
        assert result == ["m.onnx"]

    async def test_raises_when_inference_server_rejects(self):
        patcher, _ = _mock_models_api("failure", message="disk full")

        with patcher, pytest.raises(RuntimeError, match="disk full"):
            await push_artifact_tar_to_inference_server(MagicMock(), build_artifact_tar({"m.onnx": b"weights"}))


def _client_for(host):
    return MagicMock(configuration=MagicMock(host=host))


class TestInferenceServerIsColocated:
    """The gate that decides whether the push happens at all. Getting it wrong in
    either direction is expensive: a false positive leaves a remote inference_server
    on stale weights, a false negative round-trips a multi-hundred-MB model through
    base64 back into the directory it was just written to.
    """

    @pytest.mark.parametrize("host", ["http://localhost:8001", "http://127.0.0.1:8001", "http://[::1]:8001"])
    def test_loopback_is_colocated(self, host):
        assert inference_server_is_colocated(_client_for(host)) is True

    def test_this_hosts_own_name_is_colocated(self):
        assert inference_server_is_colocated(_client_for(f"http://{socket.gethostname()}:8001")) is True

    @pytest.mark.parametrize(
        "host",
        [
            "http://inference:8001",  # sibling container on the compose network
            "http://gpu-box.lan:8001",  # Settings -> Servers override
            "http://192.0.2.10:8001",
        ],
    )
    def test_another_host_is_not_colocated(self, host):
        assert inference_server_is_colocated(_client_for(host)) is False

    def test_unusable_configuration_pushes_rather_than_skipping(self):
        """An unparseable host must not be read as "co-located" — skipping the push
        is the one outcome that fails silently."""
        assert inference_server_is_colocated(MagicMock(configuration=None)) is False
        assert inference_server_is_colocated(_client_for(MagicMock())) is False
        assert inference_server_is_colocated(_client_for("")) is False
