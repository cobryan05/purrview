"""Tests for _inference_push — the shared mechanism every model-landing task uses
to keep inference_server's copy of a model in sync over HTTP (no shared filesystem
required). Exercised directly here; TrainModelTask/ImportModelTask/
ImportCuratedModelTask each only need to prove they call it with the right bytes.
"""

import base64
import io
import tarfile
from unittest.mock import MagicMock, patch

import pytest

from apps.tasks_server.tasks._inference_push import (
    build_artifact_tar,
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
