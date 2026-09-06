"""Tests for ImportModelTask.

Stubs the network boundary (httpx.AsyncClient.get) with a tiny real ONNX fixture
built in-process via the `onnx` package's helper API — no network access, no
external fixture file needed. Mirrors RunClassifierTask/ApplyFilterChainTask's
pattern of instantiating the task directly with a mocked manager.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import onnx
import pytest
from onnx import TensorProto, helper

from apps.helpers.curatedModels import CURATED_BASE_MODELS
from apps.tasks_server.tasks.ImportModelTask import ImportModelTask

_MOCK_MANAGER = MagicMock()
_MOCK_MANAGER.update_task_heartbeat = AsyncMock()


def _make_task(params: dict) -> ImportModelTask:
    task = ImportModelTask(task_config_uuid="cfg-uuid", params=params, manager=_MOCK_MANAGER)
    task._push_to_inference_server = AsyncMock()
    return task


def _build_onnx_bytes(class_names: dict) -> bytes:
    """Build a minimal but real ONNX model with `description` + `names` metadata,
    matching what apps.helpers.metadataUtils.get_model_metadata parses."""
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph)
    model.metadata_props.append(onnx.StringStringEntryProto(key="description", value="Ultralytics YOLOv8n model"))
    model.metadata_props.append(onnx.StringStringEntryProto(key="names", value=str(class_names)))
    return model.SerializeToString()


_ONNX_BYTES = _build_onnx_bytes({0: "person", 1: "bicycle"})


def _fake_get_factory(content: bytes, raise_error: bool = False):
    async def _fake_get(self, url, *args, **kwargs):
        response = MagicMock()
        response.content = content

        def _raise():
            if raise_error:
                raise RuntimeError("simulated HTTP error")

        response.raise_for_status = _raise
        return response

    return _fake_get


class TestInit:
    async def test_init_resolves_curated_entry(self, tmp_path):
        task = _make_task({"curated_key": CURATED_BASE_MODELS[0]["key"], "model_name": "my_model"})
        with patch("apps.tasks_server.tasks.ImportModelTask._MODELS_ROOT", tmp_path):
            await task._init(task._params, None)
        assert task._model_name == "my_model"
        assert task._onnx_url == CURATED_BASE_MODELS[0]["onnx_url"]
        assert task._model_family == CURATED_BASE_MODELS[0]["model_family"]

    async def test_unknown_curated_key_raises(self, tmp_path):
        task = _make_task({"curated_key": "bogus-key", "model_name": "my_model"})
        with patch("apps.tasks_server.tasks.ImportModelTask._MODELS_ROOT", tmp_path):
            with pytest.raises(ValueError, match="Unknown curated_key"):
                await task._init(task._params, None)

    async def test_requires_conversion_key_raises(self, tmp_path):
        conversion_entry = next(e for e in CURATED_BASE_MODELS if e.get("requires_conversion"))
        task = _make_task({"curated_key": conversion_entry["key"], "model_name": "my_model"})
        with patch("apps.tasks_server.tasks.ImportModelTask._MODELS_ROOT", tmp_path):
            with pytest.raises(ValueError, match="requires training-server conversion"):
                await task._init(task._params, None)

    async def test_invalid_model_name_raises(self, tmp_path):
        task = _make_task({"curated_key": CURATED_BASE_MODELS[0]["key"], "model_name": "../escape"})
        with patch("apps.tasks_server.tasks.ImportModelTask._MODELS_ROOT", tmp_path):
            with pytest.raises(ValueError, match="Invalid model_name"):
                await task._init(task._params, None)

    async def test_overwrite_false_existing_model_raises_fast(self, tmp_path, monkeypatch):
        (tmp_path / "existing.onnx").write_bytes(b"already here")
        task = _make_task({"curated_key": CURATED_BASE_MODELS[0]["key"], "model_name": "existing", "overwrite": False})

        get_calls = []
        monkeypatch.setattr("httpx.AsyncClient.get", _wrap_tracking(_fake_get_factory(_ONNX_BYTES), get_calls))

        with patch("apps.tasks_server.tasks.ImportModelTask._MODELS_ROOT", tmp_path):
            with pytest.raises(FileExistsError):
                await task._init(task._params, None)

        # Must fail fast, before any download attempt.
        assert get_calls == []


def _wrap_tracking(fn, calls):
    async def _wrapped(self, url, *args, **kwargs):
        calls.append(url)
        return await fn(self, url, *args, **kwargs)

    return _wrapped


class TestRun:
    async def test_downloads_writes_onnx_and_sidecar(self, tmp_path, monkeypatch):
        monkeypatch.setattr("httpx.AsyncClient.get", _fake_get_factory(_ONNX_BYTES))

        task = _make_task({"curated_key": "yolo9_t-coco", "model_name": "base_model", "overwrite": False})
        with patch("apps.tasks_server.tasks.ImportModelTask._MODELS_ROOT", tmp_path):
            await task._init(task._params, None)
            result = await task._run()

        onnx_path = tmp_path / "base_model.onnx"
        sidecar_path = tmp_path / "base_model.json"
        assert onnx_path.exists()
        assert onnx_path.read_bytes() == _ONNX_BYTES
        assert sidecar_path.exists()

        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["classes"] == ["person", "bicycle"]
        assert sidecar["model_family"] == "libreyolo9"
        assert sidecar["device"] == "auto"
        assert sidecar["class_map"] == {}

        assert result["model_name"] == "base_model"
        assert result["classes"] == ["person", "bicycle"]

    async def test_overwrite_drops_stale_sidecar_classes(self, tmp_path, monkeypatch):
        # Seed an existing model + sidecar with different classes and a class_map
        # that must NOT survive the overwrite (fresh classes only).
        (tmp_path / "base_model.onnx").write_bytes(b"old bytes")
        (tmp_path / "base_model.json").write_text(
            json.dumps({"classes": ["stale_class"], "class_map": {"stale_class": "some-label-uuid"}})
        )

        monkeypatch.setattr("httpx.AsyncClient.get", _fake_get_factory(_ONNX_BYTES))

        task = _make_task({"curated_key": "yolo9_t-coco", "model_name": "base_model", "overwrite": True})
        with patch("apps.tasks_server.tasks.ImportModelTask._MODELS_ROOT", tmp_path):
            await task._init(task._params, None)
            result = await task._run()

        sidecar = json.loads((tmp_path / "base_model.json").read_text())
        assert sidecar["classes"] == ["person", "bicycle"]
        assert sidecar["class_map"] == {}
        assert "stale_class" not in sidecar["classes"]
        assert result["classes"] == ["person", "bicycle"]

    async def test_download_http_error_propagates_without_writing_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("httpx.AsyncClient.get", _fake_get_factory(_ONNX_BYTES, raise_error=True))

        task = _make_task({"curated_key": "yolo9_t-coco", "model_name": "failed_dl", "overwrite": False})
        with patch("apps.tasks_server.tasks.ImportModelTask._MODELS_ROOT", tmp_path):
            await task._init(task._params, None)
            with pytest.raises(RuntimeError, match="simulated HTTP error"):
                await task._run()

        assert not (tmp_path / "failed_dl.onnx").exists()
        assert not (tmp_path / "failed_dl.json").exists()

    async def test_download_garbage_bytes_rejected_without_writing_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("httpx.AsyncClient.get", _fake_get_factory(b"not an onnx file at all"))

        task = _make_task({"curated_key": "yolo9_t-coco", "model_name": "garbage_dl", "overwrite": False})
        with patch("apps.tasks_server.tasks.ImportModelTask._MODELS_ROOT", tmp_path):
            await task._init(task._params, None)
            with pytest.raises(ValueError, match="Not a valid ONNX model"):
                await task._run()

        assert not (tmp_path / "garbage_dl.onnx").exists()
        assert not (tmp_path / "garbage_dl.json").exists()

    async def test_cancel_before_start_skips_download(self, tmp_path, monkeypatch):
        get_calls = []
        monkeypatch.setattr("httpx.AsyncClient.get", _wrap_tracking(_fake_get_factory(_ONNX_BYTES), get_calls))

        task = _make_task({"curated_key": "yolo9_t-coco", "model_name": "base_model2", "overwrite": False})
        with patch("apps.tasks_server.tasks.ImportModelTask._MODELS_ROOT", tmp_path):
            await task._init(task._params, None)
            task._cancel_flag.set()
            result = await task._run()

        assert result == {"cancelled": True}
        assert get_calls == []
        assert not (tmp_path / "base_model2.onnx").exists()

    async def test_deinit_is_noop(self, tmp_path):
        task = _make_task({"curated_key": "yolo9_t-coco", "model_name": "m"})
        with patch("apps.tasks_server.tasks.ImportModelTask._MODELS_ROOT", tmp_path):
            await task._init(task._params, None)
        await task._deinit()  # should not raise

    async def test_pushes_to_inference_server_on_success(self, tmp_path, monkeypatch):
        monkeypatch.setattr("httpx.AsyncClient.get", _fake_get_factory(_ONNX_BYTES))

        task = _make_task({"curated_key": "yolo9_t-coco", "model_name": "pushed_model", "overwrite": False})
        with patch("apps.tasks_server.tasks.ImportModelTask._MODELS_ROOT", tmp_path):
            await task._init(task._params, None)
            await task._run()

        task._push_to_inference_server.assert_awaited_once_with(_ONNX_BYTES)

    async def test_download_error_skips_inference_push(self, tmp_path, monkeypatch):
        monkeypatch.setattr("httpx.AsyncClient.get", _fake_get_factory(_ONNX_BYTES, raise_error=True))

        task = _make_task({"curated_key": "yolo9_t-coco", "model_name": "failed_push", "overwrite": False})
        with patch("apps.tasks_server.tasks.ImportModelTask._MODELS_ROOT", tmp_path):
            await task._init(task._params, None)
            with pytest.raises(RuntimeError):
                await task._run()

        task._push_to_inference_server.assert_not_awaited()


class TestParamsSchema:
    def test_required_params_present(self):
        schema = ImportModelTask.params_schema()
        assert "curated_key" in schema
        assert "model_name" in schema
        assert "overwrite" in schema
        assert schema["curated_key"]["type"] == "select"
        # Only direct-ONNX entries — "requires_conversion" entries go through
        # ImportCuratedModelTask instead (see test_import_curated_model_task.py).
        option_values = {o["value"] for o in schema["curated_key"]["options"]}
        assert option_values == {e["key"] for e in CURATED_BASE_MODELS if not e.get("requires_conversion")}

    def test_requires_conversion_entries_excluded_from_options(self):
        schema = ImportModelTask.params_schema()
        option_values = {o["value"] for o in schema["curated_key"]["options"]}
        conversion_keys = {e["key"] for e in CURATED_BASE_MODELS if e.get("requires_conversion")}
        assert conversion_keys, "expected at least one requires_conversion entry in the catalogue"
        assert option_values.isdisjoint(conversion_keys)
