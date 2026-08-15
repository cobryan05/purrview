"""tasks_server honours the Settings → Servers URL overrides.

Regression coverage for a split-brain bug: image_tagging_server rebuilt its
`/api/models` proxy from the `inference_server_url` setting, but tasks_server
built its inference client from APPS_CONFIG at import time and never re-read the
setting. Pointing the UI at a remote inference host therefore left every
background task (StreamWatcher, ApplyFilterChain, PreviewModelRelabel, …) still
talking to localhost. `training_server_url` had the identical defect.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import db_client
import inference_client
import pytest

from apps import APPS, APPS_CONFIG
from apps.tasks_server.manager import Manager

_DEFAULT_INFERENCE = "http://localhost:8001"
_DEFAULT_TRAINING = "http://localhost:8005"


def _make_manager() -> Manager:
    db_api = db_client.ApiClient(db_client.Configuration("http://localhost:8003"))
    inf_api = inference_client.ApiClient(inference_client.Configuration(_DEFAULT_INFERENCE))
    return Manager(
        db_api_client=db_api,
        inference_api_client=inf_api,
        default_training_url=_DEFAULT_TRAINING,
    )


def _patch_overrides(inference: str | None, training: str | None):
    """Patch the db_server settings read that the manager resolves overrides through."""
    return patch(
        "apps.tasks_server.manager.read_url_overrides",
        AsyncMock(return_value=(inference, training)),
    )


class TestDefaults:
    def test_inference_defaults_to_injected_client(self):
        mgr = _make_manager()
        assert mgr.get_inference_api_client().configuration.host == _DEFAULT_INFERENCE

    def test_training_defaults_to_injected_url(self):
        mgr = _make_manager()
        assert mgr.get_training_base_url() == _DEFAULT_TRAINING

    def test_default_training_url_falls_back_to_apps_config(self):
        """Omitting default_training_url keeps the pre-existing compiled-in default."""
        db_api = db_client.ApiClient(db_client.Configuration("http://localhost:8003"))
        inf_api = inference_client.ApiClient(inference_client.Configuration(_DEFAULT_INFERENCE))
        mgr = Manager(db_api_client=db_api, inference_api_client=inf_api)
        cfg = APPS_CONFIG[APPS.TRAINING_SERVER]
        assert mgr.get_training_base_url() == f"http://{cfg.host}:{cfg.port}"


class TestOverridesApplied:
    async def test_inference_override_rebuilds_client(self):
        mgr = _make_manager()
        with _patch_overrides("http://gpu-box:8001", None):
            await mgr.reload_url_overrides()
        assert mgr.get_inference_api_client().configuration.host == "http://gpu-box:8001"

    async def test_training_override_applied(self):
        mgr = _make_manager()
        with _patch_overrides(None, "http://trainer-box:8005"):
            await mgr.reload_url_overrides()
        assert mgr.get_training_base_url() == "http://trainer-box:8005"

    async def test_both_overrides_applied_together(self):
        mgr = _make_manager()
        with _patch_overrides("http://gpu-box:8001", "http://trainer-box:8005"):
            await mgr.reload_url_overrides()
        assert mgr.get_inference_api_client().configuration.host == "http://gpu-box:8001"
        assert mgr.get_training_base_url() == "http://trainer-box:8005"

    async def test_trailing_slash_is_normalised(self):
        """The UI's URL input accepts a trailing slash; the client host must not
        keep it or every request path ends up double-slashed."""
        mgr = _make_manager()
        with _patch_overrides("http://gpu-box:8001/", "http://trainer-box:8005/"):
            await mgr.reload_url_overrides()
        assert mgr.get_inference_api_client().configuration.host == "http://gpu-box:8001"
        assert mgr.get_training_base_url() == "http://trainer-box:8005"

    async def test_server_config_reports_effective_urls(self):
        mgr = _make_manager()
        with _patch_overrides("http://gpu-box:8001", "http://trainer-box:8005"):
            await mgr.reload_url_overrides()
        cfg = await mgr.get_server_config()
        assert cfg["inference_server"] == "http://gpu-box:8001"
        assert cfg["training_server"] == "http://trainer-box:8005"


class TestOverrideLifecycle:
    async def test_unchanged_override_does_not_rebuild_client(self):
        """The worker re-reads every pass; rebuilding an identical client each
        time would churn urllib3 connection pools for no reason."""
        mgr = _make_manager()
        with _patch_overrides("http://gpu-box:8001", None):
            await mgr.reload_url_overrides()
            first = mgr.get_inference_api_client()
            await mgr.reload_url_overrides()
            second = mgr.get_inference_api_client()
        assert first is second

    async def test_clearing_override_reverts_to_default(self):
        mgr = _make_manager()
        with _patch_overrides("http://gpu-box:8001", "http://trainer-box:8005"):
            await mgr.reload_url_overrides()
        with _patch_overrides(None, None):
            await mgr.reload_url_overrides()
        assert mgr.get_inference_api_client().configuration.host == _DEFAULT_INFERENCE
        assert mgr.get_training_base_url() == _DEFAULT_TRAINING

    async def test_unreadable_settings_leave_defaults_intact(self):
        """db_server unreachable at boot must not wedge the client to something
        broken — read_url_overrides reports None, meaning "use the default"."""
        mgr = _make_manager()
        with _patch_overrides(None, None):
            await mgr.reload_url_overrides()
        assert mgr.get_inference_api_client().configuration.host == _DEFAULT_INFERENCE
        assert mgr.get_training_base_url() == _DEFAULT_TRAINING


class TestTasksResolveThroughManager:
    """The point of the fix: tasks must ask the manager, not APPS_CONFIG."""

    @pytest.mark.parametrize(
        "module_path, cls_name",
        [
            ("apps.tasks_server.tasks.TrainModelTask", "TrainModelTask"),
            ("apps.tasks_server.tasks.ConvertModelTask", "ConvertModelTask"),
            ("apps.tasks_server.tasks.ImportCuratedModelTask", "ImportCuratedModelTask"),
        ],
    )
    def test_task_module_does_not_build_training_url_from_apps_config(self, module_path, cls_name):
        import importlib
        import inspect

        mod = importlib.import_module(module_path)
        src = inspect.getsource(mod)
        assert "APPS.TRAINING_SERVER" not in src, (
            f"{cls_name} still builds its training URL from APPS_CONFIG; it must use "
            "manager.get_training_base_url() so the Settings override is honoured."
        )

    async def test_train_model_task_uses_manager_training_url(self):
        """_init resolves the base URL before it does any DB work, so the later
        (unmocked) training-config lookup failing is irrelevant to this assertion."""
        from apps.tasks_server.tasks.TrainModelTask import TrainModelTask

        manager = MagicMock()
        manager.update_task_heartbeat = AsyncMock()
        manager.get_training_base_url.return_value = "http://trainer-box:8005"

        task = TrainModelTask(task_config_uuid="cfg", params={}, manager=manager)
        with pytest.raises(RuntimeError):
            await task._init({"training_config_uuid": "tc-uuid"}, None)

        manager.get_training_base_url.assert_called_once()
        assert task._training_base_url == "http://trainer-box:8005"
