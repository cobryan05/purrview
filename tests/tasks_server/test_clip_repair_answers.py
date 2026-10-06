"""Answering the "this camera's clips look damaged" notice.

Both answers are one-way doors from the notice's point of view — it stops being offered
either way — so the thing worth testing is that each one writes what it claims to.

The read-merge-write on enable is the one that has teeth: `update_source` replaces a
source's `params` wholesale, so an enable that sent only `repair_corruption` would blank
the camera's RTSP url, its broker, its capture interval. That is the failure this file
exists to catch.
"""

from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from apps.helpers.recordings import clipRepairAdvice
from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask

RTSP_PARAMS = {
    "video_path": "rtsp://cam.example/unicast",
    "rtsp_relay": True,
    "loop": False,
    "min_capture_interval": 0.0,
}


@pytest.fixture(autouse=True)
def _fresh_advice():
    clipRepairAdvice.reset()
    yield
    clipRepairAdvice.reset()


@pytest.fixture
def sources_api():
    """A sources API holding one RTSP camera, with its update calls recorded."""
    api = MagicMock()
    # SimpleNamespace, not MagicMock: `name` is a MagicMock constructor kwarg, so a
    # mock source's `.name` is the mock's own repr rather than the camera's name.
    api.get_sources.return_value = MagicMock(
        sources=[SimpleNamespace(uuid="src-1", name="Garage", typename="FfmpegImageProvider", params=dict(RTSP_PARAMS))]
    )
    with patch("apps.tasks_server.manager.db_client.SourcesApi", return_value=api):
        yield api


@pytest.fixture
def settings_api():
    api = MagicMock()
    with patch("apps.tasks_server.manager.db_client.SettingsApi", return_value=api):
        yield api


class TestTurningRepairOn:
    async def test_the_rest_of_the_source_config_survives(self, http_client, sources_api, settings_api):
        """`update_source` replaces `params` wholesale. Writing only the new key would
        take the camera's url and credentials with it."""
        res = await http_client.post("/api/clip-repair/enable/src-1")
        assert res.status_code == 200

        (payload,), _ = sources_api.update_source.call_args
        assert payload.provider_params == {**RTSP_PARAMS, "repair_corruption": True}

    async def test_the_source_is_not_renamed_or_retyped(self, http_client, sources_api, settings_api):
        """The same whole-object write carries the name and provider: resending them
        wrong would rename the camera or swap its provider class."""
        await http_client.post("/api/clip-repair/enable/src-1")

        (payload,), _ = sources_api.update_source.call_args
        assert payload.source_name == "Garage"
        assert payload.image_provider == "FfmpegImageProvider"

    async def test_the_suggestion_stops_being_offered(self, http_client, sources_api, settings_api):
        clipRepairAdvice.record_probe("src-1", 999, "a.mp4")

        await http_client.post("/api/clip-repair/enable/src-1")

        assert clipRepairAdvice.findings() == []
        assert clipRepairAdvice.wants_probe("src-1") is False

    async def test_turning_it_on_is_not_also_recorded_as_a_decline(self, http_client, sources_api, settings_api):
        """The source row is the record. Writing the uuid into the declined list too
        would mean a user who later turned it back off never heard about it again."""
        await http_client.post("/api/clip-repair/enable/src-1")

        settings_api.set_setting.assert_not_called()

    async def test_the_running_watcher_picks_it_up_without_a_restart(
        self, http_client, manager, sources_api, settings_api
    ):
        """A task reads its config once, in `_init`. Left to that, the notice would report
        success and the very next clip would break up exactly as before — the failure
        looks like the feature doing nothing, which is why it needs pinning.
        """
        provider = SimpleNamespace(repair_corruption=False)
        watcher = SimpleNamespace(
            _pipelines=[SimpleNamespace(source_uuid="src-1", provider=provider)],
        )
        watcher.adopt_repair_corruption = partial(StreamWatcherTask.adopt_repair_corruption, watcher)
        manager._running_tasks = {"t1": SimpleNamespace(task=watcher)}

        await http_client.post("/api/clip-repair/enable/src-1")

        assert provider.repair_corruption is True

    async def test_another_camera_on_the_same_watcher_is_left_alone(
        self, http_client, manager, sources_api, settings_api
    ):
        """The switch is per camera, and a watcher usually holds several."""
        mine = SimpleNamespace(repair_corruption=False)
        theirs = SimpleNamespace(repair_corruption=False)
        watcher = SimpleNamespace(
            _pipelines=[
                SimpleNamespace(source_uuid="src-1", provider=mine),
                SimpleNamespace(source_uuid="src-2", provider=theirs),
            ],
        )
        watcher.adopt_repair_corruption = partial(StreamWatcherTask.adopt_repair_corruption, watcher)
        manager._running_tasks = {"t1": SimpleNamespace(task=watcher)}

        await http_client.post("/api/clip-repair/enable/src-1")

        assert mine.repair_corruption is True
        assert theirs.repair_corruption is False

    async def test_a_task_that_is_not_a_watcher_is_not_asked(self, http_client, manager, sources_api, settings_api):
        """Every running task is swept, and most have no such method — the enable must
        not fail because a conversion job happened to be running."""
        manager._running_tasks = {"t1": SimpleNamespace(task=SimpleNamespace())}

        res = await http_client.post("/api/clip-repair/enable/src-1")

        assert res.json()["status"] == "success"

    async def test_an_unknown_source_fails_without_writing(self, http_client, sources_api, settings_api):
        res = await http_client.post("/api/clip-repair/enable/nope")

        assert res.json()["status"] != "success"
        sources_api.update_source.assert_not_called()


class TestDecliningIt:
    async def test_the_decline_is_persisted(self, http_client, sources_api, settings_api):
        """It has to outlive the container: the watcher that raised the suggestion
        restarts far more often than the user changes their mind."""
        await http_client.post("/api/clip-repair/dismiss/src-1")

        (key, payload), _ = settings_api.set_setting.call_args
        assert key == clipRepairAdvice.CLIP_REPAIR_DISMISSED_SETTING_KEY
        assert "src-1" in payload.value

    async def test_the_camera_is_left_exactly_as_recorded(self, http_client, sources_api, settings_api):
        await http_client.post("/api/clip-repair/dismiss/src-1")

        sources_api.update_source.assert_not_called()

    async def test_declining_one_camera_keeps_the_others_declined(self, http_client, sources_api, settings_api):
        """The setting is a whole-list replace, so a second decline has to send both or
        it silently un-declines the first."""
        await http_client.post("/api/clip-repair/dismiss/src-1")
        await http_client.post("/api/clip-repair/dismiss/src-2")

        (_, payload), _ = settings_api.set_setting.call_args
        assert "src-1" in payload.value and "src-2" in payload.value

    async def test_the_suggestion_stops_being_offered(self, http_client, sources_api, settings_api):
        clipRepairAdvice.record_probe("src-1", 999, "a.mp4")

        await http_client.post("/api/clip-repair/dismiss/src-1")

        assert clipRepairAdvice.findings() == []
        assert clipRepairAdvice.wants_probe("src-1") is False
