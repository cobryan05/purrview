"""How the watcher reads the `mqtt_broker` setting the Settings tab writes.

The settings table stores strings, so the Settings panel writes this key as JSON. The
task used to accept only a dict — with a JSON string it fell through to
`MqttSettings.from_dict({})` and connected to localhost:1883 instead of the configured
broker, silently, with no error anywhere.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask


def _task():
    return StreamWatcherTask.__new__(StreamWatcherTask)


def _settings_api_returning(value):
    """Patch the SettingsApi the task imports inside `_load_mqtt_settings`."""

    class _Api:
        def __init__(self, _client):
            pass

        def get_setting(self, key):
            assert key == "mqtt_broker"
            return SimpleNamespace(value=value)

    return patch("db_client.api.settings_api.SettingsApi", _Api)


class TestLoadMqttSettings:
    async def test_json_string_is_parsed(self):
        stored = json.dumps(
            {
                "host": "broker.local",
                "port": 8883,
                "prefix": "home/cams",
                "username": "u",
                "password": "p",
                "tls": True,
                "client_id": "ww-1",
            }
        )
        with _settings_api_returning(stored):
            settings = await _task()._load_mqtt_settings(object())

        assert settings is not None
        assert settings.host == "broker.local"
        assert settings.port == 8883
        assert settings.prefix == "home/cams"
        assert settings.username == "u"
        assert settings.password == "p"
        assert settings.tls is True
        assert settings.client_id == "ww-1"

    async def test_dict_value_still_works(self):
        with _settings_api_returning({"host": "broker.local"}):
            settings = await _task()._load_mqtt_settings(object())
        assert settings is not None
        assert settings.host == "broker.local"
        assert settings.port == 1883

    @pytest.mark.parametrize(
        "stored",
        [None, "", "{}", json.dumps({"host": ""}), "not json at all"],
        ids=["unset", "blank", "empty-object", "blank-host", "garbage"],
    )
    async def test_unset_or_hostless_config_disables_mqtt(self, stored):
        """No host means "no broker" — never a silent localhost fallback."""
        with _settings_api_returning(stored):
            assert await _task()._load_mqtt_settings(object()) is None
