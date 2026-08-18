"""Broker connectivity probe behind the Settings → MQTT Broker panel."""

from unittest.mock import MagicMock, patch


class TestMqttBrokerProbe:
    async def test_blank_host_is_rejected_without_connecting(self, manager):
        with patch("paho.mqtt.client.Client") as client_cls:
            result = await manager.test_mqtt_broker(host="   ")
        assert result["success"] is False
        assert "host is required" in result["message"]
        client_cls.assert_not_called()

    async def test_connect_failure_surfaces_the_error(self, manager):
        client = MagicMock()
        client.connect.side_effect = OSError("Connection refused")
        with patch("paho.mqtt.client.Client", return_value=client):
            result = await manager.test_mqtt_broker(host="broker.local", port=1883)
        assert result["success"] is False
        assert "Connection refused" in result["message"]

    async def test_successful_connack_reports_host_and_port(self, manager):
        client = MagicMock()

        def _fire_connack():
            client.on_connect(client, None, {}, 0)

        client.loop_start.side_effect = _fire_connack
        with patch("paho.mqtt.client.Client", return_value=client):
            result = await manager.test_mqtt_broker(host="broker.local", port=8883, tls=True)

        assert result["success"] is True
        assert "broker.local:8883" in result["message"]
        client.tls_set.assert_called_once()
        client.disconnect.assert_called_once()

    async def test_bad_credentials_report_the_reason_not_the_code(self, manager):
        client = MagicMock()
        client.loop_start.side_effect = lambda: client.on_connect(client, None, {}, 4)
        with patch("paho.mqtt.client.Client", return_value=client):
            result = await manager.test_mqtt_broker(host="broker.local", username="u", password="p")

        assert result["success"] is False
        assert result["message"] == "bad username or password"
        client.username_pw_set.assert_called_once_with("u", "p")

    async def test_silent_broker_times_out_rather_than_hanging(self, manager):
        client = MagicMock()  # never fires on_connect
        with patch("paho.mqtt.client.Client", return_value=client):
            result = await manager.test_mqtt_broker(host="broker.local", timeout=0.05)
        assert result["success"] is False
        assert "No CONNACK" in result["message"]
        client.loop_stop.assert_called_once()


class TestMqttTestEndpoint:
    async def test_endpoint_passes_form_values_through(self, http_client, manager, monkeypatch):
        seen = {}

        async def fake_probe(**kwargs):
            seen.update(kwargs)
            return {"success": True, "message": "Connected"}

        monkeypatch.setattr(manager, "test_mqtt_broker", fake_probe)

        resp = await http_client.post(
            "/api/test-mqtt",
            json={"host": "broker.local", "port": 8883, "username": "u", "password": "p", "tls": True},
        )
        assert resp.status_code == 200
        assert resp.json() == {"success": True, "message": "Connected"}
        assert seen == {
            "host": "broker.local",
            "port": 8883,
            "username": "u",
            "password": "p",
            "tls": True,
        }

    async def test_endpoint_defaults_optional_fields(self, http_client, manager, monkeypatch):
        seen = {}

        async def fake_probe(**kwargs):
            seen.update(kwargs)
            return {"success": False, "message": "nope"}

        monkeypatch.setattr(manager, "test_mqtt_broker", fake_probe)

        resp = await http_client.post("/api/test-mqtt", json={"host": "broker.local"})
        assert resp.status_code == 200
        assert seen["port"] == 1883
        assert seen["username"] is None
        assert seen["tls"] is False
