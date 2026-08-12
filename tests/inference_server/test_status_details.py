"""The Service Status panel's detail view for the inference server.

The collapsed row answers "how many models are loaded"; the detail sections have
to answer "which ones, and what is holding them" — the question you actually ask
when the GPU is busy.
"""

import time

from apps.inference_server.manager import InferenceModel, Manager, Pin


def _pin(pin_id: str, expiry_in: float = 300.0) -> Pin:
    return Pin(pin_id=pin_id, expiry=time.time() + expiry_in, timeout=expiry_in)


def _loaded(manager: Manager, name: str, *pins: Pin) -> None:
    manager._pinned_models[name] = InferenceModel(
        model_name=name,
        path=f"/models/{name}.onnx",
        pins={p.pin_id: p for p in pins},
    )


class TestInferenceStatusDetails:
    def test_no_models_loaded_reports_an_empty_section(self, manager):
        status = manager.get_status()

        assert status["models"]["loaded"] == 0
        assert status["models"]["pins"] == 0
        section = status["details"][0]
        assert section["rows"] == []
        assert "No models loaded" in section["empty"]

    def test_each_loaded_model_is_named_with_its_pin_count(self, manager):
        _loaded(manager, "yolo_a", _pin("holder-1"), _pin("holder-2"))
        _loaded(manager, "yolo_b", _pin("holder-3"))

        status = manager.get_status()
        rows = status["details"][0]["rows"]

        assert [r["label"] for r in rows] == ["yolo_a", "yolo_b"]
        assert "2 pins" in rows[0]["value"]
        assert "1 pin" in rows[1]["value"]
        assert status["models"]["pins"] == 3
        assert status["details"][0]["title"] == "Loaded models (2)"

    def test_summary_carries_the_total_pin_count(self, manager):
        _loaded(manager, "yolo_a", _pin("holder-1"), _pin("holder-2"))

        summary = manager.get_status()["summary"]

        assert "loaded: 1/" in summary
        assert "2 pins" in summary

    def test_keepwarm_only_model_is_distinguished_from_a_held_one(self, manager):
        """An implicitly-pinned model is nobody's — it is kept warm between
        pinless recognize() calls and will drop on expiry. A row that reads the
        same as a genuinely held model hides that."""
        _loaded(manager, "warm", _pin(Manager.IMPLICIT_PIN_ID))
        _loaded(manager, "held", _pin("holder-1"))

        rows = {r["label"]: r["value"] for r in manager.get_status()["details"][0]["rows"]}

        assert "keep-warm" in rows["warm"]
        assert "keep-warm" not in rows["held"]

    def test_row_reports_time_until_the_soonest_expiry(self, manager):
        _loaded(manager, "yolo_a", _pin("soon", expiry_in=90), _pin("later", expiry_in=3600))

        value = manager.get_status()["details"][0]["rows"][0]["value"]

        # 90s → "1m 30s"; the far-off pin must not be the one reported.
        assert "next expiry in 1m" in value

    async def test_status_endpoint_serves_the_details(self, http_client, manager):
        _loaded(manager, "yolo_a", _pin("holder-1"))

        resp = await http_client.get("/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "inference_server"
        assert data["details"][0]["rows"][0]["label"] == "yolo_a"
