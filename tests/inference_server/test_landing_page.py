"""The inference server's landing page, info endpoint, and idle unload.

The inference server is commonly the only thing running on a GPU box, where
GET / rendered an empty shell and nothing told you whether onnxruntime had
fallen back to CPU.
"""

from apps.inference_server.manager import Manager


class TestLandingPage:
    async def test_root_serves_a_page(self, http_client):
        resp = await http_client.get("/")

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        body = resp.text
        assert "WhiskerWatcher Inference Server" in body
        assert '"/status"' in body
        assert '"/api/info"' in body

    async def test_page_is_self_contained(self, http_client):
        """A GPU host running only this service has no /static and no CDN."""
        body = (await http_client.get("/")).text

        assert "/static/" not in body
        assert "https://" not in body


class TestInfoEndpoint:
    async def test_reports_runtime_and_models(self, http_client, models_dir):
        resp = await http_client.get("/api/info")

        assert resp.status_code == 200
        body = resp.json()
        assert body["models_path"] == str(models_dir)
        assert body["models_bytes"] > 0
        assert [s["title"] for s in body["details"]][0] == "Runtime"
        model_section = next(s for s in body["details"] if s["title"].startswith("Models on disk"))
        assert {r["label"] for r in model_section["rows"]} == {"yolo_test", "yolo_test2"}

    async def test_summary_states_cpu_or_gpu(self, http_client):
        body = (await http_client.get("/api/info")).json()

        assert body["summary"] in ("GPU (CUDA) available", "CPU only")
        assert body["runtime"]["gpu"] is ("CUDAExecutionProvider" in body["runtime"]["providers"])

    async def test_survives_a_missing_models_directory(self, manager, http_client, tmp_path):
        manager._models_path = str(tmp_path / "gone")

        body = (await http_client.get("/api/info")).json()

        model_section = next(s for s in body["details"] if s["title"].startswith("Models on disk"))
        assert model_section["rows"] == []
        assert "No .onnx models found" in model_section["empty"]


class TestUnloadIdle:
    async def test_unloads_keepwarm_models(self, http_client, manager):
        await manager.pin_model("yolo_test", pin_id=Manager.IMPLICIT_PIN_ID)

        resp = await http_client.post("/api/models/unload-idle")

        assert resp.status_code == 200
        assert resp.json()["unloaded"] == ["yolo_test"]
        assert "yolo_test" not in manager._pinned_models

    async def test_leaves_explicitly_pinned_models_alone(self, http_client, manager):
        """A held pin means a caller is mid-run; dropping its session fails that run."""
        await manager.pin_model("yolo_test")

        resp = await http_client.post("/api/models/unload-idle")

        assert resp.json()["unloaded"] == []
        assert "yolo_test" in manager._pinned_models

    async def test_a_model_with_both_pins_is_kept(self, http_client, manager):
        await manager.pin_model("yolo_test", pin_id=Manager.IMPLICIT_PIN_ID)
        await manager.pin_model("yolo_test")

        assert (await http_client.post("/api/models/unload-idle")).json()["unloaded"] == []
        assert "yolo_test" in manager._pinned_models

    async def test_no_loaded_models_is_not_an_error(self, http_client):
        resp = await http_client.post("/api/models/unload-idle")

        assert resp.status_code == 200
        assert resp.json()["unloaded"] == []
