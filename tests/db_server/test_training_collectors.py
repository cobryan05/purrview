"""Tests for the training_collectors CRUD endpoints."""

_PARAMS = {
    "output_dir": "collected/run1",
    "model_label_config": [{"modelName": "yolo", "labelValues": {"label-cat": {"min": 0.4, "max": 0.9}}}],
    "enable_dedup": True,
}


async def _create(http_client, name: str, **kwargs) -> dict:
    r = await http_client.post("/api/training-collectors", json={"name": name, **kwargs})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["collector"]


class TestTrainingCollectors:
    async def test_params_survive_the_round_trip(self, http_client):
        """The params blob is the whole point of the row, and it is nested — a
        shallow serialisation would flatten the bands and lose the ranges."""
        created = await _create(http_client, "Ring collector", params=_PARAMS)

        r = await http_client.get(f"/api/training-collectors/{created['uuid']}")
        assert r.json()["collector"]["params"] == _PARAMS

    async def test_duplicate_name_is_refused(self, http_client):
        """Watchers and jobs name collectors in the UI, so two identically named
        rows would be indistinguishable at the point of selection."""
        await _create(http_client, "Ring collector")

        r = await http_client.post("/api/training-collectors", json={"name": "Ring collector"})
        assert r.json()["status"] == "failure"

    async def test_updating_the_name_leaves_the_params_alone(self, http_client):
        """`params: None` means no change, not "clear" — a rename from the list
        screen sends no params and must not wipe thirty knobs."""
        created = await _create(http_client, "Ring collector", params=_PARAMS)

        r = await http_client.patch(
            f"/api/training-collectors/{created['uuid']}",
            json={"name": "Renamed"},
        )
        body = r.json()
        assert body["collector"]["name"] == "Renamed"
        assert body["collector"]["params"] == _PARAMS

    async def test_updating_params_replaces_them_wholesale(self, http_client):
        """The editor submits the whole form, so a merge would resurrect knobs the
        user just cleared."""
        created = await _create(http_client, "Ring collector", params=_PARAMS)

        r = await http_client.patch(
            f"/api/training-collectors/{created['uuid']}",
            json={"params": {"output_dir": "elsewhere"}},
        )
        assert r.json()["collector"]["params"] == {"output_dir": "elsewhere"}

    async def test_deleting_a_missing_collector_reports_failure(self, http_client):
        r = await http_client.delete("/api/training-collectors/does-not-exist")
        assert r.json()["status"] == "failure"

    async def test_delete_removes_it_from_the_list(self, http_client):
        created = await _create(http_client, "Ring collector")
        await http_client.delete(f"/api/training-collectors/{created['uuid']}")

        r = await http_client.get("/api/training-collectors")
        assert [c["uuid"] for c in r.json()["collectors"]] == []
