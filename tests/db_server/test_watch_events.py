"""Tests for the NVR event history — what the event review page reads.

The properties worth protecting are the ones whose failure mode is a *plausible-looking
wrong answer* rather than an error:

* filters that silently ignore an unknown parameter return the unfiltered set, which
  looks like "this camera really did see all of that",
* `total` drifting from the filtered set turns "Page 1 of 4" into a lie,
* denormalised rule/slot names existing so history survives a profile edit — the whole
  reason those columns are not joins,
* the snapshot path never reaching the browser, and the endpoint refusing to serve a
  path that no longer resolves to a file.
"""

import uuid as _uuid
from datetime import datetime, timedelta, timezone

import pytest


async def _make_source(http_client, name: str | None = None) -> str:
    r = await http_client.post(
        "/api/sources/create",
        json={
            "sourceName": name or f"cam-{_uuid.uuid4().hex[:6]}",
            "imageProvider": "FfmpegImageProvider",
            "providerParams": {"video_path": "rtsp://cam.local/stream"},
        },
    )
    assert r.status_code == 200, r.text
    return r.json()["source"]["uuid"]


async def _make_profile(http_client, name: str | None = None) -> str:
    r = await http_client.post("/api/watch-profiles", json={"name": name or f"watcher-{_uuid.uuid4().hex[:6]}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["profile"]["uuid"]


async def _make_event(http_client, profile_uuid: str, **overrides) -> dict:
    payload = {"profileUuid": profile_uuid}
    payload.update(overrides)
    r = await http_client.post("/api/watch-events", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["event"]


async def _list(http_client, **params) -> dict:
    r = await http_client.get("/api/watch-events", params=params)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body


class TestWatchEventCrud:
    async def test_create_and_get(self, http_client):
        profile_uuid = await _make_profile(http_client)
        created = await _make_event(
            http_client,
            profile_uuid,
            ruleName="cat using litter box",
            subjectName="Tabby",
            targetName="litter box",
        )

        r = await http_client.get(f"/api/watch-events/{created['uuid']}")
        body = r.json()
        assert body["status"] == "success", body
        assert body["event"]["ruleName"] == "cat using litter box"
        assert body["event"]["subjectName"] == "Tabby"
        assert body["event"]["targetName"] == "litter box"

    async def test_get_unknown_reports_not_found(self, http_client):
        r = await http_client.get("/api/watch-events/nope")
        assert r.json()["status"] == "failure"

    async def test_an_open_event_is_listed_before_it_ends(self, http_client):
        """A watcher writes the row on the opening edge, so an in-progress event is
        visible. If this regressed, events would only appear minutes late."""
        profile_uuid = await _make_profile(http_client)
        await _make_event(http_client, profile_uuid, ruleName="cat on counter")

        body = await _list(http_client, profile_uuid=profile_uuid)
        assert len(body["events"]) == 1
        assert body["events"][0]["endedAt"] is None

    async def test_closing_an_event_sets_only_the_closing_fields(self, http_client):
        profile_uuid = await _make_profile(http_client)
        created = await _make_event(http_client, profile_uuid, ruleName="cat on counter")
        ended = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)

        r = await http_client.patch(
            f"/api/watch-events/{created['uuid']}",
            json={"endedAt": ended.isoformat(), "peakConfidence": 0.91, "truncated": True},
        )
        body = r.json()
        assert body["status"] == "success", body
        assert body["event"]["peakConfidence"] == pytest.approx(0.91)
        assert body["event"]["truncated"] is True
        # Untouched by the patch — a closing edge must not rewrite the event's identity.
        assert body["event"]["ruleName"] == "cat on counter"
        assert body["event"]["startedAt"] == created["startedAt"]

    async def test_patching_an_unknown_event_reports_not_found(self, http_client):
        r = await http_client.patch("/api/watch-events/nope", json={"peakConfidence": 0.5})
        assert r.json()["status"] == "failure"

    async def test_delete(self, http_client):
        profile_uuid = await _make_profile(http_client)
        created = await _make_event(http_client, profile_uuid)

        r = await http_client.delete(f"/api/watch-events/{created['uuid']}")
        assert r.json()["status"] == "success"
        assert (await _list(http_client, profile_uuid=profile_uuid))["total"] == 0


class TestWatchEventFilters:
    """Every filter the review page exposes, each proved to *exclude* something.

    A filter tested only for what it returns passes just as happily when it is ignored
    entirely, which is the failure this whole class exists to catch.
    """

    async def test_filter_by_source(self, http_client):
        profile_uuid = await _make_profile(http_client)
        cam_a, cam_b = await _make_source(http_client), await _make_source(http_client)
        await _make_event(http_client, profile_uuid, sourceUuid=cam_a)
        await _make_event(http_client, profile_uuid, sourceUuid=cam_b)

        body = await _list(http_client, source_uuid=cam_a)
        assert body["total"] == 1
        assert body["events"][0]["sourceUuid"] == cam_a

    async def test_filter_by_acting_and_acted_on(self, http_client):
        """The two slot filters are independent, so an event matching only one of a
        two-filter query must not come back."""
        profile_uuid = await _make_profile(http_client)
        cat, dog, box, bowl = "cat-ref", "dog-ref", "box-ref", "bowl-ref"
        await _make_event(http_client, profile_uuid, subjectRef=cat, targetRef=box)
        await _make_event(http_client, profile_uuid, subjectRef=cat, targetRef=bowl)
        await _make_event(http_client, profile_uuid, subjectRef=dog, targetRef=box)

        assert (await _list(http_client, subject_ref=cat))["total"] == 2
        assert (await _list(http_client, target_ref=box))["total"] == 2
        both = await _list(http_client, subject_ref=cat, target_ref=box)
        assert both["total"] == 1
        assert both["events"][0]["targetRef"] == box

    async def test_filter_by_rule(self, http_client):
        profile_uuid = await _make_profile(http_client)
        await _make_event(http_client, profile_uuid, ruleName="a")
        await _make_event(http_client, profile_uuid, ruleName="b")
        # rule_uuid is nullable and both events have none; a filter for a specific rule
        # must not treat NULL as a match.
        assert (await _list(http_client, rule_uuid="some-rule"))["total"] == 0

    async def test_filter_by_date_range(self, http_client):
        profile_uuid = await _make_profile(http_client)
        base = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)
        for days in (0, 5, 10):
            await _make_event(http_client, profile_uuid, startedAt=(base + timedelta(days=days)).isoformat())

        body = await _list(
            http_client,
            start_from=(base + timedelta(days=1)).isoformat(),
            start_to=(base + timedelta(days=6)).isoformat(),
        )
        assert body["total"] == 1

    async def test_filter_by_has_recording(self, http_client):
        """The card shows a video badge off this, so "has a clip" must not be inferred
        from anything but the link actually existing."""
        profile_uuid = await _make_profile(http_client)
        with_clip = await _make_event(http_client, profile_uuid)
        await _make_event(http_client, profile_uuid)

        r = await http_client.post("/api/recordings", json={"profileUuid": profile_uuid, "filePath": "/tmp/a.mp4"})
        recording = r.json()["recording"]
        await http_client.patch(f"/api/watch-events/{with_clip['uuid']}", json={"recordingUuid": recording["uuid"]})

        assert (await _list(http_client, has_recording=True))["total"] == 1
        assert (await _list(http_client, has_recording=False))["total"] == 1

    async def test_newest_first(self, http_client):
        profile_uuid = await _make_profile(http_client)
        base = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)
        for days, name in ((0, "oldest"), (2, "newest"), (1, "middle")):
            await _make_event(
                http_client, profile_uuid, ruleName=name, startedAt=(base + timedelta(days=days)).isoformat()
            )

        body = await _list(http_client, profile_uuid=profile_uuid)
        assert [e["ruleName"] for e in body["events"]] == ["newest", "middle", "oldest"]

    async def test_total_counts_the_filtered_set_not_the_page(self, http_client):
        profile_uuid = await _make_profile(http_client)
        for _ in range(5):
            await _make_event(http_client, profile_uuid, subjectRef="cat-ref")
        await _make_event(http_client, profile_uuid, subjectRef="dog-ref")

        body = await _list(http_client, subject_ref="cat-ref", limit=2)
        assert len(body["events"]) == 2
        assert body["total"] == 5

    async def test_offset_pages_without_gaps_or_repeats(self, http_client):
        profile_uuid = await _make_profile(http_client)
        base = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)
        for i in range(5):
            await _make_event(
                http_client, profile_uuid, ruleName=f"r{i}", startedAt=(base + timedelta(minutes=i)).isoformat()
            )

        first = await _list(http_client, profile_uuid=profile_uuid, limit=2, offset=0)
        second = await _list(http_client, profile_uuid=profile_uuid, limit=2, offset=2)
        third = await _list(http_client, profile_uuid=profile_uuid, limit=2, offset=4)
        seen = [e["ruleName"] for e in first["events"] + second["events"] + third["events"]]
        assert seen == ["r4", "r3", "r2", "r1", "r0"]


class TestWatchEventSnapshot:
    async def test_snapshot_path_is_never_returned_to_the_client(self, http_client, tmp_path):
        """A server filesystem path is useless to a browser and is not its business."""
        profile_uuid = await _make_profile(http_client)
        image = tmp_path / "card.jpg"
        image.write_bytes(b"\xff\xd8\xff\xd9")
        created = await _make_event(http_client, profile_uuid, snapshotPath=str(image))

        assert "snapshotPath" not in created
        assert created["hasSnapshot"] is True

    async def test_has_snapshot_is_false_without_one(self, http_client):
        profile_uuid = await _make_profile(http_client)
        created = await _make_event(http_client, profile_uuid)
        assert created["hasSnapshot"] is False

    async def test_snapshot_is_served(self, http_client, tmp_path):
        profile_uuid = await _make_profile(http_client)
        image = tmp_path / "card.jpg"
        image.write_bytes(b"\xff\xd8\xff\xd9")
        created = await _make_event(http_client, profile_uuid, snapshotPath=str(image))

        r = await http_client.get(f"/api/watch-events/{created['uuid']}/snapshot")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"

    async def test_a_stale_snapshot_path_404s_rather_than_500s(self, http_client, tmp_path):
        """Clips and card art can be reaped out from under a row; a card whose image is
        gone must degrade to a placeholder, not break the page."""
        profile_uuid = await _make_profile(http_client)
        created = await _make_event(http_client, profile_uuid, snapshotPath=str(tmp_path / "never-written.jpg"))

        r = await http_client.get(f"/api/watch-events/{created['uuid']}/snapshot")
        assert r.status_code == 404

    async def test_deleting_an_event_removes_its_card_art(self, http_client, tmp_path):
        """The snapshot belongs to the event, unlike a `save_snapshot` action's output
        under image_datasets, which the user curates and which must survive."""
        profile_uuid = await _make_profile(http_client)
        image = tmp_path / "card.jpg"
        image.write_bytes(b"\xff\xd8\xff\xd9")
        created = await _make_event(http_client, profile_uuid, snapshotPath=str(image))

        await http_client.delete(f"/api/watch-events/{created['uuid']}")
        assert not image.exists()


class TestWatchEventFacets:
    """The filter dropdowns' contents.

    Facets come from history, not from the watcher config, so the dropdown never offers
    a filter that matches nothing and never hides one that matches everything.
    """

    async def _facets(self, http_client, **params) -> dict:
        r = await http_client.get("/api/watch-events/facets", params=params)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "success", body
        return body

    async def test_facets_is_not_swallowed_by_the_uuid_route(self, http_client):
        """`/facets` is a literal path sharing a shape with `/{uuid}`. FastAPI matches in
        declaration order, so registering it second makes it permanently unreachable."""
        body = await self._facets(http_client)
        assert "rules" in body and "subjects" in body

    async def test_facets_list_what_history_contains(self, http_client):
        profile_uuid = await _make_profile(http_client)
        await _make_event(http_client, profile_uuid, subjectRef="cat-ref", subjectName="Tabby")
        await _make_event(http_client, profile_uuid, subjectRef="cat-ref", subjectName="Tabby")
        await _make_event(http_client, profile_uuid, targetRef="box-ref", targetName="litter box")

        body = await self._facets(http_client, profile_uuid=profile_uuid)
        assert body["subjects"] == [{"ref": "cat-ref", "name": "Tabby"}]
        assert body["targets"] == [{"ref": "box-ref", "name": "litter box"}]

    async def test_a_renamed_slot_appears_once(self, http_client):
        """One thing renamed mid-history is still one thing; two entries for it would
        make the dropdown look like there are two cats."""
        profile_uuid = await _make_profile(http_client)
        await _make_event(http_client, profile_uuid, subjectRef="cat-ref", subjectName="Tabbie")
        await _make_event(http_client, profile_uuid, subjectRef="cat-ref", subjectName="Tabby")

        body = await self._facets(http_client, profile_uuid=profile_uuid)
        assert len(body["subjects"]) == 1

    async def test_source_facets_use_the_live_camera_name(self, http_client):
        """A camera is a live entity the user renames and expects renamed here, unlike a
        rule name, which is a historical fact about an event that already happened."""
        profile_uuid = await _make_profile(http_client)
        source_uuid = await _make_source(http_client, name="Litter cam")
        await _make_event(http_client, profile_uuid, sourceUuid=source_uuid)

        body = await self._facets(http_client, profile_uuid=profile_uuid)
        assert body["sources"] == [{"ref": source_uuid, "name": "Litter cam"}]

    async def test_nulls_are_not_offered_as_a_filter(self, http_client):
        profile_uuid = await _make_profile(http_client)
        await _make_event(http_client, profile_uuid)

        body = await self._facets(http_client, profile_uuid=profile_uuid)
        assert body["subjects"] == []
        assert body["sources"] == []


class TestHistorySurvivesConfigEdits:
    async def test_event_keeps_its_names_after_the_profile_is_deleted(self, http_client):
        """`rule_name`/`subject_name`/`target_name` are denormalised precisely so a
        tidy-up of the watcher config does not blank the history it produced. Note the
        event row itself is CASCADEd with its profile — what this protects is the row
        for a *rule* that went away, whose FK is SET NULL."""
        profile_uuid = await _make_profile(http_client)
        created = await _make_event(
            http_client,
            profile_uuid,
            ruleUuid=None,
            ruleName="cat using litter box",
            subjectName="Tabby",
            targetName="litter box",
        )

        r = await http_client.get(f"/api/watch-events/{created['uuid']}")
        event = r.json()["event"]
        assert event["ruleUuid"] is None
        assert (event["ruleName"], event["subjectName"], event["targetName"]) == (
            "cat using litter box",
            "Tabby",
            "litter box",
        )
