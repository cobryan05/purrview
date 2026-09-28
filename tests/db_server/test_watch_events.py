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


async def _close_event(http_client, uuid: str, **fields) -> dict:
    payload = {"endedAt": datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc).isoformat()}
    payload.update(fields)
    r = await http_client.patch(f"/api/watch-events/{uuid}", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["event"]


async def _make_closed_event(http_client, profile_uuid: str, **overrides) -> dict:
    """An event that has run its course — the only kind the lifetime tally counts.

    The tally is owed at the closing edge, not at creation, because an open event's
    subject is still the watcher's running opinion and is revised when it closes.
    """
    event = await _make_event(http_client, profile_uuid, **overrides)
    await _close_event(http_client, event["uuid"])
    return event


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


class TestReviewStateFlagging:
    """`review_state` writability, added for the Home Assistant "labels are wrong"
    action (WP-1). The conditional is the point: flagging has to both show up under
    its own filter and disappear from the old one, not just "save".
    """

    async def test_flagging_is_filterable_and_excludes_the_old_state(self, http_client):
        profile_uuid = await _make_profile(http_client)
        flagged = await _make_event(http_client, profile_uuid, ruleName="flagged one")
        await _make_event(http_client, profile_uuid, ruleName="left alone")

        r = await http_client.patch(f"/api/watch-events/{flagged['uuid']}", json={"reviewState": "flagged"})
        body = r.json()
        assert body["status"] == "success", body
        assert body["event"]["reviewState"] == "flagged"

        flagged_list = await _list(http_client, review_state="flagged")
        assert {e["uuid"] for e in flagged_list["events"]} == {flagged["uuid"]}

        unreviewed_list = await _list(http_client, review_state="unreviewed")
        assert flagged["uuid"] not in {e["uuid"] for e in unreviewed_list["events"]}

    async def test_unknown_review_state_is_rejected_and_leaves_value_unchanged(self, http_client):
        profile_uuid = await _make_profile(http_client)
        event = await _make_event(http_client, profile_uuid)

        r = await http_client.patch(f"/api/watch-events/{event['uuid']}", json={"reviewState": "bogus"})
        assert r.status_code == 422

        r = await http_client.get(f"/api/watch-events/{event['uuid']}")
        assert r.json()["event"]["reviewState"] == "unreviewed"


class TestPinningViaTheApi:
    """The HTTP surface of `WatchEvent.pinned` — the db-level sweeps/delete-refusal
    behavior itself is covered against the manager in
    tests/db_server/test_recording_retention.py::TestPinnedEvents; this is the
    request-validation layer those tests never touch.
    """

    async def test_pinning_is_filterable_and_toggles_back(self, http_client):
        profile_uuid = await _make_profile(http_client)
        pinned = await _make_event(http_client, profile_uuid, ruleName="pinned one")
        await _make_event(http_client, profile_uuid, ruleName="left alone")

        r = await http_client.patch(f"/api/watch-events/{pinned['uuid']}", json={"pinned": True})
        body = r.json()
        assert body["status"] == "success", body
        assert body["event"]["pinned"] is True

        pinned_list = await _list(http_client, pinned=True)
        assert {e["uuid"] for e in pinned_list["events"]} == {pinned["uuid"]}

        r = await http_client.patch(f"/api/watch-events/{pinned['uuid']}", json={"pinned": False})
        assert r.json()["event"]["pinned"] is False
        assert (await _list(http_client, pinned=True))["total"] == 0

    async def test_explicit_null_is_rejected_not_a_500_or_a_silent_no_op(self, http_client):
        """Unlike `cropPath`/`snapshotPath`, `pinned` has no "clear this" meaning —
        the column is NOT NULL — so this must 422 before touching the row, the same
        shape as an unknown `reviewState`."""
        profile_uuid = await _make_profile(http_client)
        event = await _make_event(http_client, profile_uuid)

        r = await http_client.patch(f"/api/watch-events/{event['uuid']}", json={"pinned": None})
        assert r.status_code == 422

        r = await http_client.get(f"/api/watch-events/{event['uuid']}")
        assert r.json()["event"]["pinned"] is False

    async def test_bulk_endpoint_pins_and_unpins(self, http_client):
        profile_uuid = await _make_profile(http_client)
        a = await _make_event(http_client, profile_uuid, ruleName="a")
        b = await _make_event(http_client, profile_uuid, ruleName="b")

        r = await http_client.post("/api/watch-events/pinned", json={"uuids": [a["uuid"], b["uuid"]]})
        body = r.json()
        assert body["status"] == "success", body
        assert body["updated"] == 2
        assert (await _list(http_client, pinned=True))["total"] == 2

        r = await http_client.post("/api/watch-events/pinned", json={"uuids": [a["uuid"]], "pinned": False})
        assert r.json()["updated"] == 1
        pinned_list = await _list(http_client, pinned=True)
        assert {e["uuid"] for e in pinned_list["events"]} == {b["uuid"]}

    async def test_deleting_a_pinned_event_is_refused(self, http_client):
        profile_uuid = await _make_profile(http_client)
        event = await _make_event(http_client, profile_uuid)
        await http_client.patch(f"/api/watch-events/{event['uuid']}", json={"pinned": True})

        r = await http_client.delete(f"/api/watch-events/{event['uuid']}")
        assert r.json()["status"] == "failure"
        assert (await http_client.get(f"/api/watch-events/{event['uuid']}")).json()["status"] == "success"


class TestCorrectingTheWatchersVerdict:
    """The event's subject/target are the watcher's *opinion*, and a human may overrule it.

    Everything else the closing edge writes is a fact about what ran. These two are the
    guess a reviewer opens the event to disagree with — "Maple drinks water" when it was
    Luna — and a correction that only rewrote the training frames left every list, group
    and filter still answering with the mistake.
    """

    async def test_a_corrected_subject_moves_between_the_filters(self, http_client):
        profile_uuid = await _make_profile(http_client)
        event = await _make_event(
            http_client,
            profile_uuid,
            ruleName="cat drinks water",
            subjectRef="maple-ref",
            subjectName="Maple",
        )

        r = await http_client.patch(
            f"/api/watch-events/{event['uuid']}",
            json={"subjectRef": "luna-ref", "subjectName": "Luna"},
        )
        body = r.json()
        assert body["status"] == "success", body
        assert body["event"]["subjectName"] == "Luna"
        # The facts about what ran are not the opinion being corrected.
        assert body["event"]["ruleName"] == "cat drinks water"

        # The half that makes it a correction rather than a cosmetic edit: the row has
        # to answer the *new* filter and stop answering the old one.
        assert {e["uuid"] for e in (await _list(http_client, subject_refs=["luna-ref"]))["events"]} == {event["uuid"]}
        assert event["uuid"] not in {
            e["uuid"] for e in (await _list(http_client, subject_refs=["maple-ref"]))["events"]
        }

    async def test_a_corrected_target_is_persisted(self, http_client):
        profile_uuid = await _make_profile(http_client)
        event = await _make_event(http_client, profile_uuid, targetRef="bowl-ref", targetName="water bowl")

        r = await http_client.patch(
            f"/api/watch-events/{event['uuid']}",
            json={"targetRef": "fountain-ref", "targetName": "fountain"},
        )
        assert r.json()["status"] == "success", r.text

        r = await http_client.get(f"/api/watch-events/{event['uuid']}")
        assert r.json()["event"]["targetRef"] == "fountain-ref"
        assert r.json()["event"]["targetName"] == "fountain"


class TestBulkDelete:
    """The selection delete behind the event grid's "Delete N events".

    The properties that matter are the ones a partial failure would quietly break: that
    the *whole* selection goes, that nothing outside it does, and that the count coming
    back describes what actually happened rather than what was asked for.
    """

    async def test_deletes_the_whole_selection_and_only_it(self, http_client):
        profile_uuid = await _make_profile(http_client)
        doomed = [(await _make_event(http_client, profile_uuid, ruleName=f"e{i}"))["uuid"] for i in range(3)]
        kept = await _make_event(http_client, profile_uuid, ruleName="keep")

        r = await http_client.post("/api/watch-events/delete", json={"uuids": doomed})
        body = r.json()
        assert body["status"] == "success", body
        assert body["deleted"] == 3

        remaining = await _list(http_client, profile_uuid=profile_uuid)
        assert [e["uuid"] for e in remaining["events"]] == [kept["uuid"]]

    async def test_a_uuid_that_is_already_gone_does_not_abandon_the_rest(self, http_client):
        """Deleting the same selection from two tabs is the ordinary way to hit this.

        If a missing uuid raised, the events after it in the list would survive — an
        arbitrary half-deleted selection, reported to the user as a failure.
        """
        profile_uuid = await _make_profile(http_client)
        first = await _make_event(http_client, profile_uuid, ruleName="first")
        second = await _make_event(http_client, profile_uuid, ruleName="second")

        r = await http_client.post(
            "/api/watch-events/delete",
            json={"uuids": [first["uuid"], "no-such-event", second["uuid"]]},
        )
        body = r.json()
        assert body["status"] == "success", body
        # Two of the three asked for — the count reports reality, not the request.
        assert body["deleted"] == 2
        assert (await _list(http_client, profile_uuid=profile_uuid))["total"] == 0

    async def test_an_empty_selection_is_a_no_op(self, http_client):
        profile_uuid = await _make_profile(http_client)
        await _make_event(http_client, profile_uuid)

        r = await http_client.post("/api/watch-events/delete", json={"uuids": []})
        assert r.json()["deleted"] == 0
        assert (await _list(http_client, profile_uuid=profile_uuid))["total"] == 1


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
        """The card shows a video badge off this, so "has media" must not be inferred
        from anything but a recording actually naming the event."""
        profile_uuid = await _make_profile(http_client)
        with_clip = await _make_event(http_client, profile_uuid)
        await _make_event(http_client, profile_uuid)

        await http_client.post(
            "/api/recordings",
            json={"profileUuid": profile_uuid, "eventUuid": with_clip["uuid"], "filePath": "/tmp/a.mp4"},
        )

        assert (await _list(http_client, has_recording=True))["total"] == 1
        assert (await _list(http_client, has_recording=False))["total"] == 1

    async def test_filter_by_recording_kind(self, http_client):
        """ "Has a timelapse" is a different question from "has video".

        An event can own a clip, a timelapse, both or neither, so the kind filter has to
        select on the kind rather than on the presence of any media — the failure it
        guards against is a clip-only event turning up under "has a timelapse".
        """
        profile_uuid = await _make_profile(http_client)
        clip_only = await _make_event(http_client, profile_uuid)
        lapse_only = await _make_event(http_client, profile_uuid)
        both = await _make_event(http_client, profile_uuid)
        await _make_event(http_client, profile_uuid)  # no media at all

        for event, kinds in ((clip_only, ["clip"]), (lapse_only, ["timelapse"]), (both, ["clip", "timelapse"])):
            for kind in kinds:
                await http_client.post(
                    "/api/recordings",
                    json={
                        "profileUuid": profile_uuid,
                        "eventUuid": event["uuid"],
                        "kind": kind,
                        "filePath": f"/tmp/{event['uuid']}-{kind}.mp4",
                    },
                )

        clips = await _list(http_client, recording_kind="clip")
        assert {e["uuid"] for e in clips["events"]} == {clip_only["uuid"], both["uuid"]}
        lapses = await _list(http_client, recording_kind="timelapse")
        assert {e["uuid"] for e in lapses["events"]} == {lapse_only["uuid"], both["uuid"]}
        # The event owning both is one row in each result, not two — same EXISTS
        # reasoning as `has_recording`, which a join would break.
        assert clips["total"] == 2 and lapses["total"] == 2

    async def test_an_event_owning_two_recordings_is_counted_once(self, http_client):
        """The filter is an EXISTS, not a join.

        A join against a one-to-many is the obvious way to write this and would count
        an event with both a clip and a timelapse twice, making "N events" disagree with
        the number of cards on the page.
        """
        profile_uuid = await _make_profile(http_client)
        event = await _make_event(http_client, profile_uuid)
        for kind in ("clip", "timelapse"):
            await http_client.post(
                "/api/recordings",
                json={
                    "profileUuid": profile_uuid,
                    "eventUuid": event["uuid"],
                    "kind": kind,
                    "filePath": f"/tmp/{kind}.mp4",
                },
            )

        listed = await _list(http_client, has_recording=True)
        assert listed["total"] == 1
        assert len(listed["events"]) == 1

    async def test_events_carry_every_recording_that_names_them(self, http_client):
        """Both kinds reach the card, so it can badge each one."""
        profile_uuid = await _make_profile(http_client)
        event = await _make_event(http_client, profile_uuid)
        for kind in ("timelapse", "clip"):
            await http_client.post(
                "/api/recordings",
                json={
                    "profileUuid": profile_uuid,
                    "eventUuid": event["uuid"],
                    "kind": kind,
                    "durationS": 4.0,
                    "filePath": f"/tmp/{kind}.mp4",
                },
            )

        listed = await _list(http_client, has_recording=True)
        recordings = listed["events"][0]["recordings"]
        assert {r["kind"] for r in recordings} == {"clip", "timelapse"}
        assert all(r["durationS"] == 4.0 for r in recordings)

    async def test_an_event_with_no_recordings_reports_an_empty_list(self, http_client):
        """Not null. The card does `.find` on it, and a null takes the whole grid down."""
        profile_uuid = await _make_profile(http_client)
        await _make_event(http_client, profile_uuid)
        assert (await _list(http_client))["events"][0]["recordings"] == []

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


class TestWatchEventCounts:
    """Lifetime (source, rule) counts — MQTT bridge counter seeding (WP-2)."""

    async def _counts(self, http_client) -> list:
        r = await http_client.get("/api/watch-events/counts")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "success", body
        return body["counts"]

    async def test_counts_group_by_source_and_rule_not_just_source(self, http_client):
        """Two rules on the same camera must not be summed into one count — a bridge
        seeding per-rule sensors from a source-only total would overstate every rule."""
        profile_uuid = await _make_profile(http_client)
        cam = await _make_source(http_client)
        rule_a, rule_b = "rule-a", "rule-b"
        for _ in range(3):
            await _make_closed_event(http_client, profile_uuid, sourceUuid=cam, ruleUuid=rule_a)
        await _make_closed_event(http_client, profile_uuid, sourceUuid=cam, ruleUuid=rule_b)

        counts = {(c["sourceUuid"], c["ruleUuid"]): c["count"] for c in await self._counts(http_client)}
        assert counts[(cam, rule_a)] == 3
        assert counts[(cam, rule_b)] == 1

    async def test_events_missing_a_rule_uuid_are_not_counted(self, http_client):
        """An event with no rule (nothing to attribute the count to) must not show up as
        a phantom `null`-keyed group."""
        profile_uuid = await _make_profile(http_client)
        cam = await _make_source(http_client)
        await _make_event(http_client, profile_uuid, sourceUuid=cam)

        assert await self._counts(http_client) == []


class TestHistorySurvivesConfigEdits:
    async def test_event_keeps_its_names_after_the_profile_is_deleted(self, http_client):
        """`rule_name`/`subject_name`/`target_name` are denormalised precisely so a
        tidy-up of the watcher config does not blank the history it produced. Note the
        event row itself is CASCADEd with its profile — what this protects is the row
        for a *rule* that went away."""
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

    async def test_saving_the_profile_does_not_orphan_the_events_its_rules_fired(self, http_client):
        """Saving a profile replaces its nested lists wholesale — every `InteractionRule`
        is deleted and re-inserted carrying its original uuid. While `watch_events.rule_uuid`
        was a foreign key with `ON DELETE SET NULL`, that delete cleared the pointer on
        every event the rule had ever produced, and the re-insert did not put it back.

        The damage was invisible until someone used the Event type filter, which then
        matched only events fired since the last save and hid all the older ones — the
        rest of the page still rendered fine off the denormalised `rule_name`.
        """
        rule_uuid = "33333333-3333-3333-3333-333333333333"
        rule = {
            "uuid": rule_uuid,
            "order": 0,
            "name": "cat using litter box",
            "enabled": True,
            "subjectAKind": "label",
            "subjectARefsJson": ["label-cat"],
            "relation": "contains",
            "subjectBKind": "label",
            "subjectBRefsJson": ["label-litter-box"],
            "threshold": 0.6,
        }
        profile_uuid = await _make_profile(http_client)
        r = await http_client.patch(f"/api/watch-profiles/{profile_uuid}", json={"rules": [rule]})
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "success", r.text

        created = await _make_event(http_client, profile_uuid, ruleUuid=rule_uuid, ruleName=rule["name"])

        # An ordinary edit that leaves the rule itself alone — the user nudged a threshold.
        r = await http_client.patch(
            f"/api/watch-profiles/{profile_uuid}",
            json={"rules": [{**rule, "threshold": 0.7}]},
        )
        assert r.status_code == 200, r.text

        r = await http_client.get(f"/api/watch-events/{created['uuid']}")
        assert r.json()["event"]["ruleUuid"] == rule_uuid

        # The two symptoms the orphaning actually produced, rather than just the column:
        # the filter stops finding the event, and the dropdown stops offering the rule.
        body = await _list(http_client, profile_uuid=profile_uuid, rule_uuid=rule_uuid)
        assert [e["uuid"] for e in body["events"]] == [created["uuid"]]

        r = await http_client.get("/api/watch-events/facets", params={"profile_uuid": profile_uuid})
        assert [f["ref"] for f in r.json()["rules"]] == [rule_uuid]


class TestClosingStrandedEvents:
    """An event a killed watcher left open would otherwise say "in progress" forever.

    Nothing in the system closed one before this: a clean stop closes its own events, a
    killed process cannot, and no sweep looked for the leftovers. The row is not merely
    cosmetic — an open event has no duration and sorts as current, so one stuck row sits
    at the top of the list indefinitely.
    """

    async def _close(self, http_client, profile_uuid: str) -> dict:
        r = await http_client.post("/api/watch-events/close-stranded", json={"profileUuid": profile_uuid})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "success", body
        return body

    async def test_an_open_event_is_closed_at_its_own_start(self, http_client):
        """Not at `now`: the true end is unknown, and `now` would claim a visit that
        lasted as long as the watcher was down."""
        profile = await _make_profile(http_client)
        started = datetime(2026, 8, 19, 9, 30, tzinfo=timezone.utc)
        event = await _make_event(http_client, profile, startedAt=started.isoformat())
        assert event["endedAt"] is None

        assert (await self._close(http_client, profile))["closed"] == 1

        row = (await _list(http_client, profileUuid=profile))["events"][0]
        assert row["endedAt"] is not None
        assert datetime.fromisoformat(row["endedAt"]) == datetime.fromisoformat(row["startedAt"])
        # So a card can say the watcher stopped rather than presenting the duration as
        # something that was measured.
        assert row["metadataJson"]["closedBy"] == "restart"

    async def test_a_closed_event_is_left_alone(self, http_client):
        profile = await _make_profile(http_client)
        ended = datetime(2026, 8, 19, 9, 45, tzinfo=timezone.utc)
        await _make_event(
            http_client,
            profile,
            startedAt=datetime(2026, 8, 19, 9, 30, tzinfo=timezone.utc).isoformat(),
            endedAt=ended.isoformat(),
        )

        assert (await self._close(http_client, profile))["closed"] == 0

        row = (await _list(http_client, profileUuid=profile))["events"][0]
        # Compared naive: these rows come back without a tz suffix, and normalising here
        # would be asserting the serialisation rather than that the row was left alone.
        assert datetime.fromisoformat(row["endedAt"]).replace(tzinfo=None) == ended.replace(tzinfo=None)
        assert "closedBy" not in row["metadataJson"]

    async def test_another_watchers_open_event_is_untouched(self, http_client):
        """A starting watcher can only speak for its own profile. Closing everything
        open would end the live events of every other watcher that is still running."""
        mine = await _make_profile(http_client)
        theirs = await _make_profile(http_client)
        await _make_event(http_client, mine)
        await _make_event(http_client, theirs)

        assert (await self._close(http_client, mine))["closed"] == 1

        still_open = (await _list(http_client, profileUuid=theirs))["events"][0]
        assert still_open["endedAt"] is None

    async def test_running_it_twice_closes_nothing_the_second_time(self, http_client):
        profile = await _make_profile(http_client)
        await _make_event(http_client, profile)
        assert (await self._close(http_client, profile))["closed"] == 1
        assert (await self._close(http_client, profile))["closed"] == 0


class TestWatchEventExclusionFilters:
    """The "not this one" half of each facet filter, plus the multi-value half.

    Both arrived together with the ≠ toggle on the Events page. An exclusion that
    quietly matches nothing, or one that also swallows the events with no value in that
    slot at all, looks on screen exactly like a correctly narrowed search.
    """

    async def test_excluding_a_camera_keeps_every_other_one(self, http_client):
        profile_uuid = await _make_profile(http_client)
        cam_a, cam_b = await _make_source(http_client), await _make_source(http_client)
        await _make_event(http_client, profile_uuid, sourceUuid=cam_a)
        await _make_event(http_client, profile_uuid, sourceUuid=cam_b)

        body = await _list(http_client, exclude_source_uuids=[cam_a])
        assert body["total"] == 1
        assert body["events"][0]["sourceUuid"] == cam_b

    async def test_excluding_a_subject_keeps_events_that_have_no_subject(self, http_client):
        """`subject_ref` is nullable — a rule with no subject slot still produces events.

        Plain `NOT IN` drops those rows, because SQL compares NULL to the list and gets
        NULL rather than true. "Not the cat" plainly includes an event with no subject
        at all, so the NULL case has to be handled explicitly.
        """
        profile_uuid = await _make_profile(http_client)
        await _make_event(http_client, profile_uuid, subjectRef="cat-ref", ruleName="cat")
        await _make_event(http_client, profile_uuid, subjectRef="dog-ref", ruleName="dog")
        await _make_event(http_client, profile_uuid, ruleName="no subject at all")

        body = await _list(http_client, exclude_subject_refs=["cat-ref"])
        assert sorted(e["ruleName"] for e in body["events"]) == ["dog", "no subject at all"]

    async def test_excluding_two_values_drops_both(self, http_client):
        profile_uuid = await _make_profile(http_client)
        for ref in ("cat-ref", "dog-ref", "fox-ref"):
            await _make_event(http_client, profile_uuid, subjectRef=ref)

        body = await _list(http_client, exclude_subject_refs=["cat-ref", "dog-ref"])
        assert [e["subjectRef"] for e in body["events"]] == ["fox-ref"]

    async def test_including_several_values_of_one_facet_widens(self, http_client):
        """Values within one facet are alternatives: an event has exactly one subject,
        so anding them would always return nothing."""
        profile_uuid = await _make_profile(http_client)
        for ref in ("cat-ref", "dog-ref", "fox-ref"):
            await _make_event(http_client, profile_uuid, subjectRef=ref)

        body = await _list(http_client, subject_refs=["cat-ref", "dog-ref"])
        assert sorted(e["subjectRef"] for e in body["events"]) == ["cat-ref", "dog-ref"]

    async def test_including_and_excluding_different_facets_compose(self, http_client):
        """Across facets the filters and together — excluding a camera must narrow the
        subject filter rather than replace it."""
        profile_uuid = await _make_profile(http_client)
        cam_a, cam_b = await _make_source(http_client), await _make_source(http_client)
        await _make_event(http_client, profile_uuid, subjectRef="cat-ref", sourceUuid=cam_a)
        await _make_event(http_client, profile_uuid, subjectRef="cat-ref", sourceUuid=cam_b)
        await _make_event(http_client, profile_uuid, subjectRef="dog-ref", sourceUuid=cam_b)

        body = await _list(http_client, subject_refs=["cat-ref"], exclude_source_uuids=[cam_a])
        assert body["total"] == 1
        assert body["events"][0]["sourceUuid"] == cam_b


async def _groups(http_client, **params) -> dict:
    r = await http_client.get("/api/watch-events/groups", params=params)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body


async def _histogram(http_client, **params) -> dict:
    r = await http_client.get("/api/watch-events/histogram", params=params)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body


class TestWatchEventGroups:
    """The grouped review view: 1277 events are 22 things that happened.

    The failure mode that matters here is not an error — it is a group card that says
    "383 events" and drills into a grid showing a different number, which is what
    happens the moment the two endpoints stop filtering identically.
    """

    async def test_events_of_one_identity_collapse_to_one_row(self, http_client):
        profile_uuid = await _make_profile(http_client)
        for _ in range(3):
            await _make_event(http_client, profile_uuid, ruleUuid="rule-a", subjectRef="cat", targetRef="bowl")
        await _make_event(http_client, profile_uuid, ruleUuid="rule-a", subjectRef="dog", targetRef="bowl")

        body = await _groups(http_client)
        assert len(body["groups"]) == 2
        assert [g["count"] for g in body["groups"]] == [3, 1], "biggest group must lead"
        assert body["total"] == 4

    async def test_group_counts_sum_to_the_grid_total_under_the_same_filter(self, http_client):
        """The drift detector for the shared filter predicate.

        Both endpoints assemble their WHERE from `_apply_watch_event_filters`. If one
        ever grows a condition the other lacks, the counts diverge — and a group card
        promising 3 events that opens onto 5 is a wrong answer, not a crash.
        """
        profile_uuid = await _make_profile(http_client)
        cam_a, cam_b = await _make_source(http_client), await _make_source(http_client)
        for cam in (cam_a, cam_a, cam_b):
            await _make_event(http_client, profile_uuid, ruleUuid="rule-a", subjectRef="cat", sourceUuid=cam)
        await _make_event(http_client, profile_uuid, ruleUuid="rule-b", subjectRef="dog", sourceUuid=cam_b)

        for params in (
            {},
            {"source_uuids": [cam_a]},
            {"exclude_source_uuids": [cam_a]},
            {"subject_refs": ["cat"]},
            {"rule_uuids": ["rule-a"], "exclude_subject_refs": ["dog"]},
        ):
            grouped = await _groups(http_client, **params)
            listed = await _list(http_client, **params)
            assert sum(g["count"] for g in grouped["groups"]) == listed["total"], params
            assert grouped["total"] == listed["total"], params

    async def test_an_identity_with_no_subject_or_target_is_its_own_group(self, http_client):
        """A rule with no slots — the live history has one — is a real identity.

        NULL is not a missing value to be dropped here: grouping that silently discarded
        the row would lose the events entirely, and the total would stop matching the
        grid's.
        """
        profile_uuid = await _make_profile(http_client)
        await _make_event(http_client, profile_uuid, ruleUuid="motion")
        await _make_event(http_client, profile_uuid, ruleUuid="motion")
        await _make_event(http_client, profile_uuid, ruleUuid="motion", subjectRef="cat")

        body = await _groups(http_client)
        bare = [g for g in body["groups"] if g["subjectRef"] is None]
        assert len(bare) == 1
        assert bare[0]["count"] == 2
        assert body["total"] == 3

    async def test_the_representative_is_the_newest_event_that_has_a_snapshot(self, http_client):
        """Not simply the newest.

        Card art comes from the representative, so a group whose newest event happens to
        lack a snapshot would render as an empty tile — with nothing on screen to say
        why, since the other events in it have pictures.
        """
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        profile_uuid = await _make_profile(http_client)
        older_with_art = await _make_event(
            http_client,
            profile_uuid,
            ruleUuid="rule-a",
            subjectRef="cat",
            startedAt=base.isoformat(),
            snapshotPath="/data/recordings/event_snapshots/a.jpg",
        )
        await _make_event(
            http_client,
            profile_uuid,
            ruleUuid="rule-a",
            subjectRef="cat",
            startedAt=(base + timedelta(hours=1)).isoformat(),
        )

        body = await _groups(http_client)
        assert body["groups"][0]["representativeUuid"] == older_with_art["uuid"]

    async def test_a_group_with_no_snapshots_at_all_still_names_a_representative(self, http_client):
        """The fallback. Returning nothing would make the card art unreachable even for
        an event that later gains a snapshot."""
        profile_uuid = await _make_profile(http_client)
        event = await _make_event(http_client, profile_uuid, ruleUuid="rule-a")

        body = await _groups(http_client)
        assert body["groups"][0]["representativeUuid"] == event["uuid"]

    async def test_an_event_owning_two_recordings_counts_once_toward_with_recording(self, http_client):
        """`withRecording` must never exceed the group's own count.

        A join would count an event owning both a clip and a timelapse twice, and
        "4 of 3 have video" is the kind of number nobody can explain later.
        """
        profile_uuid = await _make_profile(http_client)
        event = await _make_event(http_client, profile_uuid, ruleUuid="rule-a")
        for kind in ("clip", "timelapse"):
            r = await http_client.post(
                "/api/recordings",
                json={
                    "eventUuid": event["uuid"],
                    "kind": kind,
                    "path": f"/data/recordings/{kind}.mp4",
                    "durationS": 4.0,
                },
            )
            assert r.status_code == 200, r.text

        body = await _groups(http_client)
        assert body["groups"][0]["count"] == 1
        assert body["groups"][0]["withRecording"] == 1

    async def test_total_duration_ignores_open_events(self, http_client):
        """An open event has no duration yet. Counting its start as an end would report
        a measurement where there is only a guess — the same refusal the card makes."""
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        profile_uuid = await _make_profile(http_client)
        await _make_event(
            http_client,
            profile_uuid,
            ruleUuid="rule-a",
            startedAt=base.isoformat(),
            endedAt=(base + timedelta(seconds=30)).isoformat(),
        )
        await _make_event(
            http_client, profile_uuid, ruleUuid="rule-a", startedAt=(base + timedelta(hours=2)).isoformat()
        )

        body = await _groups(http_client)
        assert body["groups"][0]["count"] == 2
        assert body["groups"][0]["totalDurationS"] == pytest.approx(30.0, abs=0.5)


class TestWatchEventHistogram:
    """The timeline's data source."""

    async def test_events_in_one_hour_share_a_bucket_and_events_in_two_do_not(self, http_client):
        base = datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc)
        profile_uuid = await _make_profile(http_client)
        for minutes in (0, 20, 50):
            await _make_event(
                http_client,
                profile_uuid,
                ruleUuid="rule-a",
                startedAt=(base + timedelta(minutes=minutes)).isoformat(),
            )
        await _make_event(
            http_client, profile_uuid, ruleUuid="rule-a", startedAt=(base + timedelta(hours=1)).isoformat()
        )

        body = await _histogram(http_client, bucket="hour")
        assert body["bucket"] == "hour"
        by_start = {b["bucketStart"]: b["count"] for b in body["buckets"]}
        assert by_start == {"2026-01-01T14:00:00": 3, "2026-01-01T15:00:00": 1}

    async def test_buckets_are_split_by_identity(self, http_client):
        """One lane per identity is the whole shape of the timeline; a histogram summed
        across identities could not draw it."""
        base = datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc)
        profile_uuid = await _make_profile(http_client)
        await _make_event(http_client, profile_uuid, ruleUuid="rule-a", startedAt=base.isoformat())
        await _make_event(http_client, profile_uuid, ruleUuid="rule-b", startedAt=base.isoformat())

        body = await _histogram(http_client, bucket="hour")
        assert len(body["buckets"]) == 2
        assert {b["ruleUuid"] for b in body["buckets"]} == {"rule-a", "rule-b"}

    async def test_the_histogram_respects_the_same_filters_as_the_grid(self, http_client):
        profile_uuid = await _make_profile(http_client)
        cam_a, cam_b = await _make_source(http_client), await _make_source(http_client)
        await _make_event(http_client, profile_uuid, ruleUuid="rule-a", sourceUuid=cam_a)
        await _make_event(http_client, profile_uuid, ruleUuid="rule-a", sourceUuid=cam_b)

        body = await _histogram(http_client, source_uuids=[cam_a])
        assert sum(b["count"] for b in body["buckets"]) == (await _list(http_client, source_uuids=[cam_a]))["total"]

    async def test_an_oversized_hour_request_falls_back_to_days_without_losing_events(
        self, http_client, db_client, monkeypatch
    ):
        """The overflow answer coarsens the resolution and *says so*.

        Truncating instead would silently misreport when things happened, which is worse
        than a coarse picture — and the flag is what lets the UI tell the user the marks
        are whole days rather than hours.
        """
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        profile_uuid = await _make_profile(http_client)
        for hours in range(6):
            await _make_event(
                http_client, profile_uuid, ruleUuid="rule-a", startedAt=(base + timedelta(hours=hours)).isoformat()
            )
        monkeypatch.setattr(type(db_client), "HISTOGRAM_ROW_LIMIT", 2)

        body = await _histogram(http_client, bucket="hour")
        assert body["bucket"] == "day", "the caller must be told the resolution changed"
        assert sum(b["count"] for b in body["buckets"]) == 6, "no event may be dropped by the fallback"
        assert [b["bucketStart"] for b in body["buckets"]] == ["2026-01-01T00:00:00"]


async def _sessions(http_client, **params) -> dict:
    r = await http_client.get("/api/watch-events/sessions", params=params)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body


class TestWatchEventSessions:
    """Consecutive events of one identity, folded into the visit they describe.

    The property that matters is that the fold is computed over the *whole* filtered set:
    doing it per page — which is what a browser-side grouping would be — splits every
    session that straddles a page boundary, and groups differently depending on which
    page you happen to land on.
    """

    async def test_a_burst_of_events_becomes_one_visit(self, http_client):
        base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        profile_uuid = await _make_profile(http_client)
        # Three events 30s apart — one trip to the bowl, re-firing as the rule rearms.
        for i in range(3):
            start = base + timedelta(seconds=i * 30)
            await _make_event(
                http_client,
                profile_uuid,
                ruleUuid="rule-a",
                subjectRef="cat",
                startedAt=start.isoformat(),
                endedAt=(start + timedelta(seconds=10)).isoformat(),
            )

        body = await _sessions(http_client, gap_s=120)
        assert body["total"] == 1
        assert body["sessions"][0]["count"] == 3

    async def test_a_long_gap_starts_a_new_visit(self, http_client):
        """The whole point of the threshold. Two trips an hour apart are two visits,
        however identical the events are."""
        base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        profile_uuid = await _make_profile(http_client)
        for offset_min in (0, 60):
            start = base + timedelta(minutes=offset_min)
            await _make_event(
                http_client,
                profile_uuid,
                ruleUuid="rule-a",
                subjectRef="cat",
                startedAt=start.isoformat(),
                endedAt=(start + timedelta(seconds=10)).isoformat(),
            )

        assert (await _sessions(http_client, gap_s=120))["total"] == 2
        # Widen the threshold past the gap and the same two events become one visit.
        assert (await _sessions(http_client, gap_s=7200))["total"] == 1

    async def test_the_gap_is_measured_from_the_end_not_the_start(self, http_client):
        """A five-minute event followed 30s later by another is one visit.

        Measuring start-to-start would call them two, because the *starts* are five and
        a half minutes apart — and a long event would split from its own continuation.
        """
        base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        profile_uuid = await _make_profile(http_client)
        await _make_event(
            http_client,
            profile_uuid,
            ruleUuid="rule-a",
            subjectRef="cat",
            startedAt=base.isoformat(),
            endedAt=(base + timedelta(minutes=5)).isoformat(),
        )
        resume = base + timedelta(minutes=5, seconds=30)
        await _make_event(
            http_client,
            profile_uuid,
            ruleUuid="rule-a",
            subjectRef="cat",
            startedAt=resume.isoformat(),
            endedAt=(resume + timedelta(seconds=10)).isoformat(),
        )

        body = await _sessions(http_client, gap_s=120)
        assert body["total"] == 1, "the second event resumed the first, 30s after it ended"
        assert body["sessions"][0]["count"] == 2

    async def test_two_identities_never_share_a_visit(self, http_client):
        """Interleaved in time, but a different cat is a different visit."""
        base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        profile_uuid = await _make_profile(http_client)
        for i, subject in enumerate(("cat", "dog", "cat", "dog")):
            start = base + timedelta(seconds=i * 20)
            await _make_event(
                http_client,
                profile_uuid,
                ruleUuid="rule-a",
                subjectRef=subject,
                startedAt=start.isoformat(),
                endedAt=(start + timedelta(seconds=5)).isoformat(),
            )

        body = await _sessions(http_client, gap_s=120)
        assert body["total"] == 2
        assert {s["subjectRef"] for s in body["sessions"]} == {"cat", "dog"}
        assert all(s["count"] == 2 for s in body["sessions"])

    async def test_every_event_lands_in_exactly_one_visit(self, http_client):
        """The counts must reconcile with the grid, or the two views disagree about how
        much history there is."""
        base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        profile_uuid = await _make_profile(http_client)
        for i in range(12):
            start = base + timedelta(minutes=i * 7)
            await _make_event(
                http_client,
                profile_uuid,
                ruleUuid="rule-a",
                subjectRef="cat" if i % 2 else "dog",
                startedAt=start.isoformat(),
                endedAt=(start + timedelta(seconds=20)).isoformat(),
            )

        body = await _sessions(http_client, gap_s=600, limit=1000)
        assert sum(s["count"] for s in body["sessions"]) == (await _list(http_client))["total"]

    async def test_a_visit_is_not_split_across_a_page_boundary(self, http_client):
        """The reason this is server-side.

        Paging by session and grouping over the whole filtered set means a burst is one
        row wherever it falls; a client grouping the 60 events it received would cut the
        burst that straddled the edge into two half-visits.
        """
        base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        profile_uuid = await _make_profile(http_client)
        # Three bursts of three, far enough apart to be distinct visits.
        for burst in range(3):
            for i in range(3):
                start = base + timedelta(hours=burst, seconds=i * 20)
                await _make_event(
                    http_client,
                    profile_uuid,
                    ruleUuid="rule-a",
                    subjectRef="cat",
                    startedAt=start.isoformat(),
                    endedAt=(start + timedelta(seconds=5)).isoformat(),
                )

        first = await _sessions(http_client, gap_s=120, limit=2, offset=0)
        second = await _sessions(http_client, gap_s=120, limit=2, offset=2)
        assert first["total"] == 3
        assert [s["count"] for s in first["sessions"]] == [3, 3]
        assert [s["count"] for s in second["sessions"]] == [3]

    async def test_the_visit_spans_from_its_first_start_to_its_last_end(self, http_client):
        base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        profile_uuid = await _make_profile(http_client)
        for i in range(3):
            start = base + timedelta(seconds=i * 30)
            await _make_event(
                http_client,
                profile_uuid,
                ruleUuid="rule-a",
                subjectRef="cat",
                startedAt=start.isoformat(),
                endedAt=(start + timedelta(seconds=10)).isoformat(),
            )

        session = (await _sessions(http_client, gap_s=120))["sessions"][0]
        assert session["startedAt"].startswith("2026-01-01T12:00:00")
        assert session["endedAt"].startswith("2026-01-01T12:01:10")

    async def test_a_visit_still_running_says_so(self, http_client):
        """`endedAt` cannot answer this, which is why `isOpen` exists.

        A session always reports an extent, so an open event's missing end is coalesced
        to its start — and a visit of one open event therefore reports `startedAt ==
        endedAt`, indistinguishable from one that ended the instant it began. The card
        read that as a finished, zero-length visit: no "In progress" badge, and a
        drill-down window of zero width that resolved to no events at all.
        """
        base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        profile_uuid = await _make_profile(http_client)
        await _make_event(
            http_client,
            profile_uuid,
            ruleUuid="rule-a",
            subjectRef="cat",
            startedAt=base.isoformat(),
            endedAt=(base + timedelta(seconds=10)).isoformat(),
        )
        await _make_event(
            http_client,
            profile_uuid,
            ruleUuid="rule-a",
            subjectRef="dog",
            startedAt=base.isoformat(),
            endedAt=None,
        )

        sessions = {s["subjectRef"]: s for s in (await _sessions(http_client, gap_s=120))["sessions"]}
        assert sessions["dog"]["isOpen"] is True
        assert sessions["cat"]["isOpen"] is False
        # The coalesced end is still reported — the card needs an extent to draw — so
        # `isOpen` is the only thing separating the two.
        assert sessions["dog"]["endedAt"] == sessions["dog"]["startedAt"]

    async def test_one_open_event_does_not_make_a_closed_visit_open(self, http_client):
        """An event that closed after the open one must not clear the flag, and an open
        event anywhere in the visit must set it — the fold is over the whole session, not
        its last row."""
        base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        profile_uuid = await _make_profile(http_client)
        await _make_event(
            http_client,
            profile_uuid,
            ruleUuid="rule-a",
            subjectRef="cat",
            startedAt=base.isoformat(),
            endedAt=None,
        )
        resume = base + timedelta(seconds=30)
        await _make_event(
            http_client,
            profile_uuid,
            ruleUuid="rule-a",
            subjectRef="cat",
            startedAt=resume.isoformat(),
            endedAt=(resume + timedelta(seconds=10)).isoformat(),
        )

        body = await _sessions(http_client, gap_s=120)
        assert body["total"] == 1
        assert body["sessions"][0]["count"] == 2
        assert body["sessions"][0]["isOpen"] is True

    async def test_sessions_honour_the_same_filters_as_the_grid(self, http_client):
        profile_uuid = await _make_profile(http_client)
        cam_a, cam_b = await _make_source(http_client), await _make_source(http_client)
        for cam in (cam_a, cam_b):
            await _make_event(http_client, profile_uuid, ruleUuid="rule-a", sourceUuid=cam)

        body = await _sessions(http_client, gap_s=120, source_uuids=[cam_a], limit=1000)
        assert sum(s["count"] for s in body["sessions"]) == (await _list(http_client, source_uuids=[cam_a]))["total"]


class TestReviewStateTriage:
    """The column, the read model and the `review_state` filter all predate any way to
    write it — every event in live history reads `unreviewed` because nothing could."""

    async def test_marking_events_reviewed_makes_them_filterable(self, http_client):
        profile_uuid = await _make_profile(http_client)
        reviewed = await _make_event(http_client, profile_uuid, ruleName="seen")
        await _make_event(http_client, profile_uuid, ruleName="not seen")

        r = await http_client.post(
            "/api/watch-events/review-states",
            json={"uuids": [reviewed["uuid"]], "state": "confirmed"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["updated"] == 1

        assert [e["ruleName"] for e in (await _list(http_client, review_state="confirmed"))["events"]] == ["seen"]
        assert [e["ruleName"] for e in (await _list(http_client, review_state="unreviewed"))["events"]] == ["not seen"]

    async def test_an_unknown_state_is_refused(self, http_client):
        """A typo must not silently write a state nothing can filter for, stranding the
        events in a value no control offers."""
        profile_uuid = await _make_profile(http_client)
        event = await _make_event(http_client, profile_uuid)

        r = await http_client.post(
            "/api/watch-events/review-states",
            json={"uuids": [event["uuid"]], "state": "definitely-not-a-state"},
        )
        assert r.json()["status"] == "failure"
        assert (await _list(http_client, review_state="unreviewed"))["total"] == 1

    async def test_a_uuid_that_is_already_gone_is_not_an_error(self, http_client):
        """Same promise the bulk delete makes: two tabs triaging one selection is normal,
        and the count reports what actually moved."""
        profile_uuid = await _make_profile(http_client)
        event = await _make_event(http_client, profile_uuid)

        r = await http_client.post(
            "/api/watch-events/review-states",
            json={"uuids": [event["uuid"], "no-such-event"], "state": "dismissed"},
        )
        assert r.json()["status"] == "success"
        assert r.json()["updated"] == 1


async def _counts(http_client) -> dict:
    """`/counts` keyed by (source, rule, subject) so a test can assert one pair."""
    r = await http_client.get("/api/watch-events/counts")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return {(c["sourceUuid"], c["ruleUuid"], c["subjectRef"]): c for c in body["counts"]}


def _counts_at(rows: dict, key: tuple) -> int:
    """The tally for one (source, rule, subject), or 0 when no row exists — the two are
    the same thing to Home Assistant, and a test asserting "held at 3" must fail on
    either."""
    row = rows.get(key)
    return 0 if row is None else row["count"]


class TestLifetimeCounts:
    """The tallies that seed Home Assistant's `total_increasing` count sensors.

    These must not fall, ever, for a reason that lives outside this repo: HA reads any
    drop as a counter reset and adds the new value on top of the running total. So a
    tally that tracks *surviving* events corrupts a month of long-term statistics the
    first time retention runs — which is what `COUNT(*)` over `watch_events` did, and
    why `watch_event_counters` exists. Every test here deletes something and asserts the
    number held.

    The tally is owed at an event's *closing* edge. An open event's subject is the
    watcher's running opinion and is revised when it closes, so counting at creation
    would file a visit under a label its own card ends up contradicting — and a
    `total_increasing` counter cannot be corrected afterwards.
    """

    async def test_an_open_event_is_not_counted_until_it_closes(self, http_client):
        """Counting at creation is what made the tally and the card able to disagree."""
        source_uuid = await _make_source(http_client)
        profile_uuid = await _make_profile(http_client)
        rule_uuid, subject_ref = str(_uuid.uuid4()), str(_uuid.uuid4())
        event = await _make_event(
            http_client, profile_uuid, sourceUuid=source_uuid, ruleUuid=rule_uuid, subjectRef=subject_ref
        )
        key = (source_uuid, rule_uuid, subject_ref)
        assert _counts_at(await _counts(http_client), key) == 0

        await _close_event(http_client, event["uuid"])
        assert _counts_at(await _counts(http_client), key) == 1

    async def test_a_closing_edge_counts_the_subject_it_settled_on(self, http_client):
        """A rule may open on one label and close on another — that is the design, not a
        fault — and the tally has to follow the label the card ends up showing."""
        source_uuid = await _make_source(http_client)
        profile_uuid = await _make_profile(http_client)
        rule_uuid = str(_uuid.uuid4())
        opened_as, settled_as = str(_uuid.uuid4()), str(_uuid.uuid4())
        event = await _make_event(
            http_client, profile_uuid, sourceUuid=source_uuid, ruleUuid=rule_uuid, subjectRef=opened_as
        )
        await _close_event(http_client, event["uuid"], subjectRef=settled_as)

        counts = await _counts(http_client)
        assert _counts_at(counts, (source_uuid, rule_uuid, settled_as)) == 1
        assert _counts_at(counts, (source_uuid, rule_uuid, opened_as)) == 0

    async def test_correcting_a_closed_event_does_not_count_it_twice(self, http_client):
        """A reviewer relabelling history is not a second visit.

        The tally fires on the open-to-closed transition, so every later write to an
        already-closed row has to be free — a label correction, a pin, a snapshot path.
        """
        source_uuid = await _make_source(http_client)
        profile_uuid = await _make_profile(http_client)
        rule_uuid, subject_ref = str(_uuid.uuid4()), str(_uuid.uuid4())
        event = await _make_closed_event(
            http_client, profile_uuid, sourceUuid=source_uuid, ruleUuid=rule_uuid, subjectRef=subject_ref
        )
        key = (source_uuid, rule_uuid, subject_ref)
        assert _counts_at(await _counts(http_client), key) == 1

        await _close_event(http_client, event["uuid"], pinned=True)
        await _close_event(http_client, event["uuid"], reviewState="confirmed")
        assert _counts_at(await _counts(http_client), key) == 1

    async def test_a_deleted_event_still_counts(self, http_client):
        source_uuid = await _make_source(http_client)
        profile_uuid = await _make_profile(http_client)
        rule_uuid, subject_ref = str(_uuid.uuid4()), str(_uuid.uuid4())
        uuids = [
            (
                await _make_closed_event(
                    http_client,
                    profile_uuid,
                    sourceUuid=source_uuid,
                    ruleUuid=rule_uuid,
                    subjectRef=subject_ref,
                )
            )["uuid"]
            for _ in range(3)
        ]
        key = (source_uuid, rule_uuid, subject_ref)
        assert _counts_at(await _counts(http_client), key) == 3

        r = await http_client.post("/api/watch-events/delete", json={"uuids": uuids})
        assert r.json()["deleted"] == 3
        assert (await _list(http_client, profile_uuid=profile_uuid))["events"] == []

        assert _counts_at(await _counts(http_client), key) == 3, "retention sweep would read as a counter reset in HA"

    async def test_deleting_the_profile_does_not_zero_the_tally(self, http_client):
        """`watch_events.profile_uuid` is `ondelete="CASCADE"`, so this path took the
        whole history with it — and a profile is deleted by ordinary reconfiguration,
        not just teardown."""
        source_uuid = await _make_source(http_client)
        profile_uuid = await _make_profile(http_client)
        rule_uuid, subject_ref = str(_uuid.uuid4()), str(_uuid.uuid4())
        for _ in range(2):
            await _make_closed_event(
                http_client, profile_uuid, sourceUuid=source_uuid, ruleUuid=rule_uuid, subjectRef=subject_ref
            )

        r = await http_client.delete(f"/api/watch-profiles/{profile_uuid}")
        assert r.json()["status"] == "success", r.text

        assert _counts_at(await _counts(http_client), (source_uuid, rule_uuid, subject_ref)) == 2

    async def test_last_seen_does_not_walk_back_when_the_newest_event_is_deleted(self, http_client):
        """The companion timestamp sensor. Read as `max(started_at)` over surviving rows
        it jumps backwards on a delete, and to `unknown` once the pair's last event ages
        out — for a rule that fires twice a day, that is most of the day."""
        source_uuid = await _make_source(http_client)
        profile_uuid = await _make_profile(http_client)
        rule_uuid, subject_ref = str(_uuid.uuid4()), str(_uuid.uuid4())
        older = datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)
        newer = datetime(2026, 3, 1, 17, 0, tzinfo=timezone.utc)
        for started in (older, newer):
            event = await _make_closed_event(
                http_client,
                profile_uuid,
                sourceUuid=source_uuid,
                ruleUuid=rule_uuid,
                subjectRef=subject_ref,
                startedAt=started.isoformat(),
            )
            if started is newer:
                newest_uuid = event["uuid"]

        key = (source_uuid, rule_uuid, subject_ref)
        before = (await _counts(http_client))[key]["lastSeen"]
        assert before is not None and before.startswith("2026-03-01T17:00")

        await http_client.post("/api/watch-events/delete", json={"uuids": [newest_uuid]})

        after = (await _counts(http_client))[key]["lastSeen"]
        assert after == before

    async def test_an_event_recorded_out_of_order_does_not_move_last_seen_back(self, http_client):
        """A long event closes after a short one that began later, so inserts are not in
        start order. Taking the incoming timestamp unconditionally would rewind it."""
        source_uuid = await _make_source(http_client)
        profile_uuid = await _make_profile(http_client)
        rule_uuid, subject_ref = str(_uuid.uuid4()), str(_uuid.uuid4())
        for started in (
            datetime(2026, 3, 2, 17, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc),
        ):
            await _make_closed_event(
                http_client,
                profile_uuid,
                sourceUuid=source_uuid,
                ruleUuid=rule_uuid,
                subjectRef=subject_ref,
                startedAt=started.isoformat(),
            )

        row = (await _counts(http_client))[(source_uuid, rule_uuid, subject_ref)]
        assert row["count"] == 2
        assert row["lastSeen"].startswith("2026-03-02T17:00")

    async def test_a_rule_with_no_subject_reports_a_null_subject_not_a_sentinel(self, http_client):
        """The table stores `""` because a NULL in a composite primary key never matches
        an upsert's ON CONFLICT. That sentinel must not escape: the bridge sums these
        rows into per-subject tiers, and `""` would become a subject of its own."""
        source_uuid = await _make_source(http_client)
        profile_uuid = await _make_profile(http_client)
        rule_uuid = str(_uuid.uuid4())
        for _ in range(2):
            await _make_closed_event(http_client, profile_uuid, sourceUuid=source_uuid, ruleUuid=rule_uuid)

        rows = await _counts(http_client)
        assert (source_uuid, rule_uuid, "") not in rows
        assert rows[(source_uuid, rule_uuid, None)]["count"] == 2

    async def test_an_event_without_a_source_or_rule_is_not_tallied(self, http_client):
        """Matches the filter the endpoint has always applied. A tally keyed on a NULL
        uuid has no Home Assistant entity to publish to."""
        profile_uuid = await _make_profile(http_client)
        before = len(await _counts(http_client))
        await _make_closed_event(http_client, profile_uuid, ruleName="no ids at all")
        assert len(await _counts(http_client)) == before


class TestPresentInClipCatalogue:
    """The catalogue of what an event's clip *contains*, and the filter over it.

    Distinct from `subject_ref`/`target_ref`, which say what the rule fired about. The
    failure modes worth protecting against are all silent: a union that overwrites instead
    of merging loses whichever window landed first (the event's span or the clip's wider
    one); an exclusion built as `NOT IN` rather than `NOT EXISTS` would swallow every event
    with no catalogue at all; and a facet list built from the label table rather than from
    history would offer labels no clip contains.
    """

    async def _make_label(self, http_client, name: str) -> str:
        r = await http_client.post("/api/labels", json={"name": name, "color": "#FF0000"})
        assert r.status_code == 200, r.text
        return r.json()["label"]["uuid"]

    async def _catalogue(self, http_client, event_uuid: str, labels: dict) -> dict:
        r = await http_client.post(
            f"/api/watch-events/{event_uuid}/labels",
            json={"labels": [{"labelUuid": uuid, "peakConfidence": conf} for uuid, conf in labels.items()]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "success", body
        return body

    async def _get(self, http_client, event_uuid: str) -> dict:
        r = await http_client.get(f"/api/watch-events/{event_uuid}")
        assert r.status_code == 200, r.text
        return r.json()["event"]

    async def test_a_catalogued_label_is_readable_with_its_live_name(self, http_client):
        """The name is joined from `labels`, not denormalised, so a rename reads through to
        old clips the way it does everywhere else in the app."""
        profile_uuid = await _make_profile(http_client)
        cat = await self._make_label(http_client, "Cat")
        event = await _make_event(http_client, profile_uuid)
        await self._catalogue(http_client, event["uuid"], {cat: 0.8})

        assert [(p["labelUuid"], p["labelName"]) for p in (await self._get(http_client, event["uuid"]))["present"]] == [
            (cat, "Cat")
        ]

    async def test_a_second_write_unions_rather_than_replaces(self, http_client):
        """The watcher writes twice for a clip-bearing event — its own span at the closing
        edge, the clip's wider window when the recording is attached — and either may land
        first. A replace would make the answer depend on the order."""
        profile_uuid = await _make_profile(http_client)
        cat = await self._make_label(http_client, "Cat")
        dog = await self._make_label(http_client, "Dog")
        event = await _make_event(http_client, profile_uuid)

        await self._catalogue(http_client, event["uuid"], {cat: 0.8})
        await self._catalogue(http_client, event["uuid"], {dog: 0.5})

        present = {p["labelUuid"] for p in (await self._get(http_client, event["uuid"]))["present"]}
        assert present == {cat, dog}

    async def test_a_lower_confidence_never_lowers_a_recorded_peak(self, http_client):
        profile_uuid = await _make_profile(http_client)
        cat = await self._make_label(http_client, "Cat")
        event = await _make_event(http_client, profile_uuid)

        await self._catalogue(http_client, event["uuid"], {cat: 0.9})
        second = await self._catalogue(http_client, event["uuid"], {cat: 0.2})

        assert second["written"] == 0
        assert (await self._get(http_client, event["uuid"]))["present"][0]["peakConfidence"] == 0.9

    async def test_a_higher_confidence_raises_it(self, http_client):
        profile_uuid = await _make_profile(http_client)
        cat = await self._make_label(http_client, "Cat")
        event = await _make_event(http_client, profile_uuid)

        await self._catalogue(http_client, event["uuid"], {cat: 0.2})
        await self._catalogue(http_client, event["uuid"], {cat: 0.9})

        assert (await self._get(http_client, event["uuid"]))["present"][0]["peakConfidence"] == 0.9

    async def test_filtering_finds_events_whose_clip_merely_contained_the_label(self, http_client):
        """The whole point: the person is incidental to the rule that saved the clip, and
        the actor columns cannot answer for them at all."""
        profile_uuid = await _make_profile(http_client)
        person = await self._make_label(http_client, "Person")
        cat = await self._make_label(http_client, "Cat")
        with_person = await _make_event(http_client, profile_uuid, subjectRef="cat-ref", ruleName="cat drinking")
        without = await _make_event(http_client, profile_uuid, subjectRef="cat-ref", ruleName="cat sleeping")
        await self._catalogue(http_client, with_person["uuid"], {cat: 0.8, person: 0.6})
        await self._catalogue(http_client, without["uuid"], {cat: 0.8})

        body = await _list(http_client, present_refs=[person])
        assert [e["ruleName"] for e in body["events"]] == ["cat drinking"]
        assert body["total"] == 1

    async def test_an_event_matches_once_even_when_it_contains_several_wanted_labels(self, http_client):
        """EXISTS, not a join: a join would return the event once per matching label and
        the grid would show the same card three times with a total to match."""
        profile_uuid = await _make_profile(http_client)
        refs = [await self._make_label(http_client, name) for name in ("Cat", "Dog", "Person")]
        event = await _make_event(http_client, profile_uuid)
        await self._catalogue(http_client, event["uuid"], {ref: 0.5 for ref in refs})

        body = await _list(http_client, present_refs=refs)
        assert body["total"] == 1
        assert len(body["events"]) == 1

    async def test_excluding_a_label_keeps_events_with_no_catalogue_at_all(self, http_client):
        """Every event recorded before the catalogue existed has no rows, and "not the dog"
        plainly includes them. NOT EXISTS gets this right where NOT IN would not."""
        profile_uuid = await _make_profile(http_client)
        dog = await self._make_label(http_client, "Dog")
        catalogued = await _make_event(http_client, profile_uuid, ruleName="has a dog")
        await self._catalogue(http_client, catalogued["uuid"], {dog: 0.8})
        await _make_event(http_client, profile_uuid, ruleName="catalogued nothing")

        body = await _list(http_client, exclude_present_refs=[dog])
        assert [e["ruleName"] for e in body["events"]] == ["catalogued nothing"]

    async def test_the_facet_list_comes_from_history_not_from_the_label_tree(self, http_client):
        """A label nothing has ever been seen with must not appear in the dropdown: it
        would offer a filter that silently matches nothing."""
        profile_uuid = await _make_profile(http_client)
        seen = await self._make_label(http_client, "Seen")
        await self._make_label(http_client, "Never seen")
        event = await _make_event(http_client, profile_uuid)
        await self._catalogue(http_client, event["uuid"], {seen: 0.7})

        r = await http_client.get("/api/watch-events/facets", params={"profile_uuid": profile_uuid})
        assert r.status_code == 200, r.text
        assert [(f["ref"], f["name"]) for f in r.json()["present"]] == [(seen, "Seen")]

    async def test_deleting_an_event_takes_its_catalogue_with_it(self, http_client):
        """The FK cascade. An orphaned row would keep a deleted event's label in the facet
        dropdown forever, offering a filter that can never match."""
        profile_uuid = await _make_profile(http_client)
        cat = await self._make_label(http_client, "Cat")
        event = await _make_event(http_client, profile_uuid)
        await self._catalogue(http_client, event["uuid"], {cat: 0.8})

        r = await http_client.delete(f"/api/watch-events/{event['uuid']}")
        assert r.status_code == 200, r.text

        r = await http_client.get("/api/watch-events/facets", params={"profile_uuid": profile_uuid})
        assert r.json()["present"] == []

    async def test_cataloguing_a_deleted_event_is_a_no_op_not_an_error(self, http_client):
        """The watcher writes this after the fact, and retention or a profile save can take
        the row in between. A raised FK error there would be logged as a recording failure
        for an event nobody can see."""
        profile_uuid = await _make_profile(http_client)
        cat = await self._make_label(http_client, "Cat")
        event = await _make_event(http_client, profile_uuid)
        assert (await http_client.delete(f"/api/watch-events/{event['uuid']}")).status_code == 200

        body = await self._catalogue(http_client, event["uuid"], {cat: 0.8})
        assert body["written"] == 0
