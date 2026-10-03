"""Tracking blips — the analyser's findings, as the review surface reads them.

Same class of risk as the events history these hang off: the failure modes worth
protecting against all return a *plausible wrong answer* rather than an error.

* A replace that appends instead of replacing doubles every finding on re-analysis,
  and the count looks like the detector got worse.
* A replace that is not scoped to one recording silently discards the blips found in
  the same event's other media.
* Expiry that overwrites a *decided* blip loses the decision, and the frames get
  offered for harvest a second time.
* `harvestable` leaking tracker faults into the harvest queue pads the training set
  with frames the model already gets right — the failure the whole fault split exists
  to prevent.
"""

import json
import uuid as _uuid

import cv2
import numpy as np
import pytest

from apps.helpers.db.types import Recording


async def _make_profile(http_client) -> str:
    r = await http_client.post("/api/watch-profiles", json={"name": f"watcher-{_uuid.uuid4().hex[:6]}"})
    assert r.status_code == 200, r.text
    return r.json()["profile"]["uuid"]


async def _make_event(http_client, profile_uuid: str) -> str:
    r = await http_client.post("/api/watch-events", json={"profileUuid": profile_uuid})
    assert r.status_code == 200, r.text
    return r.json()["event"]["uuid"]


def _blip(kind="identity_flip", *, track_id=7, severity=0.5, fault="detector", harvestable=True, start=1.0, end=2.0):
    return {
        "kind": kind,
        "trackId": track_id,
        "startS": start,
        "endS": end,
        "severity": severity,
        "fault": fault,
        "harvestable": harvestable,
        "detail": {"modalLabel": "cat_a"},
    }


async def _replace(http_client, event_uuid, blips, recording_uuid=None):
    payload = {"eventUuid": event_uuid, "blips": blips}
    if recording_uuid:
        payload["recordingUuid"] = recording_uuid
    r = await http_client.post("/api/event-blips/replace", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["blips"]


async def _list(http_client, **params):
    r = await http_client.get("/api/event-blips", params=params)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body


@pytest.fixture
async def event_uuid(http_client):
    return await _make_event(http_client, await _make_profile(http_client))


class TestReplacingAnAnalysis:
    async def test_re_analysing_converges_rather_than_accumulating(self, http_client, event_uuid):
        """Analysis is deterministic over a sidecar that does not change, so running it
        twice must leave one copy — not two findings that read as a worsening detector."""
        await _replace(http_client, event_uuid, [_blip(), _blip(track_id=9)])
        await _replace(http_client, event_uuid, [_blip(), _blip(track_id=9)])

        assert (await _list(http_client, event_uuid=event_uuid))["total"] == 2

    async def test_a_shorter_result_removes_what_is_no_longer_found(self, http_client, event_uuid):
        await _replace(http_client, event_uuid, [_blip(), _blip(track_id=9), _blip(track_id=11)])
        await _replace(http_client, event_uuid, [_blip()])

        assert (await _list(http_client, event_uuid=event_uuid))["total"] == 1

    async def test_replacing_one_recording_leaves_the_others_alone(self, http_client, event_uuid):
        """An event can own a clip and a timelapse. Re-analysing one must not discard
        the findings from the other."""
        clip, timelapse = str(_uuid.uuid4()), str(_uuid.uuid4())
        await _replace(http_client, event_uuid, [_blip()], recording_uuid=clip)
        await _replace(http_client, event_uuid, [_blip(track_id=9)], recording_uuid=timelapse)

        await _replace(http_client, event_uuid, [_blip(), _blip(track_id=8)], recording_uuid=clip)

        assert (await _list(http_client, recording_uuid=timelapse))["total"] == 1
        assert (await _list(http_client, recording_uuid=clip))["total"] == 2

    async def test_deleting_the_event_takes_its_blips(self, http_client, event_uuid):
        await _replace(http_client, event_uuid, [_blip()])
        r = await http_client.delete(f"/api/watch-events/{event_uuid}")
        assert r.status_code == 200, r.text

        assert (await _list(http_client, event_uuid=event_uuid))["total"] == 0


class TestFilteringAndOrder:
    async def test_the_worst_finding_sorts_first(self, http_client, event_uuid):
        """Severity means "how sure the detector was about what it got wrong", so the
        most valuable training data heads the queue rather than merely the earliest."""
        await _replace(
            http_client,
            event_uuid,
            [_blip(severity=0.2, start=0.0), _blip(severity=0.9, start=5.0), _blip(severity=0.5, start=9.0)],
        )
        blips = (await _list(http_client, event_uuid=event_uuid))["blips"]

        assert [b["severity"] for b in blips] == [0.9, 0.5, 0.2]

    async def test_the_harvestable_filter_excludes_tracker_faults(self, http_client, event_uuid):
        """The filter the harvest queue runs on. A tracker fault reaching it would mean
        training on frames the detector already got right."""
        await _replace(
            http_client,
            event_uuid,
            [
                _blip(kind="identity_flip", fault="detector", harvestable=True),
                _blip(kind="tracker_dropout", fault="tracker", harvestable=False, track_id=9),
            ],
        )

        kept = (await _list(http_client, event_uuid=event_uuid, harvestable=True))["blips"]
        assert [b["kind"] for b in kept] == ["identity_flip"]

        excluded = (await _list(http_client, event_uuid=event_uuid, harvestable=False))["blips"]
        assert [b["kind"] for b in excluded] == ["tracker_dropout"]

    async def test_filtering_by_kind_narrows_the_set(self, http_client, event_uuid):
        await _replace(
            http_client,
            event_uuid,
            [_blip(kind="identity_flip"), _blip(kind="full_dropout", track_id=9)],
        )
        body = await _list(http_client, event_uuid=event_uuid, kinds=["full_dropout"])

        assert body["total"] == 1
        assert body["blips"][0]["kind"] == "full_dropout"

    async def test_the_total_tracks_the_filter_rather_than_the_table(self, http_client, event_uuid):
        """A total drifting from the filtered set turns "N blips" into a lie."""
        await _replace(
            http_client,
            event_uuid,
            [_blip(kind="identity_flip"), _blip(kind="full_dropout", track_id=9)],
        )

        assert (await _list(http_client, event_uuid=event_uuid, kinds=["identity_flip"]))["total"] == 1
        assert (await _list(http_client, event_uuid=event_uuid))["total"] == 2


class TestSelectingSeveralEvents:
    """The filter the bulk surfaces run on. A misspelt or ignored param answers with the
    *unfiltered* table, so the failure mode is a harvest that silently reaches events the
    user never selected."""

    async def test_only_the_named_events_come_back(self, http_client):
        profile = await _make_profile(http_client)
        picked = [await _make_event(http_client, profile) for _ in range(2)]
        left_out = await _make_event(http_client, profile)
        for event in [*picked, left_out]:
            await _replace(http_client, event, [_blip()])

        body = await _list(http_client, event_uuids=picked)

        assert body["total"] == 2
        assert {b["eventUuid"] for b in body["blips"]} == set(picked)

    async def test_harvestable_still_narrows_within_the_selection(self, http_client):
        profile = await _make_profile(http_client)
        first = await _make_event(http_client, profile)
        second = await _make_event(http_client, profile)
        await _replace(http_client, first, [_blip(kind="identity_flip")])
        await _replace(
            http_client,
            second,
            [_blip(kind="identity_switch", fault="tracker", harvestable=False)],
        )

        body = await _list(http_client, event_uuids=[first, second], harvestable=True)

        assert [b["eventUuid"] for b in body["blips"]] == [first]


class TestTriageState:
    async def test_moving_a_selection_reports_how_many_moved(self, http_client, event_uuid):
        created = await _replace(http_client, event_uuid, [_blip(), _blip(track_id=9)])
        r = await http_client.post(
            "/api/event-blips/states",
            json={"uuids": [created[0]["uuid"]], "state": "dismissed"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["updated"] == 1

        remaining = (await _list(http_client, event_uuid=event_uuid, states=["new"]))["total"]
        assert remaining == 1

    async def test_an_unknown_state_is_refused(self, http_client, event_uuid):
        created = await _replace(http_client, event_uuid, [_blip()])
        r = await http_client.post(
            "/api/event-blips/states",
            json={"uuids": [created[0]["uuid"]], "state": "probably_fine"},
        )

        assert r.json()["status"] == "failure"
        assert (await _list(http_client, event_uuid=event_uuid, states=["new"]))["total"] == 1


class TestExpiringWithTheClip:
    """Retention deletes clips on its own schedule; the finding outlives the video.

    Called by the sweep rather than by a route, so exercised against the db client
    directly.
    """

    async def test_an_untriaged_blip_becomes_unreviewable_but_is_not_deleted(self, http_client, db_client, event_uuid):
        recording = str(_uuid.uuid4())
        await _replace(http_client, event_uuid, [_blip()], recording_uuid=recording)

        moved = await db_client.expire_blips_for_recording(recording)
        assert moved == 1

        blips, total = await db_client.list_event_blips(event_uuid=event_uuid)
        assert total == 1, "the record of what went wrong must survive its clip"
        assert blips[0].state == "expired"

    async def test_a_decided_blip_keeps_its_outcome(self, http_client, db_client, event_uuid):
        """The conditional that matters: expiry must not overwrite a real decision, or
        the same frames get offered for harvest a second time."""
        recording = str(_uuid.uuid4())
        created = await _replace(http_client, event_uuid, [_blip(), _blip(track_id=9)], recording_uuid=recording)
        await db_client.set_event_blip_states([created[0]["uuid"]], "harvested")

        moved = await db_client.expire_blips_for_recording(recording)

        assert moved == 1, "only the untriaged one should have moved"
        blips, _ = await db_client.list_event_blips(event_uuid=event_uuid)
        states = {b.uuid: b.state for b in blips}
        assert states[created[0]["uuid"]] == "harvested"
        assert states[created[1]["uuid"]] == "expired"

    async def test_another_recordings_blips_are_untouched(self, http_client, db_client, event_uuid):
        swept, kept = str(_uuid.uuid4()), str(_uuid.uuid4())
        await _replace(http_client, event_uuid, [_blip()], recording_uuid=swept)
        await _replace(http_client, event_uuid, [_blip(track_id=9)], recording_uuid=kept)

        await db_client.expire_blips_for_recording(swept)

        surviving, _ = await db_client.list_event_blips(recording_uuid=kept)
        assert [b.state for b in surviving] == ["new"]


class TestBlipCountsOnTheEvent:
    async def test_a_clean_event_reports_no_counts(self, http_client, event_uuid):
        r = await http_client.get("/api/watch-events", params={"limit": 50})
        event = next(e for e in r.json()["events"] if e["uuid"] == event_uuid)

        assert event["blipCounts"] == {}

    async def test_counts_are_broken_down_by_kind(self, http_client, event_uuid):
        await _replace(
            http_client,
            event_uuid,
            [
                _blip(kind="identity_flip"),
                _blip(kind="identity_flip", track_id=9),
                _blip(kind="full_dropout", track_id=11),
            ],
        )
        r = await http_client.get("/api/watch-events", params={"limit": 50})
        event = next(e for e in r.json()["events"] if e["uuid"] == event_uuid)

        assert event["blipCounts"] == {"identity_flip": 2, "full_dropout": 1}

    async def test_tracker_fault_blips_are_not_counted(self, http_client, event_uuid):
        """Tracker faults carry no actionable content, so they must not inflate a badge
        that means "something to review here" — a user "checking off" every actionable
        finding on the event should see the badge disappear, and it can't if an
        unreviewable tracker fault is still sitting in the total."""
        await _replace(
            http_client,
            event_uuid,
            [
                _blip(kind="identity_flip", fault="detector", harvestable=True),
                _blip(kind="tracker_dropout", fault="tracker", harvestable=False, track_id=9),
            ],
        )
        r = await http_client.get("/api/watch-events", params={"limit": 50})
        event = next(e for e in r.json()["events"] if e["uuid"] == event_uuid)

        assert event["blipCounts"] == {"identity_flip": 1}

    async def test_resolved_blips_stop_counting_once_triaged(self, http_client, event_uuid):
        """The whole point of harvesting or dismissing a finding is to make the badge go
        away — a count that keeps a resolved blip forever would mean "check it off"
        never actually clears anything."""
        created = await _replace(
            http_client,
            event_uuid,
            [_blip(kind="identity_flip"), _blip(kind="full_dropout", track_id=9)],
        )
        r = await http_client.post(
            "/api/event-blips/states", json={"uuids": [created[0]["uuid"]], "state": "dismissed"}
        )
        assert r.status_code == 200, r.text

        r = await http_client.get("/api/watch-events", params={"limit": 50})
        event = next(e for e in r.json()["events"] if e["uuid"] == event_uuid)

        assert event["blipCounts"] == {"full_dropout": 1}

    async def test_the_has_blips_filter_separates_flagged_from_clean(self, http_client):
        profile = await _make_profile(http_client)
        flagged = await _make_event(http_client, profile)
        clean = await _make_event(http_client, profile)
        await _replace(http_client, flagged, [_blip()])

        # snake_case: query parameters are plain function args, unlike response bodies,
        # which are camelCased by the alias generator. A misspelling here is silently
        # ignored and returns the *unfiltered* set — which is what this asserts against.
        r = await http_client.get("/api/watch-events", params={"has_blips": True, "limit": 50})
        assert [e["uuid"] for e in r.json()["events"]] == [flagged]

        r = await http_client.get("/api/watch-events", params={"has_blips": False, "limit": 50})
        assert clean in [e["uuid"] for e in r.json()["events"]]
        assert flagged not in [e["uuid"] for e in r.json()["events"]]


def _write_clip(path, *, seconds=2, fps=10.0):
    """A tiny real clip cv2 can seek in — the decode is what's under test, not a mock."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(int(seconds * fps)):
        writer.write(np.full((48, 64, 3), (i * 5) % 256, dtype=np.uint8))
    writer.release()


class TestBlipCropPreview:
    """Before/after crops for an `identity_switch` blip, pulled straight from the clip
    and its sidecar rather than run through the harvest task's preview-candidate
    pipeline — this backs a live look, not a durable artifact."""

    async def _make_recording(self, db_client, tmp_path, *, track_id=1, times=(0.0, 0.5, 1.0, 1.5)):
        clip = tmp_path / "clip.mp4"
        _write_clip(clip)
        overlay = tmp_path / "clip.boxes.json"
        overlay.write_text(
            json.dumps(
                {
                    "frames": [
                        {"t": t, "tracks": [{"id": track_id, "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}]} for t in times
                    ]
                }
            )
        )
        return await db_client.create_recording(
            Recording(kind="clip", file_path=str(clip), overlay_path=str(overlay), status="complete")
        )

    async def test_returns_a_crop_from_each_side_of_the_switch(self, db_client, manager, tmp_path, event_uuid):
        recording = await self._make_recording(db_client, tmp_path)

        created = await db_client.replace_event_blips(
            event_uuid,
            [
                {
                    "kind": "identity_switch",
                    "trackId": 1,
                    "startS": 1.0,
                    "endS": 1.5,
                    "severity": 0.4,
                    "fault": "tracker",
                    "harvestable": False,
                    "detail": {"modalLabel": "cat_a", "sawLabel": "cat_b"},
                },
            ],
            recording_uuid=recording.uuid,
        )
        blip_uuid = created[0].uuid

        result = await manager.get_blip_crop_preview(blip_uuid)

        assert result["status"] == "success", result
        assert result["before"]["dataUrl"].startswith("data:image/jpeg;base64,")
        assert result["before"]["label"] == "cat_a"
        assert result["after"]["dataUrl"].startswith("data:image/jpeg;base64,")
        assert result["after"]["label"] == "cat_b"
        # Different instants of a clip whose brightness changes every frame — a real
        # regression here would be the seek landing on the same frame for both.
        assert result["before"]["dataUrl"] != result["after"]["dataUrl"]

    async def test_a_missing_blip_fails_without_raising(self, manager):
        result = await manager.get_blip_crop_preview("no-such-blip")
        assert result["status"] == "failure"

    async def test_a_blip_with_no_clip_on_disk_fails_without_raising(self, db_client, manager, event_uuid):
        created = await db_client.replace_event_blips(
            event_uuid,
            [
                {
                    "kind": "identity_switch",
                    "trackId": 1,
                    "startS": 1.0,
                    "endS": 1.5,
                    "fault": "tracker",
                    "harvestable": False,
                    "detail": {"modalLabel": "cat_a", "sawLabel": "cat_b"},
                },
            ],
            recording_uuid="rec-with-no-row",
        )
        result = await manager.get_blip_crop_preview(created[0].uuid)
        assert result["status"] == "failure"
