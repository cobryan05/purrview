"""Tests for Watch Profiles — the StreamWatcher's configuration entity.

The properties worth protecting here are the ones whose failure mode is *silence*:

* a rule pointing at a zone that no longer exists never fires,
* regenerating zone/rule uuids on save breaks every reference without an error,
* deleting a source that a watcher uses must fail loudly rather than quietly
  emptying the watcher (RESTRICT, not CASCADE),
* hysteresis fields must be present, because a bare threshold flaps into an MQTT
  storm and a dozen clips per visit.
"""

import uuid as _uuid

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_ZONE_A_UUID = "11111111-1111-1111-1111-111111111111"


def _zone(uuid: str = _ZONE_A_UUID, name: str = "litter box area") -> dict:
    return {
        "uuid": uuid,
        "name": name,
        "kind": "box",
        "pointsJson": [[0.1, 0.1], [0.6, 0.7]],
        "referenceWidth": 1920,
        "referenceHeight": 1080,
    }


def _rule(uuid: str = "22222222-2222-2222-2222-222222222222", **overrides) -> dict:
    rule = {
        "uuid": uuid,
        "order": 0,
        "name": "cat using litter box",
        "enabled": True,
        "subjectAKind": "label",
        "subjectARef": "label-cat",
        "relation": "contains",
        "subjectBKind": "zone",
        "subjectBRef": _ZONE_A_UUID,
        "threshold": 0.6,
        "minDurationS": 2.0,
        "rearmDelayS": 10.0,
        "maxDurationS": 300.0,
        "eventLabel": "litterbox",
    }
    rule.update(overrides)
    return rule


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
    body = r.json()
    assert body["status"] == "success", body
    return body["source"]["uuid"]


async def _make_label(http_client, name: str = "Cat") -> str:
    r = await http_client.post("/api/labels", json={"name": name, "color": "#FF0000"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["label"]["uuid"]


async def _create_profile(http_client, name: str, **overrides) -> dict:
    """POST a profile, normalising both rejection routes into one shape.

    Payload-shape violations are caught by the pydantic write models and come back as
    FastAPI's 422; cross-entity violations (a rule naming a zone no source defines) are
    caught in db_client and come back as 200 + status=failure. Tests care that the bad
    config was refused, not which layer refused it.
    """
    payload = {"name": name}
    payload.update(overrides)
    r = await http_client.post("/api/watch-profiles", json=payload)
    if r.status_code == 422:
        return {"status": "failure", "message": r.text}
    assert r.status_code == 200, r.text
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# CRUD
# ─────────────────────────────────────────────────────────────────────────────


class TestWatchProfileCrud:
    async def test_create_minimal_profile(self, http_client):
        body = await _create_profile(http_client, "House cams")
        assert body["status"] == "success", body
        profile = body["profile"]
        assert profile["name"] == "House cams"
        assert profile["trackerTypename"] == "IouTracker"
        # Analysis rate/width are first-class because per-frame inference is the
        # throughput ceiling; they must have real defaults, not None.
        assert profile["defaultAnalysisFps"] == 5.0
        assert profile["defaultAnalysisWidth"] == 640

    async def test_list_and_get(self, http_client):
        created = (await _create_profile(http_client, "House cams"))["profile"]
        r = await http_client.get("/api/watch-profiles")
        assert [p["uuid"] for p in r.json()["profiles"]] == [created["uuid"]]

        r = await http_client.get(f"/api/watch-profiles/{created['uuid']}")
        assert r.json()["profile"]["name"] == "House cams"

    async def test_duplicate_name_is_rejected(self, http_client):
        await _create_profile(http_client, "House cams")
        body = await _create_profile(http_client, "House cams")
        assert body["status"] == "failure"

    async def test_get_unknown_profile_reports_not_found(self, http_client):
        r = await http_client.get("/api/watch-profiles/nope")
        assert r.json()["status"] == "failure"
        assert "not found" in r.json()["message"].lower()

    async def test_delete(self, http_client):
        created = (await _create_profile(http_client, "House cams"))["profile"]
        r = await http_client.delete(f"/api/watch-profiles/{created['uuid']}")
        assert r.json()["status"] == "success"
        r = await http_client.get("/api/watch-profiles")
        assert r.json()["profiles"] == []

    async def test_delete_unknown_reports_not_found(self, http_client):
        r = await http_client.delete("/api/watch-profiles/nope")
        assert r.json()["status"] == "failure"


class TestNestedWrites:
    async def test_create_with_sources_zones_rules_and_actions(self, http_client):
        source_uuid = await _make_source(http_client)
        # A real label: label_lifetimes has a real FK, so a made-up uuid is rejected.
        label_uuid = await _make_label(http_client)
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid, "order": 0, "zones": [_zone()]}],
            rules=[_rule()],
            actions=[{"kind": "publish_mqtt", "trigger": "event_started", "paramsJson": {"retain": False}}],
            labelLifetimes=[{"labelUuid": label_uuid, "ttlS": 2.0, "isAnchor": False}],
        )
        assert body["status"] == "success", body
        profile = body["profile"]
        assert len(profile["sources"]) == 1
        assert len(profile["sources"][0]["zones"]) == 1
        assert profile["sources"][0]["zones"][0]["name"] == "litter box area"
        assert len(profile["rules"]) == 1
        assert len(profile["actions"]) == 1
        assert len(profile["labelLifetimes"]) == 1

    async def test_update_preserves_zone_and_rule_uuids(self, http_client):
        """Regenerating uuids on save would silently break every zone-based rule."""
        source_uuid = await _make_source(http_client)
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": source_uuid, "zones": [_zone()]}],
                rules=[_rule()],
            )
        )["profile"]

        r = await http_client.patch(
            f"/api/watch-profiles/{profile['uuid']}",
            json={
                "description": "now with a description",
                "sources": [{"sourceUuid": source_uuid, "zones": [_zone()]}],
                "rules": [_rule()],
            },
        )
        updated = r.json()["profile"]
        assert updated["sources"][0]["zones"][0]["uuid"] == _ZONE_A_UUID
        assert updated["rules"][0]["subjectBRef"] == _ZONE_A_UUID
        assert updated["description"] == "now with a description"

    async def test_omitting_a_list_leaves_it_unchanged(self, http_client):
        source_uuid = await _make_source(http_client)
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": source_uuid, "zones": [_zone()]}],
                rules=[_rule()],
            )
        )["profile"]

        r = await http_client.patch(f"/api/watch-profiles/{profile['uuid']}", json={"enabled": False})
        updated = r.json()["profile"]
        assert updated["enabled"] is False
        # None means "no change" — not "delete everything".
        assert len(updated["sources"]) == 1
        assert len(updated["rules"]) == 1

    async def test_empty_list_removes_all(self, http_client):
        source_uuid = await _make_source(http_client)
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": source_uuid, "zones": [_zone()]}],
                rules=[_rule()],
            )
        )["profile"]

        r = await http_client.patch(f"/api/watch-profiles/{profile['uuid']}", json={"rules": []})
        assert r.json()["profile"]["rules"] == []
        assert len(r.json()["profile"]["sources"]) == 1

    async def test_deleting_a_profile_cascades_to_its_children(self, http_client, db_client):
        from sqlmodel import select

        from apps.helpers.db.types import InteractionRule, WatchZone

        source_uuid = await _make_source(http_client)
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": source_uuid, "zones": [_zone()]}],
                rules=[_rule()],
            )
        )["profile"]
        await http_client.delete(f"/api/watch-profiles/{profile['uuid']}")

        async with db_client._async_session_maker() as session:
            assert (await session.exec(select(InteractionRule))).all() == []
            assert (await session.exec(select(WatchZone))).all() == []

    async def test_a_source_listed_twice_is_rejected(self, http_client):
        source_uuid = await _make_source(http_client)
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid}, {"sourceUuid": source_uuid}],
        )
        assert body["status"] == "failure"
        assert "twice" in body["message"]


class TestReferentialIntegrity:
    async def test_rule_referencing_an_unknown_zone_is_rejected(self, http_client):
        """A dangling zone reference means the rule silently never fires."""
        source_uuid = await _make_source(http_client)
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid, "zones": []}],
            rules=[_rule()],
        )
        assert body["status"] == "failure"
        assert "zone" in body["message"].lower()

    async def test_action_targeting_an_unknown_rule_is_rejected(self, http_client):
        body = await _create_profile(
            http_client,
            "House cams",
            actions=[{"kind": "record_clip", "ruleUuid": "no-such-rule"}],
        )
        assert body["status"] == "failure"
        assert "rule" in body["message"].lower()

    async def test_deleting_a_source_in_use_fails_loudly(self, http_client):
        """RESTRICT, not CASCADE: a silent unlink would empty the watcher unnoticed."""
        source_uuid = await _make_source(http_client)
        await _create_profile(http_client, "House cams", sources=[{"sourceUuid": source_uuid}])

        r = await http_client.post("/api/sources/delete", json={"sourceUuids": [source_uuid]})
        assert r.status_code == 200
        assert r.json()["status"] == "failure", "deleting an in-use source must not silently succeed"


class TestRuleValidation:
    async def test_two_subject_relation_requires_a_second_subject(self, http_client):
        body = await _create_profile(
            http_client,
            "House cams",
            rules=[_rule(relation="iou", subjectBKind="none", subjectBRef=None)],
        )
        assert body["status"] == "failure"

    async def test_present_is_arity_one(self, http_client):
        body = await _create_profile(
            http_client,
            "House cams",
            rules=[_rule(relation="present", subjectBKind="none", subjectBRef=None)],
        )
        assert body["status"] == "success", body

    async def test_present_rejects_a_second_subject(self, http_client):
        source_uuid = await _make_source(http_client)
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid, "zones": [_zone()]}],
            rules=[_rule(relation="present")],
        )
        assert body["status"] == "failure"

    async def test_zone_subject_requires_a_ref(self, http_client):
        body = await _create_profile(
            http_client,
            "House cams",
            rules=[_rule(subjectBKind="zone", subjectBRef=None)],
        )
        assert body["status"] == "failure"

    async def test_max_duration_must_be_positive(self, http_client):
        source_uuid = await _make_source(http_client)
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid, "zones": [_zone()]}],
            rules=[_rule(maxDurationS=0)],
        )
        # It is the runaway-event cap; zero would let one stuck detection record forever.
        assert body["status"] == "failure"

    async def test_defaults_include_hysteresis(self, http_client):
        """A bare threshold flaps; the defaults must already guard against it."""
        source_uuid = await _make_source(http_client)
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": source_uuid, "zones": [_zone()]}],
                rules=[
                    {
                        "name": "minimal",
                        "subjectAKind": "label",
                        "subjectARef": "label-cat",
                        "subjectBKind": "zone",
                        "subjectBRef": _ZONE_A_UUID,
                    }
                ],
            )
        )["profile"]
        rule = profile["rules"][0]
        assert rule["minDurationS"] > 0
        assert rule["rearmDelayS"] > 0
        assert rule["maxDurationS"] > 0
        # Containment, not IoU: a cat's box is far smaller than a litter box's, so IoU
        # stays low even at perfect containment.
        assert rule["relation"] == "contains"


class TestZoneValidation:
    async def test_box_needs_exactly_two_points(self, http_client):
        source_uuid = await _make_source(http_client)
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid, "zones": [{**_zone(), "pointsJson": [[0.1, 0.1]]}]}],
        )
        assert body["status"] == "failure"

    async def test_polygon_needs_three_points(self, http_client):
        source_uuid = await _make_source(http_client)
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[
                {
                    "sourceUuid": source_uuid,
                    "zones": [{**_zone(), "kind": "polygon", "pointsJson": [[0.1, 0.1], [0.2, 0.2]]}],
                }
            ],
        )
        assert body["status"] == "failure"

    async def test_pixel_coordinates_are_rejected(self, http_client):
        """Zones are stored normalised so they survive a camera resolution change."""
        source_uuid = await _make_source(http_client)
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid, "zones": [{**_zone(), "pointsJson": [[100, 100], [500, 500]]}]}],
        )
        assert body["status"] == "failure"
        assert "normalised" in body["message"]

    async def test_polygon_with_three_points_is_accepted(self, http_client):
        source_uuid = await _make_source(http_client)
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[
                {
                    "sourceUuid": source_uuid,
                    "zones": [{**_zone(), "kind": "polygon", "pointsJson": [[0.1, 0.1], [0.5, 0.2], [0.3, 0.8]]}],
                }
            ],
        )
        assert body["status"] == "success", body


class TestValidateEndpoint:
    async def test_a_profile_with_no_sources_is_invalid(self, http_client):
        profile = (await _create_profile(http_client, "House cams"))["profile"]
        r = await http_client.get(f"/api/watch-profiles/{profile['uuid']}/validate")
        report = r.json()
        assert report["valid"] is False
        assert any("no sources" in e for e in report["errors"])

    async def test_a_source_with_no_models_is_an_error(self, http_client):
        source_uuid = await _make_source(http_client)
        profile = (await _create_profile(http_client, "House cams", sources=[{"sourceUuid": source_uuid}]))["profile"]
        r = await http_client.get(f"/api/watch-profiles/{profile['uuid']}/validate")
        report = r.json()
        assert report["valid"] is False
        assert any("model stack" in e for e in report["errors"])

    async def test_no_enabled_rules_is_a_warning_not_an_error(self, http_client):
        source_uuid = await _make_source(http_client)
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                defaultModelGroupUuid=None,
                sources=[{"sourceUuid": source_uuid}],
            )
        )["profile"]
        report = (await http_client.get(f"/api/watch-profiles/{profile['uuid']}/validate")).json()
        assert any("never raise an event" in w for w in report["warnings"])

    async def test_preroll_longer_than_the_buffer_warns(self, http_client):
        """A 30 s pre-roll against a 20 s buffer is a misconfiguration, not a surprise."""
        source_uuid = await _make_source(http_client)
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": source_uuid, "recordRetentionS": 20.0, "zones": [_zone()]}],
                rules=[_rule()],
                actions=[{"kind": "record_clip", "paramsJson": {"preroll_s": 30}}],
            )
        )["profile"]
        report = (await http_client.get(f"/api/watch-profiles/{profile['uuid']}/validate")).json()
        assert any("pre-roll" in w for w in report["warnings"])

    async def test_high_analysis_fps_warns_about_throughput(self, http_client):
        source_uuid = await _make_source(http_client)
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": source_uuid, "analysisFps": 25.0}],
            )
        )["profile"]
        report = (await http_client.get(f"/api/watch-profiles/{profile['uuid']}/validate")).json()
        assert any("throughput ceiling" in w for w in report["warnings"])

    async def test_unknown_profile_reports_not_found(self, http_client):
        r = await http_client.get("/api/watch-profiles/nope/validate")
        assert r.json()["status"] == "failure"


class TestWatchEventsAndRecordings:
    async def test_events_list_is_empty_initially(self, http_client):
        r = await http_client.get("/api/watch-events")
        assert r.json()["status"] == "success"
        assert r.json()["events"] == []

    async def test_recordings_list_is_empty_initially(self, http_client):
        r = await http_client.get("/api/recordings")
        assert r.json()["status"] == "success"
        assert r.json()["recordings"] == []

    async def test_unknown_recording_reports_not_found(self, http_client):
        r = await http_client.get("/api/recordings/nope")
        assert r.json()["status"] == "failure"

    async def test_recording_file_404s_when_missing_on_disk(self, http_client, db_client):
        from apps.helpers.db.types import Recording

        rec = await db_client.create_recording(Recording(file_path="/nonexistent/clip.mp4", status="complete"))
        r = await http_client.get(f"/api/recordings/{rec.uuid}/file")
        assert r.status_code == 404

    async def test_recording_file_is_served_with_range_support(self, http_client, db_client, tmp_path):
        from apps.helpers.db.types import Recording

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"0123456789")
        rec = await db_client.create_recording(Recording(file_path=str(clip), status="complete"))

        r = await http_client.get(f"/api/recordings/{rec.uuid}/file")
        assert r.status_code == 200
        assert r.content == b"0123456789"

        # Range support is what lets a <video> element seek.
        r = await http_client.get(f"/api/recordings/{rec.uuid}/file", headers={"Range": "bytes=2-5"})
        assert r.status_code == 206
        assert r.content == b"2345"

    async def test_deleting_a_recording_removes_its_file(self, http_client, db_client, tmp_path):
        from apps.helpers.db.types import Recording

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"data")
        rec = await db_client.create_recording(Recording(file_path=str(clip), status="complete"))

        r = await http_client.delete(f"/api/recordings/{rec.uuid}")
        assert r.json()["status"] == "success"
        assert not clip.exists()
        assert await db_client.get_recording(rec.uuid) is None


class TestOpenEvents:
    async def test_open_events_are_listed_separately(self, http_client, db_client):
        """Resume has to find and close events that outlived the previous process."""
        from datetime import datetime, timezone

        from apps.helpers.db.types import WatchEvent

        profile = (await _create_profile(http_client, "House cams"))["profile"]
        await db_client.create_watch_event(WatchEvent(profile_uuid=profile["uuid"]))
        closed = await db_client.create_watch_event(WatchEvent(profile_uuid=profile["uuid"]))
        await db_client.close_watch_event(closed.uuid, ended_at=datetime.now(timezone.utc))

        open_events = await db_client.list_open_watch_events(profile_uuid=profile["uuid"])
        assert len(open_events) == 1
        assert open_events[0].ended_at is None

    async def test_closing_an_event_links_its_recording(self, http_client, db_client):
        from datetime import datetime, timezone

        from apps.helpers.db.types import Recording, WatchEvent

        profile = (await _create_profile(http_client, "House cams"))["profile"]
        event = await db_client.create_watch_event(WatchEvent(profile_uuid=profile["uuid"]))
        rec = await db_client.create_recording(Recording(status="complete"))

        updated = await db_client.close_watch_event(
            event.uuid, ended_at=datetime.now(timezone.utc), peak_confidence=0.9, recording_uuid=rec.uuid
        )
        assert updated.recording_uuid == rec.uuid
        assert updated.peak_confidence == 0.9
