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
_MASK_A_UUID = "33333333-3333-3333-3333-333333333333"


def _zone(uuid: str = _ZONE_A_UUID, name: str = "litter box area", role: str = "zone") -> dict:
    return {
        "uuid": uuid,
        "name": name,
        "role": role,
        "kind": "box",
        "pointsJson": [[0.1, 0.1], [0.6, 0.7]],
        "referenceWidth": 1920,
        "referenceHeight": 1080,
    }


def _mask(uuid: str = _MASK_A_UUID, name: str = "neighbour's driveway") -> dict:
    return _zone(uuid=uuid, name=name, role="mask")


async def _set_regions(http_client, source_uuid: str, regions: list[dict]) -> dict:
    """Draw regions on a source.

    Regions live on the source rather than inside a watcher, so a test that wants a rule
    pointing at a zone has to create the zone here first — the profile payload no longer
    carries one.
    """
    r = await http_client.post(
        "/api/sources/regions/set",
        json={"sourceUuid": source_uuid, "regions": regions},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _rule(uuid: str = "22222222-2222-2222-2222-222222222222", **overrides) -> dict:
    rule = {
        "uuid": uuid,
        "order": 0,
        "name": "cat using litter box",
        "enabled": True,
        "subjectAKind": "label",
        "subjectARefsJson": ["label-cat"],
        "relation": "contains",
        "subjectBKind": "zone",
        "subjectBRefsJson": [_ZONE_A_UUID],
        "threshold": 0.6,
        "minDurationS": 2.0,
        "rearmDelayS": 10.0,
        "maxDurationS": 300.0,
        "eventLabel": "litterbox",
    }
    # Most tests care about one ref per slot and read better naming it directly.
    for side in ("A", "B"):
        key = f"subject{side}Ref"
        if key in overrides:
            ref = overrides.pop(key)
            overrides[f"subject{side}RefsJson"] = [] if ref is None else [ref]
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
    async def test_create_with_sources_rules_and_actions(self, http_client):
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone()])
        # A real label: label_lifetimes has a real FK, so a made-up uuid is rejected.
        label_uuid = await _make_label(http_client)
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid, "order": 0}],
            rules=[_rule()],
            actions=[{"kind": "publish_mqtt", "trigger": "event_started", "paramsJson": {"retain": False}}],
            labelLifetimes=[{"labelUuid": label_uuid, "ttlS": 2.0, "isAnchor": False}],
        )
        assert body["status"] == "success", body
        profile = body["profile"]
        assert len(profile["sources"]) == 1
        assert len(profile["rules"]) == 1
        assert len(profile["actions"]) == 1
        assert len(profile["labelLifetimes"]) == 1

    async def test_a_sources_mqtt_name_round_trips(self, http_client):
        source_uuid = await _make_source(http_client)
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid, "mqttName": "litterbox"}],
        )
        assert body["status"] == "success", body
        assert body["profile"]["sources"][0]["mqttName"] == "litterbox"

    async def test_a_blank_mqtt_name_is_stored_as_no_override(self, http_client):
        """Otherwise "" and None behave alike but compare differently everywhere."""
        source_uuid = await _make_source(http_client)
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid, "mqttName": "   "}],
        )
        assert body["profile"]["sources"][0]["mqttName"] is None

    async def test_a_source_without_an_mqtt_name_keeps_none(self, http_client):
        """Every watcher predating the field must publish exactly where it did."""
        source_uuid = await _make_source(http_client)
        body = await _create_profile(http_client, "House cams", sources=[{"sourceUuid": source_uuid}])
        assert body["profile"]["sources"][0]["mqttName"] is None

    async def test_update_preserves_rule_uuids_and_zone_references(self, http_client):
        """Regenerating uuids on save would silently break every zone-based rule."""
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone()])
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": source_uuid}],
                rules=[_rule()],
            )
        )["profile"]

        r = await http_client.patch(
            f"/api/watch-profiles/{profile['uuid']}",
            json={
                "description": "now with a description",
                "sources": [{"sourceUuid": source_uuid}],
                "rules": [_rule()],
            },
        )
        updated = r.json()["profile"]
        assert updated["rules"][0]["uuid"] == "22222222-2222-2222-2222-222222222222"
        assert updated["rules"][0]["subjectBRefsJson"] == [_ZONE_A_UUID]
        assert updated["description"] == "now with a description"

    async def test_regions_survive_a_profile_rewrite(self, http_client):
        """A watcher save must not touch the camera's geometry.

        The point of moving regions onto the source: they outlive any one watcher's
        configuration, so rewriting a profile's whole source list leaves them alone.
        """
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone(), _mask()])
        profile = (await _create_profile(http_client, "House cams", sources=[{"sourceUuid": source_uuid}]))["profile"]

        await http_client.patch(
            f"/api/watch-profiles/{profile['uuid']}",
            json={"sources": [{"sourceUuid": source_uuid}]},
        )

        r = await http_client.post("/api/sources/regions/get", json={"sourceUuid": source_uuid})
        assert sorted(x["uuid"] for x in r.json()["regions"]) == sorted([_ZONE_A_UUID, _MASK_A_UUID])

    async def test_omitting_a_list_leaves_it_unchanged(self, http_client):
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone()])
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": source_uuid}],
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
        await _set_regions(http_client, source_uuid, [_zone()])
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": source_uuid}],
                rules=[_rule()],
            )
        )["profile"]

        r = await http_client.patch(f"/api/watch-profiles/{profile['uuid']}", json={"rules": []})
        assert r.json()["profile"]["rules"] == []
        assert len(r.json()["profile"]["sources"]) == 1

    async def test_deleting_a_profile_cascades_to_its_children_but_not_to_regions(self, http_client, db_client):
        from sqlmodel import select

        from apps.helpers.db.types import InteractionRule, SourceRegion

        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone()])
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": source_uuid}],
                rules=[_rule()],
            )
        )["profile"]
        await http_client.delete(f"/api/watch-profiles/{profile['uuid']}")

        async with db_client._async_session_maker() as session:
            assert (await session.exec(select(InteractionRule))).all() == []
            # The camera's geometry belongs to the camera. Deleting a watcher that used
            # it must not erase it for every other watcher.
            assert len((await session.exec(select(SourceRegion))).all()) == 1

    async def test_a_source_listed_twice_is_rejected(self, http_client):
        source_uuid = await _make_source(http_client)
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid}, {"sourceUuid": source_uuid}],
        )
        assert body["status"] == "failure"
        assert "twice" in body["message"]


class TestModelSelection:
    """A watcher runs either a Model Stack or a single model, at both levels.

    Wrapping one model in a one-member stack just to run it was the only way to do
    this, which made the stack a piece of ceremony rather than a choice.
    """

    async def test_a_source_can_name_a_single_model(self, http_client):
        source_uuid = await _make_source(http_client)
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid, "modelName": "cat_detector"}],
        )
        assert body["status"] == "success", body
        src = body["profile"]["sources"][0]
        assert src["modelName"] == "cat_detector"
        assert src["modelGroupUuid"] is None

    async def test_a_profile_default_can_be_a_single_model(self, http_client):
        body = await _create_profile(http_client, "House cams", defaultModelName="cat_detector")
        assert body["status"] == "success", body
        assert body["profile"]["defaultModelName"] == "cat_detector"

    async def test_a_source_naming_both_is_rejected(self, http_client):
        """Both set has no reading — silently preferring one makes the other dropdown a lie."""
        source_uuid = await _make_source(http_client)
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[
                {
                    "sourceUuid": source_uuid,
                    "modelName": "cat_detector",
                    "modelGroupUuid": "22222222-2222-2222-2222-222222222222",
                }
            ],
        )
        assert body["status"] == "failure", body

    async def test_a_profile_default_naming_both_is_rejected(self, http_client):
        body = await _create_profile(
            http_client,
            "House cams",
            defaultModelName="cat_detector",
            defaultModelGroupUuid="22222222-2222-2222-2222-222222222222",
        )
        assert body["status"] == "failure", body

    async def test_switching_a_source_from_a_model_to_none_clears_it(self, http_client):
        """The editor sends both keys on every save; an explicit null must actually clear."""
        source_uuid = await _make_source(http_client)
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": source_uuid, "modelName": "cat_detector"}],
            )
        )["profile"]

        r = await http_client.patch(
            f"/api/watch-profiles/{profile['uuid']}",
            json={"sources": [{"sourceUuid": source_uuid, "modelName": None, "modelGroupUuid": None}]},
        )
        assert r.status_code == 200, r.text
        assert r.json()["profile"]["sources"][0]["modelName"] is None

    async def test_clearing_a_profile_default_model_persists(self, http_client):
        profile = (await _create_profile(http_client, "House cams", defaultModelName="cat_detector"))["profile"]
        r = await http_client.patch(
            f"/api/watch-profiles/{profile['uuid']}",
            json={"defaultModelName": None, "defaultModelGroupUuid": None},
        )
        assert r.status_code == 200, r.text
        assert r.json()["profile"]["defaultModelName"] is None


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
            actions=[{"kind": "record_clip", "ruleUuidsJson": ["no-such-rule"]}],
        )
        assert body["status"] == "failure"
        assert "rule" in body["message"].lower()

    async def test_action_targeting_a_source_not_in_the_profile_is_rejected(self, http_client):
        """The source axis needs the same guard as the rule axis.

        A scope naming a camera the watcher does not watch is silently never satisfied,
        so the action simply never fires — the hardest kind of misconfiguration to spot,
        because everything looks saved and correct.
        """
        source_uuid = await _make_source(http_client)
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid}],
            actions=[{"kind": "record_clip", "sourceUuidsJson": ["no-such-source"]}],
        )
        assert body["status"] == "failure"
        assert "source" in body["message"].lower()

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
        await _set_regions(http_client, source_uuid, [_zone()])
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid}],
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
        await _set_regions(http_client, source_uuid, [_zone()])
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid}],
            rules=[_rule(maxDurationS=0)],
        )
        # It is the runaway-event cap; zero would let one stuck detection record forever.
        assert body["status"] == "failure"

    async def test_defaults_include_hysteresis(self, http_client):
        """A bare threshold flaps; the defaults must already guard against it."""
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone()])
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": source_uuid}],
                rules=[
                    {
                        "name": "minimal",
                        "subjectAKind": "label",
                        "subjectARefsJson": ["label-cat"],
                        "subjectBKind": "zone",
                        "subjectBRefsJson": [_ZONE_A_UUID],
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
        # No margin unless asked for: a rule written before the column existed must keep
        # matching on the raw boxes, not on silently widened ones.
        assert rule["margin"] == 0.0


class TestRuleMargin:
    """The "near" knob: boxes are widened before the relation is measured."""

    async def _profile_with(self, http_client, rule: dict) -> dict:
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone()])
        return await _create_profile(
            http_client,
            f"cams-{_uuid.uuid4().hex[:6]}",
            sources=[{"sourceUuid": source_uuid}],
            rules=[rule],
        )

    async def test_a_margin_round_trips(self, http_client):
        body = await self._profile_with(http_client, _rule(margin=0.75))
        assert body["status"] == "success", body
        assert body["profile"]["rules"][0]["margin"] == pytest.approx(0.75)

    async def test_a_negative_margin_round_trips(self, http_client):
        """The other half of the control: demand a deeper overlap than the boxes show."""
        body = await self._profile_with(http_client, _rule(margin=-0.3))
        assert body["status"] == "success", body
        assert body["profile"]["rules"][0]["margin"] == pytest.approx(-0.3)

    async def test_an_out_of_range_margin_is_rejected(self, http_client):
        """-1.0 erodes a box to nothing, so anything past it can never fire."""
        assert (await self._profile_with(http_client, _rule(margin=-2.0)))["status"] == "failure"
        assert (await self._profile_with(http_client, _rule(margin=5.0)))["status"] == "failure"

    async def test_present_rejects_a_margin(self, http_client):
        """It compares no boxes, so a margin there is a control that does nothing."""
        body = await self._profile_with(
            http_client,
            _rule(relation="present", subjectBKind="none", subjectBRef=None, margin=0.5),
        )
        assert body["status"] == "failure"

    async def test_distance_rejects_a_margin(self, http_client):
        """Its threshold is already a proximity measure and its inputs are centres."""
        body = await self._profile_with(http_client, _rule(relation="distance", margin=0.5))
        assert body["status"] == "failure"

    async def test_present_accepts_a_zero_margin(self, http_client):
        """The editor always sends the field, so zero must not be read as "set"."""
        body = await self._profile_with(
            http_client,
            _rule(relation="present", subjectBKind="none", subjectBRef=None, margin=0.0),
        )
        assert body["status"] == "success", body


class TestRegionValidation:
    """Regions are written through their own endpoint now, so geometry is validated there."""

    async def _post_regions(self, http_client, regions: list[dict]) -> dict:
        source_uuid = await _make_source(http_client)
        r = await http_client.post(
            "/api/sources/regions/set",
            json={"sourceUuid": source_uuid, "regions": regions},
        )
        if r.status_code == 422:
            return {"status": "failure", "message": r.text}
        return r.json()

    async def test_box_needs_exactly_two_points(self, http_client):
        body = await self._post_regions(http_client, [{**_zone(), "pointsJson": [[0.1, 0.1]]}])
        assert body["status"] == "failure"

    async def test_polygon_needs_three_points(self, http_client):
        body = await self._post_regions(
            http_client, [{**_zone(), "kind": "polygon", "pointsJson": [[0.1, 0.1], [0.2, 0.2]]}]
        )
        assert body["status"] == "failure"

    async def test_pixel_coordinates_are_rejected(self, http_client):
        """Regions are stored normalised so they survive a camera resolution change."""
        body = await self._post_regions(http_client, [{**_zone(), "pointsJson": [[100, 100], [500, 500]]}])
        assert body["status"] == "failure"
        assert "normalised" in body["message"]

    async def test_two_zones_may_not_share_a_name(self, http_client):
        body = await self._post_regions(http_client, [_zone(), _zone(uuid="44444444-4444-4444-4444-444444444444")])
        assert body["status"] == "failure"

    async def test_a_zone_and_a_mask_may_share_a_name(self, http_client):
        """They are different things in different lists; only a collision within one is confusing."""
        body = await self._post_regions(http_client, [_zone(name="driveway"), _mask(name="driveway")])
        assert body["status"] == "success", body

    async def test_unknown_source_is_rejected(self, http_client):
        r = await http_client.post(
            "/api/sources/regions/set", json={"sourceUuid": "no-such-source", "regions": [_zone()]}
        )
        assert r.json()["status"] == "failure"


class TestSourceRegionCrud:
    async def test_regions_round_trip_and_keep_their_uuids(self, http_client):
        source_uuid = await _make_source(http_client)
        body = await _set_regions(http_client, source_uuid, [_zone(), _mask()])
        assert body["status"] == "success", body

        r = await http_client.post("/api/sources/regions/get", json={"sourceUuid": source_uuid})
        regions = r.json()["regions"]
        assert {x["uuid"] for x in regions} == {_ZONE_A_UUID, _MASK_A_UUID}
        assert {x["role"] for x in regions} == {"zone", "mask"}

    async def test_role_filter(self, http_client):
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone(), _mask()])
        r = await http_client.post("/api/sources/regions/get", json={"sourceUuid": source_uuid, "role": "mask"})
        assert [x["uuid"] for x in r.json()["regions"]] == [_MASK_A_UUID]

    async def test_listing_every_source_at_once(self, http_client):
        """The rule editor offers zones across all of a watcher's sources in one call."""
        first = await _make_source(http_client)
        second = await _make_source(http_client)
        await _set_regions(http_client, first, [_zone()])
        await _set_regions(http_client, second, [_zone(uuid="55555555-5555-5555-5555-555555555555")])

        r = await http_client.post("/api/sources/regions/get", json={})
        assert len(r.json()["regions"]) == 2

    async def test_removing_a_zone_reports_the_rules_it_orphans(self, http_client):
        """Silently breaking a rule is the failure mode zones-by-uuid invites."""
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone()])
        await _create_profile(http_client, "House cams", sources=[{"sourceUuid": source_uuid}], rules=[_rule()])

        body = await _set_regions(http_client, source_uuid, [])
        assert body["status"] == "success"
        assert body["orphanedRules"] == ["cat using litter box"]

    async def test_removing_several_zones_at_once_reports_every_rule(self, http_client):
        """The orphan check takes the whole removed set in one pass, so it must not
        report only the rules broken by whichever region it happened to look at first."""
        second_zone = "66666666-6666-6666-6666-666666666666"
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone(), _zone(uuid=second_zone, name="water bowl area")])
        await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid}],
            rules=[
                _rule(),
                _rule(
                    uuid="77777777-7777-7777-7777-777777777777",
                    name="cat drinking",
                    subjectBRef=second_zone,
                ),
            ],
        )

        body = await _set_regions(http_client, source_uuid, [])
        assert body["status"] == "success"
        assert body["orphanedRules"] == ["cat drinking", "cat using litter box"]

    async def test_a_surviving_zone_orphans_nothing(self, http_client):
        """Only the removed regions are checked — re-saving an unchanged list is quiet."""
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone()])
        await _create_profile(http_client, "House cams", sources=[{"sourceUuid": source_uuid}], rules=[_rule()])

        body = await _set_regions(http_client, source_uuid, [_zone(name="litter tray")])
        assert body["orphanedRules"] == []

    async def test_deleting_a_source_takes_its_regions_with_it(self, http_client, db_client):
        from sqlmodel import select

        from apps.helpers.db.types import SourceRegion

        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone()])
        r = await http_client.post("/api/sources/delete", json={"sourceUuids": [source_uuid]})
        assert r.json()["status"] == "success", r.text

        async with db_client._async_session_maker() as session:
            assert (await session.exec(select(SourceRegion))).all() == []


class TestMultiRefRules:
    """A rule slot holds a list, so one rule covers several cats and several litter boxes."""

    async def test_a_rule_may_name_several_labels(self, http_client):
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone()])
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid}],
            rules=[_rule(subjectARefsJson=["label-a", "label-b", "label-c"])],
        )
        assert body["status"] == "success", body
        assert body["profile"]["rules"][0]["subjectARefsJson"] == ["label-a", "label-b", "label-c"]

    async def test_a_rule_may_name_several_zones(self, http_client):
        source_uuid = await _make_source(http_client)
        second_zone = _zone(uuid="66666666-6666-6666-6666-666666666666", name="upstairs box")
        await _set_regions(http_client, source_uuid, [_zone(), second_zone])
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid}],
            rules=[_rule(subjectBRefsJson=[_ZONE_A_UUID, second_zone["uuid"]])],
        )
        assert body["status"] == "success", body

    async def test_an_empty_slot_is_rejected(self, http_client):
        """A slot naming nothing can never fire, and reads as merely unconfigured."""
        body = await _create_profile(http_client, "House cams", rules=[_rule(subjectARefsJson=[])])
        assert body["status"] == "failure"

    async def test_duplicate_refs_are_rejected(self, http_client):
        body = await _create_profile(http_client, "House cams", rules=[_rule(subjectARefsJson=["label-a", "label-a"])])
        assert body["status"] == "failure"

    async def test_a_zone_ref_not_on_any_of_the_sources_is_rejected(self, http_client):
        """A rule naming a zone no source defines never fires — invisible at runtime."""
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone()])
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid}],
            rules=[_rule(subjectBRefsJson=["77777777-7777-7777-7777-777777777777"])],
        )
        assert body["status"] == "failure"
        assert "not drawn" in body["message"]

    async def test_a_mask_uuid_is_not_a_valid_zone_ref(self, http_client):
        """Masks aren't addressable by rules — they act on everything or not at all."""
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone(), _mask()])
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid}],
            rules=[_rule(subjectBRefsJson=[_MASK_A_UUID])],
        )
        assert body["status"] == "failure"


class TestMaskOptOut:
    async def test_a_watcher_may_disable_a_mask(self, http_client):
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_mask()])
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid, "disabledMaskUuidsJson": [_MASK_A_UUID]}],
        )
        assert body["status"] == "success", body
        assert body["profile"]["sources"][0]["disabledMaskUuidsJson"] == [_MASK_A_UUID]

    async def test_masks_apply_by_default(self, http_client):
        """Zero configuration for the common case: a mask added later protects old watchers."""
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_mask()])
        body = await _create_profile(http_client, "House cams", sources=[{"sourceUuid": source_uuid}])
        assert body["profile"]["sources"][0]["disabledMaskUuidsJson"] == []

    async def test_disabling_a_mask_that_is_not_on_the_source_is_rejected(self, http_client):
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_mask()])
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid, "disabledMaskUuidsJson": ["88888888-8888-8888-8888-888888888888"]}],
        )
        assert body["status"] == "failure"

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


class TestRuleSourceScope:
    """A rule may be restricted to particular sources; empty means every source.

    The case it exists for is `present`: "a cat is in the scene" is true in every room
    all day, and only worth an event in one of them. Scoping the action instead would
    leave the event raised, logged and browsable on every camera.
    """

    async def test_a_rule_is_unscoped_by_default(self, http_client):
        """Empty means all — a rule written before this column existed still runs everywhere."""
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone()])
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid}],
            rules=[_rule()],
        )
        assert body["status"] == "success", body
        assert body["profile"]["rules"][0]["sourceUuidsJson"] == []

    async def test_a_scope_round_trips(self, http_client):
        kitchen = await _make_source(http_client, "kitchen")
        hallway = await _make_source(http_client, "hallway")
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": kitchen}, {"sourceUuid": hallway}],
            rules=[_rule(relation="present", subjectBKind="none", subjectBRef=None, sourceUuidsJson=[kitchen])],
        )
        assert body["status"] == "success", body
        assert body["profile"]["rules"][0]["sourceUuidsJson"] == [kitchen]

        r = await http_client.get(f"/api/watch-profiles/{body['profile']['uuid']}")
        assert r.json()["profile"]["rules"][0]["sourceUuidsJson"] == [kitchen]

    async def test_a_scope_survives_an_update(self, http_client):
        """A nested write replaces every child row, so this is where a dropped field shows."""
        kitchen = await _make_source(http_client, "kitchen")
        created = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": kitchen}],
            rules=[_rule(relation="present", subjectBKind="none", subjectBRef=None, sourceUuidsJson=[kitchen])],
        )
        uuid = created["profile"]["uuid"]
        r = await http_client.patch(
            f"/api/watch-profiles/{uuid}",
            json={
                "sources": [{"sourceUuid": kitchen}],
                "rules": [_rule(relation="present", subjectBKind="none", subjectBRef=None, sourceUuidsJson=[kitchen])],
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["profile"]["rules"][0]["sourceUuidsJson"] == [kitchen]

    async def test_a_rule_scoped_to_a_source_not_in_the_profile_is_rejected(self, http_client):
        """Same guard as the action's source axis: a scope naming a camera this watcher
        does not watch is never satisfied, so the rule silently never fires."""
        source_uuid = await _make_source(http_client)
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid}],
            rules=[
                _rule(
                    relation="present",
                    subjectBKind="none",
                    subjectBRef=None,
                    sourceUuidsJson=["no-such-source"],
                )
            ],
        )
        assert body["status"] == "failure"
        assert "source" in body["message"].lower()

    async def test_a_duplicated_source_in_the_scope_is_rejected(self, http_client):
        source_uuid = await _make_source(http_client)
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid}],
            rules=[
                _rule(
                    relation="present",
                    subjectBKind="none",
                    subjectBRef=None,
                    sourceUuidsJson=[source_uuid, source_uuid],
                )
            ],
        )
        assert body["status"] == "failure"


class TestActionNames:
    """An action can carry a note. The engine never reads it — it exists because three
    "Save snapshot" rows scoped to different cameras are otherwise indistinguishable in
    the list, and the parameters that tell them apart are spread across a dense row."""

    async def test_a_name_round_trips(self, http_client):
        source_uuid = await _make_source(http_client)
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": source_uuid}],
                actions=[
                    {
                        "kind": "save_timelapse",
                        "trigger": "frame_analysed",
                        "name": "Back garden timelapse every 30s",
                    }
                ],
            )
        )["profile"]
        assert profile["actions"][0]["name"] == "Back garden timelapse every 30s"

    async def test_an_omitted_name_is_null_not_empty(self, http_client):
        """Every action that existed before this field has none, and an empty string
        would render as a blank label rather than as "unnamed"."""
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                actions=[{"kind": "publish_mqtt", "trigger": "event_started"}],
            )
        )["profile"]
        assert profile["actions"][0]["name"] is None

    async def test_a_blank_name_is_stored_as_null(self, http_client):
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                actions=[{"kind": "publish_mqtt", "trigger": "event_started", "name": "   "}],
            )
        )["profile"]
        assert profile["actions"][0]["name"] is None

    async def test_a_name_survives_an_update(self, http_client):
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                actions=[{"kind": "publish_mqtt", "trigger": "event_started", "name": "shout at me"}],
            )
        )["profile"]
        r = await http_client.patch(
            f"/api/watch-profiles/{profile['uuid']}",
            json={
                "name": "House cams",
                "actions": [
                    {"kind": "publish_mqtt", "trigger": "event_started", "name": "shout at me louder"},
                ],
            },
        )
        assert r.json()["profile"]["actions"][0]["name"] == "shout at me louder"


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
        assert any("no models selected" in e for e in report["errors"])

    async def test_a_source_with_only_a_bare_model_is_not_an_error(self, http_client):
        """A single model is a complete selection — it does not need a one-member stack."""
        source_uuid = await _make_source(http_client)
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": source_uuid, "modelName": "cat_detector"}],
            )
        )["profile"]
        report = (await http_client.get(f"/api/watch-profiles/{profile['uuid']}/validate")).json()
        assert not any("no models selected" in e for e in report["errors"]), report

    async def test_a_profile_default_bare_model_covers_its_sources(self, http_client):
        source_uuid = await _make_source(http_client)
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                defaultModelName="cat_detector",
                sources=[{"sourceUuid": source_uuid}],
            )
        )["profile"]
        report = (await http_client.get(f"/api/watch-profiles/{profile['uuid']}/validate")).json()
        assert not any("no models selected" in e for e in report["errors"]), report

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
        await _set_regions(http_client, source_uuid, [_zone()])
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": source_uuid, "recordRetentionS": 20.0}],
                rules=[_rule()],
                actions=[{"kind": "record_clip", "paramsJson": {"preroll_s": 30}}],
            )
        )["profile"]
        report = (await http_client.get(f"/api/watch-profiles/{profile['uuid']}/validate")).json()
        assert any("pre-roll" in w for w in report["warnings"])

    async def test_a_rule_scoped_away_from_its_zone_is_an_error(self, http_client):
        """A zone belongs to one camera, so a rule narrowed to another can never fire.

        Only reachable by editing the source scope after the rule, which is exactly when
        nobody re-reads the rule — and the editor shows a fully configured row either way.
        """
        kitchen = await _make_source(http_client, "kitchen")
        hallway = await _make_source(http_client, "hallway")
        await _set_regions(http_client, kitchen, [_zone()])
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": kitchen}, {"sourceUuid": hallway}],
                rules=[_rule(sourceUuidsJson=[hallway])],
            )
        )["profile"]
        report = (await http_client.get(f"/api/watch-profiles/{profile['uuid']}/validate")).json()
        assert report["valid"] is False
        assert any("none of its zones are drawn on" in e for e in report["errors"]), report

    async def test_a_rule_scoped_to_the_source_its_zone_is_on_is_fine(self, http_client):
        kitchen = await _make_source(http_client, "kitchen")
        hallway = await _make_source(http_client, "hallway")
        await _set_regions(http_client, kitchen, [_zone()])
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": kitchen}, {"sourceUuid": hallway}],
                rules=[_rule(sourceUuidsJson=[kitchen])],
            )
        )["profile"]
        report = (await http_client.get(f"/api/watch-profiles/{profile['uuid']}/validate")).json()
        assert not any("none of its zones are drawn on" in e for e in report["errors"]), report

    async def test_a_rule_scoped_only_to_disabled_sources_warns(self, http_client):
        kitchen = await _make_source(http_client, "kitchen")
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": kitchen, "enabled": False}],
                rules=[_rule(relation="present", subjectBKind="none", subjectBRef=None, sourceUuidsJson=[kitchen])],
            )
        )["profile"]
        report = (await http_client.get(f"/api/watch-profiles/{profile['uuid']}/validate")).json()
        assert any("never be evaluated" in w for w in report["warnings"]), report

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

    async def test_a_timelapse_faster_than_the_analysis_rate_warns(self, http_client):
        """A timelapse can only keep a frame the watcher pulled, and nothing at run time
        says the interval was quietly widened to the analysis period."""
        source_uuid = await _make_source(http_client)
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                # One frame every 10 s, against an action asking for one every 5.
                sources=[{"sourceUuid": source_uuid, "analysisFps": 0.1}],
                actions=[{"kind": "save_timelapse", "trigger": "frame_analysed", "paramsJson": {"interval_s": 5}}],
            )
        )["profile"]
        report = (await http_client.get(f"/api/watch-profiles/{profile['uuid']}/validate")).json()
        assert any("written at the slower rate" in w for w in report["warnings"])

    async def test_a_timelapse_folder_starting_with_a_placeholder_is_an_error(self, http_client):
        """The retention sweep bounds itself to the literal part of the path. With none,
        the only folder that would bound it is the dataset root itself."""
        source_uuid = await _make_source(http_client)
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": source_uuid}],
                actions=[
                    {
                        "kind": "save_timelapse",
                        "trigger": "frame_analysed",
                        "paramsJson": {"output_dir": "{source}/frames"},
                    }
                ],
            )
        )["profile"]
        report = (await http_client.get(f"/api/watch-profiles/{profile['uuid']}/validate")).json()
        assert any("starts with a placeholder" in e for e in report["errors"])
        assert report["valid"] is False

    async def test_a_folder_with_a_real_prefix_is_accepted(self, http_client):
        source_uuid = await _make_source(http_client)
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": source_uuid}],
                actions=[
                    {
                        "kind": "save_timelapse",
                        "trigger": "frame_analysed",
                        "paramsJson": {"output_dir": "timelapse/{source}"},
                    }
                ],
            )
        )["profile"]
        report = (await http_client.get(f"/api/watch-profiles/{profile['uuid']}/validate")).json()
        assert not any("placeholder" in e for e in report["errors"])

    async def test_a_timelapse_slower_than_the_analysis_rate_is_silent(self, http_client):
        source_uuid = await _make_source(http_client)
        profile = (
            await _create_profile(
                http_client,
                "House cams",
                sources=[{"sourceUuid": source_uuid, "analysisFps": 5.0}],
                actions=[{"kind": "save_timelapse", "trigger": "frame_analysed", "paramsJson": {"interval_s": 30}}],
            )
        )["profile"]
        report = (await http_client.get(f"/api/watch-profiles/{profile['uuid']}/validate")).json()
        assert not any("slower rate" in w for w in report["warnings"])

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


class TestTrackerListing:
    """The Watchers pane builds its tracker picker from this, so a new tracker must
    appear without any UI change — that is the whole point of the registry."""

    async def test_lists_registered_trackers_with_schemas(self, http_client):
        resp = await http_client.get("/api/watch-profiles/trackers")
        assert resp.status_code == 200
        body = resp.json()

        by_typename = {t["typename"]: t for t in body["trackers"]}
        assert {"IouTracker", "ByteTracker"} <= set(by_typename)
        assert by_typename["ByteTracker"]["displayName"] == "ByteTrack (two-stage)"
        # Schema *keys* are data, not model fields, so they stay snake_case and reach the
        # form machinery exactly as the tracker declared them — same as provider schemas.
        assert "track_thresh" in by_typename["ByteTracker"]["paramsSchema"]

    async def test_route_is_not_shadowed_by_the_uuid_route(self, http_client):
        """`/trackers` must not be read as a profile uuid — it is declared first."""
        resp = await http_client.get("/api/watch-profiles/trackers")

        assert resp.json().get("trackers"), "the parameterised route swallowed the literal one"


class TestIncludeDescendants:
    """The flag is stored, not expanded at write time.

    Expanding on save would freeze the rule at the moment it was written: a label added
    to the tree later would silently stop matching, which is the maintenance the flag
    exists to remove.
    """

    async def test_the_flag_round_trips(self, http_client):
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone()])
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid}],
            rules=[_rule(subjectAIncludeDescendants=True)],
        )
        assert body["status"] == "success", body
        assert body["profile"]["rules"][0]["subjectAIncludeDescendants"] is True

    async def test_it_defaults_off(self, http_client):
        """An existing rule meant exactly the labels it named."""
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone()])
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid}],
            rules=[_rule()],
        )
        assert body["profile"]["rules"][0]["subjectAIncludeDescendants"] is False
        assert body["profile"]["rules"][0]["subjectBIncludeDescendants"] is False

    async def test_the_refs_are_not_expanded_on_save(self, http_client):
        """Storing the closure is exactly what this design avoids."""
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone()])
        parent = await _make_label(http_client, "Cat")
        r = await http_client.post("/api/labels", json={"name": "Tabby", "color": "#abc", "parentUuid": parent})
        assert r.json()["status"] == "success"

        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid}],
            rules=[_rule(subjectARefsJson=[parent], subjectAIncludeDescendants=True)],
        )
        assert body["profile"]["rules"][0]["subjectARefsJson"] == [parent]

    async def test_a_label_subject_b_may_include_descendants(self, http_client):
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone()])
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid}],
            rules=[
                _rule(
                    subjectBKind="label",
                    subjectBRefsJson=["label-carrier"],
                    subjectBIncludeDescendants=True,
                )
            ],
        )
        assert body["status"] == "success", body

    async def test_a_zone_slot_may_not_include_descendants(self, http_client):
        """Zones have no hierarchy, so the flag would be a control that does nothing."""
        source_uuid = await _make_source(http_client)
        await _set_regions(http_client, source_uuid, [_zone()])
        body = await _create_profile(
            http_client,
            "House cams",
            sources=[{"sourceUuid": source_uuid}],
            rules=[_rule(subjectBIncludeDescendants=True)],
        )
        assert body["status"] == "failure"
        assert "no hierarchy" in body["message"]
