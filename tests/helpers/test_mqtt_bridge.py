"""Tests for the MQTT bridge's discovery generation, counter seeding and availability.

Per HA_INTEGRATION_PLAN.md WP-2: pure logic gets tests, the live connection does not.
Every test here exercises the pure functions directly (or `MqttBridge` with a fake
publisher standing in for the network) — nothing opens a socket.

The two regressions worth guarding: rule/source entities must survive a rename (their
`unique_id` is uuid-based even though the topic path is name-based), and a rule/source
that disappears from config must have its discovery retracted, not left retained
forever.
"""

import json

from apps.helpers.mqtt.bridge import (
    DiscoveryState,
    MqttBridge,
    ProfileView,
    RuleView,
    SourceView,
    build_all_discovery,
    build_seed_publishes,
    build_slug_index,
    build_source_discovery,
    reconcile_discovery,
    source_config_topic,
    state_from_retained,
)

PREFIX = "whiskerwatcher"


def _profile(**overrides) -> ProfileView:
    kwargs = dict(
        uuid="profile-1",
        name="Living room watcher",
        sources=(SourceView(uuid="src-1", name="Kitchen cam"),),
        rules=(RuleView(uuid="rule-1", name="cat at bowl"),),
    )
    kwargs.update(overrides)
    return ProfileView(**kwargs)


class TestSourceDiscoveryShape:
    def test_rule_gets_the_documented_entity_set(self):
        profile = _profile()
        payload = build_source_discovery(profile, profile.sources[0], PREFIX)
        cmps = payload["cmps"]
        # One component per documented entity, keyed distinctly, plus the per-source
        # ones. Not an existence check on any one of them — the count is what proves
        # nothing collapsed two components onto the same key.
        rule_components = {k for k in cmps if k.startswith("rule-1_")}
        assert rule_components == {
            "rule-1_state",
            "rule-1_count",
            "rule-1_last_seen",
            "rule-1_last_duration",
            "rule-1_peak_confidence",
            "rule-1_event",
        }

    def test_a_rule_scoped_to_another_source_is_excluded(self):
        """Empty source_uuids means "every source"; a non-empty list is a real scope,
        and a source outside it must not get that rule's entities."""
        profile = _profile(
            sources=(SourceView(uuid="src-1", name="Kitchen"), SourceView(uuid="src-2", name="Litter room")),
            rules=(RuleView(uuid="rule-1", name="cat at bowl", source_uuids=("src-1",)),),
        )
        kitchen, litter_room = profile.sources
        assert any(k.startswith("rule-1_") for k in build_source_discovery(profile, kitchen, PREFIX)["cmps"])
        assert not any(k.startswith("rule-1_") for k in build_source_discovery(profile, litter_room, PREFIX)["cmps"])

    def test_a_disabled_rule_is_excluded(self):
        profile = _profile(rules=(RuleView(uuid="rule-1", name="cat at bowl", enabled=False),))
        cmps = build_source_discovery(profile, profile.sources[0], PREFIX)["cmps"]
        assert not any(k.startswith("rule-1_") for k in cmps)

    def test_device_identity_is_the_source_uuid_not_its_name(self):
        profile = _profile()
        payload = build_source_discovery(profile, profile.sources[0], PREFIX)
        assert payload["dev"]["ids"] == "src-1"


class TestRuleUniqueIdSurvivesRename:
    """The bug HA_INTEGRATION_PLAN.md WP-2 step 5a exists to fix: a rule rename must
    change the topic path (cosmetic) but never the unique_id (identity)."""

    def test_unique_id_is_stable_across_a_rename_but_the_topic_path_changes(self):
        before = _profile(rules=(RuleView(uuid="rule-1", name="cat at bowl"),))
        after = _profile(rules=(RuleView(uuid="rule-1", name="bowl visit"),))

        before_state = build_source_discovery(before, before.sources[0], PREFIX)["cmps"]["rule-1_state"]
        after_state = build_source_discovery(after, after.sources[0], PREFIX)["cmps"]["rule-1_state"]

        assert before_state["unique_id"] == after_state["unique_id"]
        assert before_state["state_topic"] != after_state["state_topic"]
        assert "cat-at-bowl" in before_state["state_topic"]
        assert "bowl-visit" in after_state["state_topic"]

    def test_source_device_identity_is_stable_across_a_rename(self):
        before = _profile(sources=(SourceView(uuid="src-1", name="Kitchen cam"),))
        after = _profile(sources=(SourceView(uuid="src-1", name="Dining room cam"),))

        assert (
            build_source_discovery(before, before.sources[0], PREFIX)["dev"]["ids"]
            == build_source_discovery(after, after.sources[0], PREFIX)["dev"]["ids"]
        )


class TestDiscoveryReconciliation:
    """Rule/source deletion must retract, not just stop updating."""

    def test_a_deleted_rule_is_tombstoned_before_it_is_omitted(self):
        """Omitting a component from `cmps` does not remove its entity — HA requires it
        be republished as an empty config naming only its platform first, and omitted
        only on a later update. Publishing the clean message straight away leaves an
        entity in HA that nothing can remove but hand-editing the entity registry.
        """
        with_rule = _profile(rules=(RuleView(uuid="rule-1", name="cat at bowl"),))
        without_rule = _profile(rules=())
        topic = source_config_topic("src-1")

        _, state = reconcile_discovery(DiscoveryState(), build_all_discovery([with_rule], PREFIX))
        removal_pass, state = reconcile_discovery(state, build_all_discovery([without_rule], PREFIX))

        tombstones = {k: v for k, v in removal_pass[topic]["cmps"].items() if k.startswith("rule-1_")}
        assert tombstones, "the deleted rule's components must still be named, as tombstones"
        # Each carries its platform and nothing else — that is what makes it a removal
        # rather than a malformed entity.
        assert all(set(v) == {"p"} and v["p"] for v in tombstones.values())

        # …and the pass after that publishes clean, which is the second half HA asks
        # for. Without this the tombstones would be re-sent forever.
        follow_up, _ = reconcile_discovery(state, build_all_discovery([without_rule], PREFIX))
        assert not any(k.startswith("rule-1_") for k in follow_up[topic]["cmps"])

    def test_an_unchanged_rule_is_never_tombstoned(self):
        """The tombstone must key off disappearance, not off being republished — every
        desired topic is republished every pass."""
        profile = _profile()
        topic = source_config_topic("src-1")

        _, state = reconcile_discovery(DiscoveryState(), build_all_discovery([profile], PREFIX))
        again, _ = reconcile_discovery(state, build_all_discovery([profile], PREFIX))

        assert all(set(v) != {"p"} for v in again[topic]["cmps"].values())

    def test_deleting_a_rule_does_not_retract_the_device_it_was_on(self):
        """Only the rule went away, not the camera. Retracting the whole device would
        take the camera, telemetry and every other rule down with it."""
        with_rule = _profile(rules=(RuleView(uuid="rule-1", name="cat at bowl"),))
        without_rule = _profile(rules=())

        _, state = reconcile_discovery(DiscoveryState(), build_all_discovery([with_rule], PREFIX))
        to_publish, _ = reconcile_discovery(state, build_all_discovery([without_rule], PREFIX))

        topic = source_config_topic("src-1")
        assert to_publish[topic] is not None
        assert "camera" in to_publish[topic]["cmps"]

    def test_state_read_back_from_the_broker_can_retract_across_a_restart(self):
        """The bug the retained read-back exists to fix. With an in-memory state, a
        restart forgets what was published, so a rule deleted while tasks_server was
        down can never be retracted — the process concludes it has told HA nothing and
        can only ever add. Adopting the broker's retained configs is what makes the
        first reconcile after a restart able to remove.
        """
        with_rule = _profile(rules=(RuleView(uuid="rule-1", name="cat at bowl"),))
        without_rule = _profile(rules=())
        topic = source_config_topic("src-1")

        # What the broker is holding, published by a previous run of the process.
        published = build_all_discovery([with_rule], PREFIX)
        retained = {t: json.dumps(p).encode() for t, p in published.items()}

        # A fresh process: no memory at all, only what the broker replays.
        adopted = state_from_retained(retained)
        to_publish, _ = reconcile_discovery(adopted, build_all_discovery([without_rule], PREFIX))

        tombstones = {k: v for k, v in to_publish[topic]["cmps"].items() if k.startswith("rule-1_")}
        assert tombstones, "a rule deleted while the process was down must still be retracted"
        assert all(set(v) == {"p"} for v in tombstones.values())

    def test_an_empty_retained_payload_is_not_adopted_as_a_live_config(self):
        """Empty payloads are retractions someone already published. Adopting one would
        resurrect the entity it removed, on the very next reconcile."""
        assert state_from_retained({source_config_topic("src-1"): b""}).components == {}

    def test_an_unparseable_retained_config_is_skipped_not_fatal(self):
        """One malformed retained message must not stop the bridge coming up — and the
        configs around it must still be adopted."""
        good = build_all_discovery([_profile()], PREFIX)
        retained = {t: json.dumps(p).encode() for t, p in good.items()}
        retained["homeassistant/device/whiskerwatcher_junk/config"] = b"{not json"

        state = state_from_retained(retained)

        assert source_config_topic("src-1") in state.topics
        assert "homeassistant/device/whiskerwatcher_junk/config" not in state.topics

    def test_a_deleted_source_is_retracted_with_an_empty_payload(self):
        with_source = [_profile()]
        without_source = [_profile(sources=(), rules=())]

        _, previous_topics = reconcile_discovery(DiscoveryState(), build_all_discovery(with_source, PREFIX))
        to_publish, remaining = reconcile_discovery(previous_topics, build_all_discovery(without_source, PREFIX))

        topic = source_config_topic("src-1")
        assert to_publish[topic] is None
        assert topic not in remaining.topics

    def test_an_unchanged_profile_produces_no_retractions(self):
        profile = [_profile()]
        _, previous_topics = reconcile_discovery(DiscoveryState(), build_all_discovery(profile, PREFIX))
        to_publish, _ = reconcile_discovery(previous_topics, build_all_discovery(profile, PREFIX))
        assert all(payload is not None for payload in to_publish.values())

    def test_two_profiles_sharing_one_source_merge_rather_than_clobber(self):
        shared_source = SourceView(uuid="src-1", name="Kitchen cam")
        profile_a = ProfileView(
            uuid="p-a", name="A", sources=(shared_source,), rules=(RuleView(uuid="rule-a", name="rule a"),)
        )
        profile_b = ProfileView(
            uuid="p-b", name="B", sources=(shared_source,), rules=(RuleView(uuid="rule-b", name="rule b"),)
        )

        desired = build_all_discovery([profile_a, profile_b], PREFIX)
        cmps = desired[source_config_topic("src-1")]["cmps"]
        assert any(k.startswith("rule-a_") for k in cmps)
        assert any(k.startswith("rule-b_") for k in cmps)


class TestCounterSeeding:
    def test_seeded_counter_matches_the_grouped_count(self):
        profiles = [_profile()]
        counts = [{"source_uuid": "src-1", "rule_uuid": "rule-1", "count": 42}]
        seeded = build_seed_publishes(profiles, counts)
        assert seeded["living-room-watcher/kitchen-cam/rule/cat-at-bowl/count"] == 42

    def test_a_count_for_a_deleted_rule_is_skipped_not_seeded_under_a_bare_uuid(self):
        """A count whose rule no longer exists in any profile has no slug to seed
        onto — it must be dropped, not published under some fallback topic that would
        look like a deliberately-created entity."""
        profiles = [_profile(rules=())]
        counts = [{"source_uuid": "src-1", "rule_uuid": "rule-1", "count": 5}]
        assert build_seed_publishes(profiles, counts) == {}


class _FakePublisher:
    def __init__(self):
        self.calls = []

    def publish(self, topic, payload, retain=False, qos=0, coalesce=False, absolute=False):
        self.calls.append({"topic": topic, "payload": payload, "retain": retain, "absolute": absolute})


class TestAvailability:
    def test_a_stopped_watcher_publishes_offline_on_its_own_topic_only(self):
        bridge = MqttBridge(db_api_client=None)
        fake = _FakePublisher()
        bridge._publisher = fake

        bridge.set_watcher_availability("Living room watcher", online=False)

        assert len(fake.calls) == 1
        call = fake.calls[0]
        assert call["topic"] == "watcher/living-room-watcher/status"
        assert call["payload"] == "offline"
        # Not the app-level LWT topic, and not marked absolute (it's under our own
        # prefix, unlike a discovery config topic).
        assert call["topic"] != "status"
        assert call["absolute"] is False

    def test_online_and_offline_publish_different_payloads(self):
        bridge = MqttBridge(db_api_client=None)
        fake = _FakePublisher()
        bridge._publisher = fake

        bridge.set_watcher_availability("Living room watcher", online=True)
        bridge.set_watcher_availability("Living room watcher", online=False)

        assert [c["payload"] for c in fake.calls] == ["online", "offline"]

    def test_without_a_connected_publisher_it_is_a_silent_no_op(self):
        bridge = MqttBridge(db_api_client=None)
        bridge.set_watcher_availability("Living room watcher", online=True)  # must not raise


def _connected_bridge(profiles) -> tuple:
    """An `MqttBridge` wired to a fake publisher and primed with `profiles`, as a
    reconcile would leave it. No socket, no DB."""
    bridge = MqttBridge(db_api_client=None)
    fake = _FakePublisher()
    bridge._publisher = fake
    bridge._prefix = PREFIX
    bridge._slugs = build_slug_index(profiles)
    return bridge, fake


def _published(fake, suffix: str):
    return [c for c in fake.calls if c["topic"].endswith(suffix)]


class TestLivePublishingMatchesDiscovery:
    """The regression that matters most here, and the one that throws nothing when it
    breaks: discovery tells HA which topic an entity reads, and the live publish has to
    write to that exact topic. Two f-strings drifting apart gives an entity that exists
    and never updates — indistinguishable, from HA's side, from a dead camera.
    """

    def test_state_count_and_attribute_topics_are_the_ones_discovery_advertised(self):
        profile = _profile()
        bridge, fake = _connected_bridge([profile])
        cmps = build_source_discovery(profile, profile.sources[0], PREFIX)["cmps"]

        bridge.publish_rule_transition(
            profile_uuid="profile-1",
            source_uuid="src-1",
            rule_uuid="rule-1",
            active=True,
            opened=True,
            last_seen="2026-01-01T00:00:00+00:00",
        )

        # Published topics are prefix-relative; discovery's are absolute.
        written = {f"{PREFIX}/{c['topic']}" for c in fake.calls}
        assert cmps["rule-1_state"]["state_topic"] in written
        assert cmps["rule-1_count"]["state_topic"] in written
        assert cmps["rule-1_state"]["json_attributes_topic"] in written
        # The templated sensors read the attributes topic rather than one of their own,
        # so they are covered by the same publish.
        assert cmps["rule-1_last_duration"]["state_topic"] in written

    def test_telemetry_topic_is_the_one_discovery_advertised(self):
        profile = _profile()
        bridge, fake = _connected_bridge([profile])
        cmps = build_source_discovery(profile, profile.sources[0], PREFIX)["cmps"]

        bridge.publish_telemetry(profile_uuid="profile-1", source_uuid="src-1", payload={"analysisFps": 5})

        assert f"{PREFIX}/{fake.calls[0]['topic']}" == cmps["fps"]["state_topic"]


class TestTopicsAreProfileScoped:
    """Two watch profiles may watch one camera. Source-scoped topics made two distinct
    entities share one state topic, so they moved in lockstep and overwrote each
    other's counts.
    """

    def test_two_profiles_with_a_same_named_rule_on_one_source_do_not_share_topics(self):
        shared = SourceView(uuid="src-1", name="Kitchen cam")
        a = ProfileView(uuid="p-a", name="Daytime", sources=(shared,), rules=(RuleView(uuid="rule-a", name="motion"),))
        b = ProfileView(
            uuid="p-b", name="Overnight", sources=(shared,), rules=(RuleView(uuid="rule-b", name="motion"),)
        )
        bridge, fake = _connected_bridge([a, b])

        for profile_uuid, rule_uuid in (("p-a", "rule-a"), ("p-b", "rule-b")):
            bridge.publish_rule_transition(
                profile_uuid=profile_uuid,
                source_uuid="src-1",
                rule_uuid=rule_uuid,
                active=True,
                opened=True,
                last_seen="2026-01-01T00:00:00+00:00",
            )

        state_topics = [c["topic"] for c in _published(fake, "/state")]
        assert len(state_topics) == 2
        assert len(set(state_topics)) == 2

    def test_telemetry_is_per_watcher_not_per_source(self):
        """fps and queue depth describe a watcher's analysis of a source, so two
        profiles watching one camera must not overwrite each other's diagnostics."""
        shared = SourceView(uuid="src-1", name="Kitchen cam")
        a = ProfileView(uuid="p-a", name="Daytime", sources=(shared,))
        b = ProfileView(uuid="p-b", name="Overnight", sources=(shared,))
        bridge, fake = _connected_bridge([a, b])

        bridge.publish_telemetry(profile_uuid="p-a", source_uuid="src-1", payload={"analysisFps": 5})
        bridge.publish_telemetry(profile_uuid="p-b", source_uuid="src-1", payload={"analysisFps": 1})

        assert len({c["topic"] for c in fake.calls}) == 2


class TestLiveCounters:
    def test_an_opening_event_increments_past_the_seeded_value(self):
        profile = _profile()
        bridge, fake = _connected_bridge([profile])
        bridge._counts = {("profile-1", "src-1", "rule-1"): 42}

        bridge.publish_rule_transition(
            profile_uuid="profile-1",
            source_uuid="src-1",
            rule_uuid="rule-1",
            active=True,
            opened=True,
            last_seen="2026-01-01T00:00:00+00:00",
        )

        assert [c["payload"] for c in _published(fake, "/count")] == [43]

    def test_update_and_end_transitions_do_not_count_the_same_event_again(self):
        profile = _profile()
        bridge, fake = _connected_bridge([profile])
        bridge._counts = {("profile-1", "src-1", "rule-1"): 42}

        for active, opened in ((True, True), (True, False), (False, False)):
            bridge.publish_rule_transition(
                profile_uuid="profile-1",
                source_uuid="src-1",
                rule_uuid="rule-1",
                active=active,
                opened=opened,
                last_seen="2026-01-01T00:00:00+00:00",
            )

        # One event, three transitions, one increment.
        assert [c["payload"] for c in _published(fake, "/count")] == [43]
        assert [c["payload"] for c in _published(fake, "/state")] == ["ON", "ON", "OFF"]

    def test_a_rename_does_not_restart_the_running_total(self):
        """The counter is keyed by uuid, not by topic. Keyed by topic, renaming a rule
        would orphan its total and restart the sensor at 1."""
        before = _profile(rules=(RuleView(uuid="rule-1", name="cat at bowl"),))
        after = _profile(rules=(RuleView(uuid="rule-1", name="bowl visit"),))
        bridge, fake = _connected_bridge([before])
        bridge._counts = {("profile-1", "src-1", "rule-1"): 42}

        kwargs = dict(
            profile_uuid="profile-1",
            source_uuid="src-1",
            rule_uuid="rule-1",
            active=True,
            opened=True,
            last_seen="2026-01-01T00:00:00+00:00",
        )
        bridge.publish_rule_transition(**kwargs)
        bridge._slugs = build_slug_index([after])  # the rename lands on the next reconcile
        bridge.publish_rule_transition(**kwargs)

        counts = _published(fake, "/count")
        assert [c["payload"] for c in counts] == [43, 44]
        # …onto the new topic, which is the half a uuid-keyed counter still has to get
        # right for HA to follow the entity across the rename.
        assert "bowl-visit" in counts[-1]["topic"]


class TestLivePublishingGuards:
    def test_a_rule_the_bridge_has_not_reconciled_yet_publishes_nothing(self):
        """No discovery entry exists for it, so there is no entity waiting — publishing
        under a guessed topic would create a retained orphan."""
        bridge, fake = _connected_bridge([_profile()])

        bridge.publish_rule_transition(
            profile_uuid="profile-1",
            source_uuid="src-1",
            rule_uuid="rule-brand-new",
            active=True,
            opened=True,
            last_seen="2026-01-01T00:00:00+00:00",
        )

        assert fake.calls == []

    def test_without_a_connected_publisher_it_is_a_silent_no_op(self):
        bridge = MqttBridge(db_api_client=None)
        bridge.publish_rule_transition(
            profile_uuid="profile-1",
            source_uuid="src-1",
            rule_uuid="rule-1",
            active=True,
            opened=True,
            last_seen="2026-01-01T00:00:00+00:00",
        )
        bridge.publish_telemetry(profile_uuid="profile-1", source_uuid="src-1", payload={})

    def test_attributes_are_published_before_state(self):
        """The binary_sensor reads its attributes from a separate topic; state first
        would briefly pair a new ON with the previous event's attributes."""
        bridge, fake = _connected_bridge([_profile()])

        bridge.publish_rule_transition(
            profile_uuid="profile-1",
            source_uuid="src-1",
            rule_uuid="rule-1",
            active=True,
            opened=True,
            last_seen="2026-01-01T00:00:00+00:00",
            duration_s=12.345,
            peak_confidence=0.9,
        )

        topics = [c["topic"] for c in fake.calls]
        assert topics.index(next(t for t in topics if t.endswith("/attributes"))) < topics.index(
            next(t for t in topics if t.endswith("/state"))
        )

    def test_attribute_payload_carries_what_the_value_templates_read(self):
        bridge, fake = _connected_bridge([_profile()])

        bridge.publish_rule_transition(
            profile_uuid="profile-1",
            source_uuid="src-1",
            rule_uuid="rule-1",
            active=False,
            opened=False,
            last_seen="2026-01-01T00:00:00+00:00",
            duration_s=12.345,
            peak_confidence=0.912345,
            subject="subject_a",
        )

        payload = _published(fake, "/attributes")[0]["payload"]
        assert payload["lastDurationS"] == 12.35
        assert payload["peakConfidence"] == 0.9123
        assert payload["subject"] == "subject_a"
