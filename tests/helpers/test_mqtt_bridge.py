"""Tests for the MQTT bridge's discovery generation, counter seeding and availability.

Per HA_INTEGRATION_PLAN.md WP-2: pure logic gets tests, the live connection does not.
Every test here exercises the pure functions directly (or `MqttBridge` with a fake
publisher standing in for the network) — nothing opens a socket.

The two regressions worth guarding: rule/source entities must survive a rename (their
`unique_id` is uuid-based even though the topic path is name-based), and a rule/source
that disappears from config must have its discovery retracted, not left retained
forever.
"""

from apps.helpers.mqtt.bridge import (
    MqttBridge,
    ProfileView,
    RuleView,
    SourceView,
    build_all_discovery,
    build_seed_publishes,
    build_source_discovery,
    reconcile_discovery,
    source_config_topic,
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

    def test_a_deleted_rule_is_removed_from_the_republished_device_message(self):
        with_rule = _profile(rules=(RuleView(uuid="rule-1", name="cat at bowl"),))
        without_rule = _profile(rules=())

        desired_before = build_all_discovery([with_rule], PREFIX)
        _, previous_topics = reconcile_discovery(frozenset(), desired_before)

        desired_after = build_all_discovery([without_rule], PREFIX)
        to_publish, _ = reconcile_discovery(previous_topics, desired_after)

        topic = source_config_topic("src-1")
        # Same device, still present (the source itself wasn't deleted) — but its
        # cmps no longer mention the deleted rule.
        assert topic in to_publish
        assert to_publish[topic] is not None
        assert not any(k.startswith("rule-1_") for k in to_publish[topic]["cmps"])

    def test_a_deleted_source_is_retracted_with_an_empty_payload(self):
        with_source = [_profile()]
        without_source = [_profile(sources=(), rules=())]

        _, previous_topics = reconcile_discovery(frozenset(), build_all_discovery(with_source, PREFIX))
        to_publish, remaining = reconcile_discovery(previous_topics, build_all_discovery(without_source, PREFIX))

        topic = source_config_topic("src-1")
        assert to_publish[topic] is None
        assert topic not in remaining

    def test_an_unchanged_profile_produces_no_retractions(self):
        profile = [_profile()]
        _, previous_topics = reconcile_discovery(frozenset(), build_all_discovery(profile, PREFIX))
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
        assert seeded["kitchen-cam/rule/cat-at-bowl/count"] == 42

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
