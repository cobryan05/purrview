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
    SubjectView,
    build_all_discovery,
    build_seed_publishes,
    build_slug_index,
    build_source_discovery,
    build_subject_discovery,
    build_subject_seed,
    reconcile_discovery,
    source_config_topic,
    source_device_id,
    state_from_retained,
    subject_config_topic,
    subjects_in_scope,
)
from apps.helpers.watch.templating import slugify

PREFIX = "encounterwatch"


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

    def test_device_identity_derives_from_the_source_uuid_not_its_name(self):
        """Rename-stability is `test_source_device_identity_is_stable_across_a_rename`;
        this pins that the identity is the *namespaced* uuid, not the bare one."""
        profile = _profile()

        payload = build_source_discovery(profile, profile.sources[0], PREFIX)

        assert payload["dev"]["ids"] == source_device_id("src-1")


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
        retained["homeassistant/device/encounterwatch_junk/config"] = b"{not json"

        state = state_from_retained(retained)

        assert source_config_topic("src-1") in state.topics
        assert "homeassistant/device/encounterwatch_junk/config" not in state.topics

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


class TestDefaultVisibility:
    """`enabled_by_default: False` is the cardinality lever — HA still creates the
    entity, so it costs nothing until switched on, but a fresh install shows a
    dashboard's worth of entities rather than ~140."""

    def _cmps(self):
        profile = _profile()
        return build_source_discovery(profile, profile.sources[0], PREFIX)["cmps"]

    def test_the_dashboard_entities_are_on_and_the_detail_is_off(self):
        cmps = self._cmps()
        on = {k for k, v in cmps.items() if v.get("enabled_by_default") is not False}
        assert {"camera", "rule-1_state", "rule-1_count"} <= on
        assert {"rule-1_last_duration", "rule-1_peak_confidence", "fps", "mqtt_dropped"} & on == set()

    def test_health_ships_enabled_because_it_is_the_camera_failure_alarm(self):
        """The one diagnostic that stays on: an alarm that ships switched off is worse
        than no alarm."""
        assert self._cmps()["health"].get("enabled_by_default") is not False

    def test_a_disabled_entity_still_has_a_working_state_topic(self):
        """Disabled is not omitted — HA replays the retained state when it is enabled,
        so the topic has to be right even while nothing is listening."""
        cmps = self._cmps()
        assert cmps["fps"]["state_topic"].startswith(PREFIX)


class TestSubjectEnumeration:
    """Subjects come from rule config expanded through the label tree — rules commonly
    name a parent label, and it is the descendant that lands in an event's
    `subject_ref`."""

    def test_only_labels_a_rule_can_match_get_a_device(self):
        profile = _profile(rules=(RuleView(uuid="rule-1", name="bowl", subject_uuids=("subj-a",)),))
        assert set(subjects_in_scope([profile])) == {"subj-a"}

    def test_a_disabled_rule_contributes_no_subjects(self):
        profile = _profile(
            rules=(RuleView(uuid="rule-1", name="bowl", enabled=False, subject_uuids=("subj-a",)),),
        )
        assert subjects_in_scope([profile]) == {}

    def test_a_subject_is_scoped_to_the_sources_its_rule_covers(self):
        """The per-(rule, source) tier must not invent counters for cameras the rule
        was never scoped to."""
        profile = _profile(
            sources=(SourceView(uuid="src-1", name="Kitchen"), SourceView(uuid="src-2", name="Hallway")),
            rules=(RuleView(uuid="rule-1", name="bowl", source_uuids=("src-1",), subject_uuids=("subj-a",)),),
        )
        _rule, sources = subjects_in_scope([profile])["subj-a"][0]
        assert [s.uuid for s in sources] == ["src-1"]

    def test_a_subject_without_a_known_label_name_gets_no_device(self):
        """A device titled with a bare uuid is worse than no device."""
        profile = _profile(rules=(RuleView(uuid="rule-1", name="bowl", subject_uuids=("subj-gone",)),))
        desired = build_all_discovery([profile], PREFIX, {})
        assert not any("subject" in topic for topic in desired)


class TestSubjectDeviceShape:
    def _subject_cmps(self, profile, subjects=None):
        subjects = subjects or {"subj-a": SubjectView(uuid="subj-a", name="Subject A")}
        desired = build_all_discovery([profile], PREFIX, subjects)
        return desired[subject_config_topic("subj-a")]["cmps"]

    def test_the_core_entities_exist_without_any_rule_named_present(self):
        """The whole point of the core: it is derived from the label and the event
        stream, so it does not depend on any particular rule existing. A rule-derived
        anchor would vanish the moment that rule was renamed or deleted."""
        profile = _profile(rules=(RuleView(uuid="rule-1", name="something obscure", subject_uuids=("subj-a",)),))
        cmps = self._subject_cmps(profile)
        assert {"detected", "count", "last_seen", "last_location"} <= set(cmps)

    def test_the_per_source_tier_is_disabled_by_default_and_the_aggregate_is_not(self):
        """The cardinality lever: the tier that multiplies out is off, the tier you
        put on a dashboard is on."""
        profile = _profile(
            sources=(SourceView(uuid="src-1", name="Kitchen"), SourceView(uuid="src-2", name="Hallway")),
            rules=(RuleView(uuid="rule-1", name="bowl", subject_uuids=("subj-a",)),),
        )
        cmps = self._subject_cmps(profile)
        assert cmps["rule-1_count"].get("enabled_by_default") is not False
        assert cmps["rule-1_src-1_count"]["enabled_by_default"] is False

    def test_subject_entities_survive_a_label_rename(self):
        profile = _profile(rules=(RuleView(uuid="rule-1", name="bowl", subject_uuids=("subj-a",)),))
        before = self._subject_cmps(profile, {"subj-a": SubjectView(uuid="subj-a", name="Subject A")})
        after = self._subject_cmps(profile, {"subj-a": SubjectView(uuid="subj-a", name="Renamed")})

        assert before["count"]["unique_id"] == after["count"]["unique_id"]
        assert before["count"]["state_topic"] != after["count"]["state_topic"]

    def test_a_subject_device_is_not_tied_to_one_watchers_availability(self):
        """A label belongs to no single watcher — tying it to one would blank the
        device whenever that watcher stopped, even with another still watching."""
        profile = _profile(rules=(RuleView(uuid="rule-1", name="bowl", subject_uuids=("subj-a",)),))
        desired = build_all_discovery([profile], PREFIX, {"subj-a": SubjectView(uuid="subj-a", name="Subject A")})
        availability = desired[subject_config_topic("subj-a")]["availability"]
        assert availability == [{"topic": f"{PREFIX}/status"}]


class TestSubjectTiersAgree:
    """Every tier is summed from one query, so a subject total is the sum of its rules
    by construction. Three separate GROUP BYs could disagree after a concurrent write,
    and a total that isn't the sum of its parts is unexplainable afterwards."""

    COUNTS = [
        {
            "source_uuid": "src-1",
            "rule_uuid": "rule-1",
            "subject_ref": "subj-a",
            "count": 5,
            "last_seen": "2026-01-02T00:00:00+00:00",
        },
        {
            "source_uuid": "src-2",
            "rule_uuid": "rule-1",
            "subject_ref": "subj-a",
            "count": 3,
            "last_seen": "2026-01-03T00:00:00+00:00",
        },
        {
            "source_uuid": "src-1",
            "rule_uuid": "rule-2",
            "subject_ref": "subj-a",
            "count": 4,
            "last_seen": "2026-01-01T00:00:00+00:00",
        },
    ]

    def test_each_tier_is_the_sum_of_the_one_below(self):
        seed = build_subject_seed(self.COUNTS)
        assert seed[("subj-a",)][0] == 12
        assert seed[("subj-a", "rule-1")][0] == 8
        assert seed[("subj-a", "rule-1", "src-1")][0] == 5

    def test_last_seen_rolls_up_as_the_most_recent_not_the_last_row(self):
        seed = build_subject_seed(self.COUNTS)
        assert seed[("subj-a",)][1] == "2026-01-03T00:00:00+00:00"
        assert seed[("subj-a", "rule-2")][1] == "2026-01-01T00:00:00+00:00"

    def test_events_with_no_subject_are_not_counted_against_a_subject(self):
        """A rule with no subject slot still produces events; they belong to the
        rule-level counters, which seed separately."""
        assert build_subject_seed([{**self.COUNTS[0], "subject_ref": None}]) == {}


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
    def test_open_events_are_counted_from_transitions_and_close_again(self):
        """The app device's open-event count. Tracked on every transition, not just the
        opening one — an event already open when the bridge connected still closes, and
        dropping that would drift the count upward for the life of the process."""
        bridge, _fake = _connected_bridge([_profile()])
        kwargs = dict(
            profile_uuid="profile-1",
            source_uuid="src-1",
            rule_uuid="rule-1",
            last_seen="2026-01-01T00:00:00+00:00",
        )

        bridge.publish_rule_transition(active=True, opened=True, **kwargs)
        assert len(bridge._open_rules) == 1
        # An update must not double-count the same open event.
        bridge.publish_rule_transition(active=True, opened=False, **kwargs)
        assert len(bridge._open_rules) == 1
        bridge.publish_rule_transition(active=False, opened=False, **kwargs)
        assert bridge._open_rules == set()

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


class TestLoadProfiles:
    """`_load_profiles`/`_load_labels` are where the bridge meets the real db_client.

    Every other test in this file builds `ProfileView`/`SourceView` by hand, so this
    path had no coverage at all — and it was broken end to end in production: it called
    a method the client does not have (`LabelsApi.get_labels`) and iterated a
    uuid-keyed mapping (`GetSourcesResponse.sources`) as if it were a list of objects.
    Every reconcile pass raised, so no discovery was ever published.

    So these stub the *transport* but not the contract: the fake API objects are checked
    against the real generated classes (a renamed or missing method fails here), and the
    responses are real generated models (a changed response shape fails here).
    """

    @staticmethod
    def _fake_api(real_cls, method_name, value):
        """A stand-in for a generated API class, pinned to the real one's surface.

        `real_cls` must actually expose `method_name` — that is the half of the
        contract a hand-written fake otherwise silently invents.
        """
        assert hasattr(real_cls, method_name), f"{real_cls.__name__} has no {method_name}()"

        class _Fake:
            def __init__(self, _client):
                pass

        setattr(_Fake, method_name, lambda self: value)
        return _Fake

    def _patch(self, monkeypatch, *, sources_resp, profiles_resp, labels_resp):
        import db_client.api.labels_api as labels_api_mod
        import db_client.api.sources_api as sources_api_mod
        import db_client.api.watch_profiles_api as profiles_api_mod

        monkeypatch.setattr(
            sources_api_mod,
            "SourcesApi",
            self._fake_api(sources_api_mod.SourcesApi, "get_sources", sources_resp),
        )
        monkeypatch.setattr(
            profiles_api_mod,
            "WatchProfilesApi",
            self._fake_api(profiles_api_mod.WatchProfilesApi, "list_watch_profiles", profiles_resp),
        )
        monkeypatch.setattr(
            labels_api_mod,
            "LabelsApi",
            self._fake_api(labels_api_mod.LabelsApi, "list_labels", labels_resp),
        )

    @staticmethod
    def _source(uuid: str, name: str):
        from db_client.models.source_read import SourceRead

        return SourceRead(uuid=uuid, name=name, typename="rtsp", params={})

    @staticmethod
    def _labels(*labels):
        from db_client.models.label_read import LabelRead
        from db_client.models.list_labels_response import ListLabelsResponse

        return ListLabelsResponse(
            status="success",
            labels=[LabelRead(uuid=u, name=n, parent_uuid=p) for u, n, p in labels],
        )

    @staticmethod
    def _profiles(rules=None):
        from db_client.models.list_watch_profiles_response import ListWatchProfilesResponse
        from db_client.models.watch_profile_read import WatchProfileRead
        from db_client.models.watch_profile_source_read import WatchProfileSourceRead

        return ListWatchProfilesResponse(
            status="success",
            profiles=[
                WatchProfileRead(
                    uuid="profile-1",
                    name="Living room watcher",
                    tracker_typename="bytetrack",
                    sources=[
                        WatchProfileSourceRead(uuid="ps-1", profile_uuid="profile-1", source_uuid="src-1", enabled=True)
                    ],
                    rules=rules or [],
                )
            ],
        )

    @staticmethod
    def _sources(mapping):
        from db_client.models.get_sources_response import GetSourcesResponse

        return GetSourcesResponse(status="success", sources=mapping)

    async def test_source_name_resolves_from_the_uuid_keyed_mapping(self, monkeypatch):
        """`GetSourcesResponse.sources` is `dict[uuid, SourceRead]`, not a list."""
        self._patch(
            monkeypatch,
            sources_resp=self._sources({"src-1": self._source("src-1", "Kitchen cam")}),
            profiles_resp=self._profiles(),
            labels_resp=self._labels(),
        )

        profiles = await MqttBridge(db_api_client=object())._load_profiles()

        assert [s.name for s in profiles[0].sources] == ["Kitchen cam"]

    async def test_source_deleted_out_from_under_a_profile_is_dropped(self, monkeypatch):
        """The "source can be deleted out from under a profile" guard still holds."""
        self._patch(
            monkeypatch,
            sources_resp=self._sources({"other": self._source("other", "Hallway")}),
            profiles_resp=self._profiles(),
            labels_resp=self._labels(),
        )

        profiles = await MqttBridge(db_api_client=object())._load_profiles()

        assert profiles[0].sources == ()

    async def test_label_ancestry_expands_a_parent_subject_to_its_descendants(self, monkeypatch):
        """A rule naming a parent label with `include_descendants` must cover the
        children — that is what lands in an event's `subject_ref`."""
        from db_client.models.interaction_rule_read import InteractionRuleRead

        rule = InteractionRuleRead(
            uuid="rule-1",
            profile_uuid="profile-1",
            name="cat at bowl",
            enabled=True,
            subject_a_kind="label",
            subject_a_refs_json=["cat"],
            subject_a_include_descendants=True,
            relation="near",
            subject_b_kind="label",
        )
        self._patch(
            monkeypatch,
            sources_resp=self._sources({"src-1": self._source("src-1", "Kitchen cam")}),
            profiles_resp=self._profiles(rules=[rule]),
            labels_resp=self._labels(("cat", "Cat", None), ("tabby", "Tabby", "cat"), ("dog", "Dog", None)),
        )

        profiles = await MqttBridge(db_api_client=object())._load_profiles()

        assert set(profiles[0].rules[0].subject_uuids) == {"cat", "tabby"}


class TestUniqueIdsDoNotCollide:
    """HA keys entities on `unique_id` and silently drops duplicates.

    A collision does not raise anywhere in this codebase — HA logs "Platform mqtt does
    not generate unique IDs ... ignoring <entity>" and registers only the first. That is
    how every camera after the first went missing in production while the broker showed
    a complete, correct set of retained configs: rule entities hang off the *source*
    device but were keyed on the rule uuid alone, and one rule covers every source in
    the profile.
    """

    @staticmethod
    def _unique_ids(messages):
        for payload in messages.values():
            for cmp in payload.get("cmps", {}).values():
                uid = cmp.get("unique_id")
                if uid is not None:
                    yield uid

    def test_one_rule_across_two_sources_yields_distinct_unique_ids(self):
        profile = _profile(
            sources=(SourceView(uuid="src-1", name="Kitchen"), SourceView(uuid="src-2", name="Hallway")),
            rules=(RuleView(uuid="rule-1", name="cat at bowl"),),
        )

        ids = list(self._unique_ids(build_all_discovery([profile], PREFIX)))

        assert len(ids) == len(set(ids)), f"duplicate unique_id: {sorted(i for i in ids if ids.count(i) > 1)}"

    def test_every_unique_id_in_a_full_discovery_set_is_distinct(self):
        """The whole set at once — sources, subjects and the app device together."""
        subject = SubjectView(uuid="cat-1", name="Tabby")
        profile = _profile(
            sources=(SourceView(uuid="src-1", name="Kitchen"), SourceView(uuid="src-2", name="Hallway")),
            rules=(
                RuleView(uuid="rule-1", name="cat at bowl", subject_uuids=("cat-1",)),
                RuleView(uuid="rule-2", name="cat in litter", subject_uuids=("cat-1",)),
            ),
        )

        ids = list(self._unique_ids(build_all_discovery([profile], PREFIX, {"cat-1": subject})))

        assert len(ids) == len(set(ids)), f"duplicate unique_id: {sorted(i for i in ids if ids.count(i) > 1)}"


class TestDeviceIdentifiersAreNamespaced:
    """Device `ids` land in HA's global MQTT device namespace, shared with every other
    integration on the broker — so they carry the app's own prefix rather than a bare
    uuid. Source devices originally published `source.uuid` alone while subjects and the
    app device were namespaced; this keeps the three consistent."""

    def test_no_device_publishes_a_bare_identifier(self):
        subject = SubjectView(uuid="cat-1", name="Tabby")
        profile = _profile(
            sources=(SourceView(uuid="src-1", name="Kitchen"),),
            rules=(RuleView(uuid="rule-1", name="cat at bowl", subject_uuids=("cat-1",)),),
        )

        messages = build_all_discovery([profile], PREFIX, {"cat-1": subject})
        ids = [payload["dev"]["ids"] for payload in messages.values()]

        assert ids, "no devices built"
        assert all(i.startswith("encounterwatch") for i in ids), ids


class TestWatcherAvailabilitySurvivesALateConnect:
    """A watcher starts before the bridge finishes connecting.

    `StreamWatcherTask` announces availability once, at watcher start. The bridge
    connects separately, on the manager's worker pass, so that call routinely lands
    while `_publisher is None`. Dropping it there is silent and permanent: nothing
    republishes the topic, `availability_mode: all` never sees its second topic, and
    every entity on every source device reads `unavailable` forever — which is exactly
    how this behaved against a real HA.
    """

    def test_availability_set_before_connect_is_published_once_connected(self):
        bridge = MqttBridge(db_api_client=None)
        assert bridge._publisher is None

        bridge.set_watcher_availability("Living room watcher", online=True)
        fake = _FakePublisher()
        bridge._publisher = fake
        bridge._publish_watcher_availability()

        assert [(c["topic"], c["payload"]) for c in fake.calls] == [("watcher/living-room-watcher/status", "online")]

    def test_the_latest_state_wins_rather_than_replaying_every_transition(self):
        """Re-asserting must not resurrect a stopped watcher as online."""
        bridge = MqttBridge(db_api_client=None)
        bridge.set_watcher_availability("Living room watcher", online=True)
        bridge.set_watcher_availability("Living room watcher", online=False)

        fake = _FakePublisher()
        bridge._publisher = fake
        bridge._publish_watcher_availability()

        assert [c["payload"] for c in fake.calls] == ["offline"]


class TestEntityNamesDoNotStutter:
    """HA builds both the friendly name and the entity_id as `<device> <entity>`.

    An entity name that repeats its device stutters — `sensor.maple_maple_present`,
    `sensor.catroomrtsp_catroomrtsp_fps`. Renaming after the fact is expensive because
    entity_ids are sticky and carry history, so the interesting property is that new
    config cannot reintroduce it: a rule named after the cat it watches is the natural
    thing for a user to write, and it must come out clean without anyone noticing.
    """

    @staticmethod
    def _names(cmps):
        return [c.get("name") for c in cmps.values()]

    def test_source_entities_do_not_repeat_the_camera_name(self):
        profile = _profile(sources=(SourceView(uuid="src-1", name="CatRoomRTSP"),))

        cmps = build_source_discovery(profile, profile.sources[0], PREFIX)["cmps"]

        assert not [n for n in self._names(cmps) if n and "catroomrtsp" in slugify(n)]

    def test_the_camera_itself_is_the_devices_primary_entity(self):
        """`name: None` is how HA is told "this entity *is* the device" — it yields
        `camera.catroomrtsp`, not `camera.catroomrtsp_catroomrtsp`."""
        profile = _profile(sources=(SourceView(uuid="src-1", name="CatRoomRTSP"),))

        assert build_source_discovery(profile, profile.sources[0], PREFIX)["cmps"]["camera"]["name"] is None

    def test_a_rule_named_after_its_subject_is_stripped_on_the_subject_device(self):
        """The recurring case: device "Maple", rule "Maple Present" → "Present"."""
        subject = SubjectView(uuid="cat-1", name="Maple")
        rule = RuleView(uuid="rule-1", name="Maple Present", subject_uuids=("cat-1",))

        cmps = build_subject_discovery(subject, [(rule, ())], PREFIX)["cmps"]

        assert cmps["rule-1_count"]["name"] == "Present"

    def test_a_rule_not_named_after_its_subject_is_left_alone(self):
        """Stripping must key off an actual prefix match, not chop the first word."""
        subject = SubjectView(uuid="cat-1", name="Maple")
        rule = RuleView(uuid="rule-1", name="Drinking Water", subject_uuids=("cat-1",))

        cmps = build_subject_discovery(subject, [(rule, ())], PREFIX)["cmps"]

        assert cmps["rule-1_count"]["name"] == "Drinking Water"

    def test_a_word_that_merely_starts_with_the_device_name_is_not_chopped(self):
        subject = SubjectView(uuid="cat-1", name="Cat")
        rule = RuleView(uuid="rule-1", name="Catcher Active", subject_uuids=("cat-1",))

        cmps = build_subject_discovery(subject, [(rule, ())], PREFIX)["cmps"]

        assert cmps["rule-1_count"]["name"] == "Catcher Active"

    def test_a_rule_named_exactly_as_its_subject_keeps_its_name(self):
        """Stripping to empty would make the rule the device's primary entity and
        collide with it. Keeping the stutter is the lesser evil."""
        subject = SubjectView(uuid="cat-1", name="Maple")
        rule = RuleView(uuid="rule-1", name="Maple", subject_uuids=("cat-1",))

        cmps = build_subject_discovery(subject, [(rule, ())], PREFIX)["cmps"]

        assert cmps["rule-1_count"]["name"] == "Maple"
