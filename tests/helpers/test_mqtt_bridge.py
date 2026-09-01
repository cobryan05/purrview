"""Tests for the MQTT bridge's discovery generation, counter seeding and availability.

Per HA_INTEGRATION_PLAN.md WP-2: pure logic gets tests, the live connection does not.
Every test here exercises the pure functions directly (or `MqttBridge` with a fake
publisher standing in for the network) — nothing opens a socket.

The two regressions worth guarding: rule/source entities must survive a rename (their
`unique_id` is uuid-based even though the topic path is name-based), and a rule/source
that disappears from config must have its discovery retracted, not left retained
forever.
"""

import asyncio
import json

from apps.helpers.mqtt.bridge import (
    APP_CONFIG_TOPIC,
    APP_DEVICE_ID,
    PRESENCE_ANY_AVAILABILITY_TOPIC,
    RULES_CONFIG_TOPIC,
    DiscoveryState,
    MqttBridge,
    ProfileView,
    RuleView,
    SourceView,
    SubjectView,
    build_all_discovery,
    build_rules_discovery,
    build_seed_publishes,
    build_slug_index,
    build_source_discovery,
    build_subject_discovery,
    build_subject_seed,
    count_entities,
    reconcile_discovery,
    source_config_topic,
    source_device_id,
    state_from_retained,
    subject_config_topic,
    subjects_in_scope,
)
from apps.helpers.watch.templating import slugify

PREFIX = "purrview"


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

    def test_another_integrations_device_config_is_neither_adopted_nor_retracted(self):
        """The read-back subscribes to `homeassistant/device/+/config`, which is every
        device-discovery config on the broker — not just ours. Adopting a foreign one
        puts it in `previous`, and the very next reconcile retracts everything in
        `previous` that we do not desire, so this app would delete another
        integration's entities out of Home Assistant.

        Nothing catches that downstream: the payload we publish is a correct
        retraction, so the only symptom is somebody else's devices silently
        disappearing.
        """
        retained = {t: json.dumps(p).encode() for t, p in build_all_discovery([_profile()], PREFIX).items()}
        foreign = "homeassistant/device/0x00124b0022ab1234/config"
        retained[foreign] = json.dumps(
            {"dev": {"ids": "0x00124b0022ab1234"}, "cmps": {"temperature": {"p": "sensor"}}}
        ).encode()

        adopted = state_from_retained(retained)
        assert foreign not in adopted.topics, "a device we did not publish must not be adopted"

        to_publish, _ = reconcile_discovery(adopted, build_all_discovery([_profile()], PREFIX))
        assert foreign not in to_publish, "a device we did not publish must never be retracted"

    def test_our_own_configs_are_still_adopted_across_a_rename_of_nothing(self):
        """The ownership filter must not throw the baby out: configs carrying our own
        device-id namespace are exactly what the read-back exists to recover."""
        retained = {t: json.dumps(p).encode() for t, p in build_all_discovery([_profile()], PREFIX).items()}

        adopted = state_from_retained(retained)

        assert source_config_topic("src-1") in adopted.topics
        assert adopted.topics == set(retained)

    def test_an_unparseable_retained_config_is_skipped_not_fatal(self):
        """One malformed retained message must not stop the bridge coming up — and the
        configs around it must still be adopted."""
        good = build_all_discovery([_profile()], PREFIX)
        retained = {t: json.dumps(p).encode() for t, p in good.items()}
        retained["homeassistant/device/purrview_junk/config"] = b"{not json"

        state = state_from_retained(retained)

        assert source_config_topic("src-1") in state.topics
        assert "homeassistant/device/purrview_junk/config" not in state.topics

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
    """`subjects_in_scope` is the per-rule *breakdown* on a label's device, not the gate
    on whether the label has one — that is `publish_to_ha`, see `TestEveryLabelGetsADevice`.

    Subjects come from rule config expanded through the label tree — rules commonly name
    a parent label, and it is the descendant that lands in an event's `subject_ref`."""

    def test_only_labels_a_rule_can_match_get_a_breakdown(self):
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

    def test_the_per_source_tier_is_opt_in_per_label_and_the_aggregate_is_not(self):
        """The cardinality lever, and the one conditional worth pinning here.

        Publishing the (label x rule x source) tier disabled-by-default was not enough:
        HA still creates a registry entry per entity, so the cost scaled with the label
        tree while being invisible. It is now absent unless the label asks for it, and
        the aggregate tier — the one a dashboard uses — is unaffected either way.

        And when the label does ask, the entities have to be *enabled*. They used to
        carry `enabled_by_default: False` as well, which made the opt-in unobservable:
        HA created them disabled, so they had no state and no history until each was
        hand-enabled, even though the bridge published to their topics the whole time.
        The per-label flag is the cardinality gate; a second gate behind it just breaks
        the first.
        """
        profile = _profile(
            sources=(SourceView(uuid="src-1", name="Kitchen"), SourceView(uuid="src-2", name="Hallway")),
            rules=(RuleView(uuid="rule-1", name="bowl", subject_uuids=("subj-a",)),),
        )

        off = self._subject_cmps(profile, {"subj-a": SubjectView(uuid="subj-a", name="Subject A")})
        assert off["rule-1_count"].get("enabled_by_default") is not False
        assert not [key for key in off if key.startswith("rule-1_src-")]

        on = self._subject_cmps(
            profile, {"subj-a": SubjectView(uuid="subj-a", name="Subject A", publish_per_source=True)}
        )
        assert on["rule-1_count"].get("enabled_by_default") is not False
        # Present *and* enabled — an opted-in breakdown that HA creates disabled is
        # indistinguishable from the toggle doing nothing.
        assert on["rule-1_src-1_count"].get("enabled_by_default") is not False
        assert on["rule-1_src-2_count"].get("enabled_by_default") is not False

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


# A rule now publishes on two tiers — `<profile>/<source>/rule/<rule>/…` for the camera
# device and `<profile>/rules/<rule>/…` for the Rules device — so matching on a suffix
# alone no longer identifies one of them. Singular vs plural is the discriminator.
def _per_source(fake, suffix: str):
    """Only the per-camera tier."""
    return [c for c in _published(fake, suffix) if "/rules/" not in c["topic"]]


def _rule_global(fake, suffix: str):
    """Only the Rules device tier."""
    return [c for c in _published(fake, suffix) if "/rules/" in c["topic"]]


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

        # Distinctness is the property under test, not the number of tiers: two
        # profiles must never write to one topic, whatever tiers exist. Asserted per
        # tier so a collision cannot hide behind the other tier's distinct topics.
        for tier in (_per_source, _rule_global):
            topics = [c["topic"] for c in tier(fake, "/state")]
            assert len(topics) == 2, topics
            assert len(set(topics)) == 2, topics

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

        assert [c["payload"] for c in _per_source(fake, "/count")] == [43]
        # The Rules device's counter is the same event summed across sources — with one
        # source that is the same number, and it must not be a second increment.
        assert [c["payload"] for c in _rule_global(fake, "/count")] == [43]

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

        # One event, three transitions, one increment — on each tier independently.
        assert [c["payload"] for c in _per_source(fake, "/count")] == [43]
        assert [c["payload"] for c in _rule_global(fake, "/count")] == [43]
        assert [c["payload"] for c in _per_source(fake, "/state")] == ["ON", "ON", "OFF"]
        assert [c["payload"] for c in _rule_global(fake, "/state")] == ["ON", "ON", "OFF"]

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

        counts = _per_source(fake, "/count")
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
        assert all(i.startswith("purrview") for i in ids), ids


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


class TestDevicesAreIdentifiableAsOneFamily:
    """HA files every discovery-created device under "devices that don't belong to a
    sub-entry" and offers no way to change that: an MQTT subentry holds exactly one
    hand-added device, and the discovery device object has no `via_device`, so there is
    no parent to nest under. Manufacturer and model are the grouping that *is*
    available, and they are what makes the set filterable as one family."""

    def test_every_device_declares_the_same_manufacturer(self):
        subject = SubjectView(uuid="cat-1", name="Tabby")
        profile = _profile(rules=(RuleView(uuid="rule-1", name="cat at bowl", subject_uuids=("cat-1",)),))

        devices = [p["dev"] for p in build_all_discovery([profile], PREFIX, {"cat-1": subject}).values()]

        assert devices
        assert {d.get("mf") for d in devices} == {"PurrView"}

    def test_the_three_device_kinds_are_distinguishable_by_model(self):
        subject = SubjectView(uuid="cat-1", name="Tabby")
        profile = _profile(rules=(RuleView(uuid="rule-1", name="cat at bowl", subject_uuids=("cat-1",)),))

        models = {p["dev"]["mdl"] for p in build_all_discovery([profile], PREFIX, {"cat-1": subject}).values()}

        assert models == {"Watched camera", "Tracked subject", "Rules", "Application"}


class TestRulesDevice:
    """The device automations are written against: one rule, one entity, independent of
    how many cameras or labels exist."""

    def _cmps(self, profiles):
        device = build_rules_discovery(profiles, PREFIX)
        return {} if device is None else device["cmps"]

    def test_one_rule_on_many_cameras_is_still_one_set_of_entities(self):
        """The reason this device exists. The camera devices give a rule N entities for
        N cameras, so an automation has to know every camera and silently stops covering
        one the moment a camera is added. Here the rule is one entity whatever N is."""
        one = _profile(sources=(SourceView(uuid="src-1", name="Kitchen"),))
        four = _profile(
            sources=tuple(SourceView(uuid=f"src-{i}", name=f"Cam {i}") for i in range(1, 5)),
        )

        assert set(self._cmps([one])) == set(self._cmps([four]))

    def test_the_state_stays_on_until_the_last_camera_closes(self):
        """The behaviour a per-camera binary_sensor cannot express, and the reason an
        automation built by OR-ing them together gets this wrong: a rule open on two
        cameras must not read OFF when only the first one ends."""
        profile = _profile(sources=(SourceView(uuid="src-1", name="A"), SourceView(uuid="src-2", name="B")))
        bridge, fake = _connected_bridge([profile])
        common = dict(profile_uuid="profile-1", rule_uuid="rule-1", last_seen="2026-01-01T00:00:00+00:00")

        bridge.publish_rule_transition(source_uuid="src-1", active=True, opened=True, **common)
        bridge.publish_rule_transition(source_uuid="src-2", active=True, opened=True, **common)
        bridge.publish_rule_transition(source_uuid="src-1", active=False, opened=False, **common)
        assert [c["payload"] for c in _rule_global(fake, "/state")] == ["ON", "ON", "ON"]

        bridge.publish_rule_transition(source_uuid="src-2", active=False, opened=False, **common)
        assert [c["payload"] for c in _rule_global(fake, "/state")][-1] == "OFF"

    def test_the_event_carries_the_subject_and_camera_that_fired_it(self):
        """What makes the per-(rule, label, camera) entity tier unnecessary for
        automations: one entity, and the combination arrives as attributes."""
        bridge, fake = _connected_bridge([_profile()])

        bridge.publish_rule_transition(
            profile_uuid="profile-1",
            source_uuid="src-1",
            rule_uuid="rule-1",
            active=True,
            opened=True,
            last_seen="2026-01-01T00:00:00+00:00",
            subject="Luna",
            source_name="Kitchen cam",
        )

        event = _rule_global(fake, "/event")[-1]["payload"]
        assert event["event_type"] == "opened"
        assert event["subject"] == "Luna"
        assert event["source"] == "Kitchen cam"

    def test_the_event_topic_is_not_retained(self):
        """HA's event entity fires on every message it receives, including the retained
        one replayed on reconnect. Retaining this would re-trigger every automation
        bound to it whenever HA restarted."""
        bridge, fake = _connected_bridge([_profile()])
        common = dict(
            profile_uuid="profile-1", source_uuid="src-1", rule_uuid="rule-1", last_seen="2026-01-01T00:00:00+00:00"
        )

        bridge.publish_rule_transition(active=True, opened=True, **common)

        assert [c["retain"] for c in _rule_global(fake, "/event")] == [False]
        # …while the state it sits beside is retained, so HA still knows the rule is on.
        assert all(c["retain"] for c in _rule_global(fake, "/state"))

    def test_only_opening_and_closing_fire_an_event(self):
        """An UPDATE is neither. Firing on it would run the automation repeatedly for
        one visit."""
        bridge, fake = _connected_bridge([_profile()])
        common = dict(
            profile_uuid="profile-1", source_uuid="src-1", rule_uuid="rule-1", last_seen="2026-01-01T00:00:00+00:00"
        )

        for active, opened in ((True, True), (True, False), (False, False)):
            bridge.publish_rule_transition(active=active, opened=opened, **common)

        assert [c["payload"]["event_type"] for c in _rule_global(fake, "/event")] == ["opened", "closed"]


class TestPublishFlagsRemoveEntities:
    """Each flag has to *retract*, not merely stop updating. A discovery config left
    retained keeps its entities alive in HA forever with no way to remove them but
    hand-editing the entity registry."""

    def _topics(self, profiles, subjects):
        return set(build_all_discovery(profiles, PREFIX, subjects).keys())

    def test_a_rule_opted_out_disappears_from_every_device_it_was_on(self):
        subjects = {"subj-a": SubjectView(uuid="subj-a", name="Luna")}
        rule = RuleView(uuid="rule-1", name="bowl", subject_uuids=("subj-a",))
        on = _profile(rules=(rule,))
        off = _profile(rules=(RuleView(uuid="rule-1", name="bowl", subject_uuids=("subj-a",), publish_to_ha=False),))

        before = build_all_discovery([on], PREFIX, subjects)
        after = build_all_discovery([off], PREFIX, subjects)

        # Camera device keeps existing; the rule's components leave it.
        camera = source_config_topic("src-1")
        assert any(k.startswith("rule-1_") for k in before[camera]["cmps"])
        assert not [k for k in after[camera]["cmps"] if k.startswith("rule-1_")]
        # The rules device held nothing else, so it goes entirely rather than empty.
        assert RULES_CONFIG_TOPIC in before
        assert RULES_CONFIG_TOPIC not in after
        # The label keeps its device — it has one because it is a label, not because a
        # rule mentioned it — but loses the per-rule counters that rule contributed.
        subject = subject_config_topic("subj-a")
        assert any(k.startswith("rule-1_") for k in before[subject]["cmps"])
        assert not [k for k in after[subject]["cmps"] if k.startswith("rule-1_")]
        assert "detected" in after[subject]["cmps"], "the tracker-driven core survives a rule going away"

    def test_a_label_opted_out_loses_its_device_but_not_the_rule(self):
        """Opting a label out must not take the rule's own entities with it — the rule
        still fires, and the Rules device is where that is visible."""
        profile = _profile(rules=(RuleView(uuid="rule-1", name="bowl", subject_uuids=("subj-a",)),))

        on = self._topics([profile], {"subj-a": SubjectView(uuid="subj-a", name="Luna")})
        off = self._topics([profile], {"subj-a": SubjectView(uuid="subj-a", name="Luna", publish_to_ha=False)})

        assert subject_config_topic("subj-a") in on
        assert subject_config_topic("subj-a") not in off
        assert RULES_CONFIG_TOPIC in off
        assert source_config_topic("src-1") in off

    def test_a_camera_opted_out_loses_its_device_but_not_the_rule(self):
        subjects = {"subj-a": SubjectView(uuid="subj-a", name="Luna")}
        rules = (RuleView(uuid="rule-1", name="bowl", subject_uuids=("subj-a",)),)
        on = _profile(sources=(SourceView(uuid="src-1", name="Kitchen"),), rules=rules)
        off = _profile(sources=(SourceView(uuid="src-1", name="Kitchen", publish_to_ha=False),), rules=rules)

        assert source_config_topic("src-1") in self._topics([on], subjects)
        after = self._topics([off], subjects)
        assert source_config_topic("src-1") not in after
        assert RULES_CONFIG_TOPIC in after

    def test_turning_a_flag_off_emits_a_retraction(self):
        """The end-to-end property: reconciling from on to off publishes the empty
        payload that actually removes the device."""
        subjects = {"subj-a": SubjectView(uuid="subj-a", name="Luna")}
        rules = (RuleView(uuid="rule-1", name="bowl", subject_uuids=("subj-a",)),)
        on = _profile(sources=(SourceView(uuid="src-1", name="Kitchen"),), rules=rules)
        off = _profile(sources=(SourceView(uuid="src-1", name="Kitchen", publish_to_ha=False),), rules=rules)

        _, previous = reconcile_discovery(DiscoveryState(), build_all_discovery([on], PREFIX, subjects))
        to_publish, _ = reconcile_discovery(previous, build_all_discovery([off], PREFIX, subjects))

        assert to_publish[source_config_topic("src-1")] is None


class TestDevicesAreGroupedUnderTheApp:
    def test_every_device_hangs_off_the_app_device(self):
        """`via_device` is what makes HA show these as one family rather than N
        unrelated rows in a flat MQTT device list."""
        subject = SubjectView(uuid="cat-1", name="Tabby")
        profile = _profile(rules=(RuleView(uuid="rule-1", name="cat at bowl", subject_uuids=("cat-1",)),))

        messages = build_all_discovery([profile], PREFIX, {"cat-1": subject})

        for topic, payload in messages.items():
            if topic == APP_CONFIG_TOPIC:
                # The hub cannot be its own parent — HA drops a device that is.
                assert "via_device" not in payload["dev"]
            else:
                assert payload["dev"]["via_device"] == APP_DEVICE_ID, topic


class TestEntityBudget:
    def test_the_count_tracks_what_discovery_actually_asks_for(self):
        """The budget sensor is only worth having if it moves with config. Adding a
        camera must change it; it is the product of the axes that makes the number
        worth watching in the first place."""
        subjects = {"subj-a": SubjectView(uuid="subj-a", name="Luna")}
        rules = (RuleView(uuid="rule-1", name="bowl", subject_uuids=("subj-a",)),)
        one = _profile(sources=(SourceView(uuid="src-1", name="A"),), rules=rules)
        two = _profile(sources=(SourceView(uuid="src-1", name="A"), SourceView(uuid="src-2", name="B")), rules=rules)

        assert count_entities(build_all_discovery([two], PREFIX, subjects)) > count_entities(
            build_all_discovery([one], PREFIX, subjects)
        )

    def test_opting_a_label_into_the_per_source_tier_costs_entities(self):
        """The tier being opt-in is only meaningful if opting in is visibly more
        expensive than not."""
        rules = (RuleView(uuid="rule-1", name="bowl", subject_uuids=("subj-a",)),)
        profile = _profile(
            sources=(SourceView(uuid="src-1", name="A"), SourceView(uuid="src-2", name="B")), rules=rules
        )

        lean = count_entities(build_all_discovery([profile], PREFIX, {"subj-a": SubjectView("subj-a", "Luna")}))
        rich = count_entities(
            build_all_discovery([profile], PREFIX, {"subj-a": SubjectView("subj-a", "Luna", publish_per_source=True)})
        )

        # Per camera: one presence sensor, plus one counter for each rule naming this
        # label. Two cameras and one rule, so four. The presence half is the term that
        # grows with the *camera* count alone and is why the gate has to stay.
        assert rich == lean + 4


class TestAnOptedOutCameraStillFeedsTheRulesDevice:
    """`publish_to_ha` on a *camera* is a camera-level opt-out, not a rule-level one.

    The Rules device answers "did this rule fire anywhere", which is precisely the
    question a per-camera opt-out must not change — and the watchers UI promises as much
    in the flag's own tooltip. The bridge used to take one early exit for both tiers, so
    unticking one camera silently removed it from the cross-camera answer.
    """

    ON = SourceView(uuid="src-1", name="Kitchen cam")
    OFF = SourceView(uuid="src-1", name="Kitchen cam", publish_to_ha=False)

    def _open(self, bridge, **kw):
        bridge.publish_rule_transition(
            profile_uuid="profile-1",
            source_uuid="src-1",
            rule_uuid="rule-1",
            last_seen="2026-01-01T00:00:00+00:00",
            **kw,
        )

    def test_the_cameras_own_topics_go_quiet(self):
        """The half that already worked, asserted so the fix cannot be mistaken for
        'the flag stopped doing anything'."""
        bridge, fake = _connected_bridge([_profile(sources=(self.OFF,))])

        self._open(bridge, active=True, opened=True)

        assert _per_source(fake, "/state") == []

    def test_but_the_rules_device_still_hears_it(self):
        bridge, fake = _connected_bridge([_profile(sources=(self.OFF,))])

        self._open(bridge, active=True, opened=True)

        assert [c["payload"] for c in _rule_global(fake, "/state")] == ["ON"]

    def test_opting_out_mid_event_does_not_strand_the_rule_on(self):
        """The one that corrupts state rather than merely omitting it.

        Flipping the flag between an event's START and END used to skip the `discard`
        that closes the event, so the Rules device's retained `state` stayed ON for the
        life of the process — with no further transition able to clear it.
        """
        bridge, fake = _connected_bridge([_profile(sources=(self.ON,))])
        self._open(bridge, active=True, opened=True)

        # What a reconcile does after the flag is unticked, mid-event.
        bridge._slugs = build_slug_index([_profile(sources=(self.OFF,))])
        self._open(bridge, active=False, opened=False)

        assert [c["payload"] for c in _rule_global(fake, "/state")] == ["ON", "OFF"]
        assert bridge._open_rules == set()

    def test_the_cross_camera_total_still_counts_it(self):
        """The Rules device's count is summed from the per-camera counters, so a camera
        that is not counted at all is one the total silently under-reports."""
        two_cams = (self.OFF, SourceView(uuid="src-2", name="Hall cam"))
        bridge, fake = _connected_bridge([_profile(sources=two_cams)])

        self._open(bridge, active=True, opened=True)
        bridge.publish_rule_transition(
            profile_uuid="profile-1",
            source_uuid="src-2",
            rule_uuid="rule-1",
            active=True,
            opened=True,
            last_seen="2026-01-01T00:01:00+00:00",
        )

        assert [c["payload"] for c in _rule_global(fake, "/count")] == [1, 2]


class TestRetractionsArePublishedBeforeReplacements:
    def test_a_device_that_changes_id_is_retracted_before_the_new_one_is_published(self):
        """Ordering, not content — and it is what a rename costs if it is wrong.

        HA keys entities by `unique_id` but derives `entity_id` from the device and
        entity *names*, which a rename of the id namespace leaves untouched. Publish the
        replacement first and both devices exist at once, so every new entity collides
        with a live old one and is uniquified to `…_2` — permanently, because the old is
        retracted a moment later and nothing renames the survivor. Long-term statistics
        are keyed on `entity_id`, so they are orphaned with it.
        """
        subjects = {}
        before = build_all_discovery([_profile()], PREFIX, subjects)
        _, previous = reconcile_discovery(DiscoveryState(), before)

        # A device whose id changed: same camera, same names, new topic — exactly the
        # shape of the `encounterwatch_` → `purrview_` rename.
        renamed = _profile(sources=(SourceView(uuid="src-9", name="Kitchen cam"),))
        to_publish, _ = reconcile_discovery(previous, build_all_discovery([renamed], PREFIX, subjects))

        order = list(to_publish)
        gone = source_config_topic("src-1")
        added = source_config_topic("src-9")
        assert to_publish[gone] is None
        assert order.index(gone) < order.index(added)

    def test_every_retraction_precedes_every_publish(self):
        """Asserted as a partition rather than on one pair, so a future edit that
        appends a retraction in the wrong place cannot pass by luck."""
        subjects = {}
        _, previous = reconcile_discovery(DiscoveryState(), build_all_discovery([_profile()], PREFIX, subjects))
        renamed = _profile(sources=(SourceView(uuid="src-9", name="Kitchen cam"),))
        to_publish, _ = reconcile_discovery(previous, build_all_discovery([renamed], PREFIX, subjects))

        kinds = [payload is None for payload in to_publish.values()]
        assert any(kinds) and not all(kinds), kinds
        assert kinds == sorted(kinds, reverse=True), kinds


class _ArmingPublisher(_FakePublisher):
    """A fake that also records subscriptions and serves retained messages back.

    Both halves matter here and neither is exercised anywhere else: arming is the first
    thing the bridge *reads* from the broker rather than only writing to it.
    """

    def __init__(self, retained=None):
        super().__init__()
        self.subscriptions = {}
        self._retained = dict(retained or {})

    def subscribe(self, topic, callback, *, absolute=False, qos=0):
        # ⚠️ Prefixed exactly as `AsyncMqttPublisher.subscribe` does. Storing the bare
        # pattern instead silently makes `deliver` match nothing, which turns every test
        # below into a vacuous pass — it is how the first cut of these tests "passed".
        self.subscriptions[topic if absolute else f"{PREFIX}/{topic.lstrip('/')}"] = callback

    async def close(self):
        self.closed = True

    async def read_retained(self, pattern, *, absolute=False, quiet_s=0.5, timeout_s=5.0):
        return dict(self._retained)

    def deliver(self, topic: str, payload: bytes) -> int:
        """Hand a message to whichever wildcard subscription matches it.

        Returns how many handlers ran, so a caller can assert the message was actually
        routed rather than quietly dropped.
        """
        delivered = 0
        for pattern, callback in self.subscriptions.items():
            if _topic_matches(pattern, topic):
                callback(topic, payload)
                delivered += 1
        return delivered


def _topic_matches(pattern: str, topic: str) -> bool:
    """MQTT `+` matching, enough for the wildcards this bridge subscribes to."""
    p, t = pattern.split("/"), topic.split("/")
    return len(p) == len(t) and all(a == "+" or a == b for a, b in zip(p, t))


def _armed_bridge(profiles, retained=None):
    """A bridge primed as a reconcile would leave it, with arming wired up."""
    bridge = MqttBridge(db_api_client=None)
    fake = _ArmingPublisher(retained)
    bridge._publisher = fake
    bridge._prefix = PREFIX
    bridge._slugs = build_slug_index(profiles)
    try:
        # What `ensure_started` captures. Inbound commands hop back onto it, so without
        # it the handler takes its no-loop fallback and never echoes.
        bridge._loop = asyncio.get_running_loop()
    except RuntimeError:
        bridge._loop = None
    bridge._ensure_armed_subscription()
    return bridge, fake


ARMED_SET_TOPIC = f"{PREFIX}/living-room-watcher/rules/cat-at-bowl/armed/set"
ARMED_STATE_TOPIC = f"{PREFIX}/living-room-watcher/rules/cat-at-bowl/armed"


class TestRuleArmingSwitch:
    """A rule's HA switch is a *runtime* gate, not an edit of the rule.

    It exists so an automation can say "record people only while the house is empty"
    without the app having to know what "empty" means, and without a presence change
    writing to the config DB. See `RuleSpec.armed`.
    """

    def test_the_rules_device_offers_a_switch_per_rule(self):
        device = build_rules_discovery([_profile()], PREFIX)

        switch = device["cmps"]["rule-1_armed"]
        assert switch["p"] == "switch"
        assert switch["command_topic"].endswith("/armed/set")
        assert switch["state_topic"].endswith("/armed")
        # The state topic must not be the command topic: a switch that read back its own
        # commands would show armed the moment one was sent, whether or not it applied.
        assert switch["state_topic"] != switch["command_topic"]
        # Not optimistic — HA only moves the toggle once the app echoes the new state.
        assert "optimistic" not in switch

    def test_a_rule_that_publishes_nothing_gets_no_switch(self):
        """Consistent with every other entity on this device: `publish_to_ha` off means
        no HA representation at all, so there is nothing to arm from there."""
        off = _profile(rules=(RuleView(uuid="rule-1", name="cat at bowl", publish_to_ha=False),))

        assert build_rules_discovery([off], PREFIX) is None

    def test_rules_start_armed(self):
        """The flag is additive: an install that never touches a switch must behave
        exactly as it did before switches existed."""
        bridge, _fake = _armed_bridge([_profile()])

        assert bridge.is_rule_armed("rule-1") is True
        assert bridge.is_rule_armed("a-rule-never-seen") is True

    async def test_an_off_command_disarms_the_rule_and_is_echoed(self):
        bridge, fake = _armed_bridge([_profile()])

        assert fake.deliver(ARMED_SET_TOPIC, b"OFF") == 1, "the command topic must be subscribed"
        await asyncio.sleep(0)  # the handler hops off paho's thread onto the loop

        assert bridge.is_rule_armed("rule-1") is False
        echoes = [c for c in fake.calls if c["topic"].endswith("/armed")]
        assert [c["payload"] for c in echoes] == ["OFF"]
        # Retained, or the switch would read `unknown` after every HA restart.
        assert echoes[0]["retain"] is True

    async def test_an_on_command_arms_it_again(self):
        """The pair, so the handler cannot pass by hard-coding either answer."""
        bridge, fake = _armed_bridge([_profile()])
        fake.deliver(ARMED_SET_TOPIC, b"OFF")
        await asyncio.sleep(0)

        fake.deliver(ARMED_SET_TOPIC, b"ON")
        await asyncio.sleep(0)

        assert bridge.is_rule_armed("rule-1") is True
        assert [c["payload"] for c in fake.calls if c["topic"].endswith("/armed")] == ["OFF", "ON"]

    async def test_a_command_for_an_unknown_rule_is_ignored(self):
        """Retained commands outlive the rules they named. Resolving the slug to nothing
        must be a no-op, not an entry under a bogus key that a watcher would then read.
        """
        bridge, fake = _armed_bridge([_profile()])

        assert fake.deliver(f"{PREFIX}/living-room-watcher/rules/a-deleted-rule/armed/set", b"OFF") == 1
        await asyncio.sleep(0)

        assert bridge._armed == {}

    async def test_an_unrecognised_payload_changes_nothing(self):
        bridge, fake = _armed_bridge([_profile()])

        assert fake.deliver(ARMED_SET_TOPIC, b"maybe") == 1
        await asyncio.sleep(0)

        assert bridge._armed == {}
        assert bridge.is_rule_armed("rule-1") is True

    def test_the_state_topic_is_not_read_back_as_a_command(self):
        """The state and command topics differ by a level precisely so the app's own
        retained echo cannot be re-consumed as an instruction — which would make every
        reconnect replay the last command back into the handler."""
        bridge, fake = _armed_bridge([_profile()])

        assert fake.deliver(ARMED_STATE_TOPIC, b"OFF") == 0, "the echo must not reach the command handler"
        # It still resolves as a *state* topic, which is what the restore path reads.
        assert bridge._rule_uuid_from_topic(ARMED_STATE_TOPIC) == "rule-1"

    async def test_arming_is_restored_from_the_retained_state(self):
        """Without this a service restart silently re-arms everything — for the case this
        feature exists for, that is the privacy-invasive direction and nothing reports it.
        """
        retained = {f"{PREFIX}/living-room-watcher/rules/cat-at-bowl/armed": b"OFF"}
        bridge, _fake = _armed_bridge([_profile()], retained=retained)

        await bridge._restore_armed_state()

        assert bridge.is_rule_armed("rule-1") is False

    async def test_a_restore_failure_leaves_everything_armed(self):
        """Degrading to the pre-feature behaviour beats refusing to bring the bridge up."""
        bridge, fake = _armed_bridge([_profile()])

        async def boom(*_a, **_kw):
            raise RuntimeError("broker said no")

        fake.read_retained = boom
        await bridge._restore_armed_state()

        assert bridge.is_rule_armed("rule-1") is True

    async def test_stopping_forgets_the_subscription_but_not_the_arming(self):
        """`ensure_started` builds a *new* publisher with an empty subscription table, so
        a bridge that thought it was still subscribed would go deaf with nothing to show
        for it. The arming itself is state of the world, not of the connection.
        """
        bridge, _fake = _armed_bridge([_profile()])
        bridge.set_rule_armed("rule-1", False)

        await bridge.stop()

        assert bridge._armed_subscribed is False
        assert bridge.is_rule_armed("rule-1") is False


class TestEveryLabelGetsADevice:
    """A label's device is justified by the label, not by a rule mentioning it.

    `Detected now` reads off the tracker rather than off events, so it answers "is she in
    the room" for anything a model can detect — no rule required. Gating the device on
    rule membership meant writing a rule you did not want purely to get that sensor.
    `publish_to_ha` on the label is the opt-out, and the only one.
    """

    def _subject(self, **overrides):
        kwargs = {"uuid": "subj-a", "name": "Luna"}
        kwargs.update(overrides)
        return {kwargs["uuid"]: SubjectView(**kwargs)}

    def test_a_label_no_rule_mentions_still_gets_its_device(self):
        ruleless = _profile(rules=())

        desired = build_all_discovery([ruleless], PREFIX, self._subject())

        assert subject_config_topic("subj-a") in desired

    def test_that_device_carries_the_tracker_driven_core(self):
        """The four that are properties of the label rather than of any rule. Asserted by
        name because "a device exists" would pass against an empty one."""
        cmps = build_all_discovery([_profile(rules=())], PREFIX, self._subject())[subject_config_topic("subj-a")][
            "cmps"
        ]

        assert {"detected", "count", "last_seen", "last_location"} <= set(cmps)
        assert cmps["detected"]["p"] == "binary_sensor"

    def test_it_carries_no_per_rule_counters(self):
        """The other half of the same fact: the core is there, and the tiers that would
        genuinely have nothing to count are not invented alongside it."""
        cmps = build_all_discovery([_profile(rules=())], PREFIX, self._subject())[subject_config_topic("subj-a")][
            "cmps"
        ]

        assert not [k for k in cmps if k.startswith("rule-")]

    def test_a_label_opted_out_gets_no_device_even_with_a_rule(self):
        """The opt-out still wins, and it is now the only thing that can suppress a
        device."""
        profile = _profile(rules=(RuleView(uuid="rule-1", name="bowl", subject_uuids=("subj-a",)),))

        desired = build_all_discovery([profile], PREFIX, self._subject(publish_to_ha=False))

        assert subject_config_topic("subj-a") not in desired

    def test_a_rule_mentioning_it_still_adds_the_breakdown(self):
        """Rule-less is not a downgrade of the old behaviour — a label a rule *does*
        mention keeps everything it had."""
        profile = _profile(rules=(RuleView(uuid="rule-1", name="bowl", subject_uuids=("subj-a",)),))

        cmps = build_all_discovery([profile], PREFIX, self._subject())[subject_config_topic("subj-a")]["cmps"]

        assert "detected" in cmps
        assert [k for k in cmps if k.startswith("rule-1_")]

    def test_the_vocabulary_is_what_bounds_the_devices(self):
        """Two labels, two devices — and nothing invents a device for a label the caller
        never named, which is what keeps a deleted label from becoming a bare-uuid card.
        """
        subjects = {
            "subj-a": SubjectView(uuid="subj-a", name="Luna"),
            "subj-b": SubjectView(uuid="subj-b", name="Clover"),
        }

        desired = build_all_discovery([_profile(rules=())], PREFIX, subjects)

        subject_topics = {t for t in desired if "subject" in t}
        assert subject_topics == {subject_config_topic("subj-a"), subject_config_topic("subj-b")}


class TestPerCameraPresenceDiscovery:
    """The per-camera split is what "where is it" is actually asked of, and it has to
    work for a label no rule mentions.

    The rest of the per-source tier hangs off `subjects_in_scope`, so it is built inside
    a loop over the label's rules. A label with no rules therefore got *nothing* from
    ticking the flag — which is the case the tracker signal exists to cover, and the one
    that made the toggle look broken.
    """

    CAMERAS = (SourceView(uuid="src-1", name="Kitchen"), SourceView(uuid="src-2", name="Garage"))

    def _cmps(self, profile, per_source):
        subjects = {"subj-a": SubjectView(uuid="subj-a", name="Luna", publish_per_source=per_source)}
        return build_all_discovery([profile], PREFIX, subjects)[subject_config_topic("subj-a")]["cmps"]

    def test_a_label_with_no_rules_still_gets_one_presence_sensor_per_camera(self):
        ruleless = _profile(sources=self.CAMERAS, rules=())

        assert not [k for k in self._cmps(ruleless, per_source=False) if k.startswith("source_")]

        on = self._cmps(ruleless, per_source=True)
        assert {"source_src-1_detected", "source_src-2_detected"} <= set(on)
        assert on["source_src-1_detected"]["p"] == "binary_sensor"
        # Enabled, for the same reason the counters are: an entity HA creates disabled
        # carries no state and no history, so the opt-in would again do nothing visible.
        assert on["source_src-1_detected"].get("enabled_by_default") is not False

    def test_an_opted_out_camera_contributes_no_presence_sensor(self):
        """The camera's own flag still wins — a camera with no device of its own must
        not acquire one sideways through a label's breakdown."""
        profile = _profile(
            sources=(
                SourceView(uuid="src-1", name="Kitchen"),
                SourceView(uuid="src-2", name="Garage", publish_to_ha=False),
            ),
            rules=(),
        )

        cmps = self._cmps(profile, per_source=True)

        assert "source_src-1_detected" in cmps
        assert "source_src-2_detected" not in cmps

    def test_presence_sensors_survive_a_label_rename(self):
        profile = _profile(sources=self.CAMERAS, rules=())
        before = build_all_discovery(
            [profile], PREFIX, {"subj-a": SubjectView(uuid="subj-a", name="Luna", publish_per_source=True)}
        )[subject_config_topic("subj-a")]["cmps"]
        after = build_all_discovery(
            [profile], PREFIX, {"subj-a": SubjectView(uuid="subj-a", name="Renamed", publish_per_source=True)}
        )[subject_config_topic("subj-a")]["cmps"]

        assert before["source_src-1_detected"]["unique_id"] == after["source_src-1_detected"]["unique_id"]
        assert before["source_src-1_detected"]["state_topic"] != after["source_src-1_detected"]["state_topic"]


def _presence_bridge(per_source=True, name="Luna"):
    """A bridge primed as a reconcile leaves it, for one subject and two cameras."""
    profile = _profile(
        sources=(SourceView(uuid="src-1", name="Kitchen"), SourceView(uuid="src-2", name="Garage")),
        rules=(),
    )
    bridge, fake = _connected_bridge([profile])
    bridge._subject_slugs = {"subj-a": slugify(name)}
    bridge._label_publish = {"subj-a": (True, per_source)}
    return bridge, fake


def _payloads(fake, topic):
    return [c["payload"] for c in fake.calls if c["topic"] == topic]


class TestPerCameraPresencePublishing:
    def test_an_arrival_publishes_that_cameras_own_answer_and_the_union(self):
        bridge, fake = _presence_bridge()

        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T1")

        assert _payloads(fake, "subject/luna/state") == ["ON"]
        assert _payloads(fake, "subject/luna/source/kitchen/state") == ["ON"]

    def test_a_second_camera_updates_the_location_while_the_first_still_sees_it(self):
        """The reason the stamp is gated on a *camera* arrival rather than on the union
        flipping. Something that walks from one room into the next while still visible
        in the first never changes the union, so a union-gated stamp would leave the
        location pinned to the room it left."""
        bridge, fake = _presence_bridge()
        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T1")

        bridge.publish_presence(source_uuid="src-2", label_uuids=["subj-a"], source_name="Garage", at="T2")

        assert _payloads(fake, "subject/luna/attributes")[-1] == {"lastSeen": "T2", "lastSource": "Garage"}
        assert _payloads(fake, "subject/luna/source/garage/state") == ["ON"]
        # The union did not change, so the device-level sensor must not be republished —
        # a retained ON per frame is the noise this whole path is gated to avoid.
        assert _payloads(fake, "subject/luna/state") == ["ON"]

    def test_a_label_with_no_rules_gets_a_location(self):
        """The original complaint. `Last location` was written only by
        `publish_subject_event`, which fires only on a rule event, so a label no rule
        named was detectable but permanently locationless."""
        bridge, fake = _presence_bridge()

        bridge.publish_presence(source_uuid="src-2", label_uuids=["subj-a"], source_name="Garage", at="T1")

        assert _payloads(fake, "subject/luna/attributes") == [{"lastSeen": "T1", "lastSource": "Garage"}]

    def test_leaving_one_camera_turns_only_that_camera_off(self):
        bridge, fake = _presence_bridge()
        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T1")
        bridge.publish_presence(source_uuid="src-2", label_uuids=["subj-a"], source_name="Garage", at="T2")

        bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T3")

        assert _payloads(fake, "subject/luna/source/kitchen/state") == ["ON", "OFF"]
        assert _payloads(fake, "subject/luna/source/garage/state") == ["ON"]
        # Still visible in the garage, so the device-level answer stands.
        assert _payloads(fake, "subject/luna/state") == ["ON"]

    def test_departure_does_not_stamp_a_sighting(self):
        bridge, fake = _presence_bridge()
        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T1")

        bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T2")

        assert _payloads(fake, "subject/luna/attributes") == [{"lastSeen": "T1", "lastSource": "Kitchen"}]

    def test_an_unchanged_view_publishes_nothing(self):
        """Presence is evaluated every analysed frame, several times a second per
        camera. Without this the bridge would be the noisiest thing on the broker."""
        bridge, fake = _presence_bridge()
        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T1")
        before = len(fake.calls)

        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T2")

        assert len(fake.calls) == before

    def test_a_label_that_did_not_ask_for_the_breakdown_gets_no_per_camera_topics(self):
        """The cardinality gate, applied to the publish as well as to discovery: a
        retained state on a topic no entity reads sits on the broker for good, and
        re-ticking the flag later would replay a stale ON into a fresh entity."""
        bridge, fake = _presence_bridge(per_source=False)

        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T1")

        assert not [c for c in fake.calls if "/source/" in c["topic"]]
        # The device-level answer and the location are not per-camera entities, so they
        # are unaffected by the gate.
        assert _payloads(fake, "subject/luna/state") == ["ON"]
        assert _payloads(fake, "subject/luna/attributes") == [{"lastSeen": "T1", "lastSource": "Kitchen"}]

    def test_a_stopped_watcher_clears_its_cameras_but_records_no_sighting(self):
        bridge, fake = _presence_bridge()
        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T1")

        bridge.forget_source_presence("src-1")

        assert _payloads(fake, "subject/luna/source/kitchen/state") == ["ON", "OFF"]
        assert _payloads(fake, "subject/luna/state") == ["ON", "OFF"]
        assert _payloads(fake, "subject/luna/attributes") == [{"lastSeen": "T1", "lastSource": "Kitchen"}]


class TestSubjectAttributesSurviveAPartialWrite:
    """The attributes topics are retained, and a retained publish replaces. Three
    writers know three different subsets of the payload, so any of them writing a whole
    payload drops the keys it does not know."""

    def test_a_writer_that_knows_only_the_timestamp_does_not_blank_the_location(self):
        """This is the reseed (`_seed_counters`), which reads its rows from the DB —
        `build_subject_seed` has no camera to report. Before the merge, every reconnect
        blanked `Last location` app-wide until each subject's next event."""
        bridge, fake = _presence_bridge()
        bridge.publish_presence(source_uuid="src-2", label_uuids=["subj-a"], source_name="Garage", at="T1")

        bridge._publish_subject_attributes("subject/luna", lastSeen="T0")

        assert _payloads(fake, "subject/luna/attributes")[-1] == {"lastSeen": "T0", "lastSource": "Garage"}

    def test_a_nameless_camera_stamps_the_time_and_leaves_the_location_alone(self):
        bridge, fake = _presence_bridge()
        bridge.publish_presence(source_uuid="src-2", label_uuids=["subj-a"], source_name="Garage", at="T1")

        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name=None, at="T2")

        assert _payloads(fake, "subject/luna/attributes")[-1] == {"lastSeen": "T2", "lastSource": "Garage"}


class TestAStaleCameraAbstainsRatherThanReportingAbsence:
    """A camera that has stopped reporting must not be allowed to say "not here".

    `presence_labels` returns an empty set both when a camera looked and saw nothing and
    when it has not looked recently enough to say. Publishing OFF for the second makes a
    stationary object — a parked car — appear to leave when its feed stutters and to
    arrive again when it recovers, which an automation triggering on arrival cannot
    distinguish from the real thing.
    """

    def test_going_stale_marks_the_camera_unavailable_rather_than_off(self):
        bridge, fake = _presence_bridge()
        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T1")

        bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T2", known=False)

        assert _payloads(fake, "presence/kitchen/status") == ["online", "offline"]
        # Still ON, deliberately: the entity is unavailable so nobody reads it, and
        # keeping the last real answer means recovery shows the truth at once instead
        # of a stale OFF that flips ON a frame later.
        assert _payloads(fake, "subject/luna/source/kitchen/state") == ["ON"]

    def test_recovery_with_the_subject_still_there_is_not_a_fresh_arrival(self):
        """The whole point. Across a feed outage the per-camera sensor must go
        unavailable and come back ON — never OFF then ON, which reads as a departure
        and an arrival."""
        bridge, fake = _presence_bridge()
        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T1")
        bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T2", known=False)

        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T3")

        # Never OFF. A repeated retained ON is idempotent — HA records no state change
        # and fires no trigger — but a single OFF in here is the departure the sensor
        # exists to not invent.
        assert "OFF" not in _payloads(fake, "subject/luna/source/kitchen/state")
        assert _payloads(fake, "presence/kitchen/status") == ["online", "offline", "online"]

    def test_a_stale_camera_is_dropped_from_the_union_not_counted_as_absent(self):
        """Two cameras, one stale. The fresh one decides — the stale one neither pins
        the subject present nor votes it absent."""
        bridge, fake = _presence_bridge()
        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T1")
        bridge.publish_presence(source_uuid="src-2", label_uuids=[], source_name="Garage", at="T1")
        assert _payloads(fake, "subject/luna/state") == ["ON"]

        bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T2", known=False)

        # Kitchen abstained, Garage says no — so the union over cameras that can still
        # speak is OFF, and it is available because Garage is reporting.
        assert _payloads(fake, "subject/luna/state") == ["ON", "OFF"]
        assert _payloads(fake, PRESENCE_ANY_AVAILABILITY_TOPIC) == ["online"]

    def test_the_device_level_answer_goes_unavailable_only_when_every_camera_is_quiet(self):
        bridge, fake = _presence_bridge()
        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T1")
        bridge.publish_presence(source_uuid="src-2", label_uuids=["subj-a"], source_name="Garage", at="T1")

        bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T2", known=False)
        assert _payloads(fake, PRESENCE_ANY_AVAILABILITY_TOPIC) == ["online"]

        bridge.publish_presence(source_uuid="src-2", label_uuids=[], source_name="Garage", at="T2", known=False)
        assert _payloads(fake, PRESENCE_ANY_AVAILABILITY_TOPIC) == ["online", "offline"]

    def test_availability_is_change_gated(self):
        """Called from the per-frame path with `known=True` for essentially every frame,
        and again from the loop sweep for every stale camera on every pass."""
        bridge, fake = _presence_bridge()
        bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T1", known=False)
        before = len(fake.calls)

        for _ in range(5):
            bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T2", known=False)

        assert len(fake.calls) == before

    def test_a_stopped_watcher_makes_its_cameras_unavailable_too(self):
        """A watcher that stopped is not a camera reporting "nothing here" either."""
        bridge, fake = _presence_bridge()
        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T1")

        bridge.forget_source_presence("src-1")

        assert _payloads(fake, "presence/kitchen/status") == ["online", "offline"]


class TestPresenceEntitiesDeclareTheirAvailability:
    """Discovery has to name the topics above, or the bridge publishes availability
    nothing reads — the silent half of this failure."""

    def _cmps(self):
        profile = _profile(
            sources=(SourceView(uuid="src-1", name="Kitchen"), SourceView(uuid="src-2", name="Garage")),
            rules=(),
        )
        subjects = {"subj-a": SubjectView(uuid="subj-a", name="Luna", publish_per_source=True)}
        return build_all_discovery([profile], PREFIX, subjects)[subject_config_topic("subj-a")]["cmps"]

    def test_a_per_camera_sensor_tracks_its_own_cameras_freshness(self):
        cmps = self._cmps()

        topics = {a["topic"] for a in cmps["source_src-2_detected"]["availability"]}
        assert f"{PREFIX}/presence/garage/status" in topics
        # Not another camera's, or one stale feed would blank every camera's sensor.
        assert f"{PREFIX}/presence/kitchen/status" not in topics
        assert cmps["source_src-2_detected"]["availability_mode"] == "all"

    def test_the_device_level_sensor_tracks_the_any_camera_flag(self):
        topics = {a["topic"] for a in self._cmps()["detected"]["availability"]}
        assert topics == {f"{PREFIX}/status", f"{PREFIX}/{PRESENCE_ANY_AVAILABILITY_TOPIC}"}

    def test_component_availability_repeats_the_app_topic(self):
        """A component's own `availability` replaces the device's rather than adding to
        it, so dropping the app-level LWT here would leave these two entities reading as
        available after the whole app went down."""
        for key in ("detected", "source_src-1_detected"):
            topics = {a["topic"] for a in self._cmps()[key]["availability"]}
            assert f"{PREFIX}/status" in topics
