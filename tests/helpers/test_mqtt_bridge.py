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
    component_config_topic,
    count_entities,
    flatten_discovery,
    legacy_bundle_topics,
    reconcile_discovery,
    source_config_topic,
    source_device_id,
    state_from_retained,
    subject_config_topic,
    subject_device_id,
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


def _desired(profiles, subjects=None):
    """The component set `publish_all_discovery` actually reconciles against.

    Flattened, because that is what the production path feeds `reconcile_discovery`.
    Passing raw bundles here would still "work" — the reconcile treats payloads as
    opaque — while testing a shape nothing publishes.
    """
    return flatten_discovery(build_all_discovery(profiles, PREFIX, subjects))


def _as_retained(desired):
    """`desired` as the broker replays it back on a fresh connection."""
    return {topic: json.dumps(payload).encode() for topic, payload in desired.items()}


def _topic_for(topics, device_id, key):
    """The one component topic for `key` on `device_id`, whatever platform it lands on.

    Looked up rather than spelled out so these tests do not pin which platform a
    component uses — that is `build_*_discovery`'s business and has its own tests.
    """
    matches = [t for t in topics if t.split("/")[2] == device_id and t.split("/")[3] == key]
    assert len(matches) == 1, f"expected one topic for {device_id}/{key}, got {matches}"
    return matches[0]


def _keys_on(topics, device_id):
    """Component keys addressed on one device, out of a set of component topics."""
    return {t.split("/")[3] for t in topics if t.split("/")[2] == device_id}


class TestComponentTopicLayout:
    """Discovery is published one component per topic rather than as a device bundle,
    because HA reprocesses a whole bundle on any change to it — see
    `component_config_topic`. The device block is what still groups them."""

    def test_a_component_carries_the_device_block_so_grouping_survives(self):
        desired = _desired([_profile()])
        topic = _topic_for(desired, source_device_id("src-1"), "camera")
        assert desired[topic]["dev"]["ids"] == source_device_id("src-1")
        assert desired[topic]["dev"]["via_device"] == "purrview_system"

    def test_the_platform_moves_into_the_topic_and_out_of_the_payload(self):
        """A payload carrying a platform the topic could contradict is exactly the kind
        of duplicated fact that rots. HA reads it from the topic."""
        desired = _desired([_profile()])
        topic = _topic_for(desired, source_device_id("src-1"), "camera")
        assert topic.split("/")[1] == "camera"
        assert "p" not in desired[topic]

    def test_a_component_keeps_its_own_availability_over_the_devices(self):
        """A per-camera presence sensor names its own camera's freshness topic, and a
        component's availability *replaces* the device's in HA rather than adding to it.
        Inheriting the device's over the top would make it go unavailable only when the
        whole app did."""
        subjects = {"subj-a": SubjectView(uuid="subj-a", name="Luna", publish_per_source=True)}
        desired = _desired([_profile()], subjects)
        device = subject_device_id("subj-a")
        own = desired[_topic_for(desired, device, "source_src-1_detected")]["availability"]
        assert any("presence" in entry["topic"] for entry in own)

    def test_every_component_of_a_device_lands_on_its_own_topic(self):
        desired = _desired([_profile()])
        bundle = build_all_discovery([_profile()], PREFIX)[source_config_topic("src-1")]
        assert _keys_on(desired, source_device_id("src-1")) == set(bundle["cmps"])


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

    def test_publish_snapshot_to_ha_adds_a_camera_component(self):
        profile = _profile(rules=(RuleView(uuid="rule-1", name="cat at bowl", publish_snapshot_to_ha=True),))
        cmps = build_source_discovery(profile, profile.sources[0], PREFIX)["cmps"]

        state_topic = cmps["rule-1_state"]["state_topic"]
        assert cmps["rule-1_snapshot"]["p"] == "camera"
        # Same rule_base as every other per-rule topic, minus "/state" - derived from
        # a real sibling topic rather than a hand-typed slug, so this doesn't silently
        # stop meaning anything if slugify's own output format ever changes.
        assert cmps["rule-1_snapshot"]["topic"] == state_topic.removesuffix("/state") + "/snapshot"

    def test_publish_snapshot_to_ha_off_by_default(self):
        """Opt-in, unlike publish_to_ha - a snapshot is a meaningfully bigger MQTT
        payload than the sensors/counters a rule already publishes for free."""
        cmps = build_source_discovery(_profile(), _profile().sources[0], PREFIX)["cmps"]

        assert "rule-1_snapshot" not in cmps

    def test_publish_snapshot_to_ha_has_no_effect_when_publish_to_ha_is_off(self):
        """Meaningless without a device to hang off - the outer publish_to_ha gate
        already excludes the whole rule, snapshot flag or not."""
        profile = _profile(
            rules=(RuleView(uuid="rule-1", name="cat at bowl", publish_to_ha=False, publish_snapshot_to_ha=True),)
        )
        cmps = build_source_discovery(profile, profile.sources[0], PREFIX)["cmps"]

        assert not any(k.startswith("rule-1_") for k in cmps)

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

    def test_a_deleted_rule_is_retracted_on_its_own_topics(self):
        """Removal is now a single empty payload per component topic. The bundle scheme
        needed a two-step tombstone here — republish the component naming only its
        platform, omit it on a later pass — because omitting it from `cmps` did not
        reliably remove the entity. A component owning its topic makes that unnecessary.
        """
        with_rule = _profile(rules=(RuleView(uuid="rule-1", name="cat at bowl"),))
        without_rule = _profile(rules=())
        device = source_device_id("src-1")

        _, state = reconcile_discovery(DiscoveryState(), _desired([with_rule]))
        to_publish, state = reconcile_discovery(state, _desired([without_rule]))

        retracted = {t for t, payload in to_publish.items() if payload is None}
        assert _keys_on(retracted, device), "the deleted rule's components must be retracted"
        assert all(k.startswith("rule-1_") for k in _keys_on(retracted, device))

        # And it settles: a retracted topic is gone from the state, so the next pass has
        # nothing to retract and says nothing at all.
        follow_up, _ = reconcile_discovery(state, _desired([without_rule]))
        assert follow_up == {}

    def test_deleting_a_rule_leaves_the_rest_of_the_camera_untouched(self):
        """The point of per-component discovery. Under the bundle scheme this
        republished the camera's whole config, so every entity on it — the camera feed,
        telemetry, every other rule — was torn down and rebuilt to delete one rule.
        """
        with_rule = _profile(rules=(RuleView(uuid="rule-1", name="cat at bowl"),))
        without_rule = _profile(rules=())
        device = source_device_id("src-1")

        _, state = reconcile_discovery(DiscoveryState(), _desired([with_rule]))
        to_publish, _ = reconcile_discovery(state, _desired([without_rule]))

        republished = {t for t, payload in to_publish.items() if payload is not None}
        assert "camera" not in _keys_on(republished, device)
        assert republished == set(), "nothing but the deleted rule should be written"

    def test_renaming_a_rule_republishes_only_that_rule(self):
        """The same guarantee for an edit rather than a deletion, which is the case that
        actually happens while working on a watcher."""
        before = _profile(
            rules=(RuleView(uuid="rule-1", name="cat at bowl"), RuleView(uuid="rule-2", name="cat at door"))
        )
        after = _profile(
            rules=(RuleView(uuid="rule-1", name="cat at the water bowl"), RuleView(uuid="rule-2", name="cat at door"))
        )
        device = source_device_id("src-1")

        _, state = reconcile_discovery(DiscoveryState(), _desired([before]))
        to_publish, _ = reconcile_discovery(state, _desired([after]))

        touched = _keys_on(set(to_publish), device)
        assert touched, "the renamed rule must be republished"
        assert all(k.startswith("rule-1_") for k in touched), f"rule-2 and the camera must be left alone: {touched}"

    def test_renaming_a_camera_does_not_rebuild_every_subject_entity(self):
        """`build_subject_discovery` embeds the whole camera list, so under the bundle
        scheme renaming one camera rewrote every subject device in the app and rebuilt
        every entity on it — `Detected now`, every counter, for every label. Only the
        per-camera components actually mention the camera."""
        subjects = {"subj-a": SubjectView(uuid="subj-a", name="Luna", publish_per_source=True)}
        before = _profile(rules=(RuleView(uuid="rule-1", name="bowl", subject_uuids=("subj-a",)),))
        after = _profile(
            sources=(SourceView(uuid="src-1", name="Kitchen camera"),),
            rules=(RuleView(uuid="rule-1", name="bowl", subject_uuids=("subj-a",)),),
        )
        device = subject_device_id("subj-a")

        _, state = reconcile_discovery(DiscoveryState(), _desired([before], subjects))
        to_publish, _ = reconcile_discovery(state, _desired([after], subjects))

        touched = _keys_on(set(to_publish), device)
        assert "detected" not in touched, "the label's own presence entity has nothing to do with a camera rename"
        assert "count" not in touched
        assert "rule-1_count" not in touched

    def test_state_read_back_from_the_broker_can_retract_across_a_restart(self):
        """The bug the retained read-back exists to fix. With an in-memory state, a
        restart forgets what was published, so a rule deleted while tasks_server was
        down can never be retracted — the process concludes it has told HA nothing and
        can only ever add. Adopting the broker's retained configs is what makes the
        first reconcile after a restart able to remove.
        """
        with_rule = _profile(rules=(RuleView(uuid="rule-1", name="cat at bowl"),))
        without_rule = _profile(rules=())
        device = source_device_id("src-1")

        # A fresh process: no memory at all, only what the broker replays.
        adopted = state_from_retained(_as_retained(_desired([with_rule])))
        to_publish, _ = reconcile_discovery(adopted, _desired([without_rule]))

        retracted = {t for t, payload in to_publish.items() if payload is None}
        gone = _keys_on(retracted, device)
        assert gone, "a rule deleted while the process was down must still be retracted"
        assert all(k.startswith("rule-1_") for k in gone)

    def test_an_empty_retained_payload_is_not_adopted_as_a_live_config(self):
        """Empty payloads are retractions someone already published. Adopting one would
        resurrect the entity it removed, on the very next reconcile."""
        topic = component_config_topic("sensor", source_device_id("src-1"), "fps")
        assert state_from_retained({topic: b""}).payloads == {}

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
        desired = _desired([_profile()])
        retained = _as_retained(desired)
        foreign = "homeassistant/sensor/0x00124b0022ab1234/temperature/config"
        retained[foreign] = json.dumps({"dev": {"ids": "0x00124b0022ab1234"}, "name": "Temperature"}).encode()

        adopted = state_from_retained(retained)
        assert foreign not in adopted.topics, "a device we did not publish must not be adopted"

        to_publish, _ = reconcile_discovery(adopted, desired)
        assert foreign not in to_publish, "a device we did not publish must never be retracted"

    def test_our_own_configs_are_still_adopted_across_a_rename_of_nothing(self):
        """The ownership filter must not throw the baby out: configs carrying our own
        device-id namespace are exactly what the read-back exists to recover."""
        desired = _desired([_profile()])

        adopted = state_from_retained(_as_retained(desired))

        assert _topic_for(adopted.topics, source_device_id("src-1"), "camera") in adopted.topics
        assert adopted.topics == set(desired)

    def test_an_unparseable_retained_config_is_skipped_not_fatal(self):
        """One malformed retained message must not stop the bridge coming up — and the
        configs around it must still be adopted."""
        desired = _desired([_profile()])
        retained = _as_retained(desired)
        junk = component_config_topic("sensor", "purrview_junk", "broken")
        retained[junk] = b"{not json"

        state = state_from_retained(retained)

        assert _topic_for(state.topics, source_device_id("src-1"), "camera") in state.topics
        assert junk not in state.topics

    def test_a_deleted_source_is_retracted_with_an_empty_payload(self):
        with_source = [_profile()]
        without_source = [_profile(sources=(), rules=())]

        before = _desired(with_source)
        _, previous_topics = reconcile_discovery(DiscoveryState(), before)
        to_publish, remaining = reconcile_discovery(previous_topics, _desired(without_source))

        device = source_device_id("src-1")
        assert _keys_on(set(before), device), "sanity: the camera had components to begin with"
        assert all(to_publish[t] is None for t in before if t.split("/")[2] == device)
        assert not _keys_on(remaining.topics, device)

    def test_an_unchanged_profile_publishes_nothing_at_all(self):
        """Not merely "no retractions": nothing.

        The reconcile runs on a timer, and a device config republished with identical
        content is not a no-op at the other end — Home Assistant reprocesses the device
        and its entities drop to unavailable and come back with no state. Publishing
        every desired topic every pass made every entity on every device flap between
        `unknown` and `unavailable` once a minute, for config nobody had touched.
        """
        profile = [_profile()]
        _, previous_topics = reconcile_discovery(DiscoveryState(), _desired(profile))
        to_publish, _ = reconcile_discovery(previous_topics, _desired(profile))
        assert to_publish == {}

    def test_a_changed_config_still_publishes(self):
        """The other half — silence must come from nothing having changed, not from the
        reconcile having stopped working."""
        before = [_profile(rules=(RuleView(uuid="rule-1", name="cat at bowl"),))]
        after = [_profile(rules=(RuleView(uuid="rule-1", name="cat at the water bowl"),))]

        _, state = reconcile_discovery(DiscoveryState(), _desired(before))
        to_publish, _ = reconcile_discovery(state, _desired(after))

        assert _keys_on(set(to_publish), source_device_id("src-1"))

    def test_a_config_already_on_the_broker_is_not_republished_after_a_restart(self):
        """The seeded half. The bridge adopts the broker's retained configs at connect,
        so a restart that changes nothing must not churn every entity in HA — which is
        the same flap, once per restart instead of once per minute."""
        desired = _desired([_profile()])

        adopted = state_from_retained(_as_retained(desired))
        to_publish, _ = reconcile_discovery(adopted, desired)

        assert to_publish == {}

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


class TestTheOldDeviceBundlesAreCleanedUp:
    """Moving off `homeassistant/device/<id>/config` leaves the previous scheme's
    messages retained on the broker. Left there, Home Assistant re-creates a full
    duplicate set of entities from them on every restart, forever — and there is no
    later version of the app that can clean them up once it has stopped claiming them.
    """

    OURS = "homeassistant/device/purrview_source_src-1/config"

    def test_our_own_bundles_are_claimed(self):
        retained = {self.OURS: b"{}", "homeassistant/device/encounterwatch_rules/config": b"{}"}
        assert legacy_bundle_topics(retained) == set(retained)

    def test_another_integrations_bundle_is_left_alone(self):
        """Same trap as `owns_config_topic`: retracting a foreign bundle is a correct,
        effective deletion of someone else's devices, with no error anywhere."""
        assert legacy_bundle_topics({"homeassistant/device/0x00124b0022ab1234/config": b"{}"}) == frozenset()

    async def _pass(self, bridge):
        await bridge.publish_all_discovery()

    def _bridge(self, legacy=()):
        bridge = MqttBridge(db_api_client=None)
        fake = _ArmingPublisher()
        bridge._publisher = fake
        bridge._prefix = PREFIX
        bridge._legacy_bundles = frozenset(legacy)
        # Non-empty so `_restore_armed_state` short-circuits; it is not what is under
        # test here and it needs a real broker read.
        bridge._armed = {"rule-1": True}

        async def _profiles():
            return [_profile()]

        bridge._load_profiles = _profiles
        return bridge, fake

    async def test_the_retraction_pass_publishes_no_component_configs(self):
        """The load-bearing assertion. Every entity in a bundle shares its `unique_id`
        with its per-component replacement, so if both reach HA before it has finished
        removing the old device, the new entity collides with the live old one and is
        uniquified to `sensor.<name>_2` — permanently. Ordering them within one pass
        makes that a race; separating the passes removes it."""
        bridge, fake = self._bridge(legacy=[self.OURS])

        await self._pass(bridge)

        # Only discovery configs are at stake; the presence-availability writes that
        # share this pass are on our own prefix and touch no entity registry.
        configs = [c for c in fake.calls if c["topic"].startswith("homeassistant/")]
        assert [c["topic"] for c in configs] == [self.OURS]
        assert configs[0]["payload"] == "", "a retraction is an empty payload"
        assert configs[0]["retain"] is True, "a non-retained retraction leaves the old config on the broker"

    async def test_the_next_pass_publishes_the_components(self):
        """And the cleanup does not repeat — a second retraction every minute would keep
        HA rebuilding forever."""
        bridge, fake = self._bridge(legacy=[self.OURS])
        await self._pass(bridge)
        fake.calls.clear()

        await self._pass(bridge)

        topics = [c["topic"] for c in fake.calls]
        assert self.OURS not in topics
        assert any(t.startswith("homeassistant/") and t.endswith("/config") for t in topics)

    async def test_a_broker_that_never_saw_the_old_scheme_loses_no_pass(self):
        """The overwhelmingly common case, and it must not cost a reconcile interval."""
        bridge, fake = self._bridge()

        await self._pass(bridge)

        assert any(t["topic"].endswith("/config") for t in fake.calls)


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


class TestARuleThatHasNeverFiredStillGetsAState:
    """Seeding only what history knows about left every never-fired rule `unknown` in
    Home Assistant — permanently, not as a blip. Discovery creates the entity, nothing
    ever publishes its state topic, and `unknown` is not `unavailable`: it reads as a
    working sensor while numeric comparisons against it silently fail.

    Zero is not a guess. `get_watch_event_counts` returns a row per pair that has fired,
    so a pair's absence *is* a count of zero.
    """

    TOPIC = "living-room-watcher/kitchen-cam/rule/cat-at-bowl/count"

    def test_a_pair_with_no_history_seeds_zero_rather_than_being_omitted(self):
        assert build_seed_publishes([_profile()], []) == {self.TOPIC: 0}

    def test_history_still_wins_over_the_zero(self):
        """The zero is a floor for pairs nothing is known about, never an overwrite —
        seeding a real counter back to 0 would read as a counter reset to a
        `total_increasing` sensor and dent its long-term statistics."""
        counts = [{"source_uuid": "src-1", "rule_uuid": "rule-1", "count": 42}]
        assert build_seed_publishes([_profile()], counts) == {self.TOPIC: 42}

    def test_an_opted_out_camera_gets_no_zero_either(self):
        """The zero follows the same gate as discovery. A retained payload on a topic no
        entity reads never expires — it sits on the broker for good, and re-enabling the
        camera later would replay it into a freshly created entity."""
        profile = _profile(sources=(SourceView(uuid="src-1", name="Kitchen cam", publish_to_ha=False),))
        assert build_seed_publishes([profile], []) == {}

    def test_a_subject_tier_with_no_history_seeds_a_count_but_no_timestamp(self):
        """`Car Leaves` on the live device: the rule counter, its per-camera breakdown
        and its `last seen` all blank, for a rule that worked and had simply not
        happened yet. The count is knowable and the timestamp is not — and a `sensor`
        with `device_class: timestamp` rejects anything that is not a real one, so the
        honest answer to "when did this last happen" stays `None`."""
        profile = _profile(rules=(RuleView(uuid="rule-1", name="car leaves", subject_uuids=("subj-a",)),))
        seed = build_subject_seed([], [profile])
        assert seed[("subj-a",)] == (0, None)
        assert seed[("subj-a", "rule-1")] == (0, None)
        assert seed[("subj-a", "rule-1", "src-1")] == (0, None)

    def test_a_subject_tier_with_history_keeps_its_count_and_timestamp(self):
        profile = _profile(rules=(RuleView(uuid="rule-1", name="car leaves", subject_uuids=("subj-a",)),))
        counts = [
            {
                "source_uuid": "src-1",
                "rule_uuid": "rule-1",
                "subject_ref": "subj-a",
                "count": 7,
                "last_seen": "2026-01-02T00:00:00+00:00",
            }
        ]
        assert build_subject_seed(counts, [profile])[("subj-a", "rule-1")] == (7, "2026-01-02T00:00:00+00:00")


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
        seed = build_subject_seed(self.COUNTS, ())
        assert seed[("subj-a",)][0] == 12
        assert seed[("subj-a", "rule-1")][0] == 8
        assert seed[("subj-a", "rule-1", "src-1")][0] == 5

    def test_last_seen_rolls_up_as_the_most_recent_not_the_last_row(self):
        seed = build_subject_seed(self.COUNTS, ())
        assert seed[("subj-a",)][1] == "2026-01-03T00:00:00+00:00"
        assert seed[("subj-a", "rule-2")][1] == "2026-01-01T00:00:00+00:00"

    def test_events_with_no_subject_are_not_counted_against_a_subject(self):
        """A rule with no subject slot still produces events; they belong to the
        rule-level counters, which seed separately."""
        assert build_subject_seed([{**self.COUNTS[0], "subject_ref": None}], ()) == {}


class TestARuleThatHasNeverFiredHasAStateOnTheRulesDevice:
    """`_publish_rule_global` only runs on a transition, so every entity on the Rules
    device sat `unknown` until its rule first fired. Observed live on `Car Leaves`: six
    blank entities for a rule that worked and had simply not happened yet.

    Seeded where the broker holds nothing, never over a real value — this runs on every
    reconcile, and an unconditional publish would blank a rule that is currently open
    each time a watcher started.
    """

    def _seeded(self, profiles, retained=(), **primed):
        bridge, fake = _connected_bridge(profiles)
        bridge._published_rule_globals = set(retained)
        for attr, value in primed.items():
            setattr(bridge, f"_{attr}", value)
        bridge._seed_rule_globals(profiles)
        return bridge, fake

    def test_a_never_fired_rule_seeds_off_and_zero(self):
        _bridge, fake = self._seeded([_profile()])
        assert _payloads(fake, "living-room-watcher/rules/cat-at-bowl/state") == ["OFF"]
        assert _payloads(fake, "living-room-watcher/rules/cat-at-bowl/count") == [0]

    def test_the_count_comes_from_history_not_from_zero(self):
        _bridge, fake = self._seeded([_profile()], counts={("profile-1", "src-1", "rule-1"): 9})
        assert _payloads(fake, "living-room-watcher/rules/cat-at-bowl/count") == [9]

    def test_a_topic_the_broker_already_holds_is_left_alone(self):
        """The load-bearing half. A rule mid-event has a retained ON, and seeding OFF
        over it would close the event as far as every automation is concerned."""
        retained = ["living-room-watcher/rules/cat-at-bowl/state"]
        _bridge, fake = self._seeded([_profile()], retained=retained)
        assert _payloads(fake, "living-room-watcher/rules/cat-at-bowl/state") == []
        assert _payloads(fake, "living-room-watcher/rules/cat-at-bowl/count") == [0]

    def test_a_rule_open_right_now_seeds_on_not_off(self):
        """Renaming a rule moves its topics, so the new ones are blank and get seeded
        mid-event. Seeding a constant OFF there would report the open event closed."""
        _bridge, fake = self._seeded([_profile()], open_rules={("profile-1", "src-1", "rule-1")})
        assert _payloads(fake, "living-room-watcher/rules/cat-at-bowl/state") == ["ON"]

    def test_seeding_is_once_not_once_per_reconcile(self):
        bridge, fake = self._seeded([_profile()])
        before = len(fake.calls)

        for _ in range(3):
            bridge._seed_rule_globals([_profile()])

        assert len(fake.calls) == before

    def test_the_event_topic_is_never_seeded(self):
        """HA's event entity fires on every message it receives, including a retained one
        replayed on reconnect — so a seeded event would re-trigger every automation bound
        to it on each restart. `unknown` until it happens is the correct state."""
        _bridge, fake = self._seeded([_profile()])
        assert _payloads(fake, "living-room-watcher/rules/cat-at-bowl/event") == []

    def test_the_attribute_backed_entities_are_not_given_a_fabricated_sighting(self):
        """`last seen`, `last subject` and `last camera` all read out of one attributes
        payload. Seeding them means asserting a sighting that did not happen — and
        `last seen` is a `device_class: timestamp`, which HA rejects unless it is real."""
        _bridge, fake = self._seeded([_profile()])
        assert _payloads(fake, "living-room-watcher/rules/cat-at-bowl/attributes") == []

    def test_a_failed_retained_read_seeds_nothing(self):
        """`None` means the broker could not be asked which topics are blank. Seeding
        blind would overwrite exactly the live values the read exists to protect."""
        bridge, fake = _connected_bridge([_profile()])
        bridge._published_rule_globals = None

        bridge._seed_rule_globals([_profile()])

        assert fake.calls == []

    def test_a_rule_opted_out_of_ha_gets_no_retained_state(self):
        """A retained payload on a topic no entity reads never expires. Re-enabling the
        rule later would replay a stale state into a freshly created entity."""
        profile = _profile(rules=(RuleView(uuid="rule-1", name="cat at bowl", publish_to_ha=False),))
        _bridge, fake = self._seeded([profile])
        assert fake.calls == []


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


class TestPublishRuleSnapshot:
    """publish_rule_snapshot: the live counterpart to the rule-1_snapshot camera
    component TestSourceDiscoveryShape covers. Two independent gates, both checked
    here rather than trusted to the caller - see the method's own docstring."""

    def test_publishes_retained_bytes_to_the_topic_discovery_advertised(self):
        profile = _profile(rules=(RuleView(uuid="rule-1", name="cat at bowl", publish_snapshot_to_ha=True),))
        bridge, fake = _connected_bridge([profile])
        cmps = build_source_discovery(profile, profile.sources[0], PREFIX)["cmps"]

        bridge.publish_rule_snapshot(
            profile_uuid="profile-1", source_uuid="src-1", rule_uuid="rule-1", jpeg_bytes=b"\xff\xd8fake-jpeg"
        )

        assert len(fake.calls) == 1
        call = fake.calls[0]
        assert f"{PREFIX}/{call['topic']}" == cmps["rule-1_snapshot"]["topic"]
        assert call["payload"] == b"\xff\xd8fake-jpeg"
        assert call["retain"] is True

    def test_a_rule_that_never_opted_in_gets_nothing_published(self):
        """The opt-in itself, not just the topic no entity reads - bytes must not be
        queued at all for a rule that never ticked the box, transiently or not."""
        profile = _profile()  # publish_snapshot_to_ha defaults to False
        bridge, fake = _connected_bridge([profile])

        bridge.publish_rule_snapshot(profile_uuid="profile-1", source_uuid="src-1", rule_uuid="rule-1", jpeg_bytes=b"x")

        assert fake.calls == []

    def test_an_unknown_rule_is_a_silent_no_op(self):
        """A rule the bridge has not reconciled discovery for yet (or a stale
        signal for one since removed) has no entity waiting - same reasoning
        publish_rule_transition's own `rule_base is None` check documents."""
        bridge, fake = _connected_bridge([_profile()])

        bridge.publish_rule_snapshot(
            profile_uuid="profile-1", source_uuid="src-1", rule_uuid="no-such-rule", jpeg_bytes=b"x"
        )

        assert fake.calls == []

    def test_without_a_connected_publisher_it_is_a_silent_no_op(self):
        profile = _profile(rules=(RuleView(uuid="rule-1", name="cat at bowl", publish_snapshot_to_ha=True),))
        bridge = MqttBridge(db_api_client=None)
        bridge._slugs = build_slug_index([profile])

        bridge.publish_rule_snapshot(
            profile_uuid="profile-1", source_uuid="src-1", rule_uuid="rule-1", jpeg_bytes=b"x"
        )  # must not raise


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
    # A broker holding nothing yet, which is what a first run looks like. Tests that
    # care about an existing retained value set this themselves.
    bridge._published_source_states = set()
    return bridge, fake


def _adopt_retained(bridge, relative_topics):
    """Prime the bridge as `ensure_started` does, through the same normalisation.

    Deliberately fed **absolute** topics, because that is what `read_retained` returns
    — its pattern is relative but the broker delivers full topics. A test that assigned
    `_published_source_states` directly in the relative form would encode the author's
    assumption instead of the broker's behaviour, and did: the mismatch made every topic
    read as unpublished, so the seed overwrote exactly the retained values it exists to
    protect, and the test still passed.
    """
    absolute = {f"{PREFIX}/{topic}": b"ON" for topic in relative_topics}
    bridge._published_source_states = {bridge._strip_prefix(topic) for topic in absolute}


def _payloads(fake, topic):
    return [c["payload"] for c in fake.calls if c["topic"] == topic]


class TestPerCameraPresenceIsSeeded:
    """A camera that has never seen the subject must still say so.

    Per-camera state was published only for labels whose presence *changed* on that
    camera, and a camera that has never seen a subject never changes: its topic was
    never written at all. Six of seven sensors on a live device sat at `unknown`
    indefinitely — not unavailable, which would at least be honest, but stateless — so
    an automation testing `is_state(..., 'off')` could never match, and every restart
    put them back there.
    """

    def test_a_camera_that_sees_nothing_still_publishes_off(self):
        bridge, fake = _presence_bridge()

        bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T1")

        assert _payloads(fake, "subject/luna/source/kitchen/state") == ["OFF"]

    def test_a_topic_that_already_has_a_value_is_left_alone(self):
        """The regression the first version of this caused.

        Seeding every subject on the first report wrote that frame's answer over a
        perfectly good retained one — and the first frame after a restart is the worst
        evidence in the run, with a Ring camera mid-escalation and one snapshot under
        the confidence floor indistinguishable from an empty room. A cat that really was
        in the garage went ON -> OFF -> ON, the OFF lasting until the next snapshot
        thirty seconds later.
        """
        bridge, fake = _presence_bridge()
        _adopt_retained(bridge, ["subject/luna/source/kitchen/state"])

        bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T1")

        assert _payloads(fake, "subject/luna/source/kitchen/state") == []

    def test_a_real_change_still_publishes_over_a_retained_value(self):
        """Only the *seed* defers to what is retained. An actual arrival is new
        evidence and must be published however stale the old answer was."""
        bridge, fake = _presence_bridge()
        _adopt_retained(bridge, ["subject/luna/source/kitchen/state"])

        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T1")

        assert _payloads(fake, "subject/luna/source/kitchen/state") == ["ON"]

    def test_nothing_is_seeded_when_the_retained_read_failed(self):
        """`None` means "the broker did not tell us". Seeding blind would overwrite
        arbitrary topics, and a blank entity is a smaller fault than a wrong one."""
        bridge, fake = _presence_bridge()
        bridge._published_source_states = None

        bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T1")

        assert _payloads(fake, "subject/luna/source/kitchen/state") == []

    def test_the_first_report_seeds_every_camera_it_covers(self):
        """Seeded per source as that source first reports, so a subject present on one
        camera is OFF on the other rather than blank."""
        bridge, fake = _presence_bridge()

        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T1")
        bridge.publish_presence(source_uuid="src-2", label_uuids=[], source_name="Garage", at="T2")

        assert _payloads(fake, "subject/luna/source/kitchen/state") == ["ON"]
        assert _payloads(fake, "subject/luna/source/garage/state") == ["OFF"]

    def test_seeding_happens_once_not_on_every_frame(self):
        """Presence is evaluated several times a second per camera. Re-seeding each
        time would make the retained-state topics the noisiest thing on the broker."""
        bridge, fake = _presence_bridge()

        for _ in range(5):
            bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T1")

        assert _payloads(fake, "subject/luna/source/kitchen/state") == ["OFF"]

    def test_a_label_not_published_per_source_is_not_seeded(self):
        """The gate the discovery side applies: a retained state on a topic no entity
        reads sits on the broker for good."""
        bridge, fake = _presence_bridge(per_source=False)

        bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T1")

        assert _payloads(fake, "subject/luna/source/kitchen/state") == []


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

    def test_recovering_to_an_empty_view_clears_the_per_camera_sensor(self):
        """The other half of "recovery shows the truth immediately". Leaving the stale
        ON in place is only safe if the report back actually corrects it — and it did
        not, because the withdrawal dropped the camera from `_presence_by_source`, so
        the first report diffed an empty view against nothing and found no change.

        The device-level union recomputes from scratch and was always right, which is
        exactly what hid this: `Detected now` said no while the camera's own sensor
        underneath it still said yes, indefinitely.
        """
        bridge, fake = _presence_bridge()
        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T1")
        bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T2", known=False)

        # Back, and the subject left while the feed was down.
        bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T3")

        assert _payloads(fake, "subject/luna/source/kitchen/state") == ["ON", "OFF"]

    def test_recovering_with_the_subject_still_there_republishes_nothing(self):
        """The complement, and the reason the correction above has to be a diff rather
        than an unconditional republish: a repeated retained ON is harmless in HA but a
        blanket re-seed on every stutter is what overwrote good retained values with a
        single mid-escalation frame last time."""
        bridge, fake = _presence_bridge()
        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T1")
        bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T2", known=False)

        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T3")

        assert _payloads(fake, "subject/luna/source/kitchen/state") == ["ON"]

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


class TestAvailabilityIsRestoredAfterTheStateItMakesReadable:
    """Home Assistant keeps applying state messages to an entity that is unavailable —
    it renders `unavailable` but remembers the value — and paints whatever it is holding
    the instant the availability topic flips to `online`.

    So publishing `online` before the new state makes HA show the *previous* answer
    first and correct itself a message later. The window is a millisecond, which is what
    makes it dangerous: invisible in the UI, and a complete state transition to the
    automation engine. On the live device this was six of the seven sub-second flips a
    Ring camera's presence sensor recorded in a day — `unavailable`, the stale value,
    then the true value 0-8 ms later, each one a spurious `Detected`/`Clear` pair.
    """

    def _order(self, fake, topic):
        return next(i for i, call in enumerate(fake.calls) if call["topic"] == topic)

    def test_recovery_publishes_the_new_state_before_marking_the_camera_believable(self):
        """The subject was there, the feed went stale, and it is gone by the time the
        camera speaks again. HA must never be told the camera is readable while it still
        holds the ON from before the outage."""
        bridge, fake = _presence_bridge()
        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T1")
        bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T2", known=False)
        fake.calls.clear()

        bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T3")

        assert _payloads(fake, "subject/luna/source/kitchen/state") == ["OFF"]
        assert self._order(fake, "subject/luna/source/kitchen/state") < self._order(fake, "presence/kitchen/status")

    def test_the_arrival_stamp_lands_before_the_camera_becomes_believable_too(self):
        """`Last location` is read off the attributes topic and blanked the same way —
        it went to empty on the live device on every recovery, for the same reason."""
        bridge, fake = _presence_bridge()
        bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T1", known=False)
        fake.calls.clear()

        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T2")

        assert self._order(fake, "subject/luna/attributes") < self._order(fake, "presence/kitchen/status")
        # The device-level pair needs the same order for the same reason, and this is
        # the case that exercises it: the union genuinely changes on the way back.
        assert self._order(fake, "subject/luna/state") < self._order(fake, PRESENCE_ANY_AVAILABILITY_TOPIC)

    def test_withdrawal_still_masks_the_camera_before_recomputing_the_union(self):
        """The mirror image, and it must not be "fixed" to match the restore. Going out,
        masking first is what stops HA rendering the intermediate union; the entity has
        to be unreadable *before* anything about it changes."""
        bridge, fake = _presence_bridge()
        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T1")
        fake.calls.clear()

        bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T2", known=False)

        assert self._order(fake, "presence/kitchen/status") < self._order(fake, "subject/luna/state")

    def test_a_camera_reporting_no_change_still_reports_itself_believable(self):
        """Moving the restore past the state publishes must not put it behind the
        unchanged-set early return: a camera whose first frame back sees exactly what it
        saw before would stay `unavailable` forever."""
        bridge, fake = _presence_bridge()
        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T1")
        bridge.publish_presence(source_uuid="src-1", label_uuids=[], source_name="Kitchen", at="T2", known=False)
        # Not a fresh set from the bridge's point of view — the withdrawal dropped the
        # source, so this re-reports what it already had.
        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T3")
        fake.calls.clear()

        bridge.publish_presence(source_uuid="src-1", label_uuids=["subj-a"], source_name="Kitchen", at="T4")

        assert _payloads(fake, "presence/kitchen/status") == []
        assert bridge._presence_known["src-1"] is True


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
