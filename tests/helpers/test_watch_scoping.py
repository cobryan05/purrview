"""Tests for per-rule and per-source action scoping.

The behaviour these pin down is what lets one watcher serve several cameras: rules are
shared, side effects are not. The empty-means-all default is load-bearing — every
watcher written before scoping existed has empty lists on every action, and must keep
behaving exactly as it did.
"""

from dataclasses import dataclass, field
from typing import List

from apps.helpers.watch.scoping import FRAME_ANALYSED, action_applies, in_scope


@dataclass
class _Row:
    """Stands in for a `WatchAction` row as the generated db client returns it."""

    kind: str = "record_clip"
    trigger: str = "event_ended"
    enabled: bool = True
    rule_uuids_json: List[str] = field(default_factory=list)
    source_uuids_json: List[str] = field(default_factory=list)


class TestInScope:
    def test_empty_means_everything(self):
        assert in_scope([], "anything") is True
        assert in_scope(None, "anything") is True

    def test_a_named_scope_matches_only_its_members(self):
        assert in_scope(["a", "b"], "b") is True
        assert in_scope(["a", "b"], "c") is False

    def test_an_unknown_value_against_a_named_scope_does_not_match(self):
        assert in_scope(["a"], None) is False


class TestActionApplies:
    def test_an_unscoped_action_fires_everywhere(self):
        """The default that keeps the common case free of configuration."""
        row = _Row()
        assert action_applies(row, "event_ended", rule_uuid="r1", source_uuid="s1") is True
        assert action_applies(row, "event_ended", rule_uuid="r9", source_uuid="s9") is True

    def test_a_source_scoped_action_fires_only_on_that_source(self):
        """The whole point: record on the litter-box camera, not on the other four."""
        row = _Row(source_uuids_json=["s1"])
        assert action_applies(row, "event_ended", rule_uuid="r1", source_uuid="s1") is True
        assert action_applies(row, "event_ended", rule_uuid="r1", source_uuid="s2") is False

    def test_a_rule_scoped_action_fires_only_for_that_rule(self):
        row = _Row(rule_uuids_json=["r1"])
        assert action_applies(row, "event_ended", rule_uuid="r1", source_uuid="s1") is True
        assert action_applies(row, "event_ended", rule_uuid="r2", source_uuid="s1") is False

    def test_both_axes_must_match(self):
        row = _Row(rule_uuids_json=["r1"], source_uuids_json=["s1"])
        assert action_applies(row, "event_ended", rule_uuid="r1", source_uuid="s1") is True
        assert action_applies(row, "event_ended", rule_uuid="r1", source_uuid="s2") is False
        assert action_applies(row, "event_ended", rule_uuid="r2", source_uuid="s1") is False

    def test_several_members_are_one_action_not_several_rows(self):
        row = _Row(source_uuids_json=["s1", "s3"])
        assert action_applies(row, "event_ended", rule_uuid="r1", source_uuid="s3") is True
        assert action_applies(row, "event_ended", rule_uuid="r1", source_uuid="s2") is False

    def test_the_trigger_must_match(self):
        row = _Row(trigger="event_started")
        assert action_applies(row, "event_ended", rule_uuid="r1", source_uuid="s1") is False

    def test_a_disabled_action_never_fires(self):
        row = _Row(enabled=False)
        assert action_applies(row, "event_ended", rule_uuid="r1", source_uuid="s1") is False

    def test_uuids_are_compared_as_strings(self):
        """JSON round-trips can hand back values that are not `str`."""
        row = _Row(source_uuids_json=[123])
        assert action_applies(row, "event_ended", rule_uuid=None, source_uuid="123") is True


class TestRulelessTriggers:
    def test_a_frame_action_ignores_its_rule_scope(self):
        """A frame belongs to no rule.

        Treating a leftover rule scope as unmatched would make the action silently
        never fire — the failure mode this carve-out exists to prevent.
        """
        row = _Row(trigger=FRAME_ANALYSED, rule_uuids_json=["r1"])
        assert action_applies(row, FRAME_ANALYSED, rule_uuid=None, source_uuid="s1") is True

    def test_a_frame_action_still_honours_its_source_scope(self):
        """Source scoping is exactly how a live feed is limited to one camera."""
        row = _Row(trigger=FRAME_ANALYSED, source_uuids_json=["s1"])
        assert action_applies(row, FRAME_ANALYSED, rule_uuid=None, source_uuid="s1") is True
        assert action_applies(row, FRAME_ANALYSED, rule_uuid=None, source_uuid="s2") is False

    def test_an_event_action_does_not_fire_on_a_frame(self):
        row = _Row(trigger="event_ended")
        assert action_applies(row, FRAME_ANALYSED, rule_uuid=None, source_uuid="s1") is False

    def test_a_frame_action_does_not_fire_on_an_event(self):
        row = _Row(trigger=FRAME_ANALYSED)
        assert action_applies(row, "event_ended", rule_uuid="r1", source_uuid="s1") is False


class TestMultiTriggerActions:
    """`fires_on` — how an action that spans an event's whole life is dispatched.

    `record_event_timelapse` sits on one trigger in the editor and needs three at
    runtime: open a buffer, sample frames, compile. Without this the row's single
    trigger was the whole answer, and the trigger it actually collects on never
    reached it.
    """

    def test_an_action_fires_on_every_trigger_it_declares(self):
        row = _Row(trigger="event_ended")
        fires_on = {"event_started", FRAME_ANALYSED, "event_ended"}
        for trigger in fires_on:
            assert action_applies(row, trigger, rule_uuid="r1", source_uuid="s1", fires_on=fires_on) is True

    def test_a_declared_trigger_outside_the_set_still_does_not_fire(self):
        row = _Row(trigger="event_ended")
        assert action_applies(row, "event_started", fires_on={"event_ended"}) is False

    def test_omitting_it_keeps_the_old_single_trigger_behaviour(self):
        """Every other action relies on this default, so it has to stay exact."""
        row = _Row(trigger="event_ended")
        assert action_applies(row, "event_ended", rule_uuid="r1", source_uuid="s1") is True
        assert action_applies(row, "event_started", rule_uuid="r1", source_uuid="s1") is False

    def test_scoping_still_applies_to_a_multi_trigger_action(self):
        """Declaring extra triggers must not smuggle an action past its source scope —
        a timelapse scoped to the litter cam must not sample the hallway."""
        row = _Row(trigger="event_ended", source_uuids_json=["s1"])
        fires_on = {"event_started", FRAME_ANALYSED, "event_ended"}
        assert action_applies(row, FRAME_ANALYSED, source_uuid="s1", fires_on=fires_on) is True
        assert action_applies(row, FRAME_ANALYSED, source_uuid="s2", fires_on=fires_on) is False

    def test_a_disabled_multi_trigger_action_fires_on_nothing(self):
        row = _Row(trigger="event_ended", enabled=False)
        fires_on = {"event_started", FRAME_ANALYSED, "event_ended"}
        assert action_applies(row, FRAME_ANALYSED, source_uuid="s1", fires_on=fires_on) is False
