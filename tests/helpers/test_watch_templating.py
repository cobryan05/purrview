"""Tests for event template parameters and the topic renderer.

The property under test is that a rule matching several labels and several zones can
route on *which* ones matched. Without that every match publishes to one topic and a
Home Assistant automation cannot tell one cat from another.
"""

import pytest

from apps.helpers.actions.action import ActionContext
from apps.helpers.actions.mqttPublishAction import DEFAULT_TOPIC_TEMPLATE, MqttPublishAction
from apps.helpers.watch.events import EventPhase, WatchEventSignal
from apps.helpers.watch.templating import (
    build_template_params,
    render_text,
    render_topic,
    slugify,
)


def _signal(**overrides) -> WatchEventSignal:
    kwargs = dict(
        rule_uuid="rule-1",
        rule_name="Cat Using Litter Box",
        phase=EventPhase.START,
        at=10.0,
        started_at=8.0,
        source_uuid="src-1",
        subject_ref="label-a",
        target_ref="zone-1",
        subject_name="Tabby",
        target_name="Carpeted Litter Box",
    )
    kwargs.update(overrides)
    return WatchEventSignal(**kwargs)


def _context(**overrides) -> ActionContext:
    kwargs = dict(
        profile_uuid="p1",
        profile_name="House Cams",
        source_uuid="src-1",
        source_name="Back Room",
    )
    kwargs.update(overrides)
    return ActionContext(**kwargs)


class TestSlugify:
    def test_topic_separators_are_collapsed(self):
        """`/`, `+` and `#` in a name would silently reshape the topic tree."""
        assert slugify("Back/Room +cam #2") == "back-room-cam-2"

    def test_a_name_that_slugs_away_to_nothing_is_not_empty(self):
        assert slugify("///") == "unnamed"

    def test_case_and_padding_are_normalised(self):
        assert slugify("  Litter Box  ") == "litter-box"


class TestTemplateParams:
    def test_slot_values_are_exposed_by_name(self):
        params = build_template_params(_signal(), _context())
        assert params["subject"] == "Tabby"
        assert params["target"] == "Carpeted Litter Box"
        assert params["source"] == "Back Room"
        assert params["watcher"] == "House Cams"

    def test_uuids_are_available_alongside_the_names(self):
        """A consumer may want a stable id while a template prints something readable."""
        params = build_template_params(_signal(), _context())
        assert params["subject_uuid"] == "label-a"
        assert params["target_uuid"] == "zone-1"

    def test_the_event_label_overrides_the_rule_name_for_routing(self):
        signal = _signal(metadata={"eventLabel": "litterbox"})
        params = build_template_params(signal, _context())
        assert params["rule"] == "litterbox"
        assert params["rule_name"] == "Cat Using Litter Box"

    def test_a_slot_falls_back_to_its_uuid_when_unresolved(self):
        """A raw uuid is obviously wrong; a placeholder like 'unknown' looks real."""
        params = build_template_params(_signal(subject_name=None), _context())
        assert params["subject"] == "label-a"

    def test_every_value_is_a_string(self):
        params = build_template_params(_signal(), _context())
        assert all(isinstance(v, str) for v in params.values())


class TestRenderTopic:
    def test_placeholders_are_substituted_and_slugified(self):
        params = build_template_params(_signal(), _context())
        assert render_topic("cats/{rule}/{source}/{subject}", params) == ("cats/cat-using-litter-box/back-room/tabby")

    def test_the_default_template_reproduces_the_historical_scheme(self):
        params = build_template_params(_signal(), _context())
        assert render_topic(DEFAULT_TOPIC_TEMPLATE, params) == "house-cams/back-room/event/cat-using-litter-box"

    def test_an_unknown_placeholder_is_left_verbatim(self):
        """A typo must be visible on the broker, not silently collapse a level."""
        params = build_template_params(_signal(), _context())
        assert render_topic("x/{sorce}/y", params) == "x/{sorce}/y"

    def test_an_empty_known_slot_collapses_its_level(self):
        """`a//b` is legal MQTT but matches nothing anyone would subscribe to."""
        params = build_template_params(_signal(), _context())
        assert render_topic("a/{event_label}/b", params) == "a/b"

    def test_a_template_with_no_placeholders_is_passed_through(self):
        assert render_topic("static/topic", {}) == "static/topic"

    def test_an_empty_template_renders_empty(self):
        assert render_topic("", {}) == ""

    def test_format_string_attribute_access_is_not_honoured(self):
        """Templates are user input; `str.format` would expose the object graph."""
        params = build_template_params(_signal(), _context())
        rendered = render_topic("{subject.__class__}", params)
        assert "class" in rendered and "type" not in rendered


class TestRenderText:
    def test_values_are_substituted_raw(self):
        params = build_template_params(_signal(), _context())
        assert render_text("{subject} entered {target}", params) == "Tabby entered Carpeted Litter Box"


class TestMqttTopicIntegration:
    def test_the_action_defaults_to_the_historical_topic(self):
        action = MqttPublishAction()
        assert action.topic_for(_signal(), _context()) == "house-cams/back-room/event/cat-using-litter-box"

    def test_a_configured_template_is_used(self):
        action = MqttPublishAction(topic_template="cats/{subject}/{target}")
        assert action.topic_for(_signal(), _context()) == "cats/tabby/carpeted-litter-box"

    def test_a_blank_template_falls_back_to_the_default(self):
        """Blank means unset, not 'publish to the empty topic'."""
        action = MqttPublishAction(topic_template="   ")
        assert action.topic_for(_signal(), _context()) == "house-cams/back-room/event/cat-using-litter-box"

    def test_the_payload_carries_the_same_slot_values(self):
        action = MqttPublishAction()
        payload = action.payload_for(_signal(), _context())
        assert payload["templateParams"]["subject"] == "Tabby"
        assert payload["subjectRef"] == "label-a"
        assert payload["targetName"] == "Carpeted Litter Box"

    def test_two_cats_route_to_two_topics(self):
        """The whole point: one rule, one topic per matched subject."""
        action = MqttPublishAction(topic_template="cats/{subject}")
        first = action.topic_for(_signal(subject_name="Tabby"), _context())
        second = action.topic_for(_signal(subject_name="Ginger"), _context())
        assert first == "cats/tabby"
        assert second == "cats/ginger"


class TestSourceMqttNameOverride:
    """A watcher's per-source MQTT name.

    The Sources pane's name is chosen to be recognisable when picking a camera out of a
    dropdown; a topic level is something a Home Assistant automation is written against
    and has to stay short and stable. The override lets them differ.
    """

    def test_the_override_wins_for_the_source_token(self):
        params = build_template_params(_signal(), _context(source_mqtt_name="litterbox"))
        assert params["source"] == "litterbox"

    def test_without_an_override_the_sources_own_name_is_used(self):
        """Every watcher predating the override must publish exactly where it did."""
        params = build_template_params(_signal(), _context())
        assert params["source"] == "Back Room"

    def test_the_real_name_stays_reachable(self):
        """Overriding the topic must not be lossy."""
        params = build_template_params(_signal(), _context(source_mqtt_name="litterbox"))
        assert params["source_config_name"] == "Back Room"

    def test_the_override_is_slugified_like_any_other_level(self):
        topic = render_topic(
            "{watcher}/{source}", build_template_params(_signal(), _context(source_mqtt_name="Litter Box"))
        )
        assert topic == "house-cams/litter-box"

    def test_the_uuid_is_unaffected(self):
        params = build_template_params(_signal(), _context(source_mqtt_name="litterbox"))
        assert params["source_uuid"] == "src-1"

    async def test_an_image_topic_uses_the_override(self):
        """The live feed is the main reason to want a short, stable source level."""
        from apps.helpers.actions.mqttPublishImageAction import MqttPublishImageAction

        action = MqttPublishImageAction()
        topic = action.topic_for(_signal(), _context(source_mqtt_name="litterbox"))
        assert topic == "house-cams/litterbox/image"
