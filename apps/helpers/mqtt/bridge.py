"""The app-level MQTT connection: Home Assistant discovery, entities and statistics.

Independent of the per-watcher `AsyncMqttPublisher` instances in `StreamWatcherTask` —
those exist only while a profile has an enabled MQTT action, and their `<prefix>/state`
LWT already means something else (the last watcher to stop declares the whole app
offline, which is a `StreamWatcherTask` quirk, not a bridge concern). This is one
connection, independent of any watcher, alive for as long as the app is.

Split deliberately into pure functions (discovery-payload shape, given a snapshot of
what profiles/rules/sources exist) and the `MqttBridge` class (the DB calls and the
actual publish). The pure half is what regresses silently — a topic template typo or a
uuid/name mixup won't throw, it'll just quietly orphan history — so it's the half worth
testing without a broker. See `HA_INTEGRATION_PLAN.md` WP-2 for the design this
implements, including the two gotchas fixed there: rule entities key `unique_id` off the
rule's uuid (never its name, which is what the topic *path* uses), and a source that
disappears from every profile gets its discovery retracted rather than left retained
forever.
"""

import asyncio
import dataclasses
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from ..actions.mqttPublisher import AsyncMqttPublisher, load_mqtt_broker_settings
from ..watch.rules import descendants_of
from ..watch.templating import slugify

logger = logging.getLogger(__name__)

HA_DISCOVERY_PREFIX = "homeassistant"
ORIGIN = {"name": "PurrView"}

#: Stamped on every device we publish.
#:
#: HA's MQTT panel splits devices into subentries and "devices that don't belong to a
#: sub-entry", and everything discovery-created lands in the latter — an MQTT subentry
#: holds exactly one device that a human added through the UI, so it is not a folder and
#: discovery cannot target one.
#:
#: Manufacturer and model are a grouping HA offers cheaply: they make the set filterable
#: and sortable as one family in the device list instead of N unrelated rows. Neither
#: touches `ids`, so no entity is re-registered by adding them.
#:
#: The stronger grouping is `via_device`, which `_device` sets — see there. An earlier
#: version of this comment claimed the discovery device object had no such key; that was
#: wrong, and the flat device list it resigned itself to was the consequence.
MANUFACTURER = "PurrView"

#: Namespace on every device id this app publishes, and — via `owns_config_topic` —
#: the test for whether a retained discovery config is ours to reconcile at all.
#:
#: Keep the id builders below derived from this rather than spelling the prefix out
#: again: the read-back trusts the two to agree, and a device published under an id
#: this constant does not match would be adopted by nobody and retracted by nobody.
DEVICE_ID_PREFIX = "purrview_"

APP_DEVICE_ID = f"{DEVICE_ID_PREFIX}system"
APP_CONFIG_TOPIC = f"{HA_DISCOVERY_PREFIX}/device/{APP_DEVICE_ID}/config"

#: One device holding every rule's cross-camera, cross-label entities.
#:
#: A device per rule would be the obvious shape and is the wrong one: rules are the
#: axis most likely to grow, and HA charges per *device* in the UI (a row in the device
#: list, a page, an area assignment) in a way it does not charge per entity. One device
#: with N entities stays one row however many rules exist.
RULES_DEVICE_ID = f"{DEVICE_ID_PREFIX}rules"
RULES_CONFIG_TOPIC = f"{HA_DISCOVERY_PREFIX}/device/{RULES_DEVICE_ID}/config"


def owns_config_topic(topic: str) -> bool:
    """Whether a `homeassistant/device/<id>/config` topic is one of ours.

    The retained read-back has to subscribe to `homeassistant/device/+/config` —
    every device-discovery config on the broker, from every integration — because
    MQTT's `+` matches a whole topic level and cannot express "ids starting with
    `purrview_`". `homeassistant/device/purrview_+/config` is not a narrower filter,
    it is an invalid one that matches nothing. So the narrowing happens here, on the
    way in, and the subscription stays wide on purpose.

    Skipping this filter is not a cosmetic bug: an adopted topic lands in `previous`,
    and `reconcile_discovery` retracts everything in `previous` it does not desire —
    so the app would publish a correct, effective deletion of another integration's
    devices, with no error anywhere and no symptom but their entities vanishing.
    """
    return topic.startswith(f"{HA_DISCOVERY_PREFIX}/device/{DEVICE_ID_PREFIX}") and topic.endswith("/config")


def _device(ids: str, name: str, model: str, via: bool = True) -> Dict[str, Any]:
    """A discovery device object, hung off the app device unless it *is* the app device.

    `via_device` is what gives Home Assistant the grouping the manufacturer/model pair
    can only approximate: the app device's page lists every camera, subject and the
    rules device as connected devices, and each of those shows "Connected via PurrView".
    That is the at-a-glance "these are one integration" view; MQTT-discovered devices
    otherwise land in one flat list with everything else on the broker.

    This contradicts an earlier comment here claiming the discovery device object has no
    `via_device`. It does — verified against HA 2026.7.4 by publishing a device
    declaring one and reading `via_device_id` back off the device registry. The key is
    not abbreviated, unlike its neighbours.

    It does not touch `ids`, so adding it re-registers nothing.
    """
    device: Dict[str, Any] = {"ids": ids, "name": name, "mf": MANUFACTURER, "mdl": model}
    if via and ids != APP_DEVICE_ID:
        device["via_device"] = APP_DEVICE_ID
    return device


#: Component keys that arrive switched off in Home Assistant. HA still creates them, so
#: they cost nothing until enabled and are one click away in the device page's "+N
#: entities not shown" — but a disabled entity holds no state machine slot, no history
#: and no statistics.
#:
#: The count is the reason: sources x rules x 6, plus a per-source diagnostic block, is
#: ~140 entities on a modest four-camera setup before subjects are counted at all. What
#: stays on is what a dashboard asks for — did it happen, how often, and the picture.
#: The detail is for when something looks wrong, and HA is a better place to choose it
#: than a settings pane here would be.
#:
#: Health is the deliberate exception among the diagnostics: it is the one that tells
#: you a camera has stopped working, and a failure alarm that ships switched off is
#: worse than no alarm at all.
DISABLED_BY_DEFAULT_RULE_SUFFIXES = ("_last_seen", "_last_duration", "_peak_confidence", "_event")
DISABLED_BY_DEFAULT_SOURCE_KEYS = frozenset(
    {
        "tracked_count",
        "fps",
        "frames_analysed",
        "mqtt_published",
        "mqtt_dropped",
        "mqtt_coalesced",
        "mqtt_queued_kb",
    }
)


def _apply_default_visibility(cmps: Dict[str, Any]) -> Dict[str, Any]:
    """Mark the long tail of a camera device's components disabled-by-default.

    Applied as a pass over the finished component set rather than a flag on each
    definition above, so the policy is legible in one place — scattered across twenty
    dict literals it becomes impossible to answer "what does a fresh install show?"
    without reading all of them.
    """
    for key, component in cmps.items():
        disabled = key in DISABLED_BY_DEFAULT_SOURCE_KEYS or key.endswith(DISABLED_BY_DEFAULT_RULE_SUFFIXES)
        if disabled:
            component["enabled_by_default"] = False
    return cmps


def source_device_id(source_uuid: str) -> str:
    return f"{DEVICE_ID_PREFIX}{source_uuid}"


def source_config_topic(source_uuid: str) -> str:
    return f"{HA_DISCOVERY_PREFIX}/device/{source_device_id(source_uuid)}/config"


# ── Profile snapshot — the "profile fixture" the discovery functions are tested against ──


@dataclass(frozen=True)
class RuleView:
    uuid: str
    name: str
    enabled: bool = True
    #: Whether this rule is represented in Home Assistant at all. Distinct from
    #: `enabled`, which stops the rule being evaluated: a rule can run, file events and
    #: keep history while publishing nothing. False removes it from every device it
    #: appears on and retracts what it had.
    publish_to_ha: bool = True
    #: Empty means "every source in the profile" — matches `InteractionRule`.
    source_uuids: Tuple[str, ...] = ()
    #: Label uuids this rule can match as its subject, already expanded through the
    #: label tree. Expansion happens above this view because a rule commonly names a
    #: *parent* label with `include_descendants`, and it is the descendant that ends up
    #: in an event's `subject_ref` — enumerating only the literal refs would miss every
    #: individual it can actually match.
    subject_uuids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceView:
    uuid: str
    name: str
    mqtt_name: Optional[str] = None
    enabled: bool = True
    #: As on `RuleView`: `enabled` stops the camera being watched, this stops only its
    #: Home Assistant device.
    publish_to_ha: bool = True


@dataclass(frozen=True)
class SubjectView:
    """A label a rule can match as its subject.

    "Subject" is not a separate concept from a label — `subject_a_kind` is `LABEL` and
    `subject_a_refs_json` holds label uuids — so this is a label the config says can
    appear as the subject of an event. Which generalises: nothing here is cat-specific,
    and a person or vehicle label works identically.
    """

    uuid: str
    name: str
    #: Whether this label gets a device at all.
    publish_to_ha: bool = True
    #: Whether it also gets the per-(rule, source) counter tier. Off by default: this is
    #: the only tier whose size is subjects x rules x sources, so it is the one that
    #: turns a growing label tree into a four-figure entity count. The rule-level event
    #: entity carries the same breakdown as attributes, which is what an automation
    #: needs; this tier exists for long-term per-camera statistics, which is a
    #: deliberate, occasional want.
    publish_per_source: bool = False


@dataclass(frozen=True)
class ProfileView:
    uuid: str
    name: str
    sources: Tuple[SourceView, ...] = field(default_factory=tuple)
    rules: Tuple[RuleView, ...] = field(default_factory=tuple)


def _topic_source(source: SourceView) -> str:
    # Same precedence as `templating.build_template_params`'s `source` slot: an
    # explicit MQTT name override wins, so a topic path stays stable independent of
    # the Sources pane's display name.
    return slugify(source.mqtt_name or source.name)


def _topic_rule(rule: RuleView) -> str:
    return slugify(rule.name)


def entity_name(device_name: str, name: str) -> str:
    """An entity's name with its device's name stripped off the front.

    HA composes both the friendly name and the entity_id as `<device> <entity>`, so a
    name that repeats the device stutters: rule "Maple Present" on subject device
    "Maple" becomes `sensor.maple_maple_present`.

    This is not only our own doing. Naming a rule after the subject it watches is the
    natural thing to do, so the collision arrives with the *config* and would come back
    with the next rule however carefully the literals here are written — which is why
    it is stripped at render time rather than fixed one entity at a time.

    Matched on a whole-word boundary so "Cat" does not maul "Catcher", and case
    -insensitively because the device name comes from user config. If nothing would be
    left, the original is kept: an empty name means "this entity *is* the device" to HA,
    which is right for a camera and wrong for a rule that happens to share its name.
    """
    device_words = slugify(device_name).split("-")
    name_words = slugify(name).split("-")
    if len(name_words) <= len(device_words) or name_words[: len(device_words)] != device_words:
        return name
    # Take the remainder from the original string rather than the slug, so casing and
    # punctuation survive: "Cat Drinking Water" → "Drinking Water", not "drinking-water".
    remainder = name.split(None, len(device_words))[len(device_words) :]
    return remainder[0] if remainder else name


# ── Topic construction ───────────────────────────────────────────────────────────────
#
# Discovery advertises these and the live publishes write to them, so they are built
# here once rather than formatted at each call site: the failure mode of two f-strings
# drifting apart is an entity that exists in HA and never updates, which throws nothing
# and looks like a dead camera.
#
# Everything is profile-scoped, matching the existing documented event/image scheme in
# `mqttPublishAction.py`. Source-scoped alone is wrong and was a real bug: multiple
# watch profiles may watch one camera, and two same-named rules under one source would
# then share a state topic and overwrite each other's counts. Telemetry is worse — fps
# and queue depth are properties of a *watcher's analysis of* a source, never of the
# source itself.


def source_base_topic(profile_slug: str, source_slug: str) -> str:
    return f"{profile_slug}/{source_slug}"


def rule_base_topic(profile_slug: str, source_slug: str, rule_slug: str) -> str:
    return f"{source_base_topic(profile_slug, source_slug)}/rule/{rule_slug}"


def telemetry_topic(profile_slug: str, source_slug: str) -> str:
    return f"{source_base_topic(profile_slug, source_slug)}/telemetry"


def rule_global_base_topic(profile_slug: str, rule_slug: str) -> str:
    """A rule's topics across every source and subject at once.

    Deliberately `<profile>/rules/<rule>` — the same hierarchy as
    `rule_base_topic`'s `<profile>/<source>/rule/<rule>` with the source segment
    dropped, which is exactly what "this rule, on any camera" means.

    Profile-scoped, unlike subject topics: a label is app-wide (two watchers seeing the
    same cat want one answer), but a rule is owned by its profile, and two profiles that
    happen to contain a rule of the same name hold two different rules.

    Plural `rules/` rather than `rule/` so it cannot collide with a *source* whose slug
    is literally "rule", which `<profile>/<source>/rule/<rule>` would otherwise make
    ambiguous.
    """
    return f"{profile_slug}/rules/{rule_slug}"


# ── Discovery payload generation (pure) ──────────────────────────────────────────────


def build_source_discovery(profile: ProfileView, source: SourceView, prefix: str) -> Dict[str, Any]:
    """The one device-discovery message for a source's HA device: itself, plus every
    rule currently scoped to it. HA's device-discovery schema bundles a whole device's
    components into one retained message (`cmps`), so adding/removing/renaming a rule
    is a matter of recomputing this and republishing — see `reconcile_discovery`.
    """
    topic_source = _topic_source(source)
    topic_profile = slugify(profile.name)
    base = f"{prefix}/{source_base_topic(topic_profile, topic_source)}"
    telemetry_state_topic = f"{prefix}/{telemetry_topic(topic_profile, topic_source)}"
    watcher_topic = f"{prefix}/watcher/{topic_profile}/status"

    cmps: Dict[str, Any] = {}
    for rule in profile.rules:
        if not rule.enabled or not rule.publish_to_ha:
            continue
        if rule.source_uuids and source.uuid not in rule.source_uuids:
            continue
        topic_rule = _topic_rule(rule)
        rule_label = entity_name(source.name, rule.name)
        rule_base = f"{prefix}/{rule_base_topic(topic_profile, topic_source, topic_rule)}"
        # The existing, documented scheme (mqttPublishAction.py) — profile-scoped,
        # unchanged by this bridge. Built off the same `base` as the bridge's own
        # topics now that those are profile-scoped too, which is the point: the two
        # schemes share a namespace rather than sitting in parallel ones.
        event_topic = f"{base}/event/{topic_rule}"

        # unique_id is uuid-based throughout: renaming the rule changes rule_base and
        # event_topic (cosmetic, human-readable) but never these, so HA keeps treating
        # it as the same entity — see HA_INTEGRATION_PLAN.md WP-2 step 5a.
        #
        # Scoped by *source as well as rule*. These entities hang off the source device,
        # but one rule applies to every source in the profile, so keying on the rule
        # alone makes all of them collide: HA registers the first and logs "Platform
        # mqtt does not generate unique IDs ... ignoring" for the rest, which silently
        # drops every camera after the first.
        cmps[f"{rule.uuid}_state"] = {
            "p": "binary_sensor",
            "unique_id": f"purrview_{source.uuid}_{rule.uuid}_state",
            "name": rule_label,
            "device_class": "occupancy",
            "state_topic": f"{rule_base}/state",
            "payload_on": "ON",
            "payload_off": "OFF",
            "json_attributes_topic": f"{rule_base}/attributes",
        }
        cmps[f"{rule.uuid}_count"] = {
            "p": "sensor",
            "unique_id": f"purrview_{source.uuid}_{rule.uuid}_count",
            "name": f"{rule_label} count",
            # The single most important setting here: HA purges raw history after
            # ~10 days but keeps long-term statistics forever for a sensor that
            # declares a state_class. Without it, "events per day last year" is gone.
            "state_class": "total_increasing",
            "state_topic": f"{rule_base}/count",
        }
        cmps[f"{rule.uuid}_last_seen"] = {
            "p": "sensor",
            "unique_id": f"purrview_{source.uuid}_{rule.uuid}_last_seen",
            "name": f"{rule_label} last seen",
            "device_class": "timestamp",
            "state_topic": f"{rule_base}/attributes",
            "value_template": "{{ value_json.lastSeen }}",
        }
        cmps[f"{rule.uuid}_last_duration"] = {
            "p": "sensor",
            "unique_id": f"purrview_{source.uuid}_{rule.uuid}_last_duration",
            "name": f"{rule_label} last duration",
            "state_class": "measurement",
            "device_class": "duration",
            "unit_of_measurement": "s",
            "state_topic": f"{rule_base}/attributes",
            "value_template": "{{ value_json.lastDurationS }}",
        }
        cmps[f"{rule.uuid}_peak_confidence"] = {
            "p": "sensor",
            "unique_id": f"purrview_{source.uuid}_{rule.uuid}_peak_confidence",
            "name": f"{rule_label} peak confidence",
            "state_class": "measurement",
            "state_topic": f"{rule_base}/attributes",
            "value_template": "{{ value_json.peakConfidence }}",
        }
        cmps[f"{rule.uuid}_event"] = {
            "p": "event",
            "unique_id": f"purrview_{source.uuid}_{rule.uuid}_event",
            "name": rule_label,
            "event_types": [rule.name],
            "state_topic": event_topic,
        }

    cmps["camera"] = {
        "p": "camera",
        "unique_id": f"purrview_{source.uuid}_camera",
        # No name: HA treats an unnamed component as the device's primary entity, so this
        # is `camera.<source>` rather than `camera.<source>_<source>`.
        "name": None,
        "topic": f"{base}/image",
    }
    cmps["tracked_count"] = {
        "p": "sensor",
        "unique_id": f"purrview_{source.uuid}_tracked_count",
        "name": "tracked objects",
        "state_class": "measurement",
        "state_topic": telemetry_state_topic,
        "value_template": "{{ value_json.trackedCount }}",
    }
    cmps["health"] = {
        "p": "binary_sensor",
        "unique_id": f"purrview_{source.uuid}_health",
        "name": "problem",
        "device_class": "problem",
        "entity_category": "diagnostic",
        "state_topic": telemetry_state_topic,
        "value_template": "{{ 'ON' if value_json.problem else 'OFF' }}",
    }
    for key, json_path, unit in (
        ("fps", "analysisFps", None),
        ("frames_analysed", "framesAnalysed", None),
        ("mqtt_published", "mqttPublished", None),
        ("mqtt_dropped", "mqttDropped", None),
        ("mqtt_coalesced", "mqttCoalesced", None),
        ("mqtt_queued_kb", "mqttQueuedKb", "kB"),
    ):
        entry: Dict[str, Any] = {
            "p": "sensor",
            "unique_id": f"purrview_{source.uuid}_{key}",
            "name": key.replace("_", " "),
            "entity_category": "diagnostic",
            "state_class": "measurement",
            "state_topic": telemetry_state_topic,
            "value_template": f"{{{{ value_json.{json_path} }}}}",
        }
        if unit:
            entry["unit_of_measurement"] = unit
        cmps[key] = entry

    return {
        # Namespaced like every other device here (`purrview_subject_…`,
        # `purrview_system`). A bare uuid would sit in HA's global MQTT device
        # namespace shared with every other integration on the broker.
        "dev": _device(source_device_id(source.uuid), source.name, "Watched camera"),
        "o": dict(ORIGIN),
        # Both must be online: the app-wide LWT and this source's watcher. Either
        # going offline should be visible as "unavailable", not a frozen last value.
        "availability": [{"topic": f"{prefix}/status"}, {"topic": watcher_topic}],
        "availability_mode": "all",
        "cmps": _apply_default_visibility(cmps),
    }


def subject_device_id(subject_uuid: str) -> str:
    return f"{DEVICE_ID_PREFIX}subject_{subject_uuid}"


def subject_config_topic(subject_uuid: str) -> str:
    return f"{HA_DISCOVERY_PREFIX}/device/{subject_device_id(subject_uuid)}/config"


def subject_base_topic(subject_slug: str) -> str:
    """Subject topics carry no profile segment, unlike rule and telemetry topics.

    A label is app-wide: two watchers seeing the same cat is precisely the case where
    one shared answer is wanted, not two. Safe because every watcher runs in one
    `tasks_server` process against a single `MqttBridge`, so these topics have exactly
    one writer however many profiles are running.
    """
    return f"subject/{subject_slug}"


def build_subject_discovery(
    subject: SubjectView,
    rules: Sequence[Tuple[RuleView, Sequence[SourceView]]],
    prefix: str,
) -> Dict[str, Any]:
    """One subject label's HA device: a fixed core, plus a breakdown per rule and per
    (rule, source).

    The core exists because the breakdown cannot be relied on. Rules are user config —
    they get renamed, disabled and deleted — so anything derived from them is the wrong
    anchor for a device. These four are properties of the label and the event stream,
    so they are there whether the profile has five rules or none, and they are what a
    dashboard card actually wants: is it here, how often, when last, and where.

    `Detected now` is the odd one out and deliberately so: it comes from the tracker,
    not from events, so it answers "is this label in front of a camera" rather than
    "did a rule fire". Those genuinely differ — a cat wandering through a room no rule
    covers is present but generates no events — and the rule-independent answer is the
    one that stays useful when rules change.
    """
    subject_slug = slugify(subject.name)
    base = f"{prefix}/{subject_base_topic(subject_slug)}"

    cmps: Dict[str, Any] = {
        "detected": {
            "p": "binary_sensor",
            "unique_id": f"purrview_subject_{subject.uuid}_detected",
            "name": "Detected now",
            "device_class": "occupancy",
            "state_topic": f"{base}/state",
            "payload_on": "ON",
            "payload_off": "OFF",
        },
        "count": {
            "p": "sensor",
            "unique_id": f"purrview_subject_{subject.uuid}_count",
            "name": "Total events",
            "state_class": "total_increasing",
            "state_topic": f"{base}/count",
        },
        "last_seen": {
            "p": "sensor",
            "unique_id": f"purrview_subject_{subject.uuid}_last_seen",
            "name": "Last seen",
            "device_class": "timestamp",
            "state_topic": f"{base}/attributes",
            "value_template": "{{ value_json.lastSeen }}",
        },
        "last_location": {
            "p": "sensor",
            "unique_id": f"purrview_subject_{subject.uuid}_last_location",
            "name": "Last location",
            "state_topic": f"{base}/attributes",
            "value_template": "{{ value_json.lastSource }}",
        },
    }

    for rule, sources in rules:
        rule_slug = slugify(rule.name)
        rule_label = entity_name(subject.name, rule.name)
        # Tier 2 — "when is this subject doing X on *any* camera", the question most
        # worth asking and the one a per-camera breakdown alone cannot answer.
        cmps[f"{rule.uuid}_count"] = {
            "p": "sensor",
            "unique_id": f"purrview_subject_{subject.uuid}_{rule.uuid}_count",
            "name": rule_label,
            "state_class": "total_increasing",
            "state_topic": f"{base}/rule/{rule_slug}/count",
        }
        cmps[f"{rule.uuid}_last_seen"] = {
            "p": "sensor",
            "unique_id": f"purrview_subject_{subject.uuid}_{rule.uuid}_last_seen",
            "name": f"{rule_label} last seen",
            "device_class": "timestamp",
            "state_topic": f"{base}/rule/{rule_slug}/attributes",
            "value_template": "{{ value_json.lastSeen }}",
        }
        # Tier 3 — the drill-down, and the only tier gated behind a per-label opt-in.
        #
        # It is the term that multiplies all three axes (subjects x rules x sources)
        # while every other tier merely adds, so it is what turns a large label tree
        # into a four-figure entity count. Publishing it disabled-by-default was not
        # enough: HA still creates a registry entry per entity, so the cost was real and
        # invisible. Now a label that wants the per-camera breakdown asks for it.
        if not subject.publish_per_source:
            continue
        for source in sources:
            source_slug = _topic_source(source)
            cmps[f"{rule.uuid}_{source.uuid}_count"] = {
                "p": "sensor",
                "unique_id": f"purrview_subject_{subject.uuid}_{rule.uuid}_{source.uuid}_count",
                "name": f"{rule_label} · {source.name}",
                "state_class": "total_increasing",
                "enabled_by_default": False,
                "state_topic": f"{base}/rule/{rule_slug}/{source_slug}/count",
            }

    return {
        "dev": _device(subject_device_id(subject.uuid), subject.name, "Tracked subject"),
        "o": dict(ORIGIN),
        # Only the app-level LWT: a subject is not owned by any one watcher, so tying
        # its availability to a particular one would blank the device whenever that
        # watcher stopped, even while another was still watching the same label.
        "availability": [{"topic": f"{prefix}/status"}],
        "cmps": cmps,
    }


def build_rules_discovery(profiles: Sequence[ProfileView], prefix: str) -> Optional[Dict[str, Any]]:
    """The `purrview_rules` device: every rule, once, independent of camera and label.

    This is the device automations are written against. The camera and subject devices
    answer "did this rule fire *here*" and "did this rule fire for *her*"; neither
    answers "did this rule fire", which is the question an automation almost always
    actually asks — the litter box wants emptying whoever used it and whichever camera
    saw it. Composing that from the per-camera entities means an automation that silently
    stops covering a camera the moment one is added, which is the kind of bug nobody
    notices until the automation was needed.

    The `event` component is the reason this device does not need a per-(rule, camera,
    label) entity to be useful. Its payload carries the source and the subject as
    attributes, so one automation triggers on the rule and branches on
    `trigger.event.data.subject` — the combination is available without an entity per
    combination, which is what keeps the cardinality flat.

    Returns None when no rule is publishable, so the device is retracted rather than
    published empty. An empty device in HA reads as broken, not as inapplicable.
    """
    cmps: Dict[str, Any] = {}
    for profile in profiles:
        profile_slug = slugify(profile.name)
        for rule in profile.rules:
            if not rule.enabled or not rule.publish_to_ha:
                continue
            rule_slug = slugify(rule.name)
            base = f"{prefix}/{rule_global_base_topic(profile_slug, rule_slug)}"
            label = entity_name("PurrView Rules", rule.name)
            # uuid-based like everywhere else: renaming a rule moves its topics but must
            # not orphan its history.
            uid = f"{RULES_DEVICE_ID}_{rule.uuid}"

            cmps[f"{rule.uuid}_state"] = {
                "p": "binary_sensor",
                "unique_id": f"{uid}_state",
                "name": label,
                "device_class": "occupancy",
                "state_topic": f"{base}/state",
                "payload_on": "ON",
                "payload_off": "OFF",
                "json_attributes_topic": f"{base}/attributes",
            }
            cmps[f"{rule.uuid}_count"] = {
                "p": "sensor",
                "unique_id": f"{uid}_count",
                "name": f"{label} count",
                # Without a state_class HA purges raw history after ~10 days and keeps
                # nothing; with it, "how many times last year" stays answerable forever.
                "state_class": "total_increasing",
                "state_topic": f"{base}/count",
            }
            cmps[f"{rule.uuid}_last_seen"] = {
                "p": "sensor",
                "unique_id": f"{uid}_last_seen",
                "name": f"{label} last seen",
                "device_class": "timestamp",
                "state_topic": f"{base}/attributes",
                "value_template": "{{ value_json.lastSeen }}",
            }
            # HA's MQTT event platform reads `event_type` out of the JSON payload and
            # exposes every other key as an attribute — which is what carries the
            # source and subject through to the automation.
            cmps[f"{rule.uuid}_event"] = {
                "p": "event",
                "unique_id": f"{uid}_event",
                "name": f"{label} event",
                "state_topic": f"{base}/event",
                "event_types": ["opened", "closed"],
            }
            # Read off the same retained attributes payload rather than topics of their
            # own: two more entities, no more messages.
            cmps[f"{rule.uuid}_last_subject"] = {
                "p": "sensor",
                "unique_id": f"{uid}_last_subject",
                "name": f"{label} last subject",
                "state_topic": f"{base}/attributes",
                "value_template": "{{ value_json.lastSubject }}",
            }
            cmps[f"{rule.uuid}_last_source"] = {
                "p": "sensor",
                "unique_id": f"{uid}_last_source",
                "name": f"{label} last camera",
                "state_topic": f"{base}/attributes",
                "value_template": "{{ value_json.lastSource }}",
            }

    if not cmps:
        return None
    return {
        "dev": _device(RULES_DEVICE_ID, "PurrView Rules", "Rules"),
        "o": dict(ORIGIN),
        # App-level availability only: a rule spans every watcher, so tying it to one
        # would blank it whenever that watcher stopped.
        "availability": [{"topic": f"{prefix}/status"}],
        "cmps": cmps,
    }


def build_app_discovery(prefix: str) -> Dict[str, Any]:
    """The `purrview_system` device: app-wide diagnostics that belong to no
    single camera."""
    return {
        "dev": _device(APP_DEVICE_ID, "PurrView", "Application"),
        "o": dict(ORIGIN),
        "availability": [{"topic": f"{prefix}/status"}],
        "cmps": {
            "storage_bytes": {
                "p": "sensor",
                "unique_id": f"{APP_DEVICE_ID}_storage_bytes",
                "name": "Storage used",
                "device_class": "data_size",
                "unit_of_measurement": "B",
                "state_class": "measurement",
                "state_topic": f"{prefix}/system",
                "value_template": "{{ value_json.storageBytes }}",
            },
            "open_events": {
                "p": "sensor",
                "unique_id": f"{APP_DEVICE_ID}_open_events",
                "name": "Open events",
                "state_class": "measurement",
                "state_topic": f"{prefix}/system",
                "value_template": "{{ value_json.openEvents }}",
            },
            "watchers_running": {
                "p": "sensor",
                "unique_id": f"{APP_DEVICE_ID}_watchers_running",
                "name": "Watchers running",
                "state_class": "measurement",
                "state_topic": f"{prefix}/system",
                "value_template": "{{ value_json.watchersRunning }}",
            },
            # How many entities this app is asking Home Assistant to hold.
            #
            # Here because the cost is otherwise invisible until it is a problem: the
            # count is a product of labels, rules and cameras, so it grows when config
            # grows and nothing in the app or in HA says so. Publishing it means the
            # number can be watched, put on a dashboard and alerted on, which is the
            # difference between a budget and a hope.
            "ha_entities": {
                "p": "sensor",
                "unique_id": f"{APP_DEVICE_ID}_ha_entities",
                "name": "HA entities published",
                "state_class": "measurement",
                "state_topic": f"{prefix}/system",
                "value_template": "{{ value_json.haEntities }}",
            },
        },
    }


def count_entities(messages: Dict[str, Dict[str, Any]]) -> int:
    """Total components across every device-discovery message — i.e. how many entities
    Home Assistant is being asked to create. Counts disabled-by-default ones too: HA
    still makes a registry entry for those, which is the cost being measured."""
    return sum(len(message.get("cmps") or {}) for message in messages.values())


def subjects_in_scope(profiles: Sequence[ProfileView]) -> Dict[str, List[Tuple[RuleView, List[SourceView]]]]:
    """subject uuid → the (rule, sources) pairs that can produce events for it.

    Only labels some enabled rule can actually match get a device. A label nothing
    references would produce a device whose every entity was permanently zero, which is
    worse than absent — it looks broken rather than inapplicable.
    """
    out: Dict[str, List[Tuple[RuleView, List[SourceView]]]] = {}
    for profile in profiles:
        enabled_sources = [s for s in profile.sources if s.enabled and s.publish_to_ha]
        for rule in profile.rules:
            if not rule.enabled or not rule.publish_to_ha:
                continue
            sources = [s for s in enabled_sources if not rule.source_uuids or s.uuid in rule.source_uuids]
            if not sources:
                continue
            for subject_uuid in rule.subject_uuids:
                out.setdefault(subject_uuid, []).append((rule, sources))
    return out


def build_subject_seed(
    counts: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, ...], Tuple[int, Optional[str]]]:
    """Roll the finest-grain counts up into every tier the bridge publishes.

    Keys are the grouping, shortest first: `(subject,)`, `(subject, rule)`, and
    `(subject, rule, source)`. Each value is `(count, last_seen)`.

    Summed from one query rather than fetched per tier so the tiers cannot disagree —
    a subject total that isn't the sum of its rules is a discrepancy nobody can account
    for afterwards, and three separate `GROUP BY`s can produce one across a concurrent
    write. Rows with no subject are skipped: they belong to rule-level counters, which
    seed separately.
    """
    out: Dict[Tuple[str, ...], Tuple[int, Optional[str]]] = {}

    def _add(key: Tuple[str, ...], count: int, last_seen: Optional[str]) -> None:
        previous_count, previous_seen = out.get(key, (0, None))
        # max() over ISO 8601 is chronological, which is the one thing that string
        # format is good for.
        merged_seen = max(filter(None, (previous_seen, last_seen)), default=None)
        out[key] = (previous_count + count, merged_seen)

    for row in counts:
        subject = row.get("subject_ref")
        if not subject:
            continue
        rule, source = row["rule_uuid"], row["source_uuid"]
        count, last_seen = int(row["count"]), row.get("last_seen")
        _add((subject,), count, last_seen)
        _add((subject, rule), count, last_seen)
        _add((subject, rule, source), count, last_seen)
    return out


def build_all_discovery(
    profiles: Sequence[ProfileView],
    prefix: str,
    subjects: Optional[Dict[str, SubjectView]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Every device-discovery message that should currently exist, keyed by config
    topic. Does not decide what to *retract* — see `reconcile_discovery`, which diffs
    this against what was published last time.

    `subjects` maps label uuid → the label, for every label that can be a rule subject.
    Omitted, no subject devices are produced — the caller has no vocabulary to name
    them with, and a device titled with a bare uuid is worse than no device.
    """
    messages: Dict[str, Dict[str, Any]] = {}
    for profile in profiles:
        for source in profile.sources:
            if not source.enabled or not source.publish_to_ha:
                continue
            topic = source_config_topic(source.uuid)
            payload = build_source_discovery(profile, source, prefix)
            if topic in messages:
                # Two profiles watching one camera (an open question in the plan,
                # §6.3) would otherwise clobber each other depending on dict
                # ordering — merge instead so neither profile's rules disappear.
                messages[topic]["cmps"].update(payload["cmps"])
            else:
                messages[topic] = payload
    for subject_uuid, rule_scopes in subjects_in_scope(profiles).items():
        subject = (subjects or {}).get(subject_uuid)
        if subject is None or not subject.publish_to_ha:
            continue
        messages[subject_config_topic(subject_uuid)] = build_subject_discovery(subject, rule_scopes, prefix)

    rules_device = build_rules_discovery(profiles, prefix)
    if rules_device is not None:
        messages[RULES_CONFIG_TOPIC] = rules_device

    # Built last: it reports the size of everything above it, so it has to be able to
    # count the finished set.
    messages[APP_CONFIG_TOPIC] = build_app_discovery(prefix)
    return messages


@dataclass(frozen=True)
class DiscoveryState:
    """What the last reconcile published: component ids and their platforms, per config
    topic. The platforms are the part that matters — removing a component requires
    naming its platform back to HA, so it cannot be reconstructed from the new config
    once the component is gone.

    In memory only. A component deleted while the app is *down* is therefore not
    detected on the next start (previous state is empty, so nothing reads as removed)
    and its entity survives in HA until something else republishes that device. Worth
    knowing; not worth a state file yet.
    """

    components: Dict[str, Dict[str, str]] = field(default_factory=dict)

    @property
    def topics(self) -> FrozenSet[str]:
        return frozenset(self.components)


def state_from_retained(retained: Dict[str, bytes]) -> DiscoveryState:
    """Rebuild `DiscoveryState` from the broker's retained discovery configs.

    The retained messages under `homeassistant/device/+/config` *are* the record of
    what Home Assistant has been told exists — published by us, kept by the broker,
    outliving any restart of ours. Reading them back means the reconcile stops
    remembering what it did and starts observing what is, which additionally cleans up
    configs this process never published: an older version of the app, a stray
    `mosquitto_pub`, a dev instance pointed at the same broker.

    That scope is deliberately "someone else publishing *our* devices", not "anyone
    else's devices". Only topics passing `owns_config_topic` are adopted — the
    subscription cannot narrow itself, and adopting a foreign config is a licence to
    delete it. See `owns_config_topic` for why the filter lives here.

    Anything unparseable is skipped rather than raised on. A config we cannot read is
    one we will not retract — the same safe direction as reading nothing at all — and
    refusing to start the bridge over one malformed retained message would be a far
    worse trade.
    """
    components: Dict[str, Dict[str, str]] = {}
    for topic, payload in retained.items():
        if not owns_config_topic(topic):
            continue
        try:
            config = json.loads(payload)
            cmps = config.get("cmps") or {}
            components[topic] = {cid: str(comp.get("p", "")) for cid, comp in cmps.items() if isinstance(comp, dict)}
        except Exception:
            logger.debug("Skipping unparseable retained discovery config on %s", topic, exc_info=True)
    return DiscoveryState(components=components)


def reconcile_discovery(
    previous: DiscoveryState, desired: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Optional[Dict[str, Any]]], DiscoveryState]:
    """What to publish this pass: every desired message, plus the two kinds of removal.

    **A whole device that disappeared** gets an empty payload (`None`, published as
    `""`) on its config topic — a deleted source or profile.

    **A single component that disappeared** — a deleted or disabled rule, on a device
    that still exists — needs more than omission. HA's device-discovery docs are
    explicit that dropping a component from `cmps` does *not* reliably remove its
    entity; the component must first be republished as an empty config carrying only
    its platform, and omitted only on a subsequent update. So a pass that notices a
    component has gone re-inserts it as `{"p": <platform>}` — a tombstone — and the
    next pass publishes the clean message, which happens by itself because every
    desired topic is republished every pass.

    Getting this wrong is invisible from here: the payload looks correct, and the
    entity simply stays in HA forever with no way to remove it but hand-editing the
    entity registry.

    Returns `(to_publish, new_state)` rather than mutating anything, so the caller
    decides when — and whether — to persist the second value.
    """
    to_publish: Dict[str, Optional[Dict[str, Any]]] = {}
    for topic, payload in desired.items():
        previous_components = previous.components.get(topic, {})
        removed = set(previous_components) - set(payload["cmps"])
        if removed:
            payload = {
                **payload,
                "cmps": {**payload["cmps"], **{cid: {"p": previous_components[cid]} for cid in removed}},
            }
        to_publish[topic] = payload

    for gone_topic in previous.topics - desired.keys():
        to_publish[gone_topic] = None

    # Records what was *desired*, never the tombstones — otherwise the next pass would
    # see them as components that had just disappeared and tombstone them again,
    # forever.
    new_state = DiscoveryState(
        components={
            topic: {cid: str(component.get("p", "")) for cid, component in payload["cmps"].items()}
            for topic, payload in desired.items()
        }
    )
    return to_publish, new_state


def build_seed_publishes(profiles: Sequence[ProfileView], counts: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Topic → lifetime count, for every (source, rule) count both present in history
    and still known to current config. A count for a uuid no longer in any profile
    (a deleted rule or source) has no topic to seed onto — its retained state, if any,
    goes away with the rest of that entity's discovery retraction, not through here.

    Resolved per profile rather than against one flat uuid→slug map, because the source
    slug is profile-scoped: `mqtt_name` lives on the profile↔source link, so one camera
    watched by two profiles can legitimately have a different slug under each.
    """
    index = build_slug_index(profiles)
    out: Dict[str, int] = {}
    for (profile_uuid, source_uuid, rule_uuid), count in build_counter_seed(profiles, counts).items():
        topic = index.count_topic(profile_uuid, source_uuid, rule_uuid)
        if topic is not None:
            out[topic] = count
    return out


def build_counter_seed(
    profiles: Sequence[ProfileView], counts: Sequence[Dict[str, Any]]
) -> Dict[Tuple[str, str, str], int]:
    """Lifetime counts keyed by `(profile_uuid, source_uuid, rule_uuid)`.

    Uuid-keyed rather than topic-keyed on purpose: this is what the live counter is
    held in, and a rule rename changes its topic. Keyed by topic, a rename would orphan
    the running total and restart the sensor at zero — recoverable (`total_increasing`
    reads a drop as a counter reset) but a needless dent in the statistics.

    A count is attributed to a profile only when that profile holds both the source and
    the rule, which is what stops two profiles sharing a camera from each claiming the
    other's events.
    """
    out: Dict[Tuple[str, str, str], int] = {}
    for profile in profiles:
        source_uuids = {s.uuid for s in profile.sources}
        rule_uuids = {r.uuid for r in profile.rules}
        for row in counts:
            if row["source_uuid"] in source_uuids and row["rule_uuid"] in rule_uuids:
                out[(profile.uuid, row["source_uuid"], row["rule_uuid"])] = row["count"]
    return out


@dataclass(frozen=True)
class SlugIndex:
    """uuid → topic-slug lookup, so a caller holding only uuids can build the same
    topics `build_source_discovery` advertised. Returns `None` for anything it has not
    seen: a rule created since the last reconcile has no HA entity yet either, so there
    is nothing useful to publish about it.
    """

    #: profile_uuid → (profile_slug, {source_uuid: slug}, {rule_uuid: slug})
    profiles: Dict[str, Tuple[str, Dict[str, str], Dict[str, str]]]

    def rule_base(self, profile_uuid: str, source_uuid: str, rule_uuid: str) -> Optional[str]:
        entry = self.profiles.get(profile_uuid)
        if entry is None:
            return None
        profile_slug, sources, rules = entry
        source_slug, rule_slug = sources.get(source_uuid), rules.get(rule_uuid)
        if source_slug is None or rule_slug is None:
            return None
        return rule_base_topic(profile_slug, source_slug, rule_slug)

    def count_topic(self, profile_uuid: str, source_uuid: str, rule_uuid: str) -> Optional[str]:
        base = self.rule_base(profile_uuid, source_uuid, rule_uuid)
        return None if base is None else f"{base}/count"

    def rule_global_base(self, profile_uuid: str, rule_uuid: str) -> Optional[str]:
        """The rule's cross-camera topics — `rule_base` without the source segment."""
        entry = self.profiles.get(profile_uuid)
        if entry is None:
            return None
        profile_slug, _sources, rules = entry
        rule_slug = rules.get(rule_uuid)
        return None if rule_slug is None else rule_global_base_topic(profile_slug, rule_slug)

    def rule_slug(self, rule_uuid: str) -> Optional[str]:
        """A rule's topic slug, without needing to know its profile.

        Subject topics are app-wide — a label is not owned by a profile — so their
        rule and source segments are looked up across every profile rather than within
        one. A rule uuid belongs to exactly one profile, so this is unambiguous.
        """
        for _slug, _sources, rules in self.profiles.values():
            if rule_uuid in rules:
                return rules[rule_uuid]
        return None

    def source_slug(self, source_uuid: str) -> Optional[str]:
        """A source's topic slug. Unlike `rule_slug` this *can* be ambiguous — one
        camera watched by two profiles may carry a different `mqtt_name` under each —
        so the first match wins and the subject tree stays stable rather than
        alternating between two spellings of the same camera."""
        for _slug, sources, _rules in self.profiles.values():
            if source_uuid in sources:
                return sources[source_uuid]
        return None

    def telemetry(self, profile_uuid: str, source_uuid: str) -> Optional[str]:
        entry = self.profiles.get(profile_uuid)
        if entry is None:
            return None
        profile_slug, sources, _rules = entry
        source_slug = sources.get(source_uuid)
        return None if source_slug is None else telemetry_topic(profile_slug, source_slug)


def build_slug_index(profiles: Sequence[ProfileView]) -> SlugIndex:
    return SlugIndex(
        profiles={
            p.uuid: (
                slugify(p.name),
                {s.uuid: _topic_source(s) for s in p.sources},
                {r.uuid: _topic_rule(r) for r in p.rules},
            )
            for p in profiles
        }
    )


def build_event_attributes(
    *,
    last_seen: str,
    duration_s: Optional[float],
    peak_confidence: Optional[float],
    subject: Optional[str],
    target: Optional[str],
    track_ids: Sequence[int],
) -> Dict[str, Any]:
    """The `attributes` payload three sensors and the binary_sensor read through
    `value_template`. One topic rather than five, so a transition is one publish and
    the values a card shows can never be half-updated.

    Keys are camelCase to match the `value_template`s in `build_source_discovery` and
    the rest of the app's JSON. `lastSeen` is ISO 8601 with an offset because its
    sensor declares `device_class: timestamp`, which HA rejects as unknown otherwise.
    """
    return {
        "lastSeen": last_seen,
        "lastDurationS": round(duration_s, 2) if duration_s is not None else None,
        "peakConfidence": round(peak_confidence, 4) if peak_confidence is not None else None,
        "subject": subject,
        "target": target,
        "trackIds": list(track_ids),
    }


def build_rule_global_attributes(
    *,
    last_seen: str,
    subject: Optional[str],
    source_name: Optional[str],
    duration_s: Optional[float],
    peak_confidence: Optional[float],
) -> Dict[str, Any]:
    """The Rules device's attributes payload: what fired this rule, and where.

    `lastSubject` and `lastSource` are the whole point of the device — the per-camera
    and per-label devices each know only their own half, and this is the tier that
    holds both. Their sensors read them straight out of this one retained message.
    """
    return {
        "lastSeen": last_seen,
        "lastSubject": subject,
        "lastSource": source_name,
        "lastDurationS": round(duration_s, 2) if duration_s is not None else None,
        "peakConfidence": round(peak_confidence, 4) if peak_confidence is not None else None,
    }


# ── The connection ────────────────────────────────────────────────────────────────


class MqttBridge:
    """Owns the app-level MQTT connection and keeps HA's view of it current.

    Safe to construct even with no broker configured — every method is a no-op until
    `ensure_started` finds one. Call `ensure_started` on a timer (it's cheap once
    connected — just an `is None` check) rather than once at boot: a broker configured
    after the app is already running should not need a restart to take effect.
    """

    def __init__(self, db_api_client: Any) -> None:
        self._db_api_client = db_api_client
        self._publisher: Optional[AsyncMqttPublisher] = None
        self._prefix: str = "purrview"
        self._last_discovery: DiscoveryState = DiscoveryState()
        # Refreshed on every reconcile, so a watcher publishing an event transition can
        # resolve its uuids to topics without a DB round trip on the hot path.
        self._slugs: SlugIndex = build_slug_index([])
        # (profile_uuid, source_uuid, rule_uuid) → lifetime count. Seeded from the DB
        # once at connect and incremented in memory thereafter: the alternative, a
        # COUNT(*) per event, puts a query on the event path to recompute a number this
        # process already knows.
        self._counts: Dict[Tuple[str, str, str], int] = {}
        # source uuid → the label uuids that source can currently see. The published
        # per-subject answer is the union across all of them, so this is kept whole
        # rather than reduced on arrival: removing a camera's contribution requires
        # knowing what it was contributing.
        self._presence_by_source: Dict[str, Set[str]] = {}
        #: subject label uuid → topic slug, refreshed with the profile snapshot.
        self._subject_slugs: Dict[str, str] = {}
        #: Rolled-up subject counters, keyed shortest-tuple-first exactly as
        #: `build_subject_seed` produces them: (subject,), (subject, rule),
        #: (subject, rule, source).
        self._subject_counts: Dict[Tuple[str, ...], int] = {}
        #: Label vocabulary, refreshed with the profile snapshot. Labels live in the DB
        #: and change without this process being told, so both are re-read per reconcile
        #: rather than cached at startup.
        self._subject_names: Dict[str, str] = {}
        self._label_ancestry: Dict[str, List[str]] = {}
        #: label uuid → (publish_to_ha, publish_per_source). Absent means the defaults,
        #: so a label the bridge has not seen behaves as it did before the flags existed.
        self._label_publish: Dict[str, Tuple[bool, bool]] = {}
        #: Entities in the last desired discovery set, for the app device's budget
        #: sensor. Zero until the first reconcile, which is honest: nothing published.
        self._entity_count: int = 0
        #: (profile, source, rule) currently mid-event, for the app device's open-event
        #: count. Tracked from published transitions because there is no "still open"
        #: query — an event's row isn't written until an action commits to filing it.
        self._open_rules: Set[Tuple[str, str, str]] = set()
        #: watcher slug → last known online state. Held independently of the
        #: connection so a watcher that starts before the bridge connects is not lost;
        #: re-asserted on every reconcile.
        self._watcher_availability: Dict[str, bool] = {}

    @property
    def connected(self) -> bool:
        return self._publisher is not None

    async def ensure_started(self) -> bool:
        if self._publisher is not None:
            return True
        settings = await load_mqtt_broker_settings(self._db_api_client)
        if settings is None:
            return False
        # A distinct client id from the per-watcher publishers (which share the
        # broker's default id) — most brokers disconnect the existing holder of a
        # client id when a new client connects with the same one, and this
        # connection must not fight every watcher's connection for it.
        settings = dataclasses.replace(settings, client_id=f"{settings.client_id}-bridge")
        publisher = AsyncMqttPublisher(settings, availability_topic=f"{settings.prefix}/status")
        await publisher.start()
        self._publisher = publisher
        self._prefix = settings.prefix
        # Ask the broker what Home Assistant currently believes exists, rather than
        # assuming nothing does. Without this the first reconcile after any restart
        # sees an empty previous state, so it can only ever add entities — anything
        # deleted while this process was down stays in HA forever. See
        # `state_from_retained`.
        #
        # The `+` is every integration's device configs, not just ours, and cannot be
        # narrowed at the broker — MQTT wildcards match whole levels, so there is no
        # subscription that means "ids starting with `purrview_`". `state_from_retained`
        # drops the ones we do not own; do not "fix" this by writing `purrview_+`, which
        # matches nothing and would silently disable the read-back entirely.
        try:
            retained = await publisher.read_retained(f"{HA_DISCOVERY_PREFIX}/device/+/config", absolute=True)
            self._last_discovery = state_from_retained(retained)
            logger.info("MQTT bridge adopted %s retained discovery config(s)", len(self._last_discovery.components))
        except Exception:
            # Starting with an empty state is the pre-existing behaviour: additive
            # only, no retraction this pass. Worth a warning, not worth refusing to
            # bring the bridge up.
            logger.warning("Could not read retained discovery configs; starting from empty", exc_info=True)
        try:
            await self.seed_counters()
        except Exception:
            logger.warning("MQTT bridge connected but counter seeding failed", exc_info=True)
        return True

    async def stop(self) -> None:
        if self._publisher is not None:
            await self._publisher.close()
            self._publisher = None

    async def reconcile(self) -> None:
        """Connect if not already connected, then republish discovery against current
        config. One call site for a caller that just wants "make MQTT match reality
        right now" — the worker loop's periodic pass, and a watcher's start/stop.
        """
        if await self.ensure_started():
            self._publish_watcher_availability()
            await self.publish_all_discovery()

    async def _load_labels(self) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, Tuple[bool, bool]]]:
        """`(uuid → name, uuid → ancestor uuids, uuid → publish flags)` for the label tree.

        The ancestry half is what makes subject enumeration correct: rules commonly
        name a *parent* label with `include_descendants`, and it is the descendant that
        lands in an event's `subject_ref`, so the literal refs alone would miss every
        individual the rule can actually match.

        The flags are read per label and **not** inherited down the tree, deliberately.
        Inheritance would make "why does this label have no entities" unanswerable
        without walking to the root, and the tree is an annotation taxonomy — its shape
        is chosen for labelling, not for what belongs in someone's house automation.
        """
        from db_client.api.labels_api import LabelsApi

        labels_resp = await asyncio.to_thread(LabelsApi(self._db_api_client).list_labels)
        labels = list(labels_resp.labels or [])
        parent_of = {label.uuid: getattr(label, "parent_uuid", None) for label in labels}

        def _ancestors(uuid: str) -> List[str]:
            chain, cursor, guard = [], parent_of.get(uuid), 0
            # Guarded rather than trusted: a cycle here would hang the reconcile pass,
            # and this walks data the bridge does not own.
            while cursor and guard < 64:
                chain.append(cursor)
                cursor, guard = parent_of.get(cursor), guard + 1
            return chain

        return (
            {label.uuid: label.name for label in labels},
            {label.uuid: _ancestors(label.uuid) for label in labels},
            {
                label.uuid: (
                    bool(getattr(label, "publish_to_ha", True)),
                    bool(getattr(label, "publish_per_source", False)),
                )
                for label in labels
            },
        )

    def _subject_views(self, profiles: Sequence[ProfileView]) -> Dict[str, SubjectView]:
        """The subject labels in scope, named. A subject whose label has since been
        deleted is dropped rather than shown as a bare uuid — its rule will stop
        matching it too, so the device would only ever be a dead one."""
        return {
            uuid: SubjectView(
                uuid=uuid,
                name=self._subject_names[uuid],
                publish_to_ha=self._label_publish.get(uuid, (True, False))[0],
                publish_per_source=self._label_publish.get(uuid, (True, False))[1],
            )
            for uuid in subjects_in_scope(profiles)
            if uuid in self._subject_names
        }

    def _expand_subjects(self, rule: Any) -> Tuple[str, ...]:
        """Every label uuid this rule can match as its subject.

        Only `subject_a` — that is the slot an event records in `subject_ref`, and it
        is what "which cat" means. `subject_b` is the *target* (the bowl, the litter
        box), which belongs to the other half of an interaction and gets no device.

        Only `LABEL` subjects. A zone-kind subject refers to geometry, which has no
        label to hang a device off and never appears as a `subject_ref`.
        """
        if str(getattr(rule, "subject_a_kind", "label")).lower() != "label":
            return ()
        refs = [str(r) for r in (getattr(rule, "subject_a_refs_json", None) or [])]
        if not getattr(rule, "subject_a_include_descendants", False):
            return tuple(refs)
        expanded = set(refs)
        for ref in refs:
            expanded.update(descendants_of(ref, self._label_ancestry))
        return tuple(sorted(expanded))

    async def _load_profiles(self) -> List[ProfileView]:
        from db_client.api.sources_api import SourcesApi
        from db_client.api.watch_profiles_api import WatchProfilesApi

        profiles_resp = await asyncio.to_thread(WatchProfilesApi(self._db_api_client).list_watch_profiles)
        sources_resp = await asyncio.to_thread(SourcesApi(self._db_api_client).get_sources)
        # `GetSourcesResponse.sources` is already keyed by uuid — iterating it yields
        # the uuid strings, not the sources.
        source_by_uuid = dict(sources_resp.sources or {})
        label_names, ancestry, label_publish = await self._load_labels()
        self._subject_names = label_names
        self._label_ancestry = ancestry
        self._label_publish = label_publish

        views: List[ProfileView] = []
        for profile in profiles_resp.profiles or []:
            sources = tuple(
                SourceView(
                    uuid=ps.source_uuid,
                    name=source_by_uuid[ps.source_uuid].name,
                    mqtt_name=ps.mqtt_name,
                    enabled=ps.enabled,
                    publish_to_ha=bool(getattr(ps, "publish_to_ha", True)),
                )
                for ps in (profile.sources or [])
                if ps.source_uuid in source_by_uuid  # a source can be deleted out from under a profile
            )
            rules = tuple(
                RuleView(
                    uuid=r.uuid,
                    name=r.name,
                    enabled=r.enabled,
                    publish_to_ha=bool(getattr(r, "publish_to_ha", True)),
                    source_uuids=tuple(r.source_uuids_json or []),
                    subject_uuids=self._expand_subjects(r),
                )
                for r in (profile.rules or [])
            )
            views.append(ProfileView(uuid=profile.uuid, name=profile.name, sources=sources, rules=rules))
        return views

    async def publish_all_discovery(self) -> None:
        """Recompute the desired discovery set from current config and publish the
        diff. Cheap and idempotent — call it whenever config might have changed
        (watcher start/stop) and periodically regardless, since nothing in this
        process is told directly when a profile is saved in db_server.
        """
        if self._publisher is None:
            return
        profiles = await self._load_profiles()
        self._slugs = build_slug_index(profiles)
        subjects = self._subject_views(profiles)
        self._subject_slugs = {uuid: slugify(view.name) for uuid, view in subjects.items()}
        desired = build_all_discovery(profiles, self._prefix, subjects)
        # Measured off `desired`, not off what this pass happens to publish: an
        # unchanged config publishes nothing, and a budget that read zero whenever
        # nothing moved would be useless.
        self._entity_count = count_entities(desired)
        to_publish, self._last_discovery = reconcile_discovery(self._last_discovery, desired)
        for topic, payload in to_publish.items():
            self._publisher.publish(topic, "" if payload is None else payload, retain=True, absolute=True)

    async def seed_counters(self) -> None:
        if self._publisher is None:
            return
        from db_client.api.watch_profiles_api import WatchProfilesApi

        profiles = await self._load_profiles()
        self._slugs = build_slug_index(profiles)
        counts_resp = await asyncio.to_thread(WatchProfilesApi(self._db_api_client).get_watch_event_counts)
        counts = [
            {
                "source_uuid": c.source_uuid,
                "rule_uuid": c.rule_uuid,
                "subject_ref": c.subject_ref,
                "count": c.count,
                "last_seen": c.last_seen,
            }
            for c in (counts_resp.counts or [])
        ]
        self._counts = build_counter_seed(profiles, counts)
        for topic, count in build_seed_publishes(profiles, counts).items():
            self._publisher.publish(topic, count, retain=True)

        # Subject tiers, all rolled up from the same rows the rule counters used, so a
        # subject's total is the sum of its rules by construction rather than by two
        # queries happening to agree.
        seed = build_subject_seed(counts)
        self._subject_counts = {key: count for key, (count, _seen) in seed.items()}
        for key, (count, last_seen) in seed.items():
            topic = self._subject_topic(key)
            if topic is None:
                continue
            self._publisher.publish(f"{topic}/count", count, retain=True)
            if last_seen:
                self._publisher.publish(f"{topic}/attributes", {"lastSeen": last_seen}, retain=True)

    # ── Live state, published as events happen ────────────────────────────────────
    #
    # Called from `StreamWatcherTask`'s dispatch. Every method here is synchronous and
    # non-throwing by design: `AsyncMqttPublisher.publish` only enqueues (bounded,
    # drop-oldest), so none of this can block the analysis loop, and a broker problem
    # must never propagate into the watcher — a camera that stops recording because
    # Home Assistant's broker went away would be a far worse failure than a gap in a
    # graph.

    def publish_rule_transition(
        self,
        *,
        profile_uuid: str,
        source_uuid: str,
        rule_uuid: str,
        active: bool,
        opened: bool,
        last_seen: str,
        duration_s: Optional[float] = None,
        peak_confidence: Optional[float] = None,
        subject: Optional[str] = None,
        target: Optional[str] = None,
        track_ids: Sequence[int] = (),
        source_name: Optional[str] = None,
    ) -> None:
        """One event transition → the retained topics HA's entities read.

        `active` and `opened` are separate because an event has three transitions and
        only two answers between them: START is both, UPDATE is active but not new (it
        must not double-count a single event), END is neither.

        Retained throughout, so an HA restart re-reads the current state instead of
        showing `unknown` until the next event — which for a rule that fires twice a
        day would be most of the day.
        """
        if self._publisher is None:
            return
        base = self._slugs.rule_base(profile_uuid, source_uuid, rule_uuid)
        if base is None:
            # Config the bridge has not seen yet (a rule added since the last
            # reconcile). It has no discovery entry either, so there is no entity
            # waiting for this; the next reconcile creates both.
            return
        key = (profile_uuid, source_uuid, rule_uuid)
        # Tracked on every transition, not just the opening one: an event that was
        # already open when the bridge connected still closes, and dropping that would
        # leave the count drifting upward for the life of the process.
        self._open_rules.add(key) if active else self._open_rules.discard(key)
        if opened:
            self._counts[key] = self._counts.get(key, 0) + 1
            self._publisher.publish(f"{base}/count", self._counts[key], retain=True)
        self._publisher.publish(
            f"{base}/attributes",
            build_event_attributes(
                last_seen=last_seen,
                duration_s=duration_s,
                peak_confidence=peak_confidence,
                subject=subject,
                target=target,
                track_ids=track_ids,
            ),
            retain=True,
        )
        # Last: the binary_sensor's `json_attributes_topic` is the topic above, and
        # publishing state first would briefly pair a new ON with the previous event's
        # attributes.
        self._publisher.publish(f"{base}/state", "ON" if active else "OFF", retain=True)

        self._publish_rule_global(
            profile_uuid=profile_uuid,
            rule_uuid=rule_uuid,
            active=active,
            opened=opened,
            last_seen=last_seen,
            subject=subject,
            source_name=source_name,
            duration_s=duration_s,
            peak_confidence=peak_confidence,
        )

    def _publish_rule_global(
        self,
        *,
        profile_uuid: str,
        rule_uuid: str,
        active: bool,
        opened: bool,
        last_seen: str,
        subject: Optional[str],
        source_name: Optional[str],
        duration_s: Optional[float],
        peak_confidence: Optional[float],
    ) -> None:
        """The Rules device's tier: this rule, across every camera and every label.

        Must run *after* `_open_rules` has been updated for this transition — the state
        it publishes is "open on any source", which is read back out of that set. A rule
        firing on two cameras must stay ON until the second one closes, which is exactly
        what the per-camera binary_sensor cannot express and what an automation written
        against one camera gets wrong.
        """
        if self._publisher is None:
            return
        base = self._slugs.rule_global_base(profile_uuid, rule_uuid)
        if base is None:
            return

        if opened:
            # Lifetime total across sources, summed from the same per-source counters
            # the camera device publishes, so the two tiers cannot disagree.
            total = sum(c for (p, _s, r), c in self._counts.items() if p == profile_uuid and r == rule_uuid)
            self._publisher.publish(f"{base}/count", total, retain=True)

        self._publisher.publish(
            f"{base}/attributes",
            build_rule_global_attributes(
                last_seen=last_seen,
                subject=subject,
                source_name=source_name,
                duration_s=duration_s,
                peak_confidence=peak_confidence,
            ),
            retain=True,
        )
        still_open = any(p == profile_uuid and r == rule_uuid for p, _s, r in self._open_rules)
        self._publisher.publish(f"{base}/state", "ON" if still_open else "OFF", retain=True)

        # UPDATE transitions are neither an open nor a close and must not fire an event
        # — an automation triggered off this would run repeatedly for one visit.
        event_type = "opened" if opened else ("closed" if not active else None)
        if event_type is not None:
            # **Not retained**, unlike everything else here. HA's event entity fires on
            # every message it receives, including the retained one replayed on each
            # reconnect — so retaining this would re-trigger every automation bound to
            # it whenever HA restarted or the broker dropped.
            self._publisher.publish(
                f"{base}/event",
                {
                    "event_type": event_type,
                    "subject": subject,
                    "source": source_name,
                    "durationS": round(duration_s, 2) if duration_s is not None else None,
                    "peakConfidence": round(peak_confidence, 4) if peak_confidence is not None else None,
                },
                retain=False,
            )

    def _subject_topic(self, key: Tuple[str, ...]) -> Optional[str]:
        """A rolled-up counter key → its topic, or None if any part is unknown to the
        current config. Mirrors the tiers `build_subject_seed` produces."""
        subject_slug = self._subject_slugs.get(key[0])
        if subject_slug is None:
            return None
        base = subject_base_topic(subject_slug)
        if len(key) == 1:
            return base
        rule_slug = self._slugs.rule_slug(key[1])
        if rule_slug is None:
            return None
        if len(key) == 2:
            return f"{base}/rule/{rule_slug}"
        source_slug = self._slugs.source_slug(key[2])
        return None if source_slug is None else f"{base}/rule/{rule_slug}/{source_slug}"

    def publish_subject_event(
        self,
        *,
        subject_uuid: str,
        rule_uuid: str,
        source_uuid: str,
        last_seen: str,
        source_name: Optional[str] = None,
    ) -> None:
        """One opening event, rolled into all three subject tiers.

        Called only on START. The tiers are incremented together from one transition
        rather than recomputed, which is what keeps them consistent between reseeds:
        they are the same number counted at three granularities, so they must move
        together or not at all.
        """
        if self._publisher is None or subject_uuid not in self._subject_slugs:
            return
        for key in ((subject_uuid,), (subject_uuid, rule_uuid), (subject_uuid, rule_uuid, source_uuid)):
            topic = self._subject_topic(key)
            if topic is None:
                continue
            self._subject_counts[key] = self._subject_counts.get(key, 0) + 1
            self._publisher.publish(f"{topic}/count", self._subject_counts[key], retain=True)
            attributes: Dict[str, Any] = {"lastSeen": last_seen}
            if len(key) == 1:
                # Only the device-level topic carries location: at rule or source
                # granularity the camera is already implied by the topic itself.
                attributes["lastSource"] = source_name
            self._publisher.publish(f"{topic}/attributes", attributes, retain=True)

    def publish_presence(self, *, source_uuid: str, label_uuids: Sequence[str]) -> None:
        """Report which labels one camera can currently see, and republish any subject
        whose overall answer changed.

        The union across cameras is the published answer: a cat in front of *any*
        camera is present. Taken here rather than in the watcher because no watcher can
        see the others — and they all share this one bridge instance, so the union is a
        plain dictionary read rather than anything that needs coordinating.

        Only changes are published. Presence is evaluated every analysed frame, several
        times a second per source, and a retained publish per frame would be the
        noisiest thing on the broker while saying nothing new.
        """
        if self._publisher is None:
            return
        incoming = set(label_uuids)
        if self._presence_by_source.get(source_uuid) == incoming:
            return
        was_present = self._present_subjects()
        self._presence_by_source[source_uuid] = incoming
        for subject_uuid in was_present ^ self._present_subjects():
            self._publish_subject_presence(subject_uuid)

    def forget_source_presence(self, source_uuid: str) -> None:
        """Drop a stopped source's contribution. Without this its last sighting would
        pin a subject present forever — the watcher is gone, so nothing will ever
        arrive to contradict it."""
        if source_uuid not in self._presence_by_source:
            return
        was_present = self._present_subjects()
        self._presence_by_source.pop(source_uuid, None)
        for subject_uuid in was_present ^ self._present_subjects():
            self._publish_subject_presence(subject_uuid)

    def _present_subjects(self) -> Set[str]:
        return set().union(*self._presence_by_source.values()) if self._presence_by_source else set()

    def _publish_subject_presence(self, subject_uuid: str) -> None:
        if self._publisher is None:
            return
        slug = self._subject_slugs.get(subject_uuid)
        if slug is None:
            return
        state = "ON" if subject_uuid in self._present_subjects() else "OFF"
        self._publisher.publish(f"{subject_base_topic(slug)}/state", state, retain=True)

    async def publish_system(self, *, watchers_running: int) -> None:
        """The `purrview_system` device's sensors — app-wide figures owned by no
        camera, so no watcher is in a position to publish them.

        Storage comes from `/api/storage/usage`, which walks directories and is
        deliberately not part of `/status` for that reason. Called once per reconcile
        pass rather than per frame, which is the right order of magnitude for a number
        that moves as clips are written and retention sweeps run.

        Failures are swallowed: the app device is diagnostics, and a storage walk
        erroring must not take the reconcile — and therefore every entity's discovery —
        down with it.
        """
        if self._publisher is None:
            return
        payload: Dict[str, Any] = {
            "watchersRunning": watchers_running,
            # Counted from the transitions this bridge has already published rather
            # than queried: `list_watch_events` has no "still open" filter, and an
            # event's row is not even written until an action commits to filing it, so
            # the DB is the wrong place to ask. This is also the number that agrees
            # with what HA's own binary_sensors are showing.
            "openEvents": len(self._open_rules),
            "haEntities": self._entity_count,
        }
        try:
            from db_client.api.recordings_api import RecordingsApi

            usage = await asyncio.to_thread(RecordingsApi(self._db_api_client).get_storage_usage)
            payload["storageBytes"] = sum(
                int(getattr(getattr(usage, bucket, None), "bytes", 0) or 0)
                for bucket in ("recordings", "timelapse", "thumbnail_cache")
            )
        except Exception:
            logger.debug("Storage usage unavailable for the MQTT system topic", exc_info=True)
        self._publisher.publish("system", payload, retain=True)

    def publish_telemetry(self, *, profile_uuid: str, source_uuid: str, payload: Dict[str, Any]) -> None:
        """The diagnostic sensors' single JSON topic. Caller decides the cadence —
        this is per analysed frame's worth of data, which is far more often than
        anything needs to see it."""
        if self._publisher is None:
            return
        topic = self._slugs.telemetry(profile_uuid, source_uuid)
        if topic is not None:
            self._publisher.publish(topic, payload, retain=True)

    def set_watcher_availability(self, profile_name: str, online: bool) -> None:
        """A stopped watcher's entities should read `unavailable`, not freeze at a
        stale value — this is the per-watcher half of `availability_mode: all`.

        Recorded before the connection check, and re-asserted by every reconcile pass,
        because a watcher starts before the bridge finishes connecting: dropping the
        call when `_publisher is None` left the availability topic unwritten, and since
        nothing else republished it, every entity on every source device sat at
        `unavailable` until the watcher happened to restart.
        """
        self._watcher_availability[slugify(profile_name)] = online
        self._publish_watcher_availability()

    def _publish_watcher_availability(self) -> None:
        """Idempotent, and cheap: a handful of retained one-word messages."""
        if self._publisher is None:
            return
        for slug, online in self._watcher_availability.items():
            self._publisher.publish(f"watcher/{slug}/status", "online" if online else "offline", retain=True)
