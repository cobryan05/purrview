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
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..actions.mqttPublisher import AsyncMqttPublisher, load_mqtt_broker_settings
from ..watch.templating import slugify

logger = logging.getLogger(__name__)

HA_DISCOVERY_PREFIX = "homeassistant"
ORIGIN = {"name": "WhiskerWatcher"}
APP_DEVICE_ID = "whiskerwatcher_system"
APP_CONFIG_TOPIC = f"{HA_DISCOVERY_PREFIX}/device/{APP_DEVICE_ID}/config"


def source_device_id(source_uuid: str) -> str:
    return f"whiskerwatcher_{source_uuid}"


def source_config_topic(source_uuid: str) -> str:
    return f"{HA_DISCOVERY_PREFIX}/device/{source_device_id(source_uuid)}/config"


# ── Profile snapshot — the "profile fixture" the discovery functions are tested against ──


@dataclass(frozen=True)
class RuleView:
    uuid: str
    name: str
    enabled: bool = True
    #: Empty means "every source in the profile" — matches `InteractionRule`.
    source_uuids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceView:
    uuid: str
    name: str
    mqtt_name: Optional[str] = None
    enabled: bool = True


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
        if not rule.enabled:
            continue
        if rule.source_uuids and source.uuid not in rule.source_uuids:
            continue
        topic_rule = _topic_rule(rule)
        rule_base = f"{prefix}/{rule_base_topic(topic_profile, topic_source, topic_rule)}"
        # The existing, documented scheme (mqttPublishAction.py) — profile-scoped,
        # unchanged by this bridge. Built off the same `base` as the bridge's own
        # topics now that those are profile-scoped too, which is the point: the two
        # schemes share a namespace rather than sitting in parallel ones.
        event_topic = f"{base}/event/{topic_rule}"

        # unique_id is uuid-based throughout: renaming the rule changes rule_base and
        # event_topic (cosmetic, human-readable) but never these, so HA keeps treating
        # it as the same entity — see HA_INTEGRATION_PLAN.md WP-2 step 5a.
        cmps[f"{rule.uuid}_state"] = {
            "p": "binary_sensor",
            "unique_id": f"whiskerwatcher_{rule.uuid}_state",
            "name": rule.name,
            "device_class": "occupancy",
            "state_topic": f"{rule_base}/state",
            "payload_on": "ON",
            "payload_off": "OFF",
            "json_attributes_topic": f"{rule_base}/attributes",
        }
        cmps[f"{rule.uuid}_count"] = {
            "p": "sensor",
            "unique_id": f"whiskerwatcher_{rule.uuid}_count",
            "name": f"{rule.name} count",
            # The single most important setting here: HA purges raw history after
            # ~10 days but keeps long-term statistics forever for a sensor that
            # declares a state_class. Without it, "events per day last year" is gone.
            "state_class": "total_increasing",
            "state_topic": f"{rule_base}/count",
        }
        cmps[f"{rule.uuid}_last_seen"] = {
            "p": "sensor",
            "unique_id": f"whiskerwatcher_{rule.uuid}_last_seen",
            "name": f"{rule.name} last seen",
            "device_class": "timestamp",
            "state_topic": f"{rule_base}/attributes",
            "value_template": "{{ value_json.lastSeen }}",
        }
        cmps[f"{rule.uuid}_last_duration"] = {
            "p": "sensor",
            "unique_id": f"whiskerwatcher_{rule.uuid}_last_duration",
            "name": f"{rule.name} last duration",
            "state_class": "measurement",
            "device_class": "duration",
            "unit_of_measurement": "s",
            "state_topic": f"{rule_base}/attributes",
            "value_template": "{{ value_json.lastDurationS }}",
        }
        cmps[f"{rule.uuid}_peak_confidence"] = {
            "p": "sensor",
            "unique_id": f"whiskerwatcher_{rule.uuid}_peak_confidence",
            "name": f"{rule.name} peak confidence",
            "state_class": "measurement",
            "state_topic": f"{rule_base}/attributes",
            "value_template": "{{ value_json.peakConfidence }}",
        }
        cmps[f"{rule.uuid}_event"] = {
            "p": "event",
            "unique_id": f"whiskerwatcher_{rule.uuid}_event",
            "name": rule.name,
            "event_types": [rule.name],
            "state_topic": event_topic,
        }

    cmps["camera"] = {
        "p": "camera",
        "unique_id": f"whiskerwatcher_{source.uuid}_camera",
        "name": source.name,
        "topic": f"{base}/image",
    }
    cmps["tracked_count"] = {
        "p": "sensor",
        "unique_id": f"whiskerwatcher_{source.uuid}_tracked_count",
        "name": f"{source.name} tracked objects",
        "state_class": "measurement",
        "state_topic": telemetry_state_topic,
        "value_template": "{{ value_json.trackedCount }}",
    }
    cmps["health"] = {
        "p": "binary_sensor",
        "unique_id": f"whiskerwatcher_{source.uuid}_health",
        "name": f"{source.name} problem",
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
            "unique_id": f"whiskerwatcher_{source.uuid}_{key}",
            "name": f"{source.name} {key.replace('_', ' ')}",
            "entity_category": "diagnostic",
            "state_class": "measurement",
            "state_topic": telemetry_state_topic,
            "value_template": f"{{{{ value_json.{json_path} }}}}",
        }
        if unit:
            entry["unit_of_measurement"] = unit
        cmps[key] = entry

    return {
        "dev": {"ids": source.uuid, "name": source.name},
        "o": dict(ORIGIN),
        # Both must be online: the app-wide LWT and this source's watcher. Either
        # going offline should be visible as "unavailable", not a frozen last value.
        "availability": [{"topic": f"{prefix}/status"}, {"topic": watcher_topic}],
        "availability_mode": "all",
        "cmps": cmps,
    }


def build_app_discovery(prefix: str) -> Dict[str, Any]:
    """The `whiskerwatcher_system` device: app-wide diagnostics that belong to no
    single camera."""
    return {
        "dev": {"ids": APP_DEVICE_ID, "name": "WhiskerWatcher"},
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
        },
    }


def build_all_discovery(profiles: Sequence[ProfileView], prefix: str) -> Dict[str, Dict[str, Any]]:
    """Every device-discovery message that should currently exist, keyed by config
    topic. Does not decide what to *retract* — see `reconcile_discovery`, which diffs
    this against what was published last time.
    """
    messages: Dict[str, Dict[str, Any]] = {}
    for profile in profiles:
        for source in profile.sources:
            if not source.enabled:
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

    Anything unparseable is skipped rather than raised on. A config we cannot read is
    one we will not retract — the same safe direction as reading nothing at all — and
    refusing to start the bridge over one malformed retained message would be a far
    worse trade.
    """
    components: Dict[str, Dict[str, str]] = {}
    for topic, payload in retained.items():
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
        self._prefix: str = "whiskerwatcher"
        self._last_discovery: DiscoveryState = DiscoveryState()
        # Refreshed on every reconcile, so a watcher publishing an event transition can
        # resolve its uuids to topics without a DB round trip on the hot path.
        self._slugs: SlugIndex = build_slug_index([])
        # (profile_uuid, source_uuid, rule_uuid) → lifetime count. Seeded from the DB
        # once at connect and incremented in memory thereafter: the alternative, a
        # COUNT(*) per event, puts a query on the event path to recompute a number this
        # process already knows.
        self._counts: Dict[Tuple[str, str, str], int] = {}

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
            await self.publish_all_discovery()

    async def _load_profiles(self) -> List[ProfileView]:
        from db_client.api.sources_api import SourcesApi
        from db_client.api.watch_profiles_api import WatchProfilesApi

        profiles_resp = await asyncio.to_thread(WatchProfilesApi(self._db_api_client).list_watch_profiles)
        sources_resp = await asyncio.to_thread(SourcesApi(self._db_api_client).get_sources)
        source_by_uuid = {s.uuid: s for s in (sources_resp.sources or [])}

        views: List[ProfileView] = []
        for profile in profiles_resp.profiles or []:
            sources = tuple(
                SourceView(
                    uuid=ps.source_uuid,
                    name=source_by_uuid[ps.source_uuid].name,
                    mqtt_name=ps.mqtt_name,
                    enabled=ps.enabled,
                )
                for ps in (profile.sources or [])
                if ps.source_uuid in source_by_uuid  # a source can be deleted out from under a profile
            )
            rules = tuple(
                RuleView(
                    uuid=r.uuid,
                    name=r.name,
                    enabled=r.enabled,
                    source_uuids=tuple(r.source_uuids_json or []),
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
        desired = build_all_discovery(profiles, self._prefix)
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
            {"source_uuid": c.source_uuid, "rule_uuid": c.rule_uuid, "count": c.count}
            for c in (counts_resp.counts or [])
        ]
        self._counts = build_counter_seed(profiles, counts)
        for topic, count in build_seed_publishes(profiles, counts).items():
            self._publisher.publish(topic, count, retain=True)

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
        if opened:
            key = (profile_uuid, source_uuid, rule_uuid)
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
        stale value — this is the per-watcher half of `availability_mode: all`."""
        if self._publisher is None:
            return
        topic = f"watcher/{slugify(profile_name)}/status"
        self._publisher.publish(topic, "online" if online else "offline", retain=True)
