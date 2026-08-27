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


# ── Discovery payload generation (pure) ──────────────────────────────────────────────


def build_source_discovery(profile: ProfileView, source: SourceView, prefix: str) -> Dict[str, Any]:
    """The one device-discovery message for a source's HA device: itself, plus every
    rule currently scoped to it. HA's device-discovery schema bundles a whole device's
    components into one retained message (`cmps`), so adding/removing/renaming a rule
    is a matter of recomputing this and republishing — see `reconcile_discovery`.
    """
    topic_source = _topic_source(source)
    topic_profile = slugify(profile.name)
    base = f"{prefix}/{topic_source}"
    watcher_topic = f"{prefix}/watcher/{topic_profile}/status"

    cmps: Dict[str, Any] = {}
    for rule in profile.rules:
        if not rule.enabled:
            continue
        if rule.source_uuids and source.uuid not in rule.source_uuids:
            continue
        topic_rule = _topic_rule(rule)
        rule_base = f"{base}/rule/{topic_rule}"
        # The existing, documented scheme (mqttPublishAction.py) — profile-scoped,
        # unchanged by this bridge.
        event_topic = f"{prefix}/{topic_profile}/{topic_source}/event/{topic_rule}"

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
        "topic": f"{prefix}/{topic_profile}/{topic_source}/image",
    }
    cmps["tracked_count"] = {
        "p": "sensor",
        "unique_id": f"whiskerwatcher_{source.uuid}_tracked_count",
        "name": f"{source.name} tracked objects",
        "state_class": "measurement",
        "state_topic": f"{base}/telemetry",
        "value_template": "{{ value_json.trackedCount }}",
    }
    cmps["health"] = {
        "p": "binary_sensor",
        "unique_id": f"whiskerwatcher_{source.uuid}_health",
        "name": f"{source.name} problem",
        "device_class": "problem",
        "entity_category": "diagnostic",
        "state_topic": f"{base}/telemetry",
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
            "state_topic": f"{base}/telemetry",
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


def reconcile_discovery(
    previous_topics: FrozenSet[str], desired: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Optional[Dict[str, Any]]], FrozenSet[str]]:
    """What to publish this pass: every desired message, plus an empty-payload
    retraction (`None`, published as `""`) for every config topic that existed last
    pass but is no longer desired — a rule/source/profile that was deleted.

    Returns `(to_publish, new_previous_topics)` rather than mutating anything, so the
    caller decides when — and whether — to persist the second value.
    """
    to_publish: Dict[str, Optional[Dict[str, Any]]] = dict(desired)
    for gone_topic in previous_topics - desired.keys():
        to_publish[gone_topic] = None
    return to_publish, frozenset(desired.keys())


def build_seed_publishes(profiles: Sequence[ProfileView], counts: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Topic → lifetime count, for every (source, rule) count both present in history
    and still known to current config. A count for a uuid no longer in any profile
    (a deleted rule or source) has no topic to seed onto — its retained state, if any,
    goes away with the rest of that entity's discovery retraction, not through here.
    """
    slug_by_source = {s.uuid: _topic_source(s) for p in profiles for s in p.sources}
    slug_by_rule = {r.uuid: _topic_rule(r) for p in profiles for r in p.rules}
    out: Dict[str, int] = {}
    for row in counts:
        source_slug = slug_by_source.get(row["source_uuid"])
        rule_slug = slug_by_rule.get(row["rule_uuid"])
        if not source_slug or not rule_slug:
            continue
        out[f"{source_slug}/rule/{rule_slug}/count"] = row["count"]
    return out


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
        self._last_discovery_topics: FrozenSet[str] = frozenset()

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
        desired = build_all_discovery(profiles, self._prefix)
        to_publish, self._last_discovery_topics = reconcile_discovery(self._last_discovery_topics, desired)
        for topic, payload in to_publish.items():
            self._publisher.publish(topic, "" if payload is None else payload, retain=True, absolute=True)

    async def seed_counters(self) -> None:
        if self._publisher is None:
            return
        from db_client.api.watch_profiles_api import WatchProfilesApi

        profiles = await self._load_profiles()
        counts_resp = await asyncio.to_thread(WatchProfilesApi(self._db_api_client).get_watch_event_counts)
        counts = [
            {"source_uuid": c.source_uuid, "rule_uuid": c.rule_uuid, "count": c.count}
            for c in (counts_resp.counts or [])
        ]
        for topic, count in build_seed_publishes(profiles, counts).items():
            self._publisher.publish(topic, count, retain=True)

    def set_watcher_availability(self, profile_name: str, online: bool) -> None:
        """A stopped watcher's entities should read `unavailable`, not freeze at a
        stale value — this is the per-watcher half of `availability_mode: all`."""
        if self._publisher is None:
            return
        topic = f"watcher/{slugify(profile_name)}/status"
        self._publisher.publish(topic, "online" if online else "offline", retain=True)
