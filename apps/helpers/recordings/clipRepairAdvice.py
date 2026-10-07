"""Findings that suggest turning a camera's clip repair on, and the decisions about them.

`repair_corruption` is deliberately a switch the user throws, not something the app
infers: a re-encode costs real CPU and only the user knows whether a camera's picture
bothers them. But the thing the switch asks them to predict — "does this camera lose
packets?" — is exactly what the repair path already measures, so a camera that is
visibly breaking up can say so instead of waiting to be guessed at.

Two states are tracked per source and both live in memory:

*findings*
    "this camera's last clip decoded with N errors". Written by the probe, read by
    `StreamWatcherTask.get_notices()`. Memory-resident because `/status` is polled by
    the notices SSE loop every few seconds and must not touch the DB.

*decided*
    Sources that need no further advice — the user turned repair on, or said no. Only
    the "no" is persisted, to the settings store under
    `CLIP_REPAIR_DISMISSED_SETTING_KEY` and read back once at watcher start: a "yes"
    lives on the source row, which a restart reads anyway, and recording it as a
    decline would silence a camera whose repair is later switched off again.

Probing is bounded by `MAX_PROBES_PER_SOURCE`. A clean camera is scanned a few times and
then left alone for the life of the process: the question is "is this camera lossy", which
a handful of clips answers, and a decode scan of every clip forever to re-answer it would
cost more than the repair it is trying to recommend.
"""

import json
import logging
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

#: Settings-store key holding the source uuids the user has answered "no" for.
#: A JSON list of uuids. Mirrored in `shared/api/settings.js` consumers by name only —
#: nothing in the frontend reads it directly, the notice carries the decision.
CLIP_REPAIR_DISMISSED_SETTING_KEY = "clip_repair_dismissed_sources"

#: Clips scanned per source before the probe gives up on finding damage. Deliberately
#: small: a lossy camera shows errors on its first damaged clip, and the cost of being
#: wrong is a suggestion that never appears rather than a wrong repair.
MAX_PROBES_PER_SOURCE = 3

#: A probe needs more evidence than a repair does. `MIN_ERRORS_TO_REPAIR` decides whether
#: an already-opted-in clip is worth re-encoding; this decides whether to interrupt the
#: user, so it sits above the handful of errors that a single damaged frame produces.
MIN_ERRORS_TO_SUGGEST = 25


class _Advice:
    def __init__(self) -> None:
        self._findings: Dict[str, dict] = {}
        self._decided: Set[str] = set()
        #: The subset of `_decided` that was an explicit "no". Only this is persisted:
        #: a "yes" is recorded on the source row itself, and writing it here too would
        #: mean a camera whose repair is later turned off is never suggested again.
        self._dismissed: Set[str] = set()
        self._probes: Dict[str, int] = {}

    def load_dismissed(self, raw: Optional[str]) -> None:
        """Seed the decided set from the settings store's JSON list. Never raises."""
        if not raw:
            return
        try:
            uuids = json.loads(raw)
        except (TypeError, ValueError):
            logger.warning("Ignoring unparseable %s", CLIP_REPAIR_DISMISSED_SETTING_KEY)
            return
        if isinstance(uuids, list):
            loaded = {str(u) for u in uuids}
            self._decided.update(loaded)
            self._dismissed.update(loaded)

    def dismissed_json(self) -> str:
        return json.dumps(sorted(self._dismissed))

    def wants_probe(self, source_uuid: Optional[str]) -> bool:
        """Whether this source is still worth a scan-only pass."""
        if not source_uuid or source_uuid in self._decided:
            return False
        if source_uuid in self._findings:
            return False
        return self._probes.get(source_uuid, 0) < MAX_PROBES_PER_SOURCE

    def record_probe(self, source_uuid: str, errors: int, clip_name: str, source_name: str = "") -> bool:
        """Note one probe's result. True when it became a finding worth surfacing."""
        self._probes[source_uuid] = self._probes.get(source_uuid, 0) + 1
        if errors < MIN_ERRORS_TO_SUGGEST or source_uuid in self._decided:
            return False
        self._findings[source_uuid] = {
            "sourceUuid": source_uuid,
            "sourceName": source_name,
            "errors": errors,
            "clip": clip_name,
        }
        return True

    def decide(self, source_uuid: str) -> None:
        """Mark a source answered, without persisting anything. For "repair turned on"."""
        self._decided.add(source_uuid)
        self._findings.pop(source_uuid, None)

    def dismiss(self, source_uuid: str) -> None:
        """Mark a source declined, which is the answer that outlives the process."""
        self._dismissed.add(source_uuid)
        self.decide(source_uuid)

    def findings(self) -> list:
        return list(self._findings.values())

    def reset(self) -> None:
        """Drop all state. For tests, which must not inherit another test's probes."""
        self._findings.clear()
        self._decided.clear()
        self._dismissed.clear()
        self._probes.clear()


_advice = _Advice()


def load_dismissed(raw: Optional[str]) -> None:
    _advice.load_dismissed(raw)


def dismissed_json() -> str:
    return _advice.dismissed_json()


def wants_probe(source_uuid: Optional[str]) -> bool:
    return _advice.wants_probe(source_uuid)


def record_probe(source_uuid: str, errors: int, clip_name: str, source_name: str = "") -> bool:
    return _advice.record_probe(source_uuid, errors, clip_name, source_name=source_name)


def decide(source_uuid: str) -> None:
    _advice.decide(source_uuid)


def dismiss(source_uuid: str) -> None:
    _advice.dismiss(source_uuid)


def findings() -> list:
    return _advice.findings()


def reset() -> None:
    _advice.reset()


def notice_for(finding: Dict[str, Any]) -> dict:
    """Render one finding as a notices-bar entry with its two answers.

    The urls are same-origin `/api/` paths on the tasks server, which owns both the
    finding and the decision — see `noticesBar.js` for the constraint the frontend
    enforces on them.
    """
    name = finding.get("sourceName") or "A camera"
    uuid = finding["sourceUuid"]
    return {
        "key": f"clip-repair-suggestion:{uuid}",
        "level": "warning",
        "message": (
            f"{name} dropped packets in its last clip ({finding['errors']} decoder errors). "
            "Turning on Repair Corrupt Frames for it holds the last good frame instead of "
            "letting the picture break up, at the cost of re-encoding damaged clips."
        ),
        "actions": [
            {"label": "Turn on repair", "url": f"/api/clip-repair/enable/{uuid}"},
            {"label": "Not now", "url": f"/api/clip-repair/dismiss/{uuid}"},
        ],
    }
