"""Per-rule suppression of detections that never move.

A false positive on a fixed object (a coat rack read as a person) re-fires a rule every
time the detector flickers, for as long as the object is there. `StationaryFilter`
remembers where a rule's subject boxes sit; once a spot has held a box for
`ignore_after_s`, tracks at that spot are dropped from the rule's input. Boxes anywhere
else are unaffected.

Spots are matched by IoU against the box that first created them, never a moving
average, so something walking slowly across the frame cannot drag a spot along with it.
A spot nothing has matched for `ignore_after_s` is forgotten.

Pure and in-memory: a restarted watcher relearns its spots.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from ..tracking.tracker import Track
from .geometry import BBox, iou

#: IoU with a spot's first box above which a track counts as the same spot.
STATIONARY_SPOT_IOU = 0.5


@dataclass
class _Spot:
    group: str
    bbox: BBox
    first_seen: float
    last_seen: float


class StationaryFilter:
    """Learns where one rule's subject boxes stay put, and hides them once they have.

    `is_subject` picks the tracks spots are learned from (the rule's subject slot); only
    those are ever hidden. `group_of` maps a label to its group, so a box that flips
    between "Person" and a child label stays one spot.
    """

    def __init__(
        self,
        ignore_after_s: float,
        is_subject: Callable[[Track], bool],
        group_of: Callable[[Optional[str]], str],
        spot_iou: float = STATIONARY_SPOT_IOU,
    ) -> None:
        self.ignore_after_s = float(ignore_after_s)
        self._is_subject = is_subject
        self._group_of = group_of
        self._spot_iou = spot_iou
        self._spots: List[_Spot] = []

    def _spot_for(self, group: str, bbox: BBox) -> Optional[_Spot]:
        best, best_iou = None, self._spot_iou
        for spot in self._spots:
            if spot.group != group:
                continue
            overlap = iou(spot.bbox, bbox)
            if overlap >= best_iou:
                best, best_iou = spot, overlap
        return best

    def apply(self, tracks: Sequence[Track], now: float) -> List[Track]:
        """The tracks the rule should see this frame.

        Only a track the detector matched this frame refreshes or creates a spot; a
        coasting track is hidden if its spot is ignored but teaches nothing.
        """
        self._spots = [s for s in self._spots if now - s.last_seen < self.ignore_after_s]
        kept: List[Track] = []
        for track in tracks:
            if not self._is_subject(track):
                kept.append(track)
                continue
            group = self._group_of(track.label_uuid)
            spot = self._spot_for(group, track.bbox)
            if track.matched_this_update:
                if spot is None:
                    spot = _Spot(group=group, bbox=tuple(track.bbox), first_seen=now, last_seen=now)
                    self._spots.append(spot)
                else:
                    spot.last_seen = now
            if spot is not None and now - spot.first_seen >= self.ignore_after_s:
                continue
            kept.append(track)
        return kept
