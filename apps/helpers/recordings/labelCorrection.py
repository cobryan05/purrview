"""Correcting a watcher's label across a whole event, server-side.

The bulk counterpart to `labelCorrectionModal.js`. The browser's single-event path
harvests its frames by seeking a real `<video>` and encoding each one off a canvas,
which cannot be looped: it needs the tab foregrounded (`whenPageVisible`), one playhead
at a time, and several seconds per clip. Correcting forty events that way is minutes of
a tab nobody may touch. So a selection is corrected here instead, from the clip file and
its sidecar, by `CorrectEventLabelsTask`.

**The frame choice is deliberately simpler than the browser's.** `pickTrainingFrames`
in the modal ranks *hard moments* first — the frames a tracking blip pointed at, the
confidence extremes — and reserves slots so the entering and leaving poses survive.
That ranking costs a blips request per event and exists because a single correction is
a considered act on one clip. A bulk correction is a different bargain: the frames come
from an even, gap-thinned spread over the moments that actually hold the label. Stated
here so the difference reads as a decision rather than as an incomplete port.

Boxes in a sidecar are already normalised to the analysis frame (`detectionTrack._normalise`
writes x/y/w/h in 0..1), which is the shape `add_bbox` stores — so unlike the browser
path, nothing here converts pixel corners.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence

#: The remap value meaning "the detector saw something that wasn't there" — remove the
#: box rather than relabel it.
#:
#: ⚠️ Mirrors `NO_BBOX` in `labelCorrectionModal.js`, which is where the value is chosen
#: and sent from. No build step shares them; `tests/helpers/test_label_correction.py`
#: asserts the two sides agree rather than pinning either to a literal.
NO_BBOX = "__no_bbox__"

#: The remap value meaning "this label was right — collect the frames it hedged on".
#: A collection instruction, not a correction: the boxes pass through untouched.
#: Mirrored from JS exactly as `NO_BBOX` is.
WEAK = "__low_conf__"

#: Closest two harvested frames may sit in clip time. Consecutive frames of a tracked
#: animal are one pose weighted twice, so spacing them is what makes the set a spread.
DEFAULT_MIN_FRAME_GAP_S = 0.75

#: Frames one event's correction may harvest, before configuration.
DEFAULT_MAX_FRAMES = 8

#: Confidence at or above which a box is not worth collecting merely for being weak.
DEFAULT_WEAK_CONF_MAX = 0.6

#: Overlap at which two boxes of the same label in one frame are the same object. The
#: tracker's box and the detection behind it routinely both appear, and filing both
#: hands the trainer one object annotated twice.
DEFAULT_DEDUPE_IOU = 0.75


def _frames(document: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [f for f in ((document or {}).get("frames") or []) if isinstance(f, dict)]


def _time_of(frame: Dict[str, Any]) -> Optional[float]:
    t = frame.get("t")
    return float(t) if isinstance(t, (int, float)) else None


def weakest_conf(frame: Dict[str, Any], wanted: Iterable[str]) -> Optional[float]:
    """The lowest confidence any of these labels' boxes carries in one frame.

    The *lowest*, not the highest: a frame holding a confident box and a hedged one of
    the same label still holds a hedged one, and that box is the reason to collect it.

    Returns None when the frame holds none of them, or none carries a confidence.
    """
    wanted = set(wanted)
    worst: Optional[float] = None
    for box in frame.get("boxes") or []:
        if not isinstance(box, dict) or box.get("labelUuid") not in wanted:
            continue
        conf = box.get("conf")
        if not isinstance(conf, (int, float)):
            continue
        if worst is None or float(conf) < worst:
            worst = float(conf)
    return worst


def frames_holding_labels(
    document: Optional[Dict[str, Any]],
    label_uuids: Iterable[str],
    max_conf: Optional[float] = None,
) -> List[float]:
    """Clip times whose frame holds a box of any of these labels, ascending.

    With `max_conf`, only the frames whose *weakest* such box is below it — the
    `WEAK` case, where the label is right and the hedged moments are the harvest. A box
    carrying no confidence at all never qualifies.
    """
    wanted = set(label_uuids)
    if not wanted:
        return []
    times: List[float] = []
    for frame in _frames(document):
        t = _time_of(frame)
        if t is None:
            continue
        if max_conf is None:
            if any(isinstance(b, dict) and b.get("labelUuid") in wanted for b in frame.get("boxes") or []):
                times.append(t)
            continue
        worst = weakest_conf(frame, wanted)
        if worst is not None and worst < max_conf:
            times.append(t)
    return sorted(times)


def pick_training_frames(
    times: Sequence[float],
    max_frames: int = DEFAULT_MAX_FRAMES,
    min_gap_s: float = DEFAULT_MIN_FRAME_GAP_S,
) -> List[float]:
    """Thin a run of candidate moments to an even, well-spaced spread.

    Two passes, in this order and for this reason:

    1. **Thin by time.** Frames closer together than `min_gap_s` show the same pose, so
       they are dropped *before* the cap is applied — otherwise the cap is spent on
       whichever burst happened to be densest rather than on distinct frames.
    2. **Spread to the cap.** If more survive than the cap allows, take evenly across
       them. Dividing by `room - 1` is fenceposts, not buckets: it is what puts the last
       pick on the final frame rather than one step short of it, and the final frame is
       the leaving pose.
    """
    room = max(0, int(max_frames))
    if room == 0:
        return []
    spaced: List[float] = []
    for t in sorted(times):
        if not spaced or t - spaced[-1] >= min_gap_s:
            spaced.append(t)
    if len(spaced) <= room:
        return spaced
    if room == 1:
        return spaced[:1]
    step = (len(spaced) - 1) / (room - 1)
    picked = [spaced[round(i * step)] for i in range(room)]
    # Dedup after rounding: a step below 1 can land two indices on one frame, and the
    # cap is a ceiling on distinct pictures, not on seeks.
    return sorted(set(picked))


def sample_at(document: Optional[Dict[str, Any]], t: float) -> Optional[Dict[str, Any]]:
    """The sidecar frame describing clip time `t`, or None.

    Matches on the sample's own time rather than searching for the nearest: `t` here is
    always a time `frames_holding_labels` returned, so it *is* a sample's `t`. The
    round-trip through the player's timebase that makes the browser's lookup fiddly has
    no equivalent server-side.
    """
    for frame in _frames(document):
        if _time_of(frame) == t:
            return frame
    return None


def apply_label_remap(boxes: Optional[Sequence[Dict[str, Any]]], remap: Dict[str, str]) -> List[Dict[str, Any]]:
    """Rewrite each box's label to what the correction says it was.

    A `NO_BBOX` label's boxes are dropped: there is no correct label for a detection
    that should never have been made. A `WEAK` label's pass through untouched — it names
    the right label, and the instruction was about which frames to collect.

    A frame left with no boxes at all is the caller's problem, not an error here: that
    is what removing the only box looks like.
    """
    kept: List[Dict[str, Any]] = []
    for box in boxes or []:
        if not isinstance(box, dict):
            continue
        label = box.get("labelUuid")
        target = remap.get(label) if label else None
        if target == NO_BBOX:
            continue
        if target and target != WEAK:
            box = {**box, "labelUuid": target}
        kept.append(box)
    return kept


def _iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Overlap of two normalised x/y/w/h boxes."""
    ax1, ay1 = float(a.get("x", 0.0)), float(a.get("y", 0.0))
    ax2, ay2 = ax1 + float(a.get("w", 0.0)), ay1 + float(a.get("h", 0.0))
    bx1, by1 = float(b.get("x", 0.0)), float(b.get("y", 0.0))
    bx2, by2 = bx1 + float(b.get("w", 0.0)), by1 + float(b.get("h", 0.0))
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def dedupe_overlapping_boxes(
    boxes: Optional[Sequence[Dict[str, Any]]],
    iou_threshold: float = DEFAULT_DEDUPE_IOU,
) -> List[Dict[str, Any]]:
    """Drop same-label boxes that describe one object twice.

    A sidecar frame carries both the detections and the tracks, and after a remap
    collapses two labels onto one they can also become duplicates of each other. The
    more confident box is the one kept.
    """
    ordered = sorted(
        range(len(boxes or [])),
        key=lambda i: -float((boxes[i] or {}).get("conf") or 0.0),
    )
    kept_indices: List[int] = []
    for i in ordered:
        box = boxes[i]
        label = box.get("labelUuid")
        if label and any(
            boxes[k].get("labelUuid") == label and _iou(box, boxes[k]) >= iou_threshold for k in kept_indices
        ):
            continue
        kept_indices.append(i)
    # The caller's original ordering, not the confidence sort used to choose.
    return [boxes[i] for i in sorted(kept_indices)]


def event_correction_for(
    event: Any,
    remap: Dict[str, str],
    label_names: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """What a remap would do to an event row's own subject and target.

    An event whose subject was wrong reads wrong in the grid, groups under the wrong
    identity and answers the wrong filter, so the row is corrected alongside the frames.

    Neither `NO_BBOX` nor `WEAK` ever touches the row: the first says nothing about what
    actually happened, and the second confirms the identity already recorded.

    Returns None when the remap touches neither slot.

    Mirrors `eventCorrectionFor` in `labelCorrectionModal.js`, which drives the dialog's
    "also fix this event" line; the write itself happens here so a group of 383 events
    costs one task rather than 383 requests from the browser.
    """
    if not remap:
        return None
    names = label_names or {}
    patch: Dict[str, Any] = {}
    changes: List[Dict[str, str]] = []
    for slot, ref_key, name_key in (
        ("subject", "subject_ref", "subject_name"),
        ("target", "target_ref", "target_name"),
    ):
        ref = getattr(event, ref_key, None)
        if not ref or ref not in remap:
            continue
        target = remap[ref]
        if target in (NO_BBOX, WEAK):
            continue
        patch[ref_key] = target
        # The name travels with the ref, always: a row that filters as one label and
        # reads as another is the worst of the three states.
        patch[name_key] = names.get(target, target)
        changes.append(
            {
                "slot": slot,
                "from": getattr(event, name_key, None) or names.get(ref, ref),
                "to": names.get(target, target),
            }
        )
    return {"patch": patch, "changes": changes} if changes else None
