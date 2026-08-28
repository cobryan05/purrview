"""Measure the ingest lag between a clip's pixels and its detection sidecar.

    docker exec -w /project encounterwatch_mono_dev \
        python scripts/measure_overlay_lag.py /data/recordings/clips/<day>/<clip>.mp4 [...]

Set `WATCH_OVERLAY_LAG_S` to the number this reports (see `OVERLAY_LAG_S` in
`apps/tasks_server/tasks/_watch_event_recorder.py`).

**Why a measurement and not a constant.** A buffer-cut clip's timeline is the relay's
record clock — when a packet reached MediaMTX from the camera. A sidecar sample's `at`
is the watcher's fetch clock — when its own copy of that same picture came back out of
ffmpeg, a decoder and a jitter buffer later. Nothing in the rawvideo pipe carries the
capture instant, so the watcher cannot know it; the gap is whatever this deployment's
ingest path costs, and it makes every box render that much late. That is a number to
measure per install, not to hardcode.

**How.** For a candidate offset `d`, every video frame at time `vt` is scored against
the boxes the sidecar reports for `vt + d`: the score is the share of that frame's
inter-frame motion energy falling inside those boxes. Boxes in the right place sit on
top of the thing that is moving, so the offset maximising that share is where the boxes
actually belong. A positive result means the sidecar's timestamps run that far ahead of
the picture they describe — i.e. boxes are drawn that late.

Pass several clips and take the median; a single clip of a mostly-still subject carries
little alignment signal, and the peak will be shallow. Clips whose subject barely moves
are worth discarding rather than averaging in — check the peak stands clear of the
baseline before trusting it.
"""

import json
import statistics
import sys

import cv2
import numpy as np

#: Search range and step, in seconds. Wide enough to cover a badly buffered RTSP path
#: and to show the peak falling away on both sides — a "best" at the edge of the range
#: means the range was too narrow, not that the lag is that large.
SEARCH_S = 6.0
STEP_S = 0.2

#: Analysis resolution for the motion map. The measurement is about *time*, not
#: geometry, so a thumbnail carries every bit of signal the full frame does at a
#: fraction of the decode cost.
GRID_W, GRID_H = 160, 90

#: Frames whose total inter-frame difference is below this carry no alignment signal —
#: a static scene scores identically at every offset and only flattens the peak.
MOTION_FLOOR = 500

#: How far the best offset's in-box motion share must stand above the median across all
#: offsets before the reading counts. Below this the "peak" is the argmax of noise.
PEAK_OVER_BASELINE = 1.15


def _boxes_at(frames, max_stale, t):
    """The sidecar's boxes in effect at `t`, mirroring the player's own hold rule."""
    found = None
    for frame in frames:
        if frame["t"] > t:
            break
        found = frame
    if not found or t - found["t"] > max_stale:
        return None
    return found["boxes"]


def _motion_frames(path):
    """(video time, motion map) for every frame with something happening in it."""
    cap = cv2.VideoCapture(path)
    samples = []
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vt = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (GRID_W, GRID_H))
        if prev is not None:
            delta = cv2.absdiff(small, prev).astype(np.float32)
            if delta.sum() > MOTION_FLOOR:
                samples.append((vt, delta))
        prev = small
    cap.release()
    return samples


def measure(path, verbose=True):
    """Best offset for one clip, or None when it carries too little motion to tell."""
    try:
        doc = json.load(open(path + ".boxes.json"))
    except OSError:
        print(f"{path}: no sidecar beside it")
        return None

    frames = doc.get("frames") or []
    max_stale = doc.get("maxStalenessS", 1.5)
    samples = _motion_frames(path)
    if verbose:
        already = doc.get("lagS", 0.0)
        print(
            f"{path.split('/')[-1]}: {len(samples)} moving frames, "
            f"{len(frames)} sidecar samples, already compensated by {already}s"
        )
    if not samples or not frames:
        return None

    scored = []
    steps = int(SEARCH_S / STEP_S)
    for i in range(-steps, steps + 1):
        offset = i * STEP_S
        shares = []
        for vt, motion in samples:
            boxes = _boxes_at(frames, max_stale, vt + offset)
            if not boxes:
                continue
            mask = np.zeros_like(motion)
            for box in boxes:
                x1 = max(0, int(box["x"] * GRID_W))
                y1 = max(0, int(box["y"] * GRID_H))
                x2 = max(0, int((box["x"] + box["w"]) * GRID_W))
                y2 = max(0, int((box["y"] + box["h"]) * GRID_H))
                mask[y1:y2, x1:x2] = 1.0
            total = motion.sum()
            if total > 0:
                shares.append(float((motion * mask).sum() / total))
        # Too few overlapping frames to mean anything — usually an offset that has
        # walked the whole sidecar off the end of the clip.
        if len(shares) < 10:
            continue
        scored.append((offset, sum(shares) / len(shares)))

    if not scored:
        return None
    best_offset, best_score = max(scored, key=lambda pair: pair[1])
    baseline = statistics.median(score for _, score in scored)
    # A peak that barely clears the median is where the subject moved too little (or too
    # much of the frame moved) for the boxes to localise it. Such a clip's "best" offset
    # is the argmax of noise, and it is roughly uniform across the search range — so
    # averaging it in does not merely add variance, it drags the estimate toward zero.
    trustworthy = best_score >= baseline * PEAK_OVER_BASELINE
    if verbose:
        print(f"  best {best_offset:+.1f}s (in-box motion {best_score:.3f} vs {baseline:.3f} median)")
        if not trustworthy:
            print("  ⚠ peak barely clears the baseline — too little motion here to trust; excluded")
    return best_offset, trustworthy


def main(paths):
    measured = [r for r in (measure(p) for p in paths) if r is not None]
    trusted = [offset for offset, ok in measured if ok]

    if not trusted:
        print(f"\nNo usable signal: {len(measured)} clip(s) measured, none with a clear peak.")
        print("Feed it more clips — ones where the subject actually crosses the frame.")
        return 1

    median = statistics.median(trusted)
    print(
        f"\nWATCH_OVERLAY_LAG_S={median:.1f}   (median of {len(trusted)} clear clip(s)" f" of {len(measured)} measured)"
    )
    # One clip is an anecdote. The sidecar is sampled at well under a frame a second, so
    # a single reading's resolution is no better than the hold between samples, and the
    # spread across clips is the only honest guide to whether the median means anything.
    if len(trusted) < 3:
        print("⚠ Too few clear clips to trust this number — treat it as a hint, not a value.")
    elif max(trusted) - min(trusted) > 1.0:
        print(
            f"⚠ Readings range {min(trusted):+.1f}s..{max(trusted):+.1f}s. A single constant "
            "cannot correct a lag that varies this much; the median is the best of a bad job."
        )
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
