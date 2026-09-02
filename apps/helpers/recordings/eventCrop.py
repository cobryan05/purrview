"""Card art cropped to what an event was actually about.

An event's wide snapshot answers "where in the room?", which is worth knowing once. The
grid asks a different question — "what was this?" — and a wall of identical shots of the
same room cannot answer it: every card on a camera looks the same, so the picture
carries no information and the user is reading the text instead.

This module owns the geometry both producers share, so the crop the watcher writes live
and the crop the backfill cuts from a clip cannot drift apart:

* **the watcher**, from `ActionContext.tracks` at the opening edge
  (`_watch_event_recorder`), and
* **the backfill**, from a clip's `.boxes.json` sidecar
  (`scripts/backfill_event_crops.py`).

Everything here is in **normalised** coordinates — `0.0–1.0` of the frame — because that
is the only space the two callers agree in. The watcher's boxes are pixels in the
*analysis* frame (downscaled to `analysis_width`), the sidecar's are already normalised,
and the frame the crop is finally cut from is the camera's own full-resolution one. A
rect in any pixel space would be right for exactly one of the three.
"""

from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np

#: Cards are ~230–320 px wide in the grid and the page is retina, so the crop is written
#: at roughly 2× the largest card. Wider buys nothing a card can show and costs bytes on
#: every event; the clip is where the full-resolution record lives.
CROP_WIDTH = 640

#: Higher than the snapshot's 80. A crop is a magnified region of an already-compressed
#: source, so its artefacts are magnified too — the quality that reads as fine across a
#: whole room reads as mush across one animal.
CROP_QUALITY = 85

#: Breathing room around the union of the boxes, as a fraction of its size. A box cut
#: exactly to the subject reads as a mistake: the animal touches all four edges and the
#: picture looks like a crop of a bigger picture, which it is.
CROP_PAD = 0.28

#: Floor on how much of the frame the crop may show, as a fraction of each axis. Without
#: it a confident 30 px box on a distant camera becomes a full-card enlargement of thirty
#: pixels — technically the subject, visually a smear. 1/3 caps the zoom at 3×.
MIN_SPAN = 1.0 / 3.0

#: The event card's media box (`.event-card__media` is `aspect-ratio: 16 / 9`). The rect
#: is grown to this before cropping rather than letting `object-fit: cover` trim it,
#: because a trim the server does not know about silently cuts off the subject the rect
#: was built around.
CARD_ASPECT = 16.0 / 9.0

#: A rect this close to the whole frame is not a crop — it is the snapshot with extra
#: steps and a second file. Below this the card would show what it already shows, so the
#: producers write nothing and the UI falls back to the wide shot.
MIN_ZOOM = 1.15

Rect = Tuple[float, float, float, float]


def focus_rect(
    boxes: Iterable[Sequence[float]],
    frame_aspect: float,
    pad: float = CROP_PAD,
    min_span: float = MIN_SPAN,
    aspect: float = CARD_ASPECT,
) -> Optional[Rect]:
    """The region a card should show, from the boxes an event fired on.

    `boxes` are `(x, y, w, h)` normalised — the union of the *event's* tracks, not of
    everything the detector saw. That distinction is the whole point: including the
    bystanders puts the crop back where it started, framing the room.

    ⚠️ **`frame_aspect` is the source frame's own width/height, and it is not optional.**
    A normalised rect's `w/h` is not its shape: on a 1280×720 frame a rect of
    `w=h=0.5` is already 16:9 in pixels. Forcing `w/h == 16/9` in *normalised* space
    therefore produces a rect 16/9 × the frame's own aspect — 3.16:1 on a 16:9 camera,
    which is a letterbox strip with the animal's head cut off. The target here is
    `aspect / frame_aspect`, and the first back-fill run shipped 883 letterboxed crops
    before this was noticed.

    Returns `(x, y, w, h)` normalised and clamped inside the frame, or None when there
    is nothing to frame. The rect is always at least `min_span` of each axis and always
    the card's shape *in pixels*, so the caller never has to reason about either.
    """
    boxes = [b for b in boxes if b is not None and len(b) >= 4 and b[2] > 0 and b[3] > 0]
    if not boxes:
        return None

    x1 = min(float(b[0]) for b in boxes)
    y1 = min(float(b[1]) for b in boxes)
    x2 = max(float(b[0]) + float(b[2]) for b in boxes)
    y2 = max(float(b[1]) + float(b[3]) for b in boxes)

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w = max((x2 - x1) * (1.0 + pad), min_span)
    h = max((y2 - y1) * (1.0 + pad), min_span)

    # Grow the short side to the card's aspect. Growing rather than cropping the long
    # one: the boxes are the thing that must survive, and trimming to fit can cut them.
    target = aspect / max(frame_aspect, 1e-6)
    if w / h < target:
        w = h * target
    else:
        h = w / target

    # Shrink *both* axes by the same factor when either overflows the frame. Capping
    # them independently (`min(w, 1.0)`, `min(h, 1.0)`) is the obvious spelling and it
    # silently undoes the aspect fitting above the moment one side saturates — which is
    # exactly the common case, a wide rect on a frame that is only so wide.
    overflow = max(w, h, 1.0)
    w, h = w / overflow, h / overflow

    # Slide inside the frame rather than shrinking: a subject at the edge stays whole,
    # it just stops being centred — which is what a photographer would do too.
    x = min(max(cx - w / 2.0, 0.0), 1.0 - w)
    y = min(max(cy - h / 2.0, 0.0), 1.0 - h)
    return (x, y, w, h)


def is_worth_cropping(rect: Optional[Rect], min_zoom: float = MIN_ZOOM) -> bool:
    """Whether this rect shows meaningfully less than the whole frame.

    Guards against writing a second file that looks like the first. Measured on the
    wider axis so a rect that fills the width but not the height still counts as no
    zoom at all.
    """
    if rect is None:
        return False
    _x, _y, w, h = rect
    return max(w, h) > 0 and (1.0 / max(w, h)) >= min_zoom


def crop_to_rect(frame: np.ndarray, rect: Rect, width: int = CROP_WIDTH) -> np.ndarray:
    """Cut `rect` out of `frame` and size it for a card.

    **Give this the camera's frame, not the analysis one.** Both are valid inputs — the
    rect is normalised — but the whole reason the crop is a separate file is that it is
    taken before the downscale, and a crop of the 480 px analysis frame is a 2–3×
    enlargement of something already thrown away.

    Never upscales: a rect smaller than `width` in the source is written at its own size
    rather than interpolated up to a card, which adds bytes and no detail.
    """
    from apps.helpers.imageUtils import downscale_to_width

    height, frame_width = frame.shape[:2]
    x, y, w, h = rect
    x0 = max(0, min(int(round(x * frame_width)), frame_width - 1))
    y0 = max(0, min(int(round(y * height)), height - 1))
    x1 = max(x0 + 1, min(int(round((x + w) * frame_width)), frame_width))
    y1 = max(y0 + 1, min(int(round((y + h) * height)), height))
    return downscale_to_width(frame[y0:y1, x0:x1], width)


def normalise_boxes(tracks: Iterable[Any], frame_width: int, frame_height: int) -> list:
    """`Track`/`Detection` pixel corners → normalised `(x, y, w, h)`.

    Both carry `bbox` as absolute `(x1, y1, x2, y2)` in whatever frame they were
    produced against, which for a watcher is the analysis frame. Anything without a
    usable box is dropped rather than raising: one malformed track must not cost the
    event its card art.
    """
    if frame_width <= 0 or frame_height <= 0:
        return []
    out = []
    for track in tracks:
        try:
            x1, y1, x2, y2 = (float(v) for v in getattr(track, "bbox", track))
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((x1 / frame_width, y1 / frame_height, (x2 - x1) / frame_width, (y2 - y1) / frame_height))
    return out
