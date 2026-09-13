"""Frame-accurate pixel extraction from a recorded clip.

Split out of `HarvestBlipFramesTask` because a second caller (the blip crop-preview
endpoint) needs the same seek, not the JPEG bytes that task encodes it to — it crops
before encoding, and cropping an already-decoded frame twice is wasted work.
"""

from typing import Optional

import numpy as np


def grab_frame_array(clip_path: str, at_s: float) -> Optional[np.ndarray]:
    """One BGR frame of the clip, at `at_s` seconds from its start.

    Seeks to shortly *before* the target and then reads forward. A bare
    `CAP_PROP_POS_MSEC` seek lands on the nearest keyframe, which on a clip with a
    two-second GOP can be well over a second away — and callers pair this frame with a
    box computed at `at_s` from the sidecar, so a frame from elsewhere would carry a box
    that does not match the picture.
    """
    import cv2

    capture = cv2.VideoCapture(clip_path)
    if not capture.isOpened():
        return None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            # A clip that will not report its rate cannot be addressed by frame index.
            # Fall back to a plain time seek and accept whatever keyframe it lands on,
            # which is still better than returning nothing at all.
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, at_s) * 1000.0)
            ok, frame = capture.read()
            return frame if ok else None

        target = int(round(at_s * fps))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total and target >= total:
            return None
        # Land a little before and read forward. Counting in *frames* rather than
        # milliseconds is what makes this exact: `CAP_PROP_POS_MSEC` read after a
        # decode reports where the next frame starts, so comparing against it stops
        # one frame early — close enough to look right and wrong enough to pair a
        # picture with a box from a neighbouring instant.
        lead_in = 15
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, target - lead_in))
        frame = None
        for _ in range(lead_in * 4 + 2):
            position = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            ok, candidate = capture.read()
            if not ok:
                break
            frame = candidate
            if position >= target:
                break
        return frame
    finally:
        capture.release()
