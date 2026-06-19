"""Perceptual hash utilities for image and bounding-box deduplication.

All public functions return/accept hex strings for consistent DB storage and
JSON sidecar serialisation.  The `from_numpy` variants are used during ingestion
(image already in memory); the disk-read variants are used during backfill.

``PHASH_JPEG_QUALITY`` is the single source of truth for the JPEG quality used
when saving training images and when computing in-memory phashes via a round-trip
encode/decode.  Both must match or the same physical image will hash differently
depending on whether the hash was computed before or after the file was written.
"""

from pathlib import Path
from typing import Optional

import imagehash
import numpy as np
from PIL import Image

# JPEG quality used when saving training images to disk AND when computing
# phashes from an in-memory frame via a round-trip encode/decode.  Keeping both
# in sync here guarantees that pre-save and post-save hashes are identical.
PHASH_JPEG_QUALITY: int = 95


def _numpy_bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR ndarray to a PIL Image (RGB)."""
    rgb = image_bgr[:, :, ::-1]
    return Image.fromarray(rgb.astype(np.uint8))


def phash_from_numpy(image_bgr: np.ndarray, hash_size: int = 8) -> str:
    """Compute pHash from an already-decoded numpy BGR array (no disk I/O).

    Returns the hash as a hex string.
    """
    pil = _numpy_bgr_to_pil(image_bgr)
    return str(imagehash.phash(pil, hash_size=hash_size))


def phash_from_numpy_crop(
    image_bgr: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    hash_size: int = 16,
) -> str:
    """Crop a normalized bbox region from a numpy BGR array and return its pHash.

    Coordinates are normalized [0.0, 1.0].  Returns the hash as a hex string.
    """
    H, W = image_bgr.shape[:2]
    x0 = max(0, int(round(x * W)))
    y0 = max(0, int(round(y * H)))
    x1 = min(W, int(round((x + w) * W)))
    y1 = min(H, int(round((y + h) * H)))
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        # Degenerate bbox — hash a blank 1×1 image so we still return a string.
        crop = np.zeros((1, 1, 3), dtype=np.uint8)
    pil = _numpy_bgr_to_pil(crop)
    return str(imagehash.phash(pil, hash_size=hash_size))


def compute_image_phash(image_path: Path, hash_size: int = 8) -> str:
    """Read an image from disk and return its pHash as a hex string."""
    pil = Image.open(image_path).convert("RGB")
    return str(imagehash.phash(pil, hash_size=hash_size))


def compute_bbox_phash(
    image_path: Path,
    x: float,
    y: float,
    w: float,
    h: float,
    hash_size: int = 16,
) -> str:
    """Crop a normalized bbox region from an image on disk and return its pHash."""
    pil_full = Image.open(image_path).convert("RGB")
    W, H = pil_full.size
    x0 = max(0, int(round(x * W)))
    y0 = max(0, int(round(y * H)))
    x1 = min(W, int(round((x + w) * W)))
    y1 = min(H, int(round((y + h) * H)))
    crop = pil_full.crop((x0, y0, x1, y1))
    if crop.size[0] == 0 or crop.size[1] == 0:
        crop = Image.new("RGB", (1, 1))
    return str(imagehash.phash(crop, hash_size=hash_size))


def hamming(a: str, b: str) -> int:
    """Hamming distance between two imagehash hex strings.

    Operates directly on the integer value of the hex strings — no imagehash
    object allocation, no numpy round-trip.  Equivalent to the old
    ``int(imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b))`` but orders of
    magnitude faster when called O(N log N) times during BK-tree construction.
    """
    return (int(a, 16) ^ int(b, 16)).bit_count()


def compute_phashes_with_jpeg_roundtrip(
    image_bgr: np.ndarray,
    bboxes_xywh: list[tuple[float, float, float, float]],
    image_hash_size: int = 8,
    bbox_hash_size: int = 16,
) -> tuple[str, list[str]]:
    """Compute image and bbox phashes via a JPEG round-trip at ``PHASH_JPEG_QUALITY``.

    The BK-tree index stores phashes derived from JPEG files written at
    ``PHASH_JPEG_QUALITY``.  Encoding/decoding here at the same quality produces
    phashes that are byte-identical to those stored values, avoiding false
    non-matches caused by raw-frame vs JPEG-decoded pixel differences.

    Args:
        image_bgr: OpenCV BGR image array (not yet saved to disk).
        bboxes_xywh: Normalized (x, y, w, h) tuples for each detection bbox.
        image_hash_size: Hash grid size for the whole-image phash.
        bbox_hash_size: Hash grid size for the per-bbox crop phash.

    Returns:
        ``(image_phash, [bbox_phash, ...])`` — all as hex strings.
    """
    import cv2  # local import: cv2 not required by the rest of hashUtils

    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, PHASH_JPEG_QUALITY])
    jpeg_frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    img_phash = phash_from_numpy(jpeg_frame, image_hash_size)
    bbox_phashes = [phash_from_numpy_crop(jpeg_frame, x, y, w, h, bbox_hash_size) for x, y, w, h in bboxes_xywh]
    return img_phash, bbox_phashes
