"""Perceptual hash utilities for image and bounding-box deduplication.

All public functions return/accept hex strings for consistent DB storage and
JSON sidecar serialisation.  The `from_numpy` variants are used during ingestion
(image already in memory); the disk-read variants are used during backfill.
"""

from pathlib import Path
from typing import Optional

import imagehash
import numpy as np
from PIL import Image


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
    """Hamming distance between two imagehash hex strings."""
    return int(imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b))
