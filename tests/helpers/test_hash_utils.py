"""Unit tests for apps/helpers/hashUtils.py.

All tests are pure computation — no DB, no filesystem (except
compute_image_phash which reads a real PNG via PIL).
"""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from apps.helpers.hashUtils import (
    compute_image_phash,
    hamming,
    phash_from_numpy,
    phash_from_numpy_crop,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _bgr(b: int, g: int, r: int, size: int = 64) -> np.ndarray:
    """Solid-colour BGR ndarray (OpenCV convention)."""
    return np.full((size, size, 3), [b, g, r], dtype=np.uint8)


def _png_bytes(r: int, g: int, b: int, size: int = 64) -> bytes:
    """PIL-generated PNG bytes for a solid-colour image."""
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (r, g, b)).save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# phash_from_numpy
# ─────────────────────────────────────────────────────────────────────────────


class TestPhashFromNumpy:
    def test_returns_hex_string(self):
        h = phash_from_numpy(_bgr(0, 128, 200), hash_size=8)
        assert isinstance(h, str)
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_size_8_produces_16_chars(self):
        # 8×8 = 64 bits → 16 hex chars (4 bits per char)
        h = phash_from_numpy(_bgr(100, 100, 100), hash_size=8)
        assert len(h) == 16

    def test_hash_size_16_produces_64_chars(self):
        # 16×16 = 256 bits → 64 hex chars
        h = phash_from_numpy(_bgr(100, 100, 100), hash_size=16)
        assert len(h) == 64

    def test_deterministic_for_same_image(self):
        img = _bgr(50, 100, 200)
        assert phash_from_numpy(img) == phash_from_numpy(img)

    def test_different_images_differ(self):
        h_dark = phash_from_numpy(_bgr(0, 0, 0))
        h_light = phash_from_numpy(_bgr(255, 255, 255))
        assert hamming(h_dark, h_light) > 0


# ─────────────────────────────────────────────────────────────────────────────
# phash_from_numpy_crop
# ─────────────────────────────────────────────────────────────────────────────


class TestPhashFromNumpyCrop:
    def test_returns_hex_string(self):
        h = phash_from_numpy_crop(_bgr(100, 50, 200, size=100), 0.1, 0.1, 0.5, 0.5)
        assert isinstance(h, str)

    def test_default_hash_size_16_gives_64_chars(self):
        h = phash_from_numpy_crop(_bgr(100, 50, 200, size=100), 0.0, 0.0, 0.8, 0.8)
        assert len(h) == 64

    def test_deterministic(self):
        img = _bgr(100, 50, 200, size=100)
        h1 = phash_from_numpy_crop(img, 0.0, 0.0, 0.5, 0.5)
        h2 = phash_from_numpy_crop(img, 0.0, 0.0, 0.5, 0.5)
        assert h1 == h2

    def test_degenerate_zero_size_crop_does_not_raise(self):
        """A bbox with zero width/height should return a valid hash, not crash."""
        img = _bgr(100, 50, 200, size=100)
        h = phash_from_numpy_crop(img, 0.5, 0.5, 0.0, 0.0)
        assert isinstance(h, str)
        assert len(h) > 0

    def test_full_crop_equals_whole_image_hash(self):
        img = _bgr(100, 50, 200, size=64)
        h_crop = phash_from_numpy_crop(img, 0.0, 0.0, 1.0, 1.0, hash_size=8)
        h_full = phash_from_numpy(img, hash_size=8)
        assert hamming(h_crop, h_full) == 0


# ─────────────────────────────────────────────────────────────────────────────
# hamming
# ─────────────────────────────────────────────────────────────────────────────


class TestHamming:
    def test_identical_hash_gives_zero(self):
        h = phash_from_numpy(_bgr(100, 150, 200))
        assert hamming(h, h) == 0

    def test_symmetric(self):
        h1 = phash_from_numpy(_bgr(0, 128, 255))
        h2 = phash_from_numpy(_bgr(255, 0, 128))
        assert hamming(h1, h2) == hamming(h2, h1)

    def test_non_negative(self):
        h1 = phash_from_numpy(_bgr(0, 0, 0))
        h2 = phash_from_numpy(_bgr(255, 255, 255))
        assert hamming(h1, h2) >= 0

    def test_black_vs_white_differs(self):
        h_black = phash_from_numpy(_bgr(0, 0, 0))
        h_white = phash_from_numpy(_bgr(255, 255, 255))
        # pHash is DCT/grayscale based, so two solid colours differ only in the DC
        # component — distance is small (typically 1) but non-zero.
        assert hamming(h_black, h_white) > 0

    def test_checkerboard_vs_gradient_has_large_distance(self):
        """Images with very different frequency content have a large pHash distance.

        Pinned to hash_size=8 rather than riding on the default: "> 10 bits
        apart" is calibrated against a 64-bit hash. Over a wider hash these
        synthetic 64x64 patterns put nearly all their energy in a handful of
        low-frequency coefficients, so the raw bit count *falls* while the
        images stay just as different — a threshold in absolute bits only means
        something alongside the width it was chosen for.
        """
        # Checkerboard — maximum high-frequency energy
        checker = np.indices((64, 64)).sum(axis=0) % 2  # 0/1 mask
        checker_bgr = (checker[:, :, None] * 255).astype(np.uint8).repeat(3, axis=2)
        # Horizontal gradient — smooth, mostly low-frequency
        grad = np.tile(np.linspace(0, 255, 64).astype(np.uint8), (64, 1))
        grad_bgr = grad[:, :, None].repeat(3, axis=2)
        h_checker = phash_from_numpy(checker_bgr, hash_size=8)
        h_grad = phash_from_numpy(grad_bgr, hash_size=8)
        assert hamming(h_checker, h_grad) > 10

    def test_similar_images_have_small_distance(self):
        # Near-identical grey values
        h1 = phash_from_numpy(_bgr(128, 128, 128))
        h2 = phash_from_numpy(_bgr(130, 130, 130))
        assert hamming(h1, h2) < 5


# ─────────────────────────────────────────────────────────────────────────────
# compute_image_phash (disk read)
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeImagePhash:
    def test_returns_hex_string(self, tmp_path):
        p = tmp_path / "img.png"
        p.write_bytes(_png_bytes(100, 150, 200))
        h = compute_image_phash(p, hash_size=8)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_deterministic_across_calls(self, tmp_path):
        p = tmp_path / "img.png"
        p.write_bytes(_png_bytes(100, 150, 200))
        assert compute_image_phash(p, hash_size=8) == compute_image_phash(p, hash_size=8)

    def test_disk_hash_matches_numpy_hash(self, tmp_path):
        """Hash from disk (PIL) should equal hash from cv2-read array."""
        import cv2

        p = tmp_path / "img.png"
        p.write_bytes(_png_bytes(100, 150, 200))
        h_disk = compute_image_phash(p, hash_size=8)
        bgr = cv2.imread(str(p))
        h_numpy = phash_from_numpy(bgr, hash_size=8)
        assert hamming(h_disk, h_numpy) == 0
