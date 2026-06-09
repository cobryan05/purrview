"""Tests for Cv2VideoImageProvider — parallel prefetch worker behaviour.

Covers the parallel-reader refactor introduced to fix the PNG-read bottleneck:

  - All image files in a directory are yielded exactly once.
  - Mixed PNG and JPEG files are handled without error.
  - Frame metadata (source path, frame_idx=0) is populated on every frame.
  - pre_encode=True populates encoded_b64 on every frame.
  - An empty directory signals exhaustion immediately (returns None).
  - Stopping before exhaustion completes without deadlock.
  - PARALLEL_READERS constant is present and > 1.
"""

import asyncio
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from apps.helpers.imageProviders.cv2VideoImageProvider import Cv2VideoImageProvider


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _write_image(path: Path, index: int = 0) -> None:
    """Write a tiny valid image to *path* using cv2."""
    frame = np.full((8, 8, 3), fill_value=(index * 10 % 256, 0, 0), dtype=np.uint8)
    cv2.imwrite(str(path), frame)


def _make_images(directory: Path, count: int, ext: str = ".png") -> list[Path]:
    """Create *count* small images and return their paths."""
    paths = []
    for i in range(count):
        p = directory / f"frame_{i:04d}{ext}"
        _write_image(p, i)
        paths.append(p)
    return paths


def _provider(tmp_path: Path, **kwargs) -> Cv2VideoImageProvider:
    """Create a Cv2VideoImageProvider rooted at *tmp_path*.

    The constructor always prefixes rootdir with FILES_ROOT, so we patch that
    env var to tmp_path and pass rootdir="" so the effective root IS tmp_path.
    """
    with patch.dict(os.environ, {"FILES_ROOT": str(tmp_path)}):
        return Cv2VideoImageProvider(rootdir="", **kwargs)


async def _collect_all(provider: Cv2VideoImageProvider, timeout: float = 10.0) -> list:
    """Drain a started provider to exhaustion, returning all non-None frames."""
    frames = []
    while True:
        img = await asyncio.wait_for(provider.getNextImage(), timeout=timeout)
        if img is None:
            break
        frames.append(img)
    return frames


# ─── Tests ────────────────────────────────────────────────────────────────────


def test_parallel_readers_constant_exists_and_is_greater_than_one():
    """PARALLEL_READERS must be defined and > 1 so parallelism is actually active."""
    assert hasattr(Cv2VideoImageProvider, "PARALLEL_READERS")
    assert Cv2VideoImageProvider.PARALLEL_READERS > 1


async def test_yields_all_image_files(tmp_path):
    """Provider must yield exactly one frame per image file in the directory."""
    _make_images(tmp_path, 6)
    p = _provider(tmp_path)
    await p.start()
    try:
        frames = await _collect_all(p)
    finally:
        await p.stop()
    assert len(frames) == 6


async def test_yields_mixed_png_and_jpg(tmp_path):
    """Provider handles a mix of .png and .jpg files without error or omission."""
    _make_images(tmp_path, 3, ".png")
    _make_images(tmp_path, 3, ".jpg")
    p = _provider(tmp_path)
    await p.start()
    try:
        frames = await _collect_all(p)
    finally:
        await p.stop()
    assert len(frames) == 6


async def test_metadata_source_and_frame_idx(tmp_path):
    """Every yielded frame must carry the source file path and frame_idx==0."""
    paths = _make_images(tmp_path, 3)
    p = _provider(tmp_path)
    await p.start()
    try:
        frames = await _collect_all(p)
    finally:
        await p.stop()

    sources = {f.metadata.source for f in frames}
    assert sources == {str(path) for path in paths}
    assert all(f.metadata.frame_idx == 0 for f in frames)


async def test_pre_encode_populates_encoded_b64(tmp_path):
    """When pre_encode=True, each frame must carry a non-empty encoded_b64 string."""
    _make_images(tmp_path, 4)
    p = _provider(tmp_path)
    p.pre_encode = True
    await p.start()
    try:
        frames = await _collect_all(p)
    finally:
        await p.stop()

    assert len(frames) == 4
    assert all(f.encoded_b64 is not None and len(f.encoded_b64) > 0 for f in frames)


async def test_without_pre_encode_encoded_b64_is_none(tmp_path):
    """When pre_encode=False (default), encoded_b64 must remain None."""
    _make_images(tmp_path, 3)
    p = _provider(tmp_path)
    assert p.pre_encode is False
    await p.start()
    try:
        frames = await _collect_all(p)
    finally:
        await p.stop()

    assert all(f.encoded_b64 is None for f in frames)


async def test_empty_directory_returns_none_immediately(tmp_path):
    """An empty directory must signal exhaustion and return None without hanging."""
    p = _provider(tmp_path)
    await p.start()
    try:
        result = await asyncio.wait_for(p.getNextImage(), timeout=5.0)
        assert result is None
    finally:
        await p.stop()


async def test_stop_before_exhaustion_does_not_deadlock(tmp_path):
    """Stopping a provider early (queue still has frames) must complete cleanly."""
    _make_images(tmp_path, Cv2VideoImageProvider.PREFETCH_QUEUE_SIZE + 4)
    p = _provider(tmp_path)
    await p.start()
    # Consume only two frames then stop early
    await asyncio.wait_for(p.getNextImage(), timeout=5.0)
    await asyncio.wait_for(p.getNextImage(), timeout=5.0)
    await asyncio.wait_for(p.stop(), timeout=5.0)


async def test_more_files_than_parallel_readers(tmp_path):
    """All files are yielded even when count >> PARALLEL_READERS."""
    n = Cv2VideoImageProvider.PARALLEL_READERS * 3 + 1
    _make_images(tmp_path, n)
    p = _provider(tmp_path)
    await p.start()
    try:
        frames = await _collect_all(p)
    finally:
        await p.stop()
    assert len(frames) == n


async def test_nonexistent_rootdir_returns_none(tmp_path):
    """A rootdir that does not exist must yield None without crashing."""
    p = _provider(tmp_path / "does_not_exist")
    await p.start()
    try:
        result = await asyncio.wait_for(p.getNextImage(), timeout=5.0)
        assert result is None
    finally:
        await p.stop()
