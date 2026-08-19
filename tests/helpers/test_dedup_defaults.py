"""Deduplication defaults: one value per setting, agreed on by both languages.

The dedup defaults exist twice — `apps/helpers/hashUtils.py` for the services and
`static/js/shared/dedupDefaults.js` for the frontend — because no build step
shares constants between them. Divergence is silent and nasty: a frontend that
assumes a different hash size than the index was built at computes phashes of
the wrong width, and `_FlatHashTable.from_items` *skips* wrong-width rows rather
than raising, so similarity search quietly returns nothing instead of failing.

These tests assert the two copies agree with *each other*, and that whatever the
values are, they are internally coherent. They deliberately do not pin the
literals — a test that restates the constant only fails when someone edits it on
purpose.
"""

import re
from pathlib import Path

from apps.helpers.hashUtils import (
    DEFAULT_BBOX_HASH_SIZE,
    DEFAULT_BBOX_THRESHOLD,
    DEFAULT_IMAGE_HASH_SIZE,
    DEFAULT_IMAGE_THRESHOLD,
)

_JS_DEFAULTS = (
    Path(__file__).resolve().parents[2]
    / "apps"
    / "image_tagging_server"
    / "static"
    / "js"
    / "shared"
    / "dedupDefaults.js"
)


def _js_const(name: str) -> int:
    """Read `export const <name> = <int>;` out of the JS mirror."""
    match = re.search(rf"^export const {name} = (\d+);", _JS_DEFAULTS.read_text(), re.MULTILINE)
    assert match, f"{name} not found in {_JS_DEFAULTS.name}"
    return int(match.group(1))


class TestDedupDefaultValues:
    def test_thresholds_fit_their_hash_range(self):
        """A threshold is a Hamming distance over hash_size² bits — one larger
        than that range matches everything, which is never what was meant."""
        assert 1 <= DEFAULT_IMAGE_THRESHOLD <= DEFAULT_IMAGE_HASH_SIZE**2
        assert 1 <= DEFAULT_BBOX_THRESHOLD <= DEFAULT_BBOX_HASH_SIZE**2


class TestFrontendMirrorMatches:
    def test_image_hash_size_matches(self):
        assert _js_const("DEFAULT_IMAGE_HASH_SIZE") == DEFAULT_IMAGE_HASH_SIZE

    def test_bbox_hash_size_matches(self):
        assert _js_const("DEFAULT_BBOX_HASH_SIZE") == DEFAULT_BBOX_HASH_SIZE

    def test_image_threshold_matches(self):
        assert _js_const("DEFAULT_IMAGE_THRESHOLD") == DEFAULT_IMAGE_THRESHOLD

    def test_bbox_threshold_matches(self):
        assert _js_const("DEFAULT_BBOX_THRESHOLD") == DEFAULT_BBOX_THRESHOLD


class TestConsumersUseTheConstants:
    """Asserted through behaviour rather than through `inspect.signature` on each
    consumer: what matters is that a hash actually comes out at the width the
    index validates against, not which default literal a parameter carries."""

    def test_default_image_phash_is_the_expected_hex_width(self, tmp_path):
        """hash_size N → N²/4 hex chars. The index validates on this width, so a
        default that disagrees with it fails every insert."""
        import numpy as np

        from apps.helpers.hashUtils import phash_from_numpy

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:32] = 255
        assert len(phash_from_numpy(frame)) == DEFAULT_IMAGE_HASH_SIZE**2 // 4
