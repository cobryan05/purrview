"""Deduplication defaults: one value per setting, agreed on by both languages.

The dedup defaults exist twice — `apps/helpers/hashUtils.py` for the services and
`static/js/shared/dedupDefaults.js` for the frontend — because no build step
shares constants between them. Divergence is silent and nasty: a frontend that
assumes a different hash size than the index was built at computes phashes of
the wrong width, and `_FlatHashTable.from_items` *skips* wrong-width rows rather
than raising, so similarity search quietly returns nothing instead of failing.

These tests pin the values and assert the two copies still match.
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
    def test_image_hash_size_is_16(self):
        assert DEFAULT_IMAGE_HASH_SIZE == 16

    def test_bbox_hash_size_is_16(self):
        assert DEFAULT_BBOX_HASH_SIZE == 16

    def test_image_threshold_is_5(self):
        assert DEFAULT_IMAGE_THRESHOLD == 5

    def test_bbox_threshold_is_75(self):
        assert DEFAULT_BBOX_THRESHOLD == 75

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
    """The defaults were previously bare literals in each consumer; these pin the
    wiring so a future edit to the constant actually reaches them."""

    def test_hash_index_defaults_to_the_constants(self):
        import inspect

        from apps.db_server.hash_index import HashIndex

        params = inspect.signature(HashIndex.__init__).parameters
        assert params["image_hash_size"].default == DEFAULT_IMAGE_HASH_SIZE
        assert params["bbox_hash_size"].default == DEFAULT_BBOX_HASH_SIZE

    def test_phash_helpers_default_to_the_constants(self):
        import inspect

        from apps.helpers.hashUtils import (
            compute_bbox_phash,
            compute_image_phash,
            phash_from_numpy,
            phash_from_numpy_crop,
        )

        assert inspect.signature(phash_from_numpy).parameters["hash_size"].default == DEFAULT_IMAGE_HASH_SIZE
        assert inspect.signature(compute_image_phash).parameters["hash_size"].default == DEFAULT_IMAGE_HASH_SIZE
        assert inspect.signature(phash_from_numpy_crop).parameters["hash_size"].default == DEFAULT_BBOX_HASH_SIZE
        assert inspect.signature(compute_bbox_phash).parameters["hash_size"].default == DEFAULT_BBOX_HASH_SIZE

    def test_default_image_phash_is_the_expected_hex_width(self, tmp_path):
        """hash_size N → N²/4 hex chars. The index validates on this width, so a
        default that disagrees with it fails every insert."""
        import numpy as np

        from apps.helpers.hashUtils import phash_from_numpy

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:32] = 255
        assert len(phash_from_numpy(frame)) == DEFAULT_IMAGE_HASH_SIZE**2 // 4
