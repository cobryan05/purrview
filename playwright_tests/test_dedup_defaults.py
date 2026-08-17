"""Frontend deduplication defaults (shared/dedupDefaults.js).

The Python-side values and the Python↔JS agreement are pinned in
tests/helpers/test_dedup_defaults.py; this covers the browser-side helpers that
turn those constants into the numbers the Settings panel and the similarity
sliders actually show.
"""

from playwright.sync_api import Page

MODULE = "/app-static/js/shared/dedupDefaults.js"


def _consts(page: Page, base_url: str) -> dict:
    page.goto(base_url)
    return page.evaluate(
        """async (mod) => {
        const m = await import(mod);
        return {
            imgSize: m.DEFAULT_IMAGE_HASH_SIZE,
            bboxSize: m.DEFAULT_BBOX_HASH_SIZE,
            imgThresh: m.DEFAULT_IMAGE_THRESHOLD,
            bboxThresh: m.DEFAULT_BBOX_THRESHOLD,
        };
    }""",
        MODULE,
    )


class TestDefaultConstants:
    def test_defaults_are_the_configured_values(self, page: Page, base_url: str):
        c = _consts(page, base_url)
        assert c["imgSize"] == 16
        assert c["bboxSize"] == 16
        assert c["imgThresh"] == 5
        assert c["bboxThresh"] == 75


class TestHashMaxFor:
    def test_hash_max_is_the_square(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate(
            """async (mod) => {
            const { hashMaxFor } = await import(mod);
            return [hashMaxFor(8), hashMaxFor(16), hashMaxFor(32)];
        }""",
            MODULE,
        )
        assert result == [64, 256, 1024]


class TestDefaultThresholdFor:
    """A threshold is meaningless without the bit-width it was chosen for, so
    changing hash size rescales it to preserve selectivity rather than keeping
    the number."""

    def _call(self, page: Page, base_url: str, kind: str, size: int) -> int:
        page.goto(base_url)
        return page.evaluate(
            """async ([mod, kind, size]) => {
            const { defaultThresholdFor } = await import(mod);
            return defaultThresholdFor(kind, size);
        }""",
            [MODULE, kind, size],
        )

    def test_at_native_size_returns_the_default_unchanged(self, page: Page, base_url: str):
        assert self._call(page, base_url, "image", 16) == 5
        assert self._call(page, base_url, "bbox", 16) == 75

    def test_scales_down_with_a_smaller_hash(self, page: Page, base_url: str):
        # 64-bit hash is a quarter of 256-bit, so the distance budget scales the same way.
        assert self._call(page, base_url, "bbox", 8) == 19  # round(75 / 4)

    def test_scales_up_with_a_larger_hash(self, page: Page, base_url: str):
        assert self._call(page, base_url, "bbox", 32) == 300  # 75 * 4

    def test_never_returns_zero(self, page: Page, base_url: str):
        """round(5/16) is 0, and a threshold of 0 only matches byte-identical
        hashes — a silently useless search rather than a strict one."""
        assert self._call(page, base_url, "image", 4) == 1
