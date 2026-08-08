"""Keyset pagination must still advance when the cursor carries no sort value.

The review modals paginate custom (`fetchImages`) result sets by sending only
`afterUuid` — `app/review/main.js` passes `afterSortValue: null` on every chunk
fetch. Every keyset implementation in `db_client` has an
`after_sort_value is None` fallback that compares `uuid` values instead, but the
rows are ordered by `created_at`. UUIDs are random with respect to that order,
so the scan finds a "past the cursor" row within the first row or two, the page
barely advances, and it can settle on re-serving the *same* page forever.

Live reproduction before the fix: the "Needs Review" modal for a 1,294-image
training config returned 251 rows per request, 251 of them duplicates, with the
cursor frozen on one uuid — an unbounded request loop that never terminated
because `has_more` stayed true.

These tests page each affected endpoint twice with a uuid-only cursor and assert
the second page does not re-serve the first.
"""

import pytest

from tests.db_server.test_training_configs import (
    _create_dataset,
    _create_label,
    _create_model_config,
    _create_training_config,
    _register_image,
)

# Enough images that a small `limit` forces a real second page.
_IMAGE_COUNT = 6
_PAGE = 2


async def _build_partial_review_config(http_client, files_root, prefix):
    """A training config whose images all land in the 'needs review' bucket:
    the parent label is verified but the child label is not."""
    parent = await _create_label(http_client, f"{prefix}-parent")
    child = await _create_label(http_client, f"{prefix}-child", parent_uuid=parent)
    ds = await _create_dataset(http_client, f"ds-{prefix}")

    for i in range(_IMAGE_COUNT):
        name = f"{prefix}_{i}.png"
        await _register_image(http_client, files_root, name)
        # Only the parent verified => partial coverage => "needs review".
        await http_client.post(
            "/api/images/labels/set",
            json={"imagePath": name, "labelUuid": parent, "status": "present"},
        )

    mc = await _create_model_config(
        http_client,
        f"mc-{prefix}",
        nodes=[{"label_uuid": parent, "model_type": "image_classifier", "architecture": "resnet_18"}],
    )
    cfg = await _create_training_config(
        http_client, f"tc-{prefix}", [ds["uuid"]], model_config_node_uuid=mc["nodes"][0]["uuid"]
    )
    return cfg, parent, child


async def _page(http_client, url, after_uuid=None):
    """Fetch one page with a uuid-only cursor — afterSortValue deliberately None,
    mirroring what the review modal sends."""
    r = await http_client.post(
        url,
        json={
            "afterUuid": after_uuid,
            "afterSortValue": None,
            "sortBy": "created_at",
            "sortOrder": "desc",
            "limit": _PAGE,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return [img["uuid"] for img in body["images"]]


class TestUuidOnlyCursorAdvances:
    async def test_partial_review_images_second_page_is_not_the_first(self, http_client, files_root):
        """`/partial-review-images` — the endpoint behind the 'Needs Review' modal
        the hang was reported against."""
        cfg, _, _ = await _build_partial_review_config(http_client, files_root, "cursor-partial")
        url = f"/api/training-configs/{cfg['uuid']}/partial-review-images"

        first = await _page(http_client, url)
        assert len(first) == _PAGE, f"fixture too small to force a second page: {first}"

        second = await _page(http_client, url, after_uuid=first[-1])
        assert set(first).isdisjoint(
            second
        ), f"second page re-served rows from the first: first={first} second={second}"

    async def test_partial_review_images_full_walk_terminates_without_duplicates(self, http_client, files_root):
        """Walking the whole bucket with uuid-only cursors must visit each image
        exactly once and stop — this is the loop `render()` runs when a curated
        filter matches nothing."""
        cfg, _, _ = await _build_partial_review_config(http_client, files_root, "cursor-walk")
        url = f"/api/training-configs/{cfg['uuid']}/partial-review-images"

        seen = []
        after = None
        for _ in range(_IMAGE_COUNT + 3):  # generous bound; a correct walk needs 3
            page = await _page(http_client, url, after_uuid=after)
            if not page:
                break
            seen.extend(page)
            after = page[-1]
            if len(page) < _PAGE:
                break
        else:
            pytest.fail(f"pagination did not terminate; collected {len(seen)} rows")

        assert len(seen) == len(set(seen)), f"duplicate rows across pages: {seen}"
        assert len(seen) == _IMAGE_COUNT, f"expected {_IMAGE_COUNT} images, walked {len(seen)}"

    async def test_needs_review_images_second_page_is_not_the_first(self, http_client, files_root):
        """`get_training_config_needs_review_images` — same fallback, different method."""
        cfg, _, _ = await _build_partial_review_config(http_client, files_root, "cursor-needs")
        url = f"/api/training-configs/{cfg['uuid']}/needs-review-images"

        first = await _page(http_client, url)
        if len(first) < _PAGE:
            pytest.skip("fixture produced no second page for this bucket")

        second = await _page(http_client, url, after_uuid=first[-1])
        assert set(first).isdisjoint(
            second
        ), f"second page re-served rows from the first: first={first} second={second}"

    async def test_images_by_label_status_second_page_is_not_the_first(self, http_client, files_root):
        """`get_training_config_images_by_label_status` — the fourth site, reached by
        the per-label cells in the coverage table."""
        cfg, parent, _ = await _build_partial_review_config(http_client, files_root, "cursor-status")

        async def page(after_uuid=None):
            r = await http_client.post(
                f"/api/training-configs/{cfg['uuid']}/images-by-label-status",
                json={
                    "labelUuid": parent,
                    "status": "present",
                    "afterUuid": after_uuid,
                    "afterSortValue": None,
                    "sortBy": "created_at",
                    "sortOrder": "desc",
                    "limit": _PAGE,
                },
            )
            assert r.status_code == 200, r.text
            return [img["uuid"] for img in r.json()["images"]]

        first = await page()
        assert len(first) == _PAGE, f"fixture too small to force a second page: {first}"

        second = await page(after_uuid=first[-1])
        assert set(first).isdisjoint(
            second
        ), f"second page re-served rows from the first: first={first} second={second}"

    async def test_image_search_second_page_is_not_the_first(self, http_client, files_root):
        """`list_images_search` — the SQL keyset behind the main Review queue.
        Same fallback branch, reached whenever a caller omits afterSortValue."""
        for i in range(_IMAGE_COUNT):
            await _register_image(http_client, files_root, f"cursor_search_{i}.png")

        async def page(after_uuid=None):
            r = await http_client.post(
                "/api/images/review",
                json={"afterUuid": after_uuid, "afterSortValue": None, "limit": _PAGE},
            )
            assert r.status_code == 200, r.text
            return [img["uuid"] for img in r.json()["images"]]

        first = await page()
        assert len(first) == _PAGE, f"fixture too small to force a second page: {first}"

        second = await page(after_uuid=first[-1])
        assert set(first).isdisjoint(
            second
        ), f"second page re-served rows from the first: first={first} second={second}"
