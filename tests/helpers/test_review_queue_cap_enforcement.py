"""Tests for `TrainingCollector._enforce_review_queue_cap` — how many images the
review-queue cap actually deletes.

`test_review_queue_eviction.py` covers the pure ranking (which images are worst).
This covers the arithmetic that decides *how many* of them go, which is the half
that was never tested and the half that lost data: eviction calls
`delete_image`, which removes the DB row, the image file and its JSON sidecar
permanently.

The shipped bug: the delete loop counted down from the review queue's
`total_count`, but db_server caches that count for 60s (`_COUNT_CACHE_TTL`) and
nothing invalidates it on delete. So a second pass inside that window read the
*pre-deletion* total while paging the real, already-trimmed queue, and deleted
the difference all over again — every pass, until the queue was gone. The
deletions were themselves what made the count stale, so the bug fed itself.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from apps.helpers.training.collector import CollectorHost, TrainingCollector
from apps.helpers.training.collectorSpec import CollectorSpec

_TAG = "task-tag-uuid"


def _image(index: int) -> SimpleNamespace:
    """One row as the generated review-queue client returns it."""
    return SimpleNamespace(
        uuid=f"img-{index}",
        filename=f"img-{index}.jpg",
        created_at=datetime(2026, 1, 1, 0, 0, index % 60),
        bboxes=[SimpleNamespace(label_uuid="label-a")],
    )


def _collector(max_unverified_images: int) -> TrainingCollector:
    collector = TrainingCollector(
        CollectorSpec.from_params(
            {
                "output_dir": "out",
                "max_unverified_images": max_unverified_images,
                "eviction_strategy": "fifo",
            }
        ),
        CollectorHost(
            log=lambda _message: None,
            db_api_client=lambda: MagicMock(),
            inference_api_client=lambda: None,
            stop_event=asyncio.Event(),
        ),
        files_root=Path("/tmp"),
    )
    collector._task_tag_uuid = _TAG
    return collector


async def _run_pass(collector: TrainingCollector, *, reported_total: int, real_images: int) -> list[str]:
    """One eviction pass where the count query reports `reported_total` but the
    queue really holds `real_images` rows. Returns the uuids it deleted.

    A stale-high `reported_total` is not hypothetical: it is exactly what the
    60s count cache serves after a previous pass has already deleted rows.
    """
    deleted: list[str] = []
    images_api = MagicMock()

    def _get_review_queue(params):
        # The count call asks for limit=1; the candidate page asks for more.
        if params.limit == 1:
            return SimpleNamespace(total_count=reported_total, images=[])
        return SimpleNamespace(
            total_count=reported_total,
            images=[_image(i) for i in range(min(real_images, params.limit))],
        )

    images_api.get_review_queue.side_effect = _get_review_queue
    images_api.delete_image.side_effect = lambda uuid: deleted.append(uuid)

    with patch("apps.helpers.training.collector.ImagesApi", return_value=images_api):
        await collector._enforce_review_queue_cap()
    return deleted


class TestReviewQueueCapDeleteCount:
    async def test_trims_to_the_cap(self):
        """The ordinary case: 700 real images, cap 500, delete exactly 200."""
        deleted = await _run_pass(_collector(500), reported_total=700, real_images=700)
        assert len(deleted) == 200

    async def test_a_stale_high_count_does_not_delete_below_the_cap(self):
        """The regression. The queue is already at the cap; a count cached from
        before the previous pass still says 700. Nothing may be deleted — the
        real queue is the 500 rows the page returned, not the stale number."""
        deleted = await _run_pass(_collector(500), reported_total=700, real_images=500)
        assert deleted == []

    async def test_a_stale_high_count_deletes_only_the_genuine_excess(self):
        """Halfway through the same failure: really 600 rows, count says 900.
        The excess over the cap is 100, not 400."""
        deleted = await _run_pass(_collector(500), reported_total=900, real_images=600)
        assert len(deleted) == 100

    async def test_repeated_passes_converge_on_the_cap_rather_than_zero(self):
        """The shape of the data loss: pass after pass inside one cache window.
        Every pass after the first must be a no-op, not another 200 deletions."""
        collector = _collector(500)
        remaining = 700
        stale_total = 700  # cached before the first pass, never invalidated
        for _ in range(5):
            deleted = await _run_pass(collector, reported_total=stale_total, real_images=remaining)
            remaining -= len(deleted)
        assert remaining == 500

    async def test_an_empty_page_deletes_nothing_even_when_the_count_is_high(self):
        """A count that outlives every row it counted must not be acted on."""
        deleted = await _run_pass(_collector(500), reported_total=700, real_images=0)
        assert deleted == []

    async def test_a_page_truncated_by_the_ceiling_never_over_deletes(self):
        """With more real rows than EVICTION_CANDIDATE_CEILING the page is a
        partial view, so the count must come from the page, not the total:
        deleting `total - cap` out of a short page would empty it."""
        collector = _collector(500)
        ceiling = TrainingCollector.EVICTION_CANDIDATE_CEILING
        deleted = await _run_pass(collector, reported_total=ceiling + 3000, real_images=ceiling + 3000)
        assert len(deleted) == ceiling - 500
