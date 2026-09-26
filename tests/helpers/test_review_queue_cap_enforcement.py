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
import logging
from datetime import datetime, timezone
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
            log=lambda _message, _level=logging.INFO: None,
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


def _low_water(cap: int) -> int:
    """Where a pass stops, given it trims under the cap rather than exactly to it.

    Derived rather than hardcoded: the margin is a fraction that rounds down, so a
    small cap legitimately has no margin at all and the same helper has to say so.
    """
    return cap - int(cap * TrainingCollector.EVICTION_LOW_WATER_MARGIN)


class TestReviewQueueCapDeleteCount:
    async def test_trims_to_just_under_the_cap(self):
        """The ordinary case: 700 real images against a cap of 500.

        Stops below the cap rather than on it, so the very next save is not over
        again — otherwise every save pays for a full page fetch and ranking pass to
        delete a single image.
        """
        deleted = await _run_pass(_collector(500), reported_total=700, real_images=700)
        assert len(deleted) == 700 - _low_water(500)

    async def test_the_overshoot_stays_small(self):
        """A permanent delete, so the margin that buys the batching must be modest.

        Asserted as a bound rather than an exact figure: the point is that it is a
        few percent, not whichever constant is in force.
        """
        cap = 500
        deleted = await _run_pass(_collector(cap), reported_total=700, real_images=700)
        overshoot = cap - (700 - len(deleted))
        assert 0 < overshoot <= cap * 0.05

    async def test_a_small_cap_is_trimmed_exactly_and_loses_nothing_extra(self):
        """The margin rounds down, so a short queue gets none of it. One extra
        permanent delete costs more there than the saved work is worth."""
        deleted = await _run_pass(_collector(4), reported_total=5, real_images=5)
        assert len(deleted) == 1

    async def test_a_stale_high_count_does_not_delete_below_the_cap(self):
        """The regression. The queue is already at the cap; a count cached from
        before the previous pass still says 700. Nothing may be deleted — the
        real queue is the 500 rows the page returned, not the stale number."""
        deleted = await _run_pass(_collector(500), reported_total=700, real_images=500)
        assert deleted == []

    async def test_a_stale_high_count_deletes_only_the_genuine_excess(self):
        """Halfway through the same failure: really 600 rows, count says 900.
        The excess is measured against the real 600, not the stale 900."""
        deleted = await _run_pass(_collector(500), reported_total=900, real_images=600)
        assert len(deleted) == 600 - _low_water(500)

    async def test_repeated_passes_converge_on_the_cap_rather_than_zero(self):
        """The shape of the data loss: pass after pass inside one cache window.
        Every pass after the first must be a no-op, not another 200 deletions."""
        collector = _collector(500)
        remaining = 700
        stale_total = 700  # cached before the first pass, never invalidated
        for _ in range(5):
            deleted = await _run_pass(collector, reported_total=stale_total, real_images=remaining)
            remaining -= len(deleted)
        assert remaining == _low_water(500)

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
        assert len(deleted) == ceiling - _low_water(500)


CONFUSABLE = "00000000-0000-0000-0000-000000000006"


def _tagged_image(index: int, *, tag_uuid=None, created_at=None, phash=None) -> SimpleNamespace:
    """A review-queue row with the fields the protections and the hash reuse read."""
    return SimpleNamespace(
        uuid=f"img-{index}",
        filename=f"img-{index}.jpg",
        created_at=created_at or datetime(2026, 1, 1, 0, 0, index % 60),
        phash=phash,
        bboxes=[
            SimpleNamespace(
                label_uuid="label-a",
                tags=[SimpleNamespace(uuid=tag_uuid)] if tag_uuid else [],
            )
        ],
    )


async def _run_with(collector: TrainingCollector, images: list) -> list[str]:
    deleted: list[str] = []
    images_api = MagicMock()

    def _get_review_queue(params):
        if params.limit == 1:
            return SimpleNamespace(total_count=len(images), images=[])
        return SimpleNamespace(total_count=len(images), images=images[: params.limit])

    images_api.get_review_queue.side_effect = _get_review_queue
    images_api.delete_image.side_effect = lambda uuid: deleted.append(uuid)
    with patch("apps.helpers.training.collector.ImagesApi", return_value=images_api):
        await collector._enforce_review_queue_cap()
    return deleted


class TestWhatTheCapRefusesToDeleteFirst:
    """Eviction ranks on appearance, labels or age, none of which can see that a frame
    is the one worth keeping. Two facts override it, and both are read off the page the
    pass already fetched — no extra query, no file touched."""

    async def test_a_confusable_frame_outlives_an_ordinary_one(self):
        """The model scored two classes near-equally here. One correction fixes a
        systematic error, so it goes only when nothing ordinary is left."""
        images = [_tagged_image(0, tag_uuid=CONFUSABLE)] + [_tagged_image(i) for i in range(1, 5)]

        deleted = await _run_with(_collector(3), images)

        assert deleted and "img-0" not in deleted

    async def test_a_frame_saved_moments_ago_is_spared(self):
        """A queue that deletes what it saved a moment ago is not a review queue —
        the image is gone before anyone could open it.

        Under a similarity strategy rather than oldest-first, because oldest-first
        spares a fresh frame by construction and the test would prove nothing. Here
        the fresh frame is a near-duplicate of another, so it is exactly what
        `least_unique` would delete first.
        """
        collector = _collector(3)
        collector._spec = collector._spec.__class__.from_params(
            {"output_dir": "out", "max_unverified_images": 3, "eviction_strategy": "least_unique"}
        )
        twin = "ffff0000ffff0000"
        fresh = _tagged_image(0, created_at=datetime.now(timezone.utc), phash=twin)
        images = [fresh, _tagged_image(1, phash=twin)] + [
            _tagged_image(i, phash=f"{i:016x}" + "0" * 0) for i in range(2, 6)
        ]

        deleted = await _run_with(collector, images)

        assert deleted, "the cap must still be enforced"
        assert "img-0" not in deleted
        assert "img-1" in deleted, "its older twin is the one that should go"

    async def test_an_entirely_protected_queue_is_still_trimmed(self):
        """Tiering rather than exempting: protection reorders eviction, it must never
        make the cap unenforceable or a burst would grow the queue without bound."""
        images = [_tagged_image(i, tag_uuid=CONFUSABLE) for i in range(5)]

        deleted = await _run_with(_collector(3), images)

        assert len(deleted) == 5 - _low_water(3)


class TestTheStoredHashIsReused:
    async def test_a_stored_hash_is_not_recomputed(self):
        """The value is written once at save time. Re-deriving it meant reading and
        decoding every candidate file on every pass, for the sake of one deletion."""
        collector = _collector(3)
        collector._spec = collector._spec.__class__.from_params(
            {"output_dir": "out", "max_unverified_images": 3, "eviction_strategy": "least_unique"}
        )
        images = [_tagged_image(i, phash=f"{i:016x}") for i in range(5)]

        with patch.object(TrainingCollector, "_safe_phash", side_effect=AssertionError("re-hashed a stored hash")):
            deleted = await _run_with(collector, images)

        assert deleted

    async def test_a_row_with_no_stored_hash_still_falls_back_to_the_file(self):
        """Rows written before hashing, or whose file could not be hashed, must not
        silently drop out of a similarity ranking."""
        collector = _collector(3)
        collector._spec = collector._spec.__class__.from_params(
            {"output_dir": "out", "max_unverified_images": 3, "eviction_strategy": "least_unique"}
        )
        images = [_tagged_image(i, phash=None) for i in range(5)]
        hashed: list[str] = []

        with patch.object(TrainingCollector, "_safe_phash", side_effect=lambda f: hashed.append(f) or "00ff00ff"):
            await _run_with(collector, images)

        assert len(hashed) == 5
