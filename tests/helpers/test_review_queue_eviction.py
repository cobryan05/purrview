"""Tests for `apps.helpers.training.reviewQueueEviction.select_for_eviction`.

Pure selection logic — no DB, no filesystem, no collector — so these assert the
actual eviction *decision* each strategy makes, not just that it runs.
"""

from datetime import datetime, timedelta

import pytest

from apps.helpers.training.collectorSpec import EVICTION_FIFO, EVICTION_LABEL_BALANCE, EVICTION_LEAST_UNIQUE
from apps.helpers.training.reviewQueueEviction import EvictionCandidate, select_for_eviction

BASE = datetime(2026, 1, 1)


def _candidate(uuid, *, age_s=0, labels=(), phash=None):
    return EvictionCandidate(
        uuid=uuid,
        filename=f"{uuid}.jpg",
        created_at=BASE + timedelta(seconds=age_s),
        label_uuids=tuple(labels),
        phash=phash,
    )


class TestFifo:
    def test_evicts_the_oldest_first(self):
        candidates = [_candidate("newest", age_s=2), _candidate("oldest", age_s=0), _candidate("middle", age_s=1)]
        assert [c.uuid for c in select_for_eviction(candidates, 2, EVICTION_FIFO)] == ["oldest", "middle"]

    def test_count_is_clamped_to_available_candidates(self):
        candidates = [_candidate("a"), _candidate("b")]
        assert len(select_for_eviction(candidates, 10, EVICTION_FIFO)) == 2

    def test_zero_or_negative_count_evicts_nothing(self):
        candidates = [_candidate("a")]
        assert select_for_eviction(candidates, 0, EVICTION_FIFO) == []
        assert select_for_eviction(candidates, -1, EVICTION_FIFO) == []

    def test_no_candidates_evicts_nothing(self):
        assert select_for_eviction([], 5, EVICTION_FIFO) == []

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            select_for_eviction([_candidate("a")], 1, "vibes")


class TestLeastUnique:
    def test_evicts_the_closest_pair_member_first(self):
        # "a" and "b" are one bit apart (near-duplicates); "c" is unrelated.
        candidates = [
            _candidate("a", phash="0000000000000000"),
            _candidate("b", phash="0000000000000001"),
            _candidate("c", phash="ffffffffffffffff"),
        ]
        evicted = select_for_eviction(candidates, 1, EVICTION_LEAST_UNIQUE)
        assert evicted[0].uuid in ("a", "b")

    def test_the_odd_one_out_survives_a_partial_eviction(self):
        candidates = [
            _candidate("a", phash="0000000000000000"),
            _candidate("b", phash="0000000000000001"),
            _candidate("c", phash="ffffffffffffffff"),
        ]
        evicted = {c.uuid for c in select_for_eviction(candidates, 2, EVICTION_LEAST_UNIQUE)}
        assert evicted == {"a", "b"}

    def test_candidates_with_no_phash_are_evicted_last_not_guessed_at(self):
        candidates = [
            _candidate("hashed", phash="0000000000000000"),
            _candidate("unhashed", phash=None),
        ]
        evicted = select_for_eviction(candidates, 1, EVICTION_LEAST_UNIQUE)
        assert [c.uuid for c in evicted] == ["hashed"]

    def test_all_unhashed_falls_back_to_fifo_order(self):
        candidates = [_candidate("newer", age_s=1, phash=None), _candidate("older", age_s=0, phash=None)]
        evicted = select_for_eviction(candidates, 1, EVICTION_LEAST_UNIQUE)
        assert [c.uuid for c in evicted] == ["older"]


class TestLabelBalance:
    def test_evicts_from_the_over_represented_label_first(self):
        candidates = [
            _candidate("cat1", labels=["cat"]),
            _candidate("cat2", labels=["cat"]),
            _candidate("cat3", labels=["cat"]),
            _candidate("raccoon1", labels=["raccoon"]),
        ]
        evicted = {c.uuid for c in select_for_eviction(candidates, 2, EVICTION_LABEL_BALANCE)}
        assert evicted.issubset({"cat1", "cat2", "cat3"})
        assert "raccoon1" not in evicted
        assert len(evicted) == 2

    def test_rebalances_as_it_goes_rather_than_draining_one_label(self):
        """Two labels tied 2-2: evicting a full batch should take one of each,
        not two of the same label just because it started the tally."""
        candidates = [
            _candidate("cat1", labels=["cat"]),
            _candidate("cat2", labels=["cat"]),
            _candidate("dog1", labels=["dog"]),
            _candidate("dog2", labels=["dog"]),
        ]
        evicted = [c.uuid for c in select_for_eviction(candidates, 2, EVICTION_LABEL_BALANCE)]
        evicted_species = {uuid[:3] for uuid in evicted}  # "cat" / "dog"
        assert evicted_species == {"cat", "dog"}

    def test_unlabeled_candidates_are_evicted_last(self):
        candidates = [
            _candidate("unlabeled", labels=[]),
            _candidate("cat", labels=["cat"]),
        ]
        evicted = select_for_eviction(candidates, 1, EVICTION_LABEL_BALANCE)
        assert [c.uuid for c in evicted] == ["cat"]

    def test_ties_fall_back_to_oldest_first(self):
        candidates = [
            _candidate("newer", age_s=1, labels=["cat"]),
            _candidate("older", age_s=0, labels=["cat"]),
        ]
        evicted = select_for_eviction(candidates, 1, EVICTION_LABEL_BALANCE)
        assert [c.uuid for c in evicted] == ["older"]
