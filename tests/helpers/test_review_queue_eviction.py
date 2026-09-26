"""Tests for `apps.helpers.training.reviewQueueEviction.select_for_eviction`.

Pure selection logic — no DB, no filesystem, no collector — so these assert the
actual eviction *decision* each strategy makes, not just that it runs.
"""

from datetime import datetime, timedelta

import pytest

from apps.helpers.training.collectorSpec import EVICTION_FIFO, EVICTION_LABEL_BALANCE, EVICTION_LEAST_UNIQUE
from apps.helpers.training.reviewQueueEviction import EvictionCandidate, select_for_eviction

BASE = datetime(2026, 1, 1)


def _candidate(uuid, *, age_s=0, labels=(), phash=None, confusable=False, recent=False):
    return EvictionCandidate(
        uuid=uuid,
        filename=f"{uuid}.jpg",
        created_at=BASE + timedelta(seconds=age_s),
        label_uuids=tuple(labels),
        phash=phash,
        confusable=confusable,
        recent=recent,
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


class TestTiers:
    """Some frames are worth more than whatever the strategy thinks of the picture.

    A frame the model scored two classes near-equally on is the scarcest thing in the
    queue, and `least_unique` actively selects against those: a systematic confusion is
    the mistake that recurs, so it looks like a near-duplicate. A frame saved moments
    ago has simply not been seen yet. Both wait until nothing ordinary is left.
    """

    def test_an_ordinary_frame_goes_before_a_confusable_one(self):
        candidates = [_candidate("confusable", age_s=0, confusable=True), _candidate("ordinary", age_s=5)]

        assert [c.uuid for c in select_for_eviction(candidates, 1, EVICTION_FIFO)] == ["ordinary"]

    def test_an_ordinary_frame_goes_before_a_recent_one(self):
        candidates = [_candidate("recent", age_s=0, recent=True), _candidate("ordinary", age_s=5)]

        assert [c.uuid for c in select_for_eviction(candidates, 1, EVICTION_FIFO)] == ["ordinary"]

    def test_a_recent_frame_goes_before_a_confusable_one(self):
        candidates = [_candidate("confusable", confusable=True), _candidate("recent", recent=True)]

        assert [c.uuid for c in select_for_eviction(candidates, 1, EVICTION_FIFO)] == ["recent"]

    def test_confusable_wins_over_recent_on_the_same_frame(self):
        """A frame that is both is protected by the stronger of the two reasons."""
        candidates = [_candidate("both", confusable=True, recent=True), _candidate("recent", recent=True)]

        assert [c.uuid for c in select_for_eviction(candidates, 1, EVICTION_FIFO)] == ["recent"]

    def test_protection_never_makes_the_cap_unenforceable(self):
        """The whole point of tiering rather than exempting: a queue that is entirely
        protected must still be trimmable, or a burst of confusable frames would let it
        grow without bound."""
        candidates = [_candidate(f"c{i}", age_s=i, confusable=True) for i in range(4)]

        evicted = select_for_eviction(candidates, 3, EVICTION_FIFO)

        assert [c.uuid for c in evicted] == ["c0", "c1", "c2"]

    def test_the_strategy_still_orders_within_a_tier(self):
        """Tiers decide which round, the strategy decides the order inside it."""
        candidates = [
            _candidate("protected-old", age_s=0, confusable=True),
            _candidate("ordinary-new", age_s=9),
            _candidate("ordinary-old", age_s=1),
        ]

        assert [c.uuid for c in select_for_eviction(candidates, 3, EVICTION_FIFO)] == [
            "ordinary-old",
            "ordinary-new",
            "protected-old",
        ]

    def test_least_unique_cannot_reach_past_a_tier_for_a_near_duplicate(self):
        """The failure this exists to stop: two confusable frames of the same recurring
        mistake look like duplicates of each other, and would otherwise be deleted
        first — before a genuinely redundant ordinary frame."""
        twin_a = _candidate("confusable-twin-a", phash="ffff0000ffff0000", confusable=True)
        twin_b = _candidate("confusable-twin-b", phash="ffff0000ffff0001", confusable=True)
        ordinary = _candidate("ordinary", phash="0f0f0f0f0f0f0f0f")

        evicted = select_for_eviction([twin_a, twin_b, ordinary], 1, EVICTION_LEAST_UNIQUE)

        assert [c.uuid for c in evicted] == ["ordinary"]

    def test_redundancy_is_measured_against_the_whole_queue_not_the_tier(self):
        """A frame that duplicates a protected one is the clearest deletion there is —
        the copy being kept is the one worth keeping, so this one adds nothing.

        Ranking each tier in isolation hid exactly that: the duplicate's twin sat in
        another tier, so it looked unique and a genuinely distinct frame went instead.
        """
        protected = _candidate("protected", phash="ffff0000ffff0000", confusable=True)
        duplicate_of_it = _candidate("duplicate", age_s=5, phash="ffff0000ffff0000")
        distinct = _candidate("distinct", age_s=1, phash="0f0f0f0f0f0f0f0f")

        evicted = select_for_eviction([protected, duplicate_of_it, distinct], 1, EVICTION_LEAST_UNIQUE)

        assert [c.uuid for c in evicted] == ["duplicate"]
