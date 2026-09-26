"""Which already-saved, still-unverified image a cap-exceeding collector deletes
first — pure selection logic, deliberately kept apart from `TrainingCollector`'s
own I/O (the DB query that finds candidates, the file reads a phash needs, the
delete call itself). Everything here operates on plain data already in hand, so
it is testable without a database, a filesystem, or a running collector.

Three strategies, one shape: given a list of candidates and how many to evict,
return that many, worst-first. The caller decides how many that is (a function
of the configured caps and how far over them the queue currently sits) and does
the actual deleting.

Above the strategies sits a tier order that none of them can override, because
*which* image is worth keeping is not a question about what it looks like. A
frame the model was confident and confused about is the scarcest thing in the
queue — one correction there fixes a systematic error rather than a single hard
frame — and `least_unique` would otherwise select against exactly those, since a
systematic confusion is by definition the mistake that recurs and so looks like
a near-duplicate. A frame saved moments ago has simply not been seen by anyone
yet. Both are evicted only once there is nothing ordinary left, which keeps the
cap enforceable: protection reorders the queue, it never shrinks what can be
deleted.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

from ..hashUtils import hamming
from .collectorSpec import EVICTION_FIFO, EVICTION_LABEL_BALANCE, EVICTION_LEAST_UNIQUE


@dataclass
class EvictionCandidate:
    """One still-unverified image the cap might delete.

    `phash` is optional and filled in by the caller only when the strategy
    needs it (`least_unique`) — computing it means reading the file back off
    disk, which the other two strategies have no reason to pay for.
    """

    uuid: str
    filename: str
    created_at: datetime
    #: Distinct labels across this image's bboxes. Empty for a collect-all
    #: collector's boxless frames, or for an annotated one whose boxes
    #: happen to carry no label (shouldn't happen, but a candidate must
    #: never crash eviction over it).
    label_uuids: tuple[str, ...] = field(default_factory=tuple)
    phash: Optional[str] = None
    #: The model scored two classes close together on one of this image's boxes —
    #: it was confident and confused, which is the scarcest and most useful thing
    #: the collector knows how to find.
    confusable: bool = False
    #: Saved too recently for anyone to have looked at it yet.
    recent: bool = False

    @property
    def tier(self) -> int:
        """Which round of eviction this candidate belongs to. Lower goes first."""
        if self.confusable:
            return TIER_CONFUSABLE
        if self.recent:
            return TIER_RECENT
        return TIER_ORDINARY


#: Eviction rounds, emptied in this order. A later tier is touched only once every
#: earlier one is exhausted, so a cap stays enforceable however much of the queue is
#: protected — the order changes, the total that can be deleted does not.
TIER_ORDINARY = 0
TIER_RECENT = 1
TIER_CONFUSABLE = 2
TIER_ORDER = (TIER_ORDINARY, TIER_RECENT, TIER_CONFUSABLE)


def select_for_eviction(candidates: Sequence[EvictionCandidate], count: int, strategy: str) -> list[EvictionCandidate]:
    """The `count` candidates to delete, worst-first, under `strategy`.

    `count` is clamped to `len(candidates)` — the caller may ask for more than
    exist when its own bookkeeping (`total_count` from a paginated query) sees
    more unverified images than fit in the bounded candidate page it fetched;
    this pass simply does what it can and a later one continues the cleanup.
    """
    count = max(0, min(count, len(candidates)))
    if count == 0:
        return []
    if strategy not in (EVICTION_LEAST_UNIQUE, EVICTION_LABEL_BALANCE, EVICTION_FIFO):
        raise ValueError(f"Unknown eviction strategy {strategy!r}")

    # Tier first, strategy within it. The strategies rank on appearance, labels or
    # age, none of which can see that a frame is the one worth keeping — so the
    # ordering they produce is applied inside each round rather than across the
    # whole queue. `least_unique` in particular ranks a recurring confusion as a
    # near-duplicate, which is the opposite of what it is worth.
    ordered: list[EvictionCandidate] = []
    for tier in TIER_ORDER:
        members = [c for c in candidates if c.tier == tier]
        if not members:
            continue
        ordered.extend(_rank_within_tier(members, strategy, candidates))
        if len(ordered) >= count:
            break
    return ordered[:count]


def _rank_within_tier(
    members: Sequence[EvictionCandidate],
    strategy: str,
    everyone: Sequence[EvictionCandidate],
) -> list[EvictionCandidate]:
    """One tier's candidates, worst-first under `strategy`.

    `everyone` is the whole queue, and only the similarity strategy uses it: whether a
    frame is redundant is a fact about the queue, not about its tier. Measuring it
    tier-locally would hide the most clear-cut deletion there is — an ordinary frame
    that duplicates a protected one, which adds nothing precisely because the copy
    being kept is the one worth keeping.
    """
    if strategy == EVICTION_LEAST_UNIQUE:
        return _select_least_unique(members, len(members), reference=everyone)
    if strategy == EVICTION_LABEL_BALANCE:
        return _select_label_balance(members, len(members))
    return _select_fifo(members, len(members))


def _select_fifo(candidates: Sequence[EvictionCandidate], count: int) -> list[EvictionCandidate]:
    return sorted(candidates, key=lambda c: c.created_at)[:count]


def _select_least_unique(
    candidates: Sequence[EvictionCandidate],
    count: int,
    reference: Optional[Sequence[EvictionCandidate]] = None,
) -> list[EvictionCandidate]:
    """Evict the candidate closest (smallest Hamming distance) to some other
    candidate first — the queue's near-duplicates, not its distinct frames.

    A candidate with no phash (the caller couldn't hash its file — missing
    from disk, read error) is excluded rather than guessed at: `hamming`
    needs two real hashes, and treating a missing one as "infinitely unique"
    or "maximally redundant" are both fabricating a fact that isn't there.
    Falls through to FIFO order for those, appended after every real-hash
    candidate has had its say, so a hashing failure never blocks eviction
    outright.
    """
    hashed = [c for c in candidates if c.phash]
    unhashed = [c for c in candidates if not c.phash]
    #: What "looks like something else" is measured against — the whole queue when the
    #: caller supplies it, so a frame that duplicates one in another tier still reads as
    #: redundant. Only these are compared against; only `candidates` can be returned.
    against = [c for c in (reference if reference is not None else candidates) if c.phash]

    def nearest_distance(target: EvictionCandidate) -> float:
        # Hamming distance to whichever OTHER hashed candidate is closest —
        # "closest" is what "least unique" means: this frame looks like
        # something else already in the queue.
        #
        # `inf` and not 0 when there is nobody to compare against (a page with a
        # single hashable candidate): "no near-duplicate found" is maximally
        # unique, and defaulting to 0 ranked the one distinct frame FIRST for
        # deletion — the exact inverse of what the strategy promises.
        return min(
            (hamming(target.phash, other.phash) for other in against if other is not target),
            default=math.inf,
        )

    ranked = sorted(hashed, key=nearest_distance)
    ordered = ranked + _select_fifo(unhashed, len(unhashed))
    return ordered[:count]


def _select_label_balance(candidates: Sequence[EvictionCandidate], count: int) -> list[EvictionCandidate]:
    """Evict from whichever label currently has the most unverified images,
    one at a time, re-tallying after each pick — a greedy pass that keeps
    pulling from today's largest pile rather than committing to one label for
    the whole batch (which would just hollow out a single label instead of
    balancing the queue).

    A candidate with no labels at all (a collect-all collector's boxless
    frames) has nothing to balance against and is treated as the LOWEST
    priority to evict — there is no over-represented subject to relieve by
    deleting it, so it only goes once every labeled candidate that could
    plausibly help balance the set has already been picked. Ties (including
    "everyone is unlabeled") fall back to oldest-first, so the result is
    still deterministic.
    """
    remaining = sorted(candidates, key=lambda c: c.created_at)
    counts: Counter[str] = Counter(label for c in remaining for label in c.label_uuids)
    evicted: list[EvictionCandidate] = []

    for _ in range(count):
        if not remaining:
            break
        worst = max((counts[label] for label in remaining[0].label_uuids), default=-1)
        victim = remaining[0]
        victim_idx = 0
        for idx, candidate in enumerate(remaining):
            score = max((counts[label] for label in candidate.label_uuids), default=-1)
            if score > worst:
                worst = score
                victim = candidate
                victim_idx = idx
        remaining.pop(victim_idx)
        for label in victim.label_uuids:
            counts[label] -= 1
        evicted.append(victim)

    return evicted
