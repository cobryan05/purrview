"""How many write transactions one canvas save costs.

SQLite allows a single writer at a time, WAL or not. While a collect run or a
training is writing, every transaction a save opens queues behind it — so the
cost of a save is not the work it does (measured at ~50ms of decode and hashing
for a 11-box frame) but the number of times it asks for the writer lock. Saves
were taking 3+ seconds against a busy database while doing ~50ms of work.

This pins the transaction count. It is a real contract, not an implementation
detail: adding a stray `await self._db_client.<some write>` to the save path is
exactly the regression that made saves slow, and it is invisible in any test
that only asserts the data landed.
"""

import uuid as uuid_mod

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from apps.helpers.db.types import UNVERIFIED_TAG_UUID, VERIFIED_TAG_UUID
from tests.db_server.test_phash import _make_png


@pytest.fixture
def image_file(files_root):
    p = files_root / "txn_test.png"
    p.write_bytes(_make_png())
    return p


@pytest.fixture
async def two_labels(http_client):
    a = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    b = await http_client.post("/api/labels", json={"name": "dog", "color": "#00FF00"})
    return a.json()["label"]["uuid"], b.json()["label"]["uuid"]


@pytest.fixture
def count_commits(monkeypatch):
    """Count every committed transaction, whoever opens it."""
    calls = {"n": 0}
    original = AsyncSession.commit

    async def _counting_commit(self, *args, **kwargs):
        calls["n"] += 1
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "commit", _counting_commit)
    return calls


def _box(label_uuid, x, tags):
    return {
        "uuid": str(uuid_mod.uuid4()),
        "labelUuid": label_uuid,
        "x": x,
        "y": 0.1,
        "width": 0.2,
        "height": 0.2,
        "tagUuids": tags,
    }


class TestSaveTransactionCount:
    async def test_a_save_does_not_open_a_transaction_per_label(
        self, http_client, image_file, two_labels, count_commits
    ):
        """The PRESENT sync used one upsert transaction per verified label, plus
        a delete — so the lock was taken N+1 times for bookkeeping alone, and a
        heavily-labelled image paid for every one of them."""
        cat, dog = two_labels
        boxes = [
            _box(cat, 0.1, [VERIFIED_TAG_UUID]),
            _box(dog, 0.4, [VERIFIED_TAG_UUID]),
            _box(cat, 0.7, [UNVERIFIED_TAG_UUID]),
        ]

        count_commits["n"] = 0
        r = await http_client.post("/api/images/metadata/update", json={"imagePath": "txn_test.png", "boxes": boxes})
        assert r.status_code == 200

        # Two verified labels present. Before batching this was ~5 commits and
        # grew with the label count; the geometry write and the label sync are
        # each one transaction, and neither scales with the number of boxes.
        assert count_commits["n"] <= 3, f"a save opened {count_commits['n']} transactions"

    async def test_the_count_does_not_grow_with_more_labels(self, http_client, image_file, two_labels, count_commits):
        """The property that matters: cost per save is flat, not per-label."""
        cat, dog = two_labels

        # Warm-up save. The first write for an image also inserts its row and
        # computes the image phash (once, since the file never changes), so
        # comparing against it would measure first-write cost, not label count.
        await http_client.post(
            "/api/images/metadata/update",
            json={"imagePath": "txn_test.png", "boxes": [_box(cat, 0.1, [VERIFIED_TAG_UUID])]},
        )

        count_commits["n"] = 0
        await http_client.post(
            "/api/images/metadata/update",
            json={"imagePath": "txn_test.png", "boxes": [_box(cat, 0.1, [VERIFIED_TAG_UUID])]},
        )
        one_label = count_commits["n"]

        count_commits["n"] = 0
        await http_client.post(
            "/api/images/metadata/update",
            json={
                "imagePath": "txn_test.png",
                "boxes": [
                    _box(cat, 0.1, [VERIFIED_TAG_UUID]),
                    _box(dog, 0.4, [VERIFIED_TAG_UUID]),
                ],
            },
        )
        two_labels_count = count_commits["n"]

        assert two_labels_count == one_label, (
            f"{one_label} transactions for one label but {two_labels_count} for two — "
            "the save is still paying per label"
        )
