"""The review queue's `total_count` must not outlive the rows it counted.

The count is expensive, so `Manager.get_review_queue` caches it — keyed on the
filter, with a 60s TTL and nothing else. A TTL is the right tool for "this is
costly and slightly stale is fine"; it is the wrong tool for "the user just
deleted every row this number describes".

The reported symptom: "Apply to filtered images" deleted all 500 matching
images, the grid went empty, and the toolbar went on saying
`Apply to Filtered Images (500)` and `Page 1 of 2 (0 cards)` — the list query
was right and the count beside it was a minute out of date. Every counter in
that toolbar is derived from `totalCount`, so one stale number made the whole
bar lie at once.

These tests go through the HTTP endpoints rather than poking the cache, because
the cache is an implementation detail and the contract is "the count agrees with
the images returned beside it".
"""

from httpx import ASGITransport, AsyncClient

from apps.db_server.manager import Manager
from apps.db_server.web import WebApp
from tests.conftest import make_png


def _png(files_root, name: str) -> str:
    (files_root / name).write_bytes(make_png())
    return name


async def _register(http_client, name: str) -> str:
    r = await http_client.post("/api/images/metadata/update", json={"imagePath": name, "boxes": []})
    assert r.status_code == 200, r.text
    return name


async def _review(http_client, **payload) -> dict:
    r = await http_client.post("/api/images/review", json=payload or {})
    assert r.status_code == 200, r.text
    return r.json()


async def _uuid_of(http_client, filename: str) -> str:
    body = await _review(http_client)
    for img in body["images"]:
        if img["filename"] == filename:
            return img["uuid"]
    raise AssertionError(f"{filename} not in review queue: {[i['filename'] for i in body['images']]}")


class TestReviewCountFollowsDeletes:
    async def test_count_drops_when_an_image_is_deleted(self, http_client, files_root):
        """The first query populates the count cache; the delete must retire it.

        Without the write epoch this returned 3 — the cached count — while
        `images` correctly held 2.
        """
        for name in ("a.png", "b.png", "c.png"):
            _png(files_root, name)
            await _register(http_client, name)

        first = await _review(http_client)
        assert first["totalCount"] == 3

        uuid = await _uuid_of(http_client, "b.png")
        r = await http_client.delete(f"/api/images/{uuid}")
        assert r.status_code == 200, r.text

        after = await _review(http_client)
        assert len(after["images"]) == 2
        assert after["totalCount"] == 2, "count served from cache after a delete"

    async def test_emptying_the_result_set_reports_zero(self, http_client, files_root):
        """The reported case, reduced: delete everything the filter matched and
        the count must reach 0, not stay at what it was."""
        names = ("x.png", "y.png")
        for name in names:
            _png(files_root, name)
            await _register(http_client, name)
        assert (await _review(http_client))["totalCount"] == 2

        for name in names:
            uuid = await _uuid_of(http_client, name)
            assert (await http_client.delete(f"/api/images/{uuid}")).status_code == 200

        after = await _review(http_client)
        assert after["images"] == []
        assert after["totalCount"] == 0, "an empty grid was still reporting a non-zero total"

    async def test_count_rises_when_an_image_is_added(self, http_client, files_root):
        """The same rule in the other direction — a collector saving frames into
        a filter the user is looking at must not be invisible for a minute."""
        _png(files_root, "one.png")
        await _register(http_client, "one.png")
        assert (await _review(http_client))["totalCount"] == 1

        _png(files_root, "two.png")
        await _register(http_client, "two.png")

        assert (await _review(http_client))["totalCount"] == 2

    async def test_repeated_identical_queries_still_agree(self, http_client, files_root):
        """The cache must still be allowed to do its job: with no write in
        between, the count is stable and matches the rows."""
        for name in ("p.png", "q.png"):
            _png(files_root, name)
            await _register(http_client, name)

        first = await _review(http_client)
        second = await _review(http_client)

        assert first["totalCount"] == second["totalCount"] == 2
        assert len(second["images"]) == 2


class TestReviewCountSurvivesConcurrentWrite:
    """A write that lands *during* the count query, not between requests.

    The cache stores `(total_count, timestamp, epoch)` and only trusts the
    count while the stored epoch still matches the live one. That guard is
    worthless if the epoch it stores was read *after* the count query
    finished: a write landing mid-query bumps the epoch before the read, so
    the stale count gets stamped with the new epoch and looks fresh forever
    (until the next write or the TTL). The fix is to tag the cache entry with
    the epoch as of when the query *started*.
    """

    async def test_write_during_count_query_is_not_masked_by_a_late_epoch_read(self, db_client, files_root):
        manager = Manager(db_client=db_client, files_root=files_root)
        manager._db_ready = True
        web_app = WebApp(app_name="test", manager=manager)
        transport = ASGITransport(app=web_app.app())

        for name in ("a.png", "b.png", "c.png"):
            (files_root / name).write_bytes(make_png())

        async with AsyncClient(transport=transport, base_url="http://test") as http_client:
            for name in ("a.png", "b.png", "c.png"):
                await _register(http_client, name)

            real_count = manager._count_images_bounded
            deleted = {"done": False}

            async def racing_count(params):
                # Simulate a delete committing (and bumping write_epoch) after the
                # count query has already read the pre-delete row set, but before
                # this coroutine returns control to the caller.
                result = await real_count(params)
                if not deleted["done"]:
                    deleted["done"] = True
                    uuid = await _uuid_of(http_client, "b.png")
                    r = await http_client.delete(f"/api/images/{uuid}")
                    assert r.status_code == 200, r.text
                return result

            manager._count_images_bounded = racing_count
            try:
                first = await _review(http_client)
            finally:
                manager._count_images_bounded = real_count

            assert first["totalCount"] == 3, "count query ran before the delete, so it should see all 3"

            after = await _review(http_client)
            assert len(after["images"]) == 2
            assert after["totalCount"] == 2, (
                "the cache entry from the raced request was stamped with the epoch "
                "read after the delete, so it looked fresh and masked the delete"
            )
