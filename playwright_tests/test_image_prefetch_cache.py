"""The prefetch cache behind `fetchImage(path, { allowCached })`.

The cache is what makes canvas next/prev cheap on a slow DB (see
TestCanvasNavPrefetch for the navigation itself). These tests pin the rules that
decide whether it is *safe*: what it may serve, what must invalidate it, and
what must never be cached at all.
"""

import json
import urllib.parse

from playwright.sync_api import Page, Route

TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

IMAGE_PATH = "folder/prefetch1.jpg"


def _route_file(page: Page, *, fail: bool = False):
    """Serve /file and record every request path. `fail` makes it 500."""
    calls: list[str] = []

    def handle(route: Route):
        path = urllib.parse.unquote(route.request.url.split("/api/images/")[1].rsplit("/file", 1)[0])
        calls.append(path)
        if fail:
            route.fulfill(status=500, content_type="application/json", body=json.dumps({"status": "failure"}))
            return
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "imageBase64": TINY_PNG_B64,
                    "mimeType": "image/png",
                    "image": {"uuid": "img-p1", "filename": path, "bboxes": []},
                }
            ),
        )

    page.route("**/api/images/*/file", handle)
    return calls


def _load(page: Page, base_url: str, fail: bool = False):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    return _route_file(page, fail=fail)


def test_a_prefetched_image_is_served_without_a_second_request(page: Page, base_url: str):
    calls = _load(page, base_url)

    page.evaluate(
        """async (path) => {
        const api = await import('/app-static/js/shared/api/images.js');
        api.prefetchImage(path);
        await api.fetchImage(path, { allowCached: true });
        await api.fetchImage(path, { allowCached: true });
    }""",
        IMAGE_PATH,
    )

    assert calls == [IMAGE_PATH], f"prefetch + 2 cached reads should be one request, got {calls}"


def test_each_caller_gets_its_own_bboxes_from_the_shared_entry(page: Page, base_url: str):
    """The cache holds the raw response, not the mapped UIImage. A canvas mutates
    the UIBbox objects it is handed, so handing two loads the same objects would
    let an edit in one canvas silently appear in the other."""
    _load(page, base_url)

    same = page.evaluate(
        """async (path) => {
        const api = await import('/app-static/js/shared/api/images.js');
        const a = await api.fetchImage(path, { allowCached: true });
        const b = await api.fetchImage(path, { allowCached: true });
        return { bboxes: a.bboxes === b.bboxes, image: a.image === b.image };
    }""",
        IMAGE_PATH,
    )

    assert same["bboxes"] is False, "two loads must not share one bbox collection"
    # The decoded bitmap IS shared on purpose — it is read-only and the expensive part.
    assert same["image"] is True, "the decoded image element should be reused"


def test_uncached_is_the_default_so_an_explicit_reload_sees_the_server(page: Page, base_url: str):
    calls = _load(page, base_url)

    page.evaluate(
        """async (path) => {
        const api = await import('/app-static/js/shared/api/images.js');
        api.prefetchImage(path);
        await api.fetchImage(path, { allowCached: true });
        await api.fetchImage(path);
    }""",
        IMAGE_PATH,
    )

    assert calls == [IMAGE_PATH, IMAGE_PATH], f"the default call must hit the network, got {calls}"


def test_a_committed_bbox_change_invalidates_the_cached_copy(page: Page, base_url: str):
    """Without this, navigating back onto an image shows it as it was before the
    edit — the failure mode that makes caching here dangerous rather than slow."""
    calls = _load(page, base_url)

    page.evaluate(
        """async (path) => {
        const api = await import('/app-static/js/shared/api/images.js');
        const { events, EventTypes } = await import('/app-static/js/shared/events/index.js');
        await api.fetchImage(path, { allowCached: true });
        events.publish(EventTypes.IMAGE_BBOXES_CHANGED, { imageUuid: 'img-p1', imageFilename: path });
        await api.fetchImage(path, { allowCached: true });
    }""",
        IMAGE_PATH,
    )

    assert calls == [IMAGE_PATH, IMAGE_PATH], f"the stale entry was served after a change, got {calls}"


def test_an_unrelated_image_change_does_not_flush_the_cache(page: Page, base_url: str):
    """Invalidation is per path — a change elsewhere must not throw away the
    window the user is about to step through."""
    calls = _load(page, base_url)

    page.evaluate(
        """async (path) => {
        const api = await import('/app-static/js/shared/api/images.js');
        const { events, EventTypes } = await import('/app-static/js/shared/events/index.js');
        await api.fetchImage(path, { allowCached: true });
        events.publish(EventTypes.IMAGE_BBOXES_CHANGED, { imageUuid: 'other', imageFilename: 'folder/other.jpg' });
        await api.fetchImage(path, { allowCached: true });
    }""",
        IMAGE_PATH,
    )

    assert calls == [IMAGE_PATH], f"an unrelated change flushed this entry, got {calls}"


def test_a_failed_fetch_is_not_cached(page: Page, base_url: str):
    """A cached rejection would make the path permanently unloadable — the retry
    would keep replaying the original error instead of reaching the server."""
    calls = _load(page, base_url, fail=True)

    errors = page.evaluate(
        """async (path) => {
        const api = await import('/app-static/js/shared/api/images.js');
        const out = [];
        for (let i = 0; i < 2; i++) {
            try { await api.fetchImage(path, { allowCached: true }); out.push('ok'); }
            catch (e) { out.push('err'); }
        }
        return out;
    }""",
        IMAGE_PATH,
    )

    assert errors == ["err", "err"]
    assert calls == [IMAGE_PATH, IMAGE_PATH], f"the retry was served a cached failure, got {calls}"


def test_the_cache_is_bounded(page: Page, base_url: str):
    """Entries pin decoded full-resolution images, so an unbounded cache would
    grow with every image navigated past."""
    calls = _load(page, base_url)

    page.evaluate(
        """async () => {
        const api = await import('/app-static/js/shared/api/images.js');
        // Fill well past the bound, then come back to the first one.
        for (let i = 0; i < 8; i++) await api.fetchImage(`folder/bounded${i}.jpg`, { allowCached: true });
        await api.fetchImage('folder/bounded0.jpg', { allowCached: true });
    }""",
    )

    assert calls.count("folder/bounded0.jpg") == 2, f"the oldest entry should have been evicted, got {calls}"
