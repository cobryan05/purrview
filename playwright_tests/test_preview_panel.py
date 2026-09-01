"""
Tests for app/review/previewPanel.js — the persisted preview-run triage panel
(`createPreviewPanel`, `openPreviewRunModal`, `openSavedPreviewsModal`), the
candidate-card variant in cardGrid.js (`buildCandidateCard`), and the "Saved
Previews" toolbar button in review/main.js.

Because the Playwright `live_server` proxies are static mocks with no DB behind
them, every `/api/preview-runs*` endpoint is served by `FakePreviewApi` below —
an in-memory, stateful fake that mirrors the real db_server contract closely
enough to exercise pagination, filtering and decision persistence.

Run with:
    docker compose --profile dev run --rm --no-deps -w /project purrview-dev \
        python3.11 -m pytest playwright_tests/test_preview_panel.py -q
"""

import json
from urllib.parse import urlsplit

import pytest
from playwright.sync_api import Page, Route, expect

LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parentUuid": None}
LABEL_DOG = {"uuid": "lbl-dog", "name": "Dog", "color": "#00ff00", "parentUuid": None}
# Child of Cat — used by the label-filter hierarchy test.
LABEL_KITTEN = {"uuid": "lbl-kitten", "name": "Kitten", "color": "#ff8800", "parentUuid": "lbl-cat"}

_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)

# Stubs the notices stream so a test can expand #notices-bar on demand — mirrors
# the fixture in test_notices_bar.py, duplicated here rather than imported so
# this file stays runnable on its own.
_FAKE_EVENT_SOURCE_SCRIPT = """
  class FakeEventSource {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSED = 2;
    static notices = null;

    constructor(url) {
      this.url = url;
      this.readyState = FakeEventSource.OPEN;
      this.onmessage = null;
      this.onerror = null;
      if (String(url).includes('/api/notices/stream')) FakeEventSource.notices = this;
    }
    close() { this.readyState = FakeEventSource.CLOSED; }
    send(payload) { this.onmessage?.({ data: JSON.stringify(payload) }); }
  }
  window.FakeEventSource = FakeEventSource;
  window.EventSource = FakeEventSource;
"""


# ---------------------------------------------------------------------------
# Mocking helpers (mirrors the conventions in test_run_model_on_results.py)
# ---------------------------------------------------------------------------


def _mock_review_queue(page: Page, images: list = None):
    images = images or []
    page.route(
        "**/api/images/review",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "images": images, "hasMore": False, "totalCount": len(images)}),
        ),
    )


def _mock_labels(page: Page, labels: list):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels") and "api/labels/" not in url,
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": labels}),
        ),
    )


def _mock_tags(page: Page, tags: list = None):
    tags = tags or []
    page.route(
        "**/api/tags/list",
        lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({"tags": tags})),
    )


def _mock_thumbnails(page: Page):
    page.route(
        "**/bbox/*/thumbnail*",
        lambda route: route.fulfill(status=200, content_type="image/png", body=_TINY_PNG),
    )


def _mock_image_thumbnails(page: Page):
    # Scoped to /api/images/.../thumbnail (the actual endpoint) rather than a
    # bare "/thumbnail" substring match — the latter also matches the JS module
    # /app-static/js/ui/thumbnailHoverPreview.js and serves it as image/png,
    # which breaks the module import chain for the whole review tab.
    page.route(
        lambda url: "/api/images/" in url and "/thumbnail" in url.split("?")[0] and "/bbox/" not in url,
        lambda route: route.fulfill(status=200, content_type="image/png", body=_TINY_PNG),
    )


def _navigate_to_review(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()
    page.locator("#tab-image-tagging .section-tab-button[data-tab='tab-review']").click()
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#review-root .review-grid")


def _open_panel(page: Page, run_uuid: str):
    """Open the triage panel directly, bypassing the run-model flow."""
    page.evaluate(
        """async (uuid) => {
            const m = await import('/app-static/js/app/review/previewPanel.js');
            window.__panel = await m.openPreviewRunModal({ runUuid: uuid });
        }""",
        run_uuid,
    )
    expect(page.get_by_test_id("preview-panel")).to_be_visible()


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_run(uuid="run-1", name="Run 1", model_name="yolo9-detector", pinned=False, **overrides):
    run = {
        "uuid": uuid,
        "name": name,
        "modelName": model_name,
        "taskTypename": "PreviewModelRelabelTask",
        "params": {},
        "scope": {},
        "stats": {},
        "pinned": pinned,
        "expiresAt": "2026-08-18T00:00:00Z",
        "createdAt": "2026-08-11T00:00:00Z",
        "updatedAt": "2026-08-11T00:00:00Z",
        "decisionCounts": {"pending": 0, "accepted": 0, "rejected": 0, "applied": 0, "stale": 0},
        "candidateCount": 0,
    }
    run.update(overrides)
    return run


def _make_candidate(
    uuid,
    run_uuid="run-1",
    image_uuid="img-1",
    image_filename="folder/image1.jpg",
    bbox_uuid="bbox-1",
    is_new=False,
    current_label="lbl-cat",
    proposed_label="lbl-dog",
    confidence=0.8,
    decision="pending",
    decision_action=None,
    source_model="yolo9-detector",
    label_changed=False,
    geometry=None,
):
    return {
        "uuid": uuid,
        "runUuid": run_uuid,
        "imageUuid": image_uuid,
        "imageFilename": image_filename,
        "bboxUuid": None if is_new else bbox_uuid,
        "isNew": is_new,
        "currentLabelUuid": None if is_new else current_label,
        "proposedLabelUuid": proposed_label,
        "confidence": confidence,
        "geometry": geometry,
        "decision": decision,
        "decisionAction": decision_action,
        "sourceModel": source_model,
        "appliedAt": None,
        "labelChangedSinceRun": label_changed,
    }


# ---------------------------------------------------------------------------
# Stateful fake for every /api/preview-runs* endpoint
# ---------------------------------------------------------------------------


class FakePreviewApi:
    """In-memory stand-in for the preview-runs REST surface.

    Applies the candidate filter/sort/paginate fields the JS actually posts,
    so pagination and filtering assertions exercise real request bodies rather
    than a hand-waved always-return-everything stub.
    """

    def __init__(self):
        self.runs = {}
        self.candidates = {}
        self.patch_bodies = []  # (runUuid, body) — PATCH /candidates
        self.run_patch_bodies = []  # (runUuid, body) — PATCH /{uuid}
        self.query_bodies = []  # (runUuid, body) — POST /candidates/query
        self.apply_calls = []  # (runUuid, body) — POST /apply
        self.delete_calls = []  # runUuid
        self.apply_result = {"modified": 0, "added": 0, "deleted": 0, "failed": 0, "stale": 0}

    def add_run(self, run, candidates):
        uuid = run["uuid"]
        self.runs[uuid] = dict(run)
        self.candidates[uuid] = {c["uuid"]: dict(c) for c in candidates}
        self._recompute(uuid)

    def _recompute(self, run_uuid):
        counts = {"pending": 0, "accepted": 0, "rejected": 0, "applied": 0, "stale": 0}
        cands = self.candidates[run_uuid]
        for c in cands.values():
            d = c.get("decision", "pending")
            counts[d] = counts.get(d, 0) + 1
        self.runs[run_uuid]["decisionCounts"] = counts
        self.runs[run_uuid]["candidateCount"] = len(cands)

    def install(self, page: Page):
        page.route(lambda url: "/api/preview-runs" in url.split("?")[0], self._handle)

    def _handle(self, route: Route):
        req = route.request
        path = urlsplit(req.url).path
        method = req.method
        idx = path.find("/api/preview-runs")
        sub = path[idx + len("/api/preview-runs") :].strip("/")
        parts = sub.split("/") if sub else []

        if not parts:
            if method == "GET":
                runs = sorted(self.runs.values(), key=lambda r: r["createdAt"], reverse=True)
                return route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "runs": runs}),
                )
            return route.fallback()

        run_uuid = parts[0]
        rest = parts[1:]

        if not rest:
            if method == "GET":
                run = self.runs.get(run_uuid)
                return route.fulfill(
                    status=200, content_type="application/json", body=json.dumps({"status": "success", "run": run})
                )
            if method == "PATCH":
                body = json.loads(req.post_data or "{}")
                self.run_patch_bodies.append((run_uuid, body))
                self.runs[run_uuid].update(body)
                return route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "run": self.runs[run_uuid]}),
                )
            if method == "DELETE":
                self.delete_calls.append(run_uuid)
                self.runs.pop(run_uuid, None)
                self.candidates.pop(run_uuid, None)
                return route.fulfill(
                    status=200, content_type="application/json", body=json.dumps({"status": "success"})
                )
            return route.fallback()

        if rest == ["candidates", "query"]:
            body = json.loads(req.post_data or "{}")
            self.query_bodies.append((run_uuid, body))
            cands = list(self.candidates.get(run_uuid, {}).values())

            decisions = body.get("decisions") or []
            if decisions:
                cands = [c for c in cands if c.get("decision") in decisions]
            label_uuids = body.get("proposedLabelUuids") or []
            if label_uuids:
                cands = [c for c in cands if c.get("proposedLabelUuid") in label_uuids]
            # Exclusion mirrors `_preview_candidate_filters`: it applies *alongside* the
            # include list rather than instead of it, so "every cat except the kitten"
            # narrows here the same way it does against the real server.
            exclude_decisions = body.get("excludeDecisions") or []
            if exclude_decisions:
                cands = [c for c in cands if c.get("decision") not in exclude_decisions]
            exclude_labels = body.get("excludeProposedLabelUuids") or []
            if exclude_labels:
                cands = [c for c in cands if c.get("proposedLabelUuid") not in exclude_labels]
            conf_min = body.get("confidenceMin")
            if conf_min is not None:
                cands = [c for c in cands if (c.get("confidence") or 0) >= conf_min]
            conf_max = body.get("confidenceMax")
            if conf_max is not None:
                cands = [c for c in cands if (c.get("confidence") or 0) <= conf_max]
            is_new = body.get("isNew")
            if is_new is not None:
                cands = [c for c in cands if bool(c.get("isNew")) == is_new]

            # The two grouping sorts are what keep a whole image / a whole class
            # contiguous across pages; the panel draws its label-group headers
            # on the assumption that the server already did this.
            sort_by = body.get("sortBy")
            group_key = {"image_confidence": "imageUuid", "label_confidence": "proposedLabelUuid"}.get(sort_by)
            if group_key:
                best = {}
                for c in cands:
                    k = c.get(group_key)
                    best[k] = max(best.get(k, 0.0), c.get("confidence") or 0.0)
                cands.sort(key=lambda c: (-best[c.get(group_key)], str(c.get(group_key)), -(c.get("confidence") or 0)))
            elif sort_by == "confidence":
                cands.sort(key=lambda c: -(c.get("confidence") or 0))

            total = len(cands)
            offset = body.get("offset", 0)
            limit = body.get("limit", 100)
            page_cands = cands[offset : offset + limit]
            return route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "candidates": page_cands, "totalCount": total}),
            )

        if rest == ["candidates"] and method == "PATCH":
            body = json.loads(req.post_data or "{}")
            self.patch_bodies.append((run_uuid, dict(body)))
            uuids = body.get("candidateUuids", [])
            for u in uuids:
                c = self.candidates.get(run_uuid, {}).get(u)
                if not c:
                    continue
                for k in ("decision", "decisionAction", "proposedLabelUuid", "geometry"):
                    if k in body:
                        c[k] = body[k]
            self._recompute(run_uuid)
            return route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "updated": len(uuids)}),
            )

        if rest == ["apply"] and method == "POST":
            body = json.loads(req.post_data or "{}")
            self.apply_calls.append((run_uuid, body))
            return route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "result": self.apply_result}),
            )

        return route.fallback()


def _wait_for_patch(page: Page, api: FakePreviewApi, min_count=1, timeout_ms=2000):
    """Polls FakePreviewApi.patch_bodies until at least `min_count` PATCH
    bodies have arrived — decisions are debounced ~400ms, so a straight
    assertion right after the click would race the timer.

    Uses page.wait_for_timeout (not time.sleep) to poll: route callbacks are
    delivered on Playwright's driver thread and only get pumped into Python
    when a Playwright call is made from the main thread, so a bare
    time.sleep loop never observes them and always times out.
    """
    step_ms = 50
    waited = 0
    while waited < timeout_ms:
        if len(api.patch_bodies) >= min_count:
            return
        page.wait_for_timeout(step_ms)
        waited += step_ms


def _setup_common(page: Page, base_url: str):
    _mock_review_queue(page)
    _mock_labels(page, [LABEL_CAT, LABEL_DOG])
    _mock_tags(page)
    _mock_thumbnails(page)
    _mock_image_thumbnails(page)
    _navigate_to_review(page, base_url)


# ---------------------------------------------------------------------------
# 1. Rendering
# ---------------------------------------------------------------------------


class TestPreviewPanelRendering:
    def test_renders_one_card_per_candidate_with_counts(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(
            _make_run(),
            [_make_candidate(f"c{i}") for i in range(3)],
        )
        api.install(page)

        _open_panel(page, "run-1")

        cards = page.get_by_test_id("preview-candidate-card")
        expect(cards).to_have_count(3)
        expect(page.get_by_test_id("preview-count")).to_contain_text("3")
        expect(page.get_by_test_id("preview-page-label")).to_have_text("Page 1 / 1")


# ---------------------------------------------------------------------------
# 2. Card content
# ---------------------------------------------------------------------------


class TestCandidateCardContent:
    def test_existing_box_shows_chips_and_confidence(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(
            _make_run(),
            [_make_candidate("c1", current_label="lbl-cat", proposed_label="lbl-dog", confidence=0.81)],
        )
        api.install(page)

        _open_panel(page, "run-1")

        card = page.locator('[data-candidate-uuid="c1"]')
        expect(card).to_contain_text("Cat")
        expect(card).to_contain_text("Dog")
        expect(card).to_contain_text("81%")
        expect(card.locator('[data-testid="preview-decision-badge"]')).to_have_count(0)

    def test_new_candidate_shows_new_badge(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(
            _make_run(),
            [
                _make_candidate(
                    "c-new",
                    is_new=True,
                    proposed_label="lbl-dog",
                    confidence=0.5,
                    geometry={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                )
            ],
        )
        api.install(page)

        _open_panel(page, "run-1")

        card = page.locator('[data-candidate-uuid="c-new"]')
        expect(card).to_contain_text("NEW")

    def test_new_candidate_thumb_is_a_region_crop_not_the_whole_frame(self, page: Page, base_url: str):
        """A proposed new box has no bbox row to crop by uuid, but it does have
        geometry — the card must show that crop, or a small proposal is a few
        pixels in the middle of the frame."""
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(
            _make_run(),
            [
                _make_candidate(
                    "c-new",
                    is_new=True,
                    proposed_label="lbl-dog",
                    geometry={"x": 0.25, "y": 0.5, "width": 0.1, "height": 0.2},
                )
            ],
        )
        api.install(page)

        _open_panel(page, "run-1")

        src = page.locator('[data-candidate-uuid="c-new"] img.thumb').get_attribute("src")
        assert "/region/thumbnail" in src, src
        assert "x=0.25" in src and "width_frac=0.1" in src and "height_frac=0.2" in src, src

    def test_new_candidate_without_geometry_falls_back_to_whole_image(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(_make_run(), [_make_candidate("c-new", is_new=True, geometry=None)])
        api.install(page)

        _open_panel(page, "run-1")

        src = page.locator('[data-candidate-uuid="c-new"] img.thumb').get_attribute("src")
        assert "/region/thumbnail" not in src, src
        assert "/thumbnail" in src, src

    def test_changed_since_run_shows_badge(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(_make_run(), [_make_candidate("c1", label_changed=True)])
        api.install(page)

        _open_panel(page, "run-1")

        card = page.locator('[data-candidate-uuid="c1"]')
        expect(card.get_by_test_id("preview-changed-badge")).to_be_visible()

    def test_accepted_decision_shows_decision_badge(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(_make_run(), [_make_candidate("c1", decision="accepted")])
        api.install(page)

        _open_panel(page, "run-1")

        card = page.locator('[data-candidate-uuid="c1"]')
        expect(card.get_by_test_id("preview-decision-badge")).to_be_visible()
        expect(card.get_by_test_id("preview-decision-badge")).to_contain_text("Accepted")


# ---------------------------------------------------------------------------
# 3 & 4. Decision actions + coalescing
# ---------------------------------------------------------------------------


def _select_card(page: Page, candidate_uuid: str):
    page.locator(f'[data-candidate-uuid="{candidate_uuid}"] input[type=checkbox]').click()


class TestDecisionActions:
    def test_accept_sends_relabel_patch(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(_make_run(), [_make_candidate("c1")])
        api.install(page)
        _open_panel(page, "run-1")

        _select_card(page, "c1")
        page.get_by_test_id("preview-accept").click()

        _wait_for_patch(page, api)
        assert len(api.patch_bodies) == 1
        run_uuid, body = api.patch_bodies[0]
        assert run_uuid == "run-1"
        assert body["candidateUuids"] == ["c1"]
        assert body["decision"] == "accepted"
        assert body["decisionAction"] == "relabel"

    def test_accept_as_delete_sends_delete_action(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(_make_run(), [_make_candidate("c1")])
        api.install(page)
        _open_panel(page, "run-1")

        _select_card(page, "c1")
        page.get_by_test_id("preview-accept-delete").click()

        _wait_for_patch(page, api)
        assert len(api.patch_bodies) == 1
        _, body = api.patch_bodies[0]
        assert body["decision"] == "accepted"
        assert body["decisionAction"] == "delete"

    def test_reject_sends_rejected(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(_make_run(), [_make_candidate("c1")])
        api.install(page)
        _open_panel(page, "run-1")

        _select_card(page, "c1")
        page.get_by_test_id("preview-reject").click()

        _wait_for_patch(page, api)
        assert len(api.patch_bodies) == 1
        _, body = api.patch_bodies[0]
        assert body["decision"] == "rejected"
        assert "decisionAction" not in body

    def test_unreview_sends_pending(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(_make_run(), [_make_candidate("c1", decision="accepted")])
        api.install(page)
        _open_panel(page, "run-1")

        _select_card(page, "c1")
        page.get_by_test_id("preview-reset").click()

        _wait_for_patch(page, api)
        assert len(api.patch_bodies) == 1
        _, body = api.patch_bodies[0]
        assert body["decision"] == "pending"


class TestSelectAllNone:
    def _run(self, api: FakePreviewApi, n=4):
        api.add_run(_make_run(), [_make_candidate(f"c{i}") for i in range(n)])

    def test_select_all_selects_every_card_on_the_page(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        self._run(api)
        api.install(page)
        _open_panel(page, "run-1")

        page.get_by_test_id("preview-select-all").click()

        expect(page.get_by_test_id("preview-selection-count")).to_have_text("4 selected")
        expect(page.locator(".review-card.selected")).to_have_count(4)

        page.get_by_test_id("preview-accept").click()
        _wait_for_patch(page, api)
        _, body = api.patch_bodies[-1]
        assert sorted(body["candidateUuids"]) == ["c0", "c1", "c2", "c3"]

    def test_select_none_clears_the_selection(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        self._run(api)
        api.install(page)
        _open_panel(page, "run-1")

        page.get_by_test_id("preview-select-all").click()
        expect(page.get_by_test_id("preview-selection-count")).to_have_text("4 selected")

        page.get_by_test_id("preview-select-none").click()

        expect(page.get_by_test_id("preview-selection-count")).to_have_text("none selected")
        expect(page.locator(".review-card.selected")).to_have_count(0)
        expect(page.locator(".review-card input[type=checkbox]:checked")).to_have_count(0)
        # With nothing selected the decision buttons have no target.
        expect(page.get_by_test_id("preview-accept")).to_be_disabled()

    def test_select_none_clears_a_click_selection_too(self, page: Page, base_url: str):
        """Deselect must also undo cards selected by clicking, not just ones the
        select-all button added."""
        _setup_common(page, base_url)
        api = FakePreviewApi()
        self._run(api)
        api.install(page)
        _open_panel(page, "run-1")

        _select_card(page, "c1")
        expect(page.get_by_test_id("preview-selection-count")).to_have_text("1 selected")

        page.get_by_test_id("preview-select-none").click()
        expect(page.get_by_test_id("preview-selection-count")).to_have_text("none selected")
        expect(page.locator(".review-card.selected")).to_have_count(0)


class TestDecisionCoalescing:
    def test_rubber_banding_decisions_send_one_patch(self, page: Page, base_url: str):
        """Accept -> Reject -> Accept in quick succession (all within the
        400ms debounce window) must coalesce into a single PATCH carrying the
        final state, not one PATCH per click."""
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(_make_run(), [_make_candidate("c1")])
        api.install(page)
        _open_panel(page, "run-1")

        _select_card(page, "c1")
        page.get_by_test_id("preview-accept").click()
        page.get_by_test_id("preview-reject").click()
        page.get_by_test_id("preview-accept").click()

        _wait_for_patch(page, api)
        page.wait_for_timeout(300)  # make sure no second (late) PATCH trickles in
        assert len(api.patch_bodies) == 1, api.patch_bodies
        _, body = api.patch_bodies[0]
        assert body["decision"] == "accepted"
        assert body["decisionAction"] == "relabel"


# ---------------------------------------------------------------------------
# 5. Pagination
# ---------------------------------------------------------------------------


class TestPagination:
    def _setup_big_run(self, page: Page, base_url: str, count=65):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(_make_run(uuid="run-big"), [_make_candidate(f"c{i}", run_uuid="run-big") for i in range(count)])
        api.install(page)
        _open_panel(page, "run-big")
        return api

    def test_prev_disabled_on_first_page_next_enabled(self, page: Page, base_url: str):
        self._setup_big_run(page, base_url)

        expect(page.get_by_test_id("preview-prev-page")).to_be_disabled()
        expect(page.get_by_test_id("preview-next-page")).to_be_enabled()
        expect(page.get_by_test_id("preview-page-label")).to_have_text("Page 1 / 2")
        expect(page.get_by_test_id("preview-candidate-card")).to_have_count(60)

    def test_next_sends_correct_offset_and_disables_on_last_page(self, page: Page, base_url: str):
        api = self._setup_big_run(page, base_url)

        page.get_by_test_id("preview-next-page").click()
        expect(page.get_by_test_id("preview-page-label")).to_have_text("Page 2 / 2")
        expect(page.get_by_test_id("preview-candidate-card")).to_have_count(5)
        expect(page.get_by_test_id("preview-next-page")).to_be_disabled()
        expect(page.get_by_test_id("preview-prev-page")).to_be_enabled()

        last_query = [b for (_, b) in api.query_bodies][-1]
        assert last_query["offset"] == 60

    def test_prev_returns_to_first_page(self, page: Page, base_url: str):
        api = self._setup_big_run(page, base_url)

        page.get_by_test_id("preview-next-page").click()
        expect(page.get_by_test_id("preview-page-label")).to_have_text("Page 2 / 2")
        page.get_by_test_id("preview-prev-page").click()
        expect(page.get_by_test_id("preview-page-label")).to_have_text("Page 1 / 2")

        last_query = [b for (_, b) in api.query_bodies][-1]
        assert last_query["offset"] == 0

    def test_the_page_size_is_selectable(self, page: Page, base_url: str):
        """A run over a whole filter returns five figures, and at a fixed 60 the pager
        reads "page 1 / 500" — which is not navigation, it is scrolling with extra
        steps. The size has to reach the *server*, not just re-slice a page."""
        api = self._setup_big_run(page, base_url, count=65)
        expect(page.get_by_test_id("preview-page-label")).to_have_text("Page 1 / 2")

        page.get_by_test_id("preview-page-size").select_option("120")

        expect(page.get_by_test_id("preview-page-label")).to_have_text("Page 1 / 1")
        expect(page.get_by_test_id("preview-candidate-card")).to_have_count(65)
        _, last_query = api.query_bodies[-1]
        assert last_query["limit"] == 120
        assert last_query["offset"] == 0

    def test_a_chosen_page_size_is_remembered(self, page: Page, base_url: str):
        """It is a working preference, not a property of the run — being asked to
        re-pick it on every run is the same complaint one step later."""
        self._setup_big_run(page, base_url, count=65)
        page.get_by_test_id("preview-page-size").select_option("30")
        expect(page.get_by_test_id("preview-page-label")).to_have_text("Page 1 / 3")

        page.get_by_test_id("preview-run-close").click()
        _open_panel(page, "run-big")

        expect(page.get_by_test_id("preview-page-size")).to_have_value("30")
        expect(page.get_by_test_id("preview-page-label")).to_have_text("Page 1 / 3")


# ---------------------------------------------------------------------------
# 6. Filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_decision_filter_requeries_and_resets_offset(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        candidates = [_make_candidate(f"c{i}", run_uuid="run-big") for i in range(65)]
        candidates[0]["decision"] = "accepted"
        candidates[1]["decision"] = "accepted"
        api.add_run(_make_run(uuid="run-big"), candidates)
        api.install(page)
        _open_panel(page, "run-big")

        # Move off page 1 first so the reset-to-0 behavior is actually exercised.
        page.get_by_test_id("preview-next-page").click()
        expect(page.get_by_test_id("preview-page-label")).to_have_text("Page 2 / 2")

        page.get_by_test_id("preview-filter-decision").locator(".ms-dropdown__trigger").click()
        page.locator(".ms-dropdown[data-filter-key='preview-decision'] input[value='accepted']").click()

        expect(page.get_by_test_id("preview-candidate-card")).to_have_count(2)
        last_run_uuid, last_query = api.query_bodies[-1]
        assert last_query["decisions"] == ["accepted"]
        assert last_query["offset"] == 0

    def test_a_decision_can_be_excluded_as_well_as_included(self, page: Page, base_url: str):
        """ "Show me everything I have not already dealt with" is the filter an hour into
        triage, and it has no expression without exclusion.

        The conditional is the point: the same click on the same row, in its "≠" state,
        has to return the *complement* of what selecting it returns."""
        _setup_common(page, base_url)
        api = FakePreviewApi()
        candidates = [_make_candidate(f"c{i}", run_uuid="run-x") for i in range(5)]
        candidates[0]["decision"] = "accepted"
        candidates[1]["decision"] = "accepted"
        api.add_run(_make_run(uuid="run-x"), candidates)
        api.install(page)
        _open_panel(page, "run-x")

        page.get_by_test_id("preview-filter-decision").locator(".ms-dropdown__trigger").click()
        panel = page.locator(".ms-dropdown[data-filter-key='preview-decision']")
        panel.locator("input[value='accepted']").click()
        expect(page.get_by_test_id("preview-candidate-card")).to_have_count(2)

        # Same row, the "≠" toggle beside it: the two accepted ones are what is left out
        # now, not what is kept.
        panel.locator("label:has(input[value='accepted']) .ms-item__not-btn").click()

        expect(page.get_by_test_id("preview-candidate-card")).to_have_count(3)
        _, q = api.query_bodies[-1]
        assert q["excludeDecisions"] == ["accepted"]
        assert q["decisions"] == [], "excluding a value must not leave it included as well"
        assert q["offset"] == 0, "a filter change starts again at page 1"

    def test_label_filter_is_hierarchical_and_expands_children_on_demand(self, page: Page, base_url: str):
        """The Label dropdown mirrors the review filter panel: the vocabulary is
        a tree, and "Include child labels" widens a parent pick to its
        descendants (expanded client-side, since the candidates endpoint filters
        on plain uuid membership)."""
        _mock_review_queue(page)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG, LABEL_KITTEN])
        _mock_tags(page)
        _mock_thumbnails(page)
        _mock_image_thumbnails(page)
        _navigate_to_review(page, base_url)

        api = FakePreviewApi()
        api.add_run(
            _make_run(),
            [
                _make_candidate("c-cat", proposed_label="lbl-cat"),
                _make_candidate("c-kitten", proposed_label="lbl-kitten"),
                _make_candidate("c-dog", proposed_label="lbl-dog"),
            ],
        )
        api.install(page)
        _open_panel(page, "run-1")

        page.get_by_test_id("preview-filter-label").locator(".ms-dropdown__trigger").click()
        panel = page.locator(".ms-dropdown[data-filter-key='preview-proposed-label']")

        # The child is rendered under its parent with a depth indent, not as a
        # flat sibling.
        expect(panel.locator(".ms-item--child input[value='lbl-kitten']")).to_have_count(1)

        panel.locator("input[value='lbl-cat']").click()
        expect(page.get_by_test_id("preview-candidate-card")).to_have_count(1)
        _, q = api.query_bodies[-1]
        assert q["proposedLabelUuids"] == ["lbl-cat"]

        # Toggle on → the parent's descendants come along.
        panel.locator(".ms-hierarchy-toggle input[type=checkbox]").click()
        expect(page.get_by_test_id("preview-candidate-card")).to_have_count(2)
        _, q = api.query_bodies[-1]
        assert sorted(q["proposedLabelUuids"]) == ["lbl-cat", "lbl-kitten"]
        assert q["offset"] == 0

        # Toggle back off → parent only again.
        panel.locator(".ms-hierarchy-toggle input[type=checkbox]").click()
        expect(page.get_by_test_id("preview-candidate-card")).to_have_count(1)
        _, q = api.query_bodies[-1]
        assert q["proposedLabelUuids"] == ["lbl-cat"]

    def test_confidence_min_requeries_and_resets_offset(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        candidates = [_make_candidate(f"c{i}", run_uuid="run-big", confidence=0.5) for i in range(65)]
        candidates[0]["confidence"] = 0.95
        api.add_run(_make_run(uuid="run-big"), candidates)
        api.install(page)
        _open_panel(page, "run-big")

        page.get_by_test_id("preview-next-page").click()
        expect(page.get_by_test_id("preview-page-label")).to_have_text("Page 2 / 2")

        conf_min = page.get_by_test_id("preview-conf-min")
        conf_min.fill("90")
        conf_min.dispatch_event("change")

        expect(page.get_by_test_id("preview-candidate-card")).to_have_count(1)
        _, last_query = api.query_bodies[-1]
        assert abs(last_query["confidenceMin"] - 0.9) < 1e-9
        assert last_query["offset"] == 0


# ---------------------------------------------------------------------------
# 6b. Grouping (Group: label)
# ---------------------------------------------------------------------------


class TestLabelGrouping:
    def _mixed_run(self, api: FakePreviewApi):
        api.add_run(
            _make_run(),
            [
                _make_candidate("c1", proposed_label="lbl-dog", confidence=0.9),
                _make_candidate("c2", proposed_label="lbl-cat", confidence=0.8),
                _make_candidate("c3", proposed_label="lbl-dog", confidence=0.4),
                _make_candidate("c4", proposed_label="lbl-cat", confidence=0.3),
            ],
        )

    def test_label_grouping_is_the_default_and_draws_one_header_per_label(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        self._mixed_run(api)
        api.install(page)
        _open_panel(page, "run-1")

        expect(page.get_by_test_id("preview-sort")).to_have_value("label_confidence")
        expect(page.get_by_test_id("preview-group-header")).to_have_count(2)
        _, last_query = api.query_bodies[-1]
        assert last_query["sortBy"] == "label_confidence"

        headers = page.get_by_test_id("preview-group-header")
        # Dog's best candidate (0.9) outranks Cat's (0.8), so Dog leads.
        expect(headers.nth(0)).to_contain_text("Dog")
        expect(headers.nth(0)).to_contain_text("2 on this page")
        expect(headers.nth(1)).to_contain_text("Cat")

    def test_no_group_headers_in_a_flat_sort(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        self._mixed_run(api)
        api.install(page)
        _open_panel(page, "run-1")

        page.get_by_test_id("preview-sort").select_option("confidence")
        expect(page.get_by_test_id("preview-candidate-card")).to_have_count(4)
        expect(page.get_by_test_id("preview-group-header")).to_have_count(0)

    def test_select_group_selects_only_that_label(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        self._mixed_run(api)
        api.install(page)
        _open_panel(page, "run-1")

        expect(page.get_by_test_id("preview-group-header")).to_have_count(2)

        page.get_by_test_id("preview-group-select").nth(0).click()
        expect(page.get_by_test_id("preview-selection-count")).to_have_text("2 selected")

        page.get_by_test_id("preview-accept").click()
        _wait_for_patch(page, api)
        _, body = api.patch_bodies[-1]
        assert sorted(body["candidateUuids"]) == ["c1", "c3"]  # the two Dogs


# ---------------------------------------------------------------------------
# 7. Apply
# ---------------------------------------------------------------------------


class TestApply:
    def test_apply_confirms_posts_and_toasts_counts(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.apply_result = {"modified": 3, "added": 1, "deleted": 0, "failed": 0, "stale": 2}
        api.add_run(
            _make_run(decisionCounts={"pending": 0, "accepted": 4, "rejected": 0, "applied": 0, "stale": 0}),
            [_make_candidate(f"c{i}", decision="accepted") for i in range(4)],
        )
        api.install(page)
        _open_panel(page, "run-1")

        apply_btn = page.get_by_test_id("preview-apply")
        expect(apply_btn).to_be_enabled()
        apply_btn.click()

        confirm_dialog = page.get_by_test_id("modal-shell-dialog").last
        expect(confirm_dialog).to_be_visible()
        confirm_dialog.locator("button", has_text="Apply").click()

        expect(page.locator("#toast-container")).to_contain_text("3 relabeled")
        expect(page.locator("#toast-container")).to_contain_text("1 added")
        expect(page.locator("#toast-container")).to_contain_text("2 skipped")

        assert len(api.apply_calls) == 1
        run_uuid, body = api.apply_calls[0]
        assert run_uuid == "run-1"
        assert "markVerified" in body


# ---------------------------------------------------------------------------
# 8. Close flushes pending decisions
# ---------------------------------------------------------------------------


class TestClosePanel:
    def test_closing_panel_flushes_pending_decision(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(_make_run(), [_make_candidate("c1")])
        api.install(page)
        _open_panel(page, "run-1")

        _select_card(page, "c1")
        page.get_by_test_id("preview-accept").click()
        # Close immediately — well inside the 400ms debounce window.
        page.get_by_test_id("preview-run-close").click()

        expect(page.get_by_test_id("preview-panel")).to_have_count(0)
        _wait_for_patch(page, api)
        assert len(api.patch_bodies) == 1, api.patch_bodies
        _, body = api.patch_bodies[0]
        assert body["decision"] == "accepted"


# ---------------------------------------------------------------------------
# 8b. Notices bar must not bury the footer
# ---------------------------------------------------------------------------


class TestModalNotClippedByNoticesBar:
    """Regression: #notices-bar is fixed above the modal layer's z-index, and the
    full-screen dialog used to size itself against the raw viewport — so once
    the notices bar grew (several services unreachable at once), it hung over
    the bottom of the panel and buried the accept/reject row where nothing
    could click it."""

    def test_accept_button_stays_above_an_expanded_notices_bar(self, page: Page, base_url: str):
        page.add_init_script(_FAKE_EVENT_SOURCE_SCRIPT)
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(_make_run(), [_make_candidate("c1")])
        api.install(page)
        _open_panel(page, "run-1")

        page.evaluate(
            "(n) => FakeEventSource.notices.send(n)",
            [
                {"key": f"{name}_unreachable", "level": "warning", "message": f"{name} is not responding"}
                for name in ["db_server", "inference_server", "tasks_server", "training_server", "stream_server"]
            ],
        )
        expect(page.locator("#notices-bar")).to_be_visible()

        # --bottom-bars-h (and the dialog height/overlay inset derived from it)
        # is recomputed by a ResizeObserver callback on #bottom-bars, which fires
        # a frame after the bar's own layout settles — not synchronously with the
        # message above. Poll rather than read the boxes once.
        accept_box, bar_box = None, None
        for _ in range(40):
            accept_box = page.get_by_test_id("preview-accept").bounding_box()
            bar_box = page.locator("#notices-bar").bounding_box()
            if accept_box["y"] + accept_box["height"] <= bar_box["y"] + 1:
                break
            page.wait_for_timeout(50)

        accept_bottom = accept_box["y"] + accept_box["height"]
        assert (
            accept_bottom <= bar_box["y"] + 1
        ), f"accept button bottom ({accept_bottom}) overlaps the notices bar top ({bar_box['y']})"


# ---------------------------------------------------------------------------
# 9. Saved previews list
# ---------------------------------------------------------------------------


class TestSavedPreviewsList:
    def _saved_previews_btn(self, page: Page):
        return page.get_by_test_id("review-saved-previews")

    def test_lists_one_row_per_run_with_decision_counts(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(
            _make_run(uuid="run-1", name="First run"),
            [_make_candidate("c1", decision="accepted"), _make_candidate("c2", decision="pending")],
        )
        api.add_run(
            _make_run(uuid="run-2", name="Second run"),
            [_make_candidate("c3", run_uuid="run-2", decision="rejected")],
        )
        api.install(page)

        self._saved_previews_btn(page).click()

        rows = page.get_by_test_id("preview-run-row")
        expect(rows).to_have_count(2)
        first_row = page.locator('[data-testid="preview-run-row"][data-run-uuid="run-1"]')
        expect(first_row).to_contain_text("First run")
        expect(first_row).to_contain_text("1 accepted")
        expect(first_row).to_contain_text("1 unreviewed")

    def test_pin_patches_pinned(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(_make_run(uuid="run-1", name="First run", pinned=False), [_make_candidate("c1")])
        api.install(page)

        self._saved_previews_btn(page).click()
        row = page.locator('[data-testid="preview-run-row"][data-run-uuid="run-1"]')
        row.get_by_test_id("preview-run-pin").click()

        expect(row.get_by_test_id("preview-run-pin")).to_contain_text("Pinned")
        assert len(api.run_patch_bodies) == 1
        run_uuid, body = api.run_patch_bodies[0]
        assert run_uuid == "run-1"
        assert body["pinned"] is True

    def test_delete_confirms_and_deletes(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(_make_run(uuid="run-1", name="First run"), [_make_candidate("c1")])
        api.install(page)

        self._saved_previews_btn(page).click()
        row = page.locator('[data-testid="preview-run-row"][data-run-uuid="run-1"]')
        row.get_by_test_id("preview-run-delete").click()

        confirm_dialog = page.get_by_test_id("modal-shell-dialog").last
        expect(confirm_dialog).to_be_visible()
        confirm_dialog.locator("button", has_text="Delete").click()

        expect(page.get_by_test_id("preview-run-row")).to_have_count(0)
        assert api.delete_calls == ["run-1"]

    def test_open_opens_triage_panel(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(_make_run(uuid="run-1", name="First run"), [_make_candidate("c1")])
        api.install(page)

        self._saved_previews_btn(page).click()
        row = page.locator('[data-testid="preview-run-row"][data-run-uuid="run-1"]')
        row.get_by_test_id("preview-run-open").click()

        panel = page.get_by_test_id("preview-panel")
        expect(panel).to_be_visible()
        expect(page.get_by_test_id("preview-run-title")).to_have_text("First run")


def _blip_geometry(
    *,
    blip_uuid="blip-1",
    event_uuid="event-1",
    at_s=4.6,
    kind="identity_flip",
    track_id=7,
    samples=9,
    track_samples=121,
    estimated=False,
):
    """Geometry as `HarvestBlipFramesTask` writes it — box plus provenance."""
    return {
        "x": 0.1,
        "y": 0.1,
        "width": 0.2,
        "height": 0.2,
        "estimated": estimated,
        "sourceBlipUuid": blip_uuid,
        "sourceEventUuid": event_uuid,
        "sourceBlipKind": kind,
        "sourceClipTimeS": at_s,
        "sourceTrackId": track_id,
        "sourceBlipSamples": samples,
        "sourceTrackSamples": track_samples,
    }


class TestTrackingIncidentGrouping:
    """Option C's review surface: the existing panel, told what it is looking at.

    A harvested frame cannot be judged from the crop alone — that ambiguity is what
    produced the mistake being reviewed. What makes it decidable is the claim in the
    group header and the clip being one click away, so both are asserted here rather
    than merely rendered.
    """

    def _open_with(self, page, base_url, candidates):
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(_make_run(), candidates)
        api.install(page)
        _open_panel(page, "run-1")
        page.get_by_test_id("preview-sort").select_option("blip")
        return api

    def _harvested(self, uuid, **geometry_kwargs):
        return _make_candidate(
            uuid,
            is_new=True,
            geometry=_blip_geometry(**geometry_kwargs),
        )

    def test_one_header_is_drawn_per_incident_not_per_frame(self, page: Page, base_url: str):
        self._open_with(
            page,
            base_url,
            [
                self._harvested("c1", blip_uuid="blip-1"),
                self._harvested("c2", blip_uuid="blip-1"),
                self._harvested("c3", blip_uuid="blip-2"),
            ],
        )

        expect(page.get_by_test_id("preview-candidate-card")).to_have_count(3)
        expect(page.get_by_test_id("preview-blip-header")).to_have_count(2)

    def test_the_header_states_the_claim_that_makes_the_frame_reviewable(self, page: Page, base_url: str):
        self._open_with(page, base_url, [self._harvested("c1", samples=9, track_samples=121)])

        claim = page.get_by_test_id("preview-blip-claim").first
        expect(claim).to_contain_text("identity flip")
        expect(claim).to_contain_text("9 of 121 frames disagree")

    def test_a_partial_claim_omits_the_ratio_rather_than_inventing_a_total(self, page: Page, base_url: str):
        """Nine out of twelve is a different story from nine out of a hundred and
        twenty-one. Without the total the ratio is the half that carries no weight."""
        self._open_with(page, base_url, [self._harvested("c1", samples=9, track_samples=None)])

        claim = page.get_by_test_id("preview-blip-claim").first
        expect(claim).not_to_contain_text("of")
        expect(claim).to_contain_text("identity flip")

    def test_the_clip_button_names_the_moment_the_frame_came_from(self, page: Page, base_url: str):
        self._open_with(page, base_url, [self._harvested("c1", at_s=4.6)])

        expect(page.get_by_test_id("preview-open-clip").first).to_have_text("▶ Clip at 4.6s")

    def test_a_frame_with_no_provenance_offers_no_clip_button(self, page: Page, base_url: str):
        """An older run, harvested before provenance existed, must not show a button
        that cannot answer."""
        geometry = _blip_geometry()
        geometry.pop("sourceEventUuid")
        self._open_with(page, base_url, [_make_candidate("c1", is_new=True, geometry=geometry)])

        expect(page.get_by_test_id("preview-blip-header")).to_have_count(1)
        expect(page.get_by_test_id("preview-open-clip")).to_have_count(0)

    def test_grouping_by_incident_asks_the_server_for_found_order(self, page: Page, base_url: str):
        """The grouping is a client-side reading of an ordering the server already has.
        Sending the local name would silently fall back to a default order and make the
        grouping look broken rather than the sort."""
        api = self._open_with(page, base_url, [self._harvested("c1")])
        # Wait for the re-query the sort change triggers to actually land. Reading the
        # recorded calls straight after `select_option` races it, and the assertion
        # would be about the *previous* sort.
        expect(page.get_by_test_id("preview-blip-header")).to_have_count(1)

        sorts = [body.get("sortBy") for _run, body in api.query_bodies]
        assert "created_at" in sorts, sorts
        assert "blip" not in sorts, sorts


class TestEstimatedGeometryIsMarked:
    def test_an_interpolated_box_is_called_out(self, page: Page, base_url: str):
        """The detector produced no box here. Accepting it unchecked is how a box that
        is twenty pixels out reaches the training set."""
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(_make_run(), [_make_candidate("c1", is_new=True, geometry=_blip_geometry(estimated=True))])
        api.install(page)
        _open_panel(page, "run-1")

        expect(page.get_by_test_id("preview-estimated-badge")).to_have_count(1)

    def test_a_real_detection_is_not_marked(self, page: Page, base_url: str):
        """The conditional half — otherwise the badge would pass while always on."""
        _setup_common(page, base_url)
        api = FakePreviewApi()
        api.add_run(_make_run(), [_make_candidate("c1", is_new=True, geometry=_blip_geometry(estimated=False))])
        api.install(page)
        _open_panel(page, "run-1")

        expect(page.get_by_test_id("preview-candidate-card")).to_have_count(1)
        expect(page.get_by_test_id("preview-estimated-badge")).to_have_count(0)


class TestTheClipEscapeHatch:
    """The part that makes option C work rather than just look tidy.

    A crop plus a claim is enough most of the time; when it is not, the moving picture
    has to be one click away. This exercises the whole path — the card publishes, the
    composition root resolves NVR's modal on demand, and the player opens over Review
    rather than navigating away from a half-triaged run.
    """

    def _mock_event_apis(self, page: Page, event_uuid="event-1"):
        page.route(
            f"**/api/watch-events/{event_uuid}",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "event": {
                            "uuid": event_uuid,
                            "profileUuid": "prof-1",
                            "sourceUuid": "src-1",
                            "ruleName": "cat using litter box",
                            "startedAt": "2026-08-20T12:00:00+00:00",
                            "endedAt": "2026-08-20T12:00:24+00:00",
                            "recordings": [],
                            "blipCounts": {},
                        },
                    }
                ),
            ),
        )
        page.route(
            "**/api/sources**",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "sources": []}),
            ),
        )

    def test_clicking_the_clip_button_opens_the_player(self, page: Page, base_url: str):
        _setup_common(page, base_url)
        self._mock_event_apis(page)
        api = FakePreviewApi()
        api.add_run(_make_run(), [_make_candidate("c1", is_new=True, geometry=_blip_geometry())])
        api.install(page)
        _open_panel(page, "run-1")
        page.get_by_test_id("preview-sort").select_option("blip")

        page.get_by_test_id("preview-open-clip").first.click()

        expect(page.get_by_test_id("event-detail-modal")).to_be_visible()

    def test_the_triage_panel_is_still_there_underneath(self, page: Page, base_url: str):
        """Deliberately not a tab switch. Taking someone out of a half-triaged run to
        answer one question about one frame loses their place in it."""
        _setup_common(page, base_url)
        self._mock_event_apis(page)
        api = FakePreviewApi()
        api.add_run(_make_run(), [_make_candidate("c1", is_new=True, geometry=_blip_geometry())])
        api.install(page)
        _open_panel(page, "run-1")
        page.get_by_test_id("preview-sort").select_option("blip")

        page.get_by_test_id("preview-open-clip").first.click()
        expect(page.get_by_test_id("event-detail-modal")).to_be_visible()

        assert page.get_by_test_id("preview-panel").count() == 1
