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
    docker compose --profile dev run --rm --no-deps -w /project whiskerwatcher-dev \
        python3.11 -m pytest playwright_tests/test_preview_panel.py -q
"""

import json
from urllib.parse import urlsplit

import pytest
from playwright.sync_api import Page, Route, expect

LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parentUuid": None}
LABEL_DOG = {"uuid": "lbl-dog", "name": "Dog", "color": "#00ff00", "parentUuid": None}

_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


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
    page.locator(".main-tab-button[data-tab='tab-review']").click()
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#review-root .review-grid", timeout=4000)


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
            conf_min = body.get("confidenceMin")
            if conf_min is not None:
                cands = [c for c in cands if (c.get("confidence") or 0) >= conf_min]
            conf_max = body.get("confidenceMax")
            if conf_max is not None:
                cands = [c for c in cands if (c.get("confidence") or 0) <= conf_max]
            is_new = body.get("isNew")
            if is_new is not None:
                cands = [c for c in cands if bool(c.get("isNew")) == is_new]

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
