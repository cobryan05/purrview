"""Tests for Verify Model saving its report instead of only rendering it.

The point of persisting a verification is that nobody has to sit and watch it:
the task writes the summary into a preview run as it finishes, so losing the
page no longer loses the run. What only a browser proves is the pairing of the
two — that the dialog actually registers a run before starting the task, and
that it only promises "you can close this" when closing really is safe.
"""

import json

from playwright.sync_api import Page, expect

from playwright_tests.test_compare_model_versions import (
    _mock_models_with_history,
    _mock_preview_endpoints,
    _version_select,
)
from playwright_tests.test_run_model_on_results import (
    LABEL_CAT,
    LABEL_DOG,
    PAGE_1_IMAGES,
    UNVERIFIED_TAG,
    _FakeTask,
    _mock_labels,
    _mock_preview_runs,
    _mock_review_queue,
    _mock_tags,
    _navigate_to_review,
)


def _pin_task_running(page: Page, task: _FakeTask):
    """Hold the task at "running" for the rest of the test.

    Registered after _FakeTask.install so it wins. Without it the mocked task
    completes on its second poll and the dialog swaps the progress area (and the
    note inside it) out for the results — whether the assertion sees the note
    would then be a race.
    """
    page.route(
        "**/api/tasks/instances/get",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "tasks": {
                        task.task_uuid: {
                            "config": {
                                "uuid": task.config_uuid,
                                "typename": "PreviewModelRelabelTask",
                                "name": "x",
                                "params": {},
                            },
                            "instance": {"uuid": task.task_uuid, "status": "running"},
                            "progress": 40.0,
                            "statusMessage": "",
                            "statusActions": [],
                            "logLines": [],
                            "isPausedInPlace": False,
                        }
                    },
                }
            ),
        ),
    )


def _setup(page: Page, base_url: str):
    """The verify dialog, open, over a two-box review scope, with its task
    held at "running" so the in-flight UI is what gets asserted."""
    _mock_review_queue(page, PAGE_1_IMAGES)
    _mock_labels(page, [LABEL_CAT, LABEL_DOG])
    _mock_tags(page, [UNVERIFIED_TAG])
    _mock_models_with_history(page)
    created, deleted = [], []
    _mock_preview_endpoints(page, created, deleted)
    runs = _mock_preview_runs(page)
    task = _FakeTask()
    task.install(page, candidates=[], stats={"no_op_count": 2, "candidate_count": 0})
    _pin_task_running(page, task)
    _navigate_to_review(page, base_url)
    page.get_by_test_id("review-verify-model").click()
    expect(page.get_by_test_id("compare-models-primary-a")).to_be_visible()
    return runs, task, deleted


def _pick_model(page: Page, name="fresh"):
    page.get_by_test_id("compare-models-primary-a").select_option(name)


class TestReportIsRegistered:
    def test_run_is_saved_as_a_report_before_the_task_starts(self, page: Page, base_url: str):
        runs, task, _ = _setup(page, base_url)
        _pick_model(page)
        page.get_by_test_id("compare-models-run").click()

        expect(page.get_by_test_id("compare-models-saved-note")).to_be_visible()
        assert len(runs["created"]) == 1, runs["created"]
        assert runs["created"][0]["params"]["report"] is True

        # The task has to be told to keep the summary and drop the candidates:
        # a verification's candidates are disagreements with ground truth, and
        # the summary is the only thing a later reader gets.
        params = task.create_bodies[0]["params"]
        assert params["preview_run_uuid"] == "run-1"
        assert params["persist_stats"] is True
        assert params["persist_candidates"] is False

    def test_a_second_model_is_registered_in_the_same_comparison(self, page: Page, base_url: str):
        runs, _, _ = _setup(page, base_url)
        _pick_model(page, "fresh")
        page.get_by_test_id("compare-models-primary-b").select_option("aged")
        page.get_by_test_id("compare-models-run").click()

        expect(page.get_by_test_id("compare-models-saved-note")).to_be_visible()
        assert len(runs["created"]) == 2, runs["created"]
        groups = {r["params"]["comparison_group"] for r in runs["created"]}
        slots = sorted(r["params"]["slot"] for r in runs["created"])
        # One group, two slots: that pairing is what lets a saved report be
        # re-opened as a side-by-side rather than two unrelated numbers.
        assert len(groups) == 1
        assert slots == ["A", "B"]

    def test_the_note_says_to_wait_when_an_archived_version_is_in_play(self, page: Page, base_url: str):
        # Closing the dialog deletes the materialised copy of a version, which
        # the still-running task needs — so this is the one case where the run
        # does not survive walking away, and the note has to say so.
        _setup(page, base_url)
        _pick_model(page, "aged")
        _version_select(page).select_option("20260901T101500Z")
        expect(page.get_by_test_id("compare-models-version-note-a")).to_be_visible()
        page.get_by_test_id("compare-models-run").click()

        note = page.get_by_test_id("compare-models-saved-note")
        expect(note).to_contain_text("keep this window open")
        assert "you can close this window" not in note.text_content()

    def test_the_note_says_you_can_leave_for_a_live_model(self, page: Page, base_url: str):
        _setup(page, base_url)
        _pick_model(page, "fresh")
        page.get_by_test_id("compare-models-run").click()

        note = page.get_by_test_id("compare-models-saved-note")
        expect(note).to_contain_text("you can close this window")
        assert "keep this window open" not in note.text_content()
