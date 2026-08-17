"""A very large image removal has to be confirmed twice, the second time slowly.

Deleting 40 images and deleting 40,000 are the same click through the same
dialog. The filter-scoped removals ("apply to all images matching the current
filter, across all pages") resolve their count server-side, so the user has
never seen most of what they are about to destroy — a filter one term looser
than they thought turns a small cleanup into a catastrophe with no change in the
interaction.

So above HIGH_RISK_REMOVAL_COUNT there is a second step: the count in large bold
digits with thousands separators, and a short cooldown before Confirm can be
pressed, because the second dialog is otherwise cleared by the same reflex that
cleared the first.

How loud that step is depends on what is actually lost. A verified box is the
one thing on an image a model cannot regenerate, so a removal carrying verified
images leads with *that* count, in red, behind the cooldown; a removal of images
no human ever touched is amber and opens armed. Painting both red is how a
warning stops being read.

Run with:
    pytest playwright_tests/test_bulk_removal_guard.py -p no:asyncio -n 0 -v
"""

from playwright.sync_api import Page, expect

MODULE = "/app-static/js/app/review/imageBulkActionModal.js"

# Starts the confirmation and parks the promise on window, so the test can drive
# the dialogs with real clicks and read the resolved value afterwards.
_START = """
async ([action, count, verified]) => {
    const m = await import('%s');
    window.__removalResult = m.confirmRemovalAction(
        action, count, 'matching the current filter', { verifiedCount: verified },
    );
    return { threshold: m.HIGH_RISK_REMOVAL_COUNT, cooldown: m.HIGH_RISK_COOLDOWN_MS };
}
""" % MODULE

_RESULT = "async () => await window.__removalResult"


def _start(page: Page, action: str, count: int, verified=None):
    """verified=None means "the caller could not tell" — the loud case."""
    return page.evaluate(_START, [action, count, verified])


def _dialog(page: Page):
    return page.locator("[data-testid='modal-shell-dialog']")


class TestBelowThreshold:
    def test_single_step_confirm(self, shared_page: Page):
        page = shared_page
        _start(page, "delete", 12)
        _dialog(page).get_by_role("button", name="Delete", exact=True).click()
        assert page.evaluate(_RESULT) is True
        expect(_dialog(page)).to_have_count(0)

    def test_no_high_risk_callout(self, shared_page: Page):
        page = shared_page
        _start(page, "delete", 100)  # the threshold itself is not "over" it
        expect(_dialog(page)).to_be_visible()
        assert page.locator("[data-testid='high-risk-callout']").count() == 0
        _dialog(page).get_by_role("button", name="Delete", exact=True).click()
        assert page.evaluate(_RESULT) is True


class TestAboveThreshold:
    def test_second_step_shows_grouped_count(self, shared_page: Page):
        page = shared_page
        _start(page, "delete", 41273, verified=0)
        _dialog(page).get_by_role("button", name="Delete", exact=True).click()

        count_el = page.locator("[data-testid='high-risk-count']")
        expect(count_el).to_be_visible()
        assert count_el.inner_text() == "41,273"

        _dialog(page).get_by_role("button", name="Cancel").click()
        assert page.evaluate(_RESULT) is False

    def test_cancelling_second_step_refuses_the_removal(self, shared_page: Page):
        page = shared_page
        _start(page, "removeFromDb", 900, verified=3)
        _dialog(page).get_by_role("button", name="Remove from DB", exact=True).click()
        expect(page.locator("[data-testid='high-risk-callout']")).to_be_visible()
        _dialog(page).get_by_role("button", name="Cancel").click()
        assert page.evaluate(_RESULT) is False

    def test_cancelling_first_step_never_reaches_the_second(self, shared_page: Page):
        page = shared_page
        _start(page, "delete", 9000)
        _dialog(page).get_by_role("button", name="Cancel").click()
        assert page.evaluate(_RESULT) is False
        assert page.locator("[data-testid='high-risk-callout']").count() == 0


class TestSeverityFollowsWhatIsLost:
    """The loud version is for work that cannot be regenerated, not for volume."""

    def test_verified_images_lead_with_their_own_count_in_red(self, shared_page: Page):
        page = shared_page
        _start(page, "delete", 41273, verified=1204)
        _dialog(page).get_by_role("button", name="Delete", exact=True).click()

        callout = page.locator("[data-testid='high-risk-callout']")
        expect(callout).to_have_attribute("data-tone", "danger")
        # The headline is the verified count, not the total: 900 machine-labelled
        # frames and 900 hand-verified ones are completely different losses.
        assert page.locator("[data-testid='high-risk-count']").inner_text() == "1,204"
        expect(page.locator("[data-testid='high-risk-note']")).to_contain_text("41,273")

        _dialog(page).get_by_role("button", name="Cancel").click()
        assert page.evaluate(_RESULT) is False

    def test_verified_removal_is_held_behind_a_cooldown(self, shared_page: Page):
        page = shared_page
        info = _start(page, "delete", 1234, verified=7)
        _dialog(page).get_by_role("button", name="Delete", exact=True).click()

        armed = page.locator("[data-testid='confirm-armed']")
        expect(armed).to_be_disabled()
        # The button keeps saying what it does while it counts.
        assert "1,234" in armed.inner_text()

        expect(armed).to_be_enabled(timeout=info["cooldown"] + 3000)
        armed.click()
        assert page.evaluate(_RESULT) is True

    def test_untouched_images_get_the_quiet_version(self, shared_page: Page):
        """Nothing a human made is in these, so the dialog says so and opens armed."""
        page = shared_page
        _start(page, "delete", 4000, verified=0)
        _dialog(page).get_by_role("button", name="Delete", exact=True).click()

        expect(page.locator("[data-testid='high-risk-callout']")).to_have_attribute("data-tone", "caution")
        assert page.locator("[data-testid='confirm-armed']").count() == 0

        _dialog(page).get_by_role("button", name="Delete 4,000 images").click()
        assert page.evaluate(_RESULT) is True

    def test_an_unknown_verified_count_is_treated_as_the_loud_case(self, shared_page: Page):
        """An absence of information is not evidence there is nothing to lose."""
        page = shared_page
        _start(page, "delete", 4000, verified=None)
        _dialog(page).get_by_role("button", name="Delete", exact=True).click()

        expect(page.locator("[data-testid='high-risk-callout']")).to_have_attribute("data-tone", "danger")
        expect(page.locator("[data-testid='confirm-armed']")).to_be_disabled()

        _dialog(page).get_by_role("button", name="Cancel").click()
        assert page.evaluate(_RESULT) is False


class TestCountingVerifiedWork:
    def _count(self, page: Page, entries):
        return page.evaluate(
            """async (entries) => {
                const m = await import('%s');
                return m.countImagesWithVerifiedWork(entries);
            }""" % MODULE,
            entries,
        )

    VERIFIED = "00000000-0000-0000-0000-000000000005"
    UNVERIFIED = "00000000-0000-0000-0000-000000000001"

    def test_counts_images_holding_a_verified_box(self, shared_page: Page):
        entries = [
            {"img": {"bboxes": [{"tags": [{"uuid": self.VERIFIED}]}]}},
            {"img": {"bboxes": [{"tags": [{"uuid": self.UNVERIFIED}]}]}},
            {"img": {"bboxes": []}},
        ]
        assert self._count(shared_page, entries) == 1

    def test_an_image_counts_once_however_many_verified_boxes_it_has(self, shared_page: Page):
        entries = [
            {"img": {"bboxes": [{"tags": [{"uuid": self.VERIFIED}]}, {"tags": [{"uuid": self.VERIFIED}]}]}},
        ]
        assert self._count(shared_page, entries) == 1

    def test_entries_without_bbox_data_are_unknown_not_zero(self, shared_page: Page):
        """Candidate-list stubs carry no bboxes; reporting 0 would silently
        downgrade the warning on a set nobody has actually checked."""
        entries = [
            {"img": {"bboxes": [{"tags": [{"uuid": self.VERIFIED}]}]}},
            {"img": {"uuid": "no-bbox-data"}},
        ]
        assert self._count(shared_page, entries) is None
