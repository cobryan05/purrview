"""
Auto-hiding chrome inside a review modal.

The page-level version of this (tab strip + review bars sliding away on
scroll-down) is covered by test_main_tabs_sticky.py and keys off `window.scrollY`
with `body.chrome-hidden`. A modal scrolls its own `overflow:auto` content div
instead, so it gets its own listener via `attachScrollChrome()`
(app/utils/scrollChrome.js), toggling `chrome-hidden` on that div.

Run with:
    docker compose exec -T mono-dev bash -lc \
      'cd /project/playwright_tests && python -m pytest test_modal_scroll_chrome.py \
       -o addopts="" -p no:xdist -q'
"""

import json

from playwright.sync_api import Page

LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parentUuid": None}

CANDIDATES = [
    {
        "uuid": f"img-{i:03d}",
        "filename": f"folder/image{i:03d}.jpg",
        "bboxes": [],
    }
    for i in range(6)
]


def _mock_api(page: Page):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels") and "api/labels/" not in url,
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": [LABEL_CAT]}),
        ),
    )
    page.route(
        "**/api/tags/list",
        lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({"tags": []})),
    )
    page.route(
        "**/api/images/labels/batch*",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"status": "success", "imageLabels": {}})
        ),
    )


def _open_modal(page: Page, base_url: str, with_controls: bool = False):
    """`with_controls` mirrors similarSearch.js, which passes a `modal-toolbar-row`
    of its own controls (threshold slider, count field) into the sticky toolbar."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.evaluate(
        """async ({ candidates, withControls }) => {
            const { showReviewModal } = await import('/app-static/js/app/review/main.js');
            let row = null;
            if (withControls) {
                row = document.createElement('div');
                row.className = 'modal-toolbar-row';
                const slider = document.createElement('input');
                slider.type = 'range';
                slider.className = 'modal-threshold-slider';
                const count = document.createElement('input');
                count.type = 'number';
                count.className = 'modal-count-input';
                row.append(slider, count);
            }
            await showReviewModal({
                title: 'Scroll Chrome Test',
                initialViewFilter: 'images',
                toolbarControls: row,
                fetchImages: async ({ afterUuid, limit }) => {
                    const start = afterUuid ? candidates.findIndex(c => c.uuid === afterUuid) + 1 : 0;
                    return candidates.slice(start, start + limit);
                },
            });
        }""",
        {"candidates": CANDIDATES, "withControls": with_controls},
    )
    page.wait_for_selector(".compare-modal-content .review-sticky-top", timeout=4000)


# A spacer makes the modal content tall enough to scroll regardless of how many
# result cards actually render, so the test isn't hostage to card layout.
_SCROLL_SCRIPT = """async () => {
    const content = document.querySelector('.compare-modal-content');
    const bars = content.querySelector('.review-sticky-top');
    if (!content || !bars) return { error: 'modal content or sticky bars missing' };

    const spacer = document.createElement('div');
    spacer.style.height = '4000px';
    spacer.style.flex = '0 0 auto';  // content is a flex column — don't let it shrink away
    content.appendChild(spacer);

    const settle = async () => {
        // One frame for the rAF-throttled handler, then past the 0.18s transform.
        await new Promise(r => requestAnimationFrame(r));
        await new Promise(r => setTimeout(r, 300));
    };
    const barsOffset = () => bars.getBoundingClientRect().top - content.getBoundingClientRect().top;

    await settle();
    const atTop = barsOffset();

    content.scrollTop = 1200;
    await settle();
    const hiddenDown = content.classList.contains('chrome-hidden');
    const offsetDown = barsOffset();
    const bodyHiddenDown = document.body.classList.contains('chrome-hidden');

    content.scrollTop = 700;
    await settle();
    const hiddenUp = content.classList.contains('chrome-hidden');
    const offsetUp = barsOffset();

    content.scrollTop = 0;
    await settle();
    const hiddenAtTop = content.classList.contains('chrome-hidden');

    return { atTop, hiddenDown, offsetDown, bodyHiddenDown, hiddenUp, offsetUp, hiddenAtTop };
}"""


class TestModalScrollChrome:
    def test_modal_bars_hide_on_scroll_down_and_return_on_scroll_up(self, page: Page, base_url: str):
        _mock_api(page)
        _open_modal(page, base_url)

        result = page.evaluate(_SCROLL_SCRIPT)

        assert result.get("error") is None, result
        assert abs(result["atTop"]) < 2, f"bars not pinned at the top of an unscrolled modal: {result['atTop']}"
        assert result["hiddenDown"], "chrome-hidden not set on the modal content after scrolling down"
        assert result["offsetDown"] <= -1, f"bars still in view after scrolling down: {result['offsetDown']}"
        assert not result["hiddenUp"], "chrome-hidden still set after scrolling up"
        assert abs(result["offsetUp"]) < 2, f"bars did not come back on scroll up: {result['offsetUp']}"
        assert not result["hiddenAtTop"], "bars must be visible once the modal is back at the top"

    def test_modal_scrolling_leaves_the_page_chrome_alone(self, page: Page, base_url: str):
        """The modal drives its own container class — it must not hide the tab
        strip behind it via body.chrome-hidden."""
        _mock_api(page)
        _open_modal(page, base_url)

        result = page.evaluate(_SCROLL_SCRIPT)

        assert result.get("error") is None, result
        assert not result["bodyHiddenDown"], "modal scrolling must not toggle body.chrome-hidden"

    def test_modal_header_stays_put_while_bars_hide(self, page: Page, base_url: str):
        """The close button lives in the modal header, outside the scroller —
        it must never slide away with the bars."""
        _mock_api(page)
        _open_modal(page, base_url)

        result = page.evaluate("""async () => {
            const content = document.querySelector('.compare-modal-content');
            const header = document.querySelector('.compare-modal-header');
            const spacer = document.createElement('div');
            spacer.style.height = '4000px';
            spacer.style.flex = '0 0 auto';
            content.appendChild(spacer);

            const settle = async () => {
                await new Promise(r => requestAnimationFrame(r));
                await new Promise(r => setTimeout(r, 300));
            };

            await settle();
            const before = header.getBoundingClientRect().top;
            content.scrollTop = 1200;
            await settle();
            return {
                before,
                after: header.getBoundingClientRect().top,
                hidden: content.classList.contains('chrome-hidden'),
                closeVisible: !!document.querySelector('.compare-modal-close-btn')?.offsetParent,
            };
        }""")

        assert result["hidden"], "precondition: bars should have hidden on scroll-down"
        assert abs(result["after"] - result["before"]) < 2, f"modal header moved: {result}"
        assert result["closeVisible"], "close button must stay reachable"


class TestFocusedControlsDoNotPinTheBars:
    """Regression: focus alone used to count as "chrome in use", so one click on
    a toolbar control pinned the bars open for the rest of the modal's life.
    The similar-search modals put a threshold slider right there — adjusting it,
    which is the whole point of those modals, permanently disabled the hiding.
    Only text entry pins the bars now (you can't type into what you can't see).
    """

    def _scroll_and_report(self, page: Page):
        return page.evaluate("""async () => {
            const content = document.querySelector('.compare-modal-content');
            const spacer = document.createElement('div');
            spacer.style.height = '4000px';
            spacer.style.flex = '0 0 auto';
            content.appendChild(spacer);
            await new Promise(r => requestAnimationFrame(r));
            content.scrollTop = 1200;
            await new Promise(r => requestAnimationFrame(r));
            await new Promise(r => setTimeout(r, 300));
            const active = document.activeElement;
            return {
                hidden: content.classList.contains('chrome-hidden'),
                activeInBars: !!active?.closest?.('.review-sticky-top'),
            };
        }""")

    def test_a_focused_slider_still_lets_the_bars_hide(self, page: Page, base_url: str):
        _mock_api(page)
        _open_modal(page, base_url, with_controls=True)

        page.locator(".compare-modal-content input.modal-threshold-slider").click()
        result = self._scroll_and_report(page)

        assert result["activeInBars"], "precondition: the slider should hold focus after a click"
        assert result["hidden"], "a focused slider must not pin the bars open"

    def test_a_focused_text_field_pins_the_bars(self, page: Page, base_url: str):
        """The other half of the rule: a field being typed into must not scroll away."""
        _mock_api(page)
        _open_modal(page, base_url, with_controls=True)

        page.locator(".compare-modal-content input.modal-count-input").click()
        result = self._scroll_and_report(page)

        assert result["activeInBars"], "precondition: the number field should hold focus after a click"
        assert not result["hidden"], "a focused text field must keep the bars visible"
