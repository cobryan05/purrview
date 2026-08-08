"""
Regression tests for the auto-hiding page chrome.

The top-level tab strip used to scroll away with the page, so switching tabs
from deep inside a long Review search result page meant scrolling all the way
back to the top first. It is now `position: sticky; top: 0` — but pinned
permanently it plus the Review toolbar block (`.review-sticky-top`: toolbar,
filter chips, action bars) ate a large slice of the viewport, so
app/utils/scrollChrome.js slides both away on scroll-down and brings them back
on scroll-up via `body.chrome-hidden`.

Run with:
    pytest playwright_tests/test_main_tabs_sticky.py -p no:xdist -v
"""

from playwright.sync_api import Page

# Makes the page tall enough to scroll, the way a full page of results does.
_ADD_SPACER = """
    const spacer = document.createElement('div');
    spacer.style.height = '4000px';
    document.querySelector('.main-tab-pane.active').appendChild(spacer);
"""


class TestScrollChrome:
    def test_chrome_hides_on_scroll_down_and_returns_on_scroll_up(self, page: Page, base_url: str):
        """Scrolling down must free the viewport; scrolling up must give it back."""
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        result = page.evaluate("""async () => {
            const header = document.querySelector('.main-tab-header');
            if (!header) return { error: 'no .main-tab-header' };
            %s
            const settle = async () => {
                // One frame for the rAF-throttled scroll handler, then past the
                // 0.18s transform transition.
                await new Promise(r => requestAnimationFrame(r));
                await new Promise(r => setTimeout(r, 300));
            };

            const atTop = header.getBoundingClientRect().top;

            window.scrollTo(0, 2000);
            await settle();
            const afterDown = header.getBoundingClientRect().bottom;
            const hiddenClass = document.body.classList.contains('chrome-hidden');

            window.scrollTo(0, 1400);
            await settle();
            const afterUp = header.getBoundingClientRect().top;

            return { atTop, afterDown, hiddenClass, afterUp, scrollY: window.scrollY };
        }""" % _ADD_SPACER)

        assert result.get("error") is None, result
        # At scroll 0 the header sits at its natural offset (main.container's
        # top padding) — sticky only pulls it to 0 once the page has scrolled.
        assert 0 <= result["atTop"] < 16, f"header not at the top of an unscrolled page: {result['atTop']}"
        assert result["hiddenClass"], "body.chrome-hidden not set after scrolling down"
        assert result["afterDown"] <= 1, f"header still on screen after scrolling down: {result['afterDown']}"
        assert abs(result["afterUp"]) < 2, f"header did not come back on scroll up: top={result['afterUp']}"

    def test_scrolling_back_to_the_top_always_shows_the_chrome(self, page: Page, base_url: str):
        """Near the top there is nothing to gain by hiding, so it stays visible."""
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        result = page.evaluate("""async () => {
            %s
            const settle = async () => {
                await new Promise(r => requestAnimationFrame(r));
                await new Promise(r => setTimeout(r, 300));
            };

            window.scrollTo(0, 2000);
            await settle();
            const hiddenMidPage = document.body.classList.contains('chrome-hidden');

            window.scrollTo(0, 0);
            await settle();
            return { hiddenMidPage, hiddenAtTop: document.body.classList.contains('chrome-hidden') };
        }""" % _ADD_SPACER)

        assert result["hiddenMidPage"], "chrome should hide mid-page"
        assert not result["hiddenAtTop"], "chrome must be visible once back at the top"

    def test_review_sticky_top_offsets_below_the_header(self, page: Page, base_url: str):
        """In-tab review bars park below the pinned header; modal ones stay at 0."""
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        result = page.evaluate("""() => {
            const header = document.querySelector('.main-tab-header');
            const headerH = header.offsetHeight;

            const inTab = document.createElement('div');
            inTab.className = 'review-sticky-top';
            document.querySelector('.main-tab-pane.active').appendChild(inTab);

            // showReviewModal mounts its panel on document.body, outside any
            // .main-tab-pane — there is no pinned header above it there.
            const inModal = document.createElement('div');
            inModal.className = 'review-sticky-top';
            document.body.appendChild(inModal);

            return {
                headerH,
                cssVar: getComputedStyle(document.documentElement)
                    .getPropertyValue('--main-tabs-h').trim(),
                inTabTop: getComputedStyle(inTab).top,
                inModalTop: getComputedStyle(inModal).top,
            };
        }""")

        assert result["headerH"] > 0
        assert result["cssVar"] == f"{result['headerH']}px", result
        assert result["inTabTop"] == f"{result['headerH']}px", result
        assert result["inModalTop"] == "0px", result
