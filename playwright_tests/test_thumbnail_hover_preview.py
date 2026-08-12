"""Hover-preview popover lifecycle (ui/thumbnailHoverPreview.js).

The popover is dismissed by `mouseleave` on the card it is anchored to. Anything
that takes the card away without the pointer leaving it first therefore has to
dismiss the popover some other way, or it is stranded on screen with nothing
left to un-hover.
"""

import base64
import json

import pytest
from playwright.sync_api import Page

from playwright_tests.waits import wait_until

# A 1x1 transparent PNG — enough for the popover's `img.naturalWidth` gate.
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)

# Show immediately: these tests are about what happens after the popover is up.
_SETUP = """async ({ removeCard }) => {
    const { createHoverPreview } = await import('/app-static/js/ui/thumbnailHoverPreview.js');
    document.querySelectorAll('.ww-test-card').forEach(el => el.remove());
    document.querySelectorAll('.thumb-hover-popover').forEach(el => el.remove());

    const host = document.createElement('div');
    host.className = 'ww-test-card-host';
    const card = document.createElement('div');
    card.className = 'ww-test-card';
    card.style.cssText = 'width:80px;height:80px;';
    host.appendChild(card);
    document.body.appendChild(host);

    const preview = createHoverPreview({ showDelayMs: 0 });
    preview.attach(card, { imagePath: 'folder/frame.jpg', imageUuid: 'img-1', allBboxes: [] });
    card.dispatchEvent(new MouseEvent('mouseenter'));

    const popover = () => document.querySelector('.thumb-hover-popover');
    const visible = async () => {
        for (let i = 0; i < 100; i++) {
            if (popover()?.classList.contains('visible')) return true;
            await new Promise(r => setTimeout(r, 20));
        }
        return false;
    };
    const shown = await visible();
    if (!shown) return { shown: false };

    if (removeCard) host.remove();

    // Give the MutationObserver callback (a microtask) room to run.
    for (let i = 0; i < 50; i++) {
        if (!popover()?.classList.contains('visible')) break;
        await new Promise(r => setTimeout(r, 20));
    }
    return { shown: true, stillVisible: popover()?.classList.contains('visible') === true };
}"""


@pytest.fixture
def preview_page(page: Page, base_url: str) -> Page:
    page.route(
        "**/api/images/**/thumbnail*",
        lambda route: route.fulfill(status=200, content_type="image/png", body=TINY_PNG),
    )
    page.route(
        "**/api/settings/**",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"status": "success", "value": None})
        ),
    )
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    return page


class TestPopoverSurvivesItsAnchor:
    def test_popover_is_dismissed_when_its_card_is_removed(self, preview_page: Page):
        """The bug: deleting an image while its hover-zoom is up left the zoom on
        screen permanently — no mouseleave ever fires for a card that is gone."""
        result = preview_page.evaluate(_SETUP, {"removeCard": True})
        assert result["shown"], "popover never appeared, so the test proves nothing"
        assert not result["stillVisible"], "popover outlived the card it was anchored to"

    def test_popover_stays_up_while_its_card_is_still_there(self, preview_page: Page):
        """The guard must not dismiss on unrelated DOM churn — the observer watches
        the whole body, so a popover that hides on any mutation would be useless."""
        result = preview_page.evaluate(_SETUP, {"removeCard": False})
        assert result["shown"], "popover never appeared, so the test proves nothing"
        assert result["stillVisible"], "popover was dismissed while its card was still in the DOM"

    def test_unrelated_dom_changes_do_not_dismiss_it(self, preview_page: Page):
        result = preview_page.evaluate("""async () => {
            const { createHoverPreview } = await import('/app-static/js/ui/thumbnailHoverPreview.js');
            document.querySelectorAll('.ww-test-card').forEach(el => el.remove());
            document.querySelectorAll('.thumb-hover-popover').forEach(el => el.remove());

            const card = document.createElement('div');
            card.className = 'ww-test-card';
            document.body.appendChild(card);

            const preview = createHoverPreview({ showDelayMs: 0 });
            preview.attach(card, { imagePath: 'folder/frame.jpg', imageUuid: 'img-2', allBboxes: [] });
            card.dispatchEvent(new MouseEvent('mouseenter'));

            const popover = () => document.querySelector('.thumb-hover-popover');
            for (let i = 0; i < 100; i++) {
                if (popover()?.classList.contains('visible')) break;
                await new Promise(r => setTimeout(r, 20));
            }
            if (!popover()?.classList.contains('visible')) return { shown: false };

            // Churn the document hard, without touching the anchor.
            for (let i = 0; i < 20; i++) {
                const noise = document.createElement('div');
                noise.className = 'ww-test-noise';
                document.body.appendChild(noise);
                noise.remove();
            }
            await new Promise(r => setTimeout(r, 100));
            return { shown: true, stillVisible: popover()?.classList.contains('visible') === true };
        }""")
        assert result["shown"], "popover never appeared, so the test proves nothing"
        assert result["stillVisible"], "unrelated DOM churn dismissed the popover"
