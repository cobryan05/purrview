/**
 * Auto-hiding chrome for a scrolling results view.
 *
 * The main tab strip, the Review toolbar block (`.review-sticky-top`: toolbar +
 * filter chips + the action bars) and the Events filter block
 * (`.event-search__sticky-top`: facet filters + toolbar) are all `position:
 * sticky` at the top of the page scroll. Pinned permanently they cost a large slice of the
 * viewport on a results page — the very page where you want to see results.
 *
 * So: scrolling *down* slides them out of the way, scrolling *up* brings them
 * straight back, which is what you're already doing when you want to reach the
 * tabs or the filters. Near the top of the scroll they're always shown.
 *
 * The show/hide is a `transform` toggled by the `chrome-hidden` class (see
 * app.css) rather than a height/display change, so revealing never reflows the
 * grid underneath — a reflow mid-scroll would shift the cards you're looking at.
 *
 * Two contexts use this:
 *   - the page (`initScrollChrome`) — scroller is the window, class lands on
 *     <body>, and the tab header plus the active tab's own bars move.
 *   - a review modal (`attachScrollChrome`) — scroller is the modal's own
 *     `overflow:auto` content div, class lands on that div, and only the review
 *     bars inside it move (the modal header keeps the close button reachable).
 */

const HIDE_CLASS = 'chrome-hidden';

/** The bars this module moves — anything focus-pinned inside one keeps it shown. */
const CHROME_SELECTOR = '.main-tab-header, .review-sticky-top, .event-search__sticky-top';

/** Ignore jitter/inertial noise smaller than this (px) so the bars don't flicker. */
const DELTA_THRESHOLD = 6;

/** Above this scroll offset the chrome is always shown — there's nothing to gain. */
const ALWAYS_SHOWN_ABOVE = 80;

/**
 * Text entry is the one thing that must not scroll away underneath the user:
 * you cannot keep typing into a field you can no longer see. Everything else in
 * the chrome — sliders, selects, checkboxes, buttons — keeps focus after a
 * plain click, so treating focus alone as "in use" pinned the bars open for the
 * rest of the session. The similar-search modal made that obvious: its
 * threshold slider sits in the toolbar, and adjusting it (the whole point of
 * that modal) meant the bars never hid again.
 *
 * @param {Element|null} el
 * @returns {boolean}
 */
function isTextEntry(el) {
  if (!el) return false;
  if (el.isContentEditable) return true;
  const tag = el.tagName;
  if (tag === 'TEXTAREA') return true;
  if (tag !== 'INPUT') return false;
  // Everything that isn't a button/toggle/slider takes typed input of some kind
  // (text, search, number, date, …), including <input> with no type at all.
  return !['button', 'submit', 'reset', 'checkbox', 'radio', 'range', 'color', 'file'].includes(el.type);
}

/**
 * Never hide out from under an open menu or a field being typed into: the
 * filter popover is anchored to the chip bar and rides along with the
 * transform, and a text field would be yanked off screen mid-edit.
 *
 * @param {ParentNode} root  where to look for an open popover — the document for
 *   the page context, the modal's content div for a modal.
 * @returns {boolean}
 */
function chromeIsBusy(root) {
  if (root.querySelector('.review-filter-popover:not([hidden])')) return true;
  // A multi-select panel is position:fixed, anchored to a trigger inside the
  // chrome — it does not follow the transform, so hiding would leave it floating
  // over nothing.
  if (root.querySelector('.ms-dropdown--open')) return true;
  const active = document.activeElement;
  if (!isTextEntry(active) || !active.closest?.(CHROME_SELECTOR)) return false;
  // In a modal only focus *inside this scroller* counts — a field focused in
  // another panel behind the modal must not pin this one's bars open.
  return root === document ? true : root.contains(active);
}

/**
 * Wire the hide-on-scroll-down / show-on-scroll-up behaviour to one scroller.
 *
 * @param {{ scroller: Window|HTMLElement, host: HTMLElement, root: ParentNode }} opts
 *   `host` is the element the HIDE_CLASS is toggled on; `root` scopes the
 *   busy check.
 * @returns {{ show: () => void, destroy: () => void }}
 */
function createScrollChrome({ scroller, host, root }) {
  const readY = () => Math.max(0, scroller === window ? window.scrollY : scroller.scrollTop);

  let lastY = readY();
  let ticking = false;

  function update() {
    ticking = false;
    const y = readY();
    const delta = y - lastY;

    // Below the threshold, leave lastY alone so a slow drift still accumulates
    // into a real direction change instead of being swallowed sample by sample.
    if (Math.abs(delta) < DELTA_THRESHOLD) return;
    lastY = y;

    const hide = delta > 0 && y > ALWAYS_SHOWN_ABOVE && !chromeIsBusy(root);
    host.classList.toggle(HIDE_CLASS, hide);
  }

  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(update);
  }

  scroller.addEventListener('scroll', onScroll, { passive: true });

  return {
    show() {
      host.classList.remove(HIDE_CLASS);
      lastY = readY();
    },
    destroy() {
      scroller.removeEventListener('scroll', onScroll);
      host.classList.remove(HIDE_CLASS);
    },
  };
}

let _page = null;

/** Install the page-level scroll listener. Idempotent. */
export function initScrollChrome() {
  if (_page) return;
  _page = createScrollChrome({ scroller: window, host: document.body, root: document });
}

/** Force the page chrome back into view (e.g. after a programmatic tab switch). */
export function showChrome() {
  _page?.show();
}

/**
 * Same behaviour for a scroll container that isn't the page — the review
 * modal's content div. The caller owns the returned handle and must call
 * `destroy()` when the container goes away.
 *
 * @param {HTMLElement} scrollEl  an `overflow:auto`/`scroll` element.
 * @returns {{ show: () => void, destroy: () => void }}
 */
export function attachScrollChrome(scrollEl) {
  return createScrollChrome({ scroller: scrollEl, host: scrollEl, root: scrollEl });
}
