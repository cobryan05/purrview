/**
 * Auto-hiding page chrome.
 *
 * The main tab strip and the Review toolbar block (`.review-sticky-top`:
 * toolbar + filter chips + the action bars) are both `position: sticky` at the
 * top of the page scroll. Pinned permanently they cost a large slice of the
 * viewport on a results page — the very page where you want to see results.
 *
 * So: scrolling *down* slides both out of the way, scrolling *up* brings them
 * straight back, which is what you're already doing when you want to reach the
 * tabs or the filters. Near the top of the page they're always shown.
 *
 * The show/hide is a `transform` toggled by `body.chrome-hidden` (see app.css)
 * rather than a height/display change, so revealing never reflows the grid
 * underneath — a reflow mid-scroll would shift the cards you're looking at.
 */

const HIDE_CLASS = 'chrome-hidden';

/** Ignore jitter/inertial noise smaller than this (px) so the bars don't flicker. */
const DELTA_THRESHOLD = 6;

/** Above this scroll offset the chrome is always shown — there's nothing to gain. */
const ALWAYS_SHOWN_ABOVE = 80;

let _lastY = 0;
let _ticking = false;
let _installed = false;

/**
 * Never hide out from under an open menu or a control being used: the filter
 * popover is anchored to the chip bar and rides along with the transform, and
 * anything focused inside the chrome (a text filter, a dropdown) would be
 * yanked off screen mid-interaction.
 *
 * @returns {boolean}
 */
function chromeIsBusy() {
  if (document.querySelector('.review-filter-popover:not([hidden])')) return true;
  const active = document.activeElement;
  return !!active && !!active.closest?.('.main-tab-header, .review-sticky-top');
}

function update() {
  _ticking = false;
  const y = Math.max(0, window.scrollY);
  const delta = y - _lastY;

  // Below the threshold, leave _lastY alone so a slow drift still accumulates
  // into a real direction change instead of being swallowed sample by sample.
  if (Math.abs(delta) < DELTA_THRESHOLD) return;
  _lastY = y;

  const hide = delta > 0 && y > ALWAYS_SHOWN_ABOVE && !chromeIsBusy();
  document.body.classList.toggle(HIDE_CLASS, hide);
}

function onScroll() {
  if (_ticking) return;
  _ticking = true;
  requestAnimationFrame(update);
}

/** Install the scroll listener. Idempotent. */
export function initScrollChrome() {
  if (_installed) return;
  _installed = true;
  _lastY = Math.max(0, window.scrollY);
  window.addEventListener('scroll', onScroll, { passive: true });
}

/** Force the chrome back into view (e.g. after a programmatic tab switch). */
export function showChrome() {
  document.body.classList.remove(HIDE_CLASS);
  _lastY = Math.max(0, window.scrollY);
}
