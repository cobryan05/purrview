/**
 * Measure `--le-chrome-h` instead of hard-coding it.
 *
 * `.list-editor-split` sizes itself as `100vh - var(--le-chrome-h)` so the
 * editor body scrolls rather than the page. That variable used to be two magic
 * numbers in app.css (`#tab-configuration: 139px`, `#tab-operations: 195px`),
 * with a comment warning they had to be re-measured by hand whenever the chrome
 * above the split changed — which the Configuration left rail does, by removing
 * the sub-tab row from above the pane entirely.
 *
 * Measuring closes that whole class of drift: the chrome is whatever sits above
 * the split right now, plus the fixed bottom bar the split must not run under.
 */

/** Splits shorter than this are not worth shrinking further. */
const MIN_SPLIT_H = 380;

/**
 * In-flow space below `el`, all the way out to the document.
 *
 * Deliberately NOT `documentElement.scrollHeight - el.bottom`. `scrollHeight` is
 * `max(content, clientHeight)`, so as soon as the page fits the viewport — which
 * is precisely the state this module works to produce — that expression collapses
 * to `innerHeight - above - splitHeight`, i.e. an identity: `chrome` recomputes to
 * whatever it already was and the split can never grow. Enlarging the window left
 * the pane at its old height with the new space dead beneath it, and a pane that
 * mounted undersized stayed undersized forever.
 *
 * Walking the ancestors measures the padding/border/margin and trailing siblings
 * that actually sit below the split, which is independent of the split's own
 * height — so the value converges in one pass whether the page overflows or not.
 *
 * @param {Element} el
 * @returns {number} pixels
 */
function inFlowSpaceBelow(el) {
  let total = 0;
  let node = el;
  while (node.parentElement) {
    const parent = node.parentElement;
    // Gap between the node's bottom and the parent's — trailing siblings plus the
    // parent's own bottom padding and border.
    total += parent.getBoundingClientRect().bottom - node.getBoundingClientRect().bottom;
    total += parseFloat(getComputedStyle(parent).marginBottom) || 0;
    node = parent;
  }
  return Math.max(0, total);
}

/**
 * @returns {number} height of the fixed bottom bar strip, or 0 when absent.
 */
function bottomBarsHeight() {
  const bars = document.getElementById('bottom-bars');
  if (!bars) return 0;
  return bars.offsetHeight || 0;
}

/**
 * Recompute `--le-chrome-h` for every currently visible list-editor split.
 *
 * Only visible ones: `getBoundingClientRect()` on a `display:none` pane returns
 * zeros, which would set a nonsense height that survives until the next pass.
 */
export function refreshListEditorHeights() {
  const barsH = bottomBarsHeight();

  // Published for `body { padding-bottom }`, so every page's bottom clearance is the
  // bar's *current* height rather than a flat reserve guessed from its resting one —
  // the notices bar expands in place, and that is precisely when content hidden under
  // the bars matters. No feedback loop: the bars are fixed, so their height does not
  // depend on the padding that reads it.
  document.documentElement.style.setProperty('--bottom-bars-h', `${barsH}px`);

  for (const split of document.querySelectorAll('.list-editor-split')) {
    // offsetParent is null for display:none subtrees — the cheap visibility test.
    if (split.offsetParent === null) continue;

    const rect = split.getBoundingClientRect();
    // Document-relative, not viewport-relative. The two only agree while the page
    // is at scroll 0 — and the whole point of this measurement is the case where
    // it is *not*, because a mis-sized split is what put a scrollbar on the page.
    // Measuring from a scrolled viewport reported the split as starting higher
    // than it does and grew it further, so the overflow never converged away.
    const above = rect.top + window.scrollY;

    // Padding and margins that sit *below* the split in normal flow — the app
    // shell has some on both `main.container` and `body`. They were unaccounted
    // for, which is what left the page ~48px taller than the viewport on every
    // list-editor screen: enough to scroll the editor bar (and its Save button)
    // out of sight, with no obvious way back because the wheel belongs to the
    // inner scroller.
    const below = inFlowSpaceBelow(split);

    // max, not sum: the bottom bars are fixed, so they overlay that in-flow space
    // rather than adding to it. Adding them would shrink the split by the bar
    // height a second time and leave a permanent gap above the bars.
    const chrome = Math.max(0, Math.round(above + Math.max(below, barsH)));

    // Guard against a measurement that would collapse the split. min-height in
    // CSS already floors the rendered size; this keeps the variable itself sane.
    const usable = window.innerHeight - chrome;
    split.style.setProperty('--le-chrome-h', usable < MIN_SPLIT_H ? `${window.innerHeight - MIN_SPLIT_H}px` : `${chrome}px`);
  }
}

/**
 * Watch for the things that move a split: window resize, and the bottom bars
 * growing (the notices bar expands in place).
 *
 * Tab changes are driven by the caller, which knows when a pane has been
 * mounted — a MutationObserver over the whole document would fire far more
 * often for the same result.
 */
export function initListEditorHeights() {
  refreshListEditorHeights();

  window.addEventListener('resize', refreshListEditorHeights);

  const bars = document.getElementById('bottom-bars');
  if (bars && typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(refreshListEditorHeights).observe(bars);
  }
}
