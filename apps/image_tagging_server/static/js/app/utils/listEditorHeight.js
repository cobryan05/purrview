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

  for (const split of document.querySelectorAll('.list-editor-split')) {
    // offsetParent is null for display:none subtrees — the cheap visibility test.
    if (split.offsetParent === null) continue;

    const top = split.getBoundingClientRect().top;
    // `top` is relative to the viewport, which is exactly what `100vh - chrome`
    // wants: everything above the split plus the bar below it.
    const chrome = Math.max(0, Math.round(top + barsH));

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
