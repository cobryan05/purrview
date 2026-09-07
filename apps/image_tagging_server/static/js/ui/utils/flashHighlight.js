/**
 * Scroll an element into view and flash it, so a change the user just caused is
 * visibly *where* it happened.
 *
 * Extracted from the review grid's "recently viewed" card treatment, which was
 * the first place to need it (closing the canvas modal drops you back into a
 * long grid with no clue which card you were just on). The watcher editor needs
 * exactly the same thing for a different reason: "+ Add rule" appends a row
 * below the fold, so on a profile with a few rules the button reads as doing
 * nothing at all.
 *
 * The flash is CSS (`@keyframes recently-viewed-flash`), not inline style, so it
 * can be themed in one place and costs no layout — it animates `outline` and
 * `box-shadow`, neither of which affects the box model. That matters in the
 * watcher editor, whose row-height budget has no slack.
 */

/** Class carrying the flash animation. Defined in app.css. */
const FLASH_CLASS = 'recently-viewed';

/**
 * @param {?HTMLElement} element — the thing that changed. No-op when null, so
 *        callers don't have to guard a `querySelector` that found nothing.
 * @param {Object} [opts]
 * @param {boolean} [opts.scroll=true] — bring it into view first. Pass false when
 *        the caller has already positioned the viewport.
 * @param {ScrollLogicalPosition} [opts.block='nearest'] — 'nearest' keeps an
 *        element that is already visible exactly where it is, which is why it is
 *        the default: scrolling a visible thing is disorienting, not helpful.
 */
export function flashHighlight(element, { scroll = true, block = 'nearest' } = {}) {
  if (!element) return;
  if (scroll) element.scrollIntoView({ behavior: 'smooth', block });
  // Remove-reflow-add, not a plain add: adding a class the element already has
  // does not restart a CSS animation, so a second flash on the same element
  // (delete two rules in a row, reopen the same card) would show nothing.
  element.classList.remove(FLASH_CLASS);
  void element.offsetWidth; // force reflow so the animation restarts
  element.classList.add(FLASH_CLASS);
  element.addEventListener('animationend', () => element.classList.remove(FLASH_CLASS), { once: true });
}
