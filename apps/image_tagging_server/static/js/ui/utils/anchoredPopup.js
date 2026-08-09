import { acquireModalLayer, registerEscHandler, unregisterEscHandler } from './modalZIndex.js';

/**
 * A floating popup anchored above-right of a trigger element, dismissed by an
 * outside click or ESC. Shared by call sites that need a simple flyout (e.g. a
 * checkbox/options menu) without the full context-menu engine in
 * ui/contextMenu/menu.js (sections/submenus/keyboard nav).
 *
 * Usage:
 *   let popup = null;
 *   function open(anchorEl) {
 *     popup = openAnchoredPopup(anchorEl, buildContentEl());
 *   }
 *   function close() { popup?.close(); popup = null; }
 */

/**
 * Wires the outside-click + ESC dismiss contract shared by every anchored
 * popup/popover, for callers that manage their own positioning (e.g. a
 * CSS-anchored popover) and so can't use openAnchoredPopup itself.
 *
 * Capture-phase mousedown rather than click: a click that starts inside the
 * popup and releases outside it (dragging across a slider or selecting text)
 * would otherwise dismiss the panel mid-interaction.
 *
 * @param {HTMLElement} contentEl  popup content; clicks inside never dismiss
 * @param {HTMLElement} anchorEl  trigger element; clicks on/inside it never dismiss
 *   (its own handler is assumed to toggle open state — dismissing here too
 *   would close-then-reopen and make the button look dead)
 * @param {() => void} onClose  called on outside mousedown or ESC
 * @returns {() => void} unbind — call when closing via another path (e.g. its own ✕ button)
 */
export function bindOutsideDismiss(contentEl, anchorEl, onClose) {
  function outsideListener(e) {
    if (contentEl.contains(e.target) || anchorEl.contains(e.target)) return;
    onClose();
  }
  document.addEventListener('mousedown', outsideListener, true);
  registerEscHandler(onClose);
  return function unbind() {
    unregisterEscHandler(onClose);
    document.removeEventListener('mousedown', outsideListener, true);
  };
}

/**
 * @param {HTMLElement} anchorEl  element the popup is positioned relative to
 * @param {HTMLElement} contentEl  popup content; caller owns its styling
 * @returns {{ close: () => void }}
 */
export function openAnchoredPopup(anchorEl, contentEl) {
  // Hide before appending so getBoundingClientRect reflects the element's
  // actual rendered size before we know where to position it.
  contentEl.style.visibility = 'hidden';
  contentEl.style.position = 'fixed';
  document.body.appendChild(contentEl);

  // Share the overlay band so a popup opened from inside a modal lands above it
  // (PR 6.1) — but only when the caller has no opinion. Popups anchored to
  // controls *inside* #bottom-bars (--layer-bottom-bars) must sit above that
  // bar, i.e. outside the modal band entirely; they say so with an inline
  // z-index or a class rule, and overwriting it dropped them behind their own
  // anchor. Read the computed value so a class rule counts, not just inline.
  // Appended first so that rule has actually matched.
  let layer = null;
  if (getComputedStyle(contentEl).zIndex === 'auto') {
    layer = acquireModalLayer();
    contentEl.style.zIndex = layer.zIndex;
  }

  const rect = anchorEl.getBoundingClientRect();
  const popupRect = contentEl.getBoundingClientRect();

  // Right-align popup to the anchor's right edge; clamp to stay on screen.
  const leftPos = Math.min(
    Math.max(4, rect.right - popupRect.width),
    window.innerWidth - popupRect.width - 4,
  );
  contentEl.style.left = leftPos + 'px';

  // Prefer above the anchor (natural for bottom-anchored chrome). If it would
  // run off the top — e.g. a menu opened from a panel header near the top of the
  // viewport — flip below instead, then clamp so it stays fully on screen.
  let topPos = rect.top - popupRect.height - 4;
  if (topPos < 4) topPos = rect.bottom + 4;
  topPos = Math.max(4, Math.min(topPos, window.innerHeight - popupRect.height - 4));
  contentEl.style.top = topPos + 'px';
  contentEl.style.visibility = '';

  let _closed = false;
  let unbindDismiss = null;
  function close() {
    if (_closed) return;
    _closed = true;
    unbindDismiss();
    layer?.release();
    contentEl.remove();
  }
  unbindDismiss = bindOutsideDismiss(contentEl, anchorEl, close);

  return { close };
}
