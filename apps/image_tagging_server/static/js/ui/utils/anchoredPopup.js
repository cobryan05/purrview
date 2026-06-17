/**
 * A floating popup anchored above-right of a trigger element, dismissed by an
 * outside click. Shared by call sites that need a simple flyout (e.g. a
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

  const rect = anchorEl.getBoundingClientRect();
  const popupRect = contentEl.getBoundingClientRect();

  // Right-align popup to the anchor's right edge; clamp to stay on screen.
  const leftPos = Math.min(
    Math.max(4, rect.right - popupRect.width),
    window.innerWidth - popupRect.width - 4,
  );
  contentEl.style.left = leftPos + 'px';
  contentEl.style.top = (rect.top - popupRect.height - 4) + 'px';
  contentEl.style.visibility = '';

  function close() {
    contentEl.remove();
    document.removeEventListener('mousedown', outsideListener, true);
  }

  function outsideListener(e) {
    if (!contentEl.contains(e.target) && e.target !== anchorEl) close();
  }
  document.addEventListener('mousedown', outsideListener, true);

  return { close };
}
