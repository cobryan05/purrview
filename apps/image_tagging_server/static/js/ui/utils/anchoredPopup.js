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
  document.body.appendChild(contentEl);

  const rect = anchorEl.getBoundingClientRect();
  const popupRect = contentEl.getBoundingClientRect();
  contentEl.style.position = 'fixed';
  contentEl.style.left = Math.max(4, rect.right - popupRect.width) + 'px';
  contentEl.style.top = (rect.top - popupRect.height - 4) + 'px';

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
