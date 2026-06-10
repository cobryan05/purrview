/**
 * DOM controller: captures `contextmenu` events and opens the context menu
 * for the nearest ancestor element that has a resolver attached.
 *
 * Canvas elements bypass this and call openContextMenuForObjects directly.
 */

import { getResolver, openContextMenuForObjects } from './registry.js';

let _installed = false;

/**
 * Install the document-level contextmenu capture handler (idempotent).
 */
export function installContextMenuController() {
  if (_installed) return;
  _installed = true;

  document.addEventListener('contextmenu', async (event) => {
    // Walk up the DOM to find the nearest element with an attached resolver
    let el = event.target;
    while (el && el !== document.body) {
      const resolver = getResolver(el);
      if (resolver) {
        event.preventDefault();
        const objects = resolver(event) || [];
        await openContextMenuForObjects(objects, { x: event.clientX, y: event.clientY });
        return;
      }
      el = el.parentElement;
    }
    // No resolver found — allow browser default
  }, true /* capture */);
}
