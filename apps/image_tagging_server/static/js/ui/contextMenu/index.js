/**
 * Context-menu system entry point.
 *
 * Import this module and call initContextMenus() once at app startup.
 */

import './providers/bboxProvider.js';    // registers 'bbox'
import './providers/imageProvider.js';   // registers 'image'
// Note: canvasProvider ('canvas' type) is intentionally NOT auto-registered here to
// preserve the right-click → tool-toggle fallthrough on empty canvas.  If you want the
// canvas surface menu, import and register it explicitly.
import { installContextMenuController } from './controller.js';

export function initContextMenus() {
  installContextMenuController();
}

export { attachContext, openContextMenuForObjects } from './registry.js';
