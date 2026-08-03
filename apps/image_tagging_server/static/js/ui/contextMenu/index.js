/**
 * Context-menu system entry point.
 *
 * Import this module and call initContextMenus() once at app startup.
 */

import './providers/bboxProvider.js';    // registers 'bbox'
import './providers/imageProvider.js';   // registers 'image'
import './providers/canvasProvider.js';  // registers 'canvas'
// Note: the canvas surface menu ('canvas' type — similar/dissimilar image search) is
// reached only via the canvas toolbar button (see app/image-tagging/toolbar.js), never
// from right-click. Right-clicking empty canvas space still falls straight through to
// the tool-toggle behavior in canvas/interaction.js's handleContextMenu(); that handler
// never builds a 'canvas' adapter, so registering this provider does not affect it.
import { installContextMenuController } from './controller.js';

export function initContextMenus() {
  installContextMenuController();
}

export { attachContext, openContextMenuForObjects } from './registry.js';
