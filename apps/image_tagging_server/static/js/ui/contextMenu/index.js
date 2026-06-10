/**
 * Context-menu system entry point.
 *
 * Import this module and call initContextMenus() once at app startup.
 */

import './providers/bboxProvider.js';   // registers 'bbox'
import './providers/imageProvider.js';  // registers 'image'
import { installContextMenuController } from './controller.js';

export function initContextMenus() {
  installContextMenuController();
}

export { attachContext, openContextMenuForObjects } from './registry.js';
