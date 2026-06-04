// files.js
import { FileBrowser } from '/app-static/js/ui/fileBrowser.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { guardUnsavedChanges } from './canvas/image.js';

let _initialized = false;
let browser = null;

/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export async function init(runtime) {
  if (_initialized) {
      browser.rerender();
      return;
  }
  _initialized = true;

  const container = document.getElementById('tab-files');
  if (!container) return;

  browser = new FileBrowser({
    container,
    initialPath: '/',
    fileNavigation: runtime.fileNavigation,
    showDbStatus: true,
    filter: entry => entry.isDir || entry.name.match(/\.(jpg|jpeg|png)$/i),
    onFileClick: async (entry, dirPath) => {
      const imageName =
        (dirPath === '/' ? '' : dirPath + '/') + entry.name;

      // Reloading the currently-open image is intentional (no guard); switching
      // to a different image first guards against unsaved changes.
      if (imageName !== runtime.state?.image?.image_path) {
        if (!await guardUnsavedChanges(runtime)) return;
      }

      /**
      * @typedef {import('@events').CanvasImageLoadPayload}
      */
      const payload = { path: imageName, showCanvas: true, runtimeId: runtime.id };
      events.publish(EventTypes.LOAD_IMAGE_ONTO_CANVAS, payload );
    }
  });

  await browser.init();
}


/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export async function next_image(runtime) {
  const nav = runtime.fileNavigation;
  if (!nav.hasNext()) return;
  if (!await guardUnsavedChanges(runtime)) return;
  nav.next();
  const payload = { path: nav.currentPath(), showCanvas: true, runtimeId: runtime.id };
  events.publish(EventTypes.LOAD_IMAGE_ONTO_CANVAS, payload);
}

/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export async function prev_image(runtime) {
  const nav = runtime.fileNavigation;
  if (!nav.hasPrev()) return;
  if (!await guardUnsavedChanges(runtime)) return;
  nav.prev();
  const payload = { path: nav.currentPath(), showCanvas: true, runtimeId: runtime.id };
  events.publish(EventTypes.LOAD_IMAGE_ONTO_CANVAS, payload);
}

export { _initialized as isInitialized };
