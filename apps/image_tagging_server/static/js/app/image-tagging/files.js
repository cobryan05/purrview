// files.js
import { FileBrowser } from '/app-static/js/ui/fileBrowser.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { guardUnsavedChanges } from './canvas/image.js';
import { toast } from '/app-static/js/ui/utils/logging.js';

let _initialized = false;
let browser = null;
let _lastBoundaryToast = 0;

/**
 * Navigate the browser to the directory containing the currently loaded image,
 * select it in the navigation state, rerender, then scroll to and flash it.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
async function revealCurrentFile(runtime) {
  const imagePath = runtime.state?.image?.image_path;
  if (!imagePath) {
    browser.rerender();
    return;
  }
  const parts = imagePath.split('/');
  const filename = parts.pop();
  const dir = parts.join('/') || '/';

  if (browser.currentPath !== dir) {
    // load() resets FileNavigation via setDirectory; re-select afterwards.
    await browser.load(dir);
    runtime.fileNavigation.select(filename);
  }
  browser.rerender();
  browser.scrollToActive();
}

export async function init(runtime) {
  if (_initialized) {
    await revealCurrentFile(runtime);
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
    showThumbnails: true,
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
  if (!nav.hasNext()) {
    await browser?.loadNextPage();
    if (!nav.hasNext()) {
      const now = Date.now();
      if (now - _lastBoundaryToast > 1200) {
        _lastBoundaryToast = now;
        toast('End of directory', 3000, 'warning');
      }
      return;
    }
  }
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
  if (!nav.hasPrev()) {
    const now = Date.now();
    if (now - _lastBoundaryToast > 1200) {
      _lastBoundaryToast = now;
      toast('Beginning of directory', 3000, 'warning');
    }
    return;
  }
  if (!await guardUnsavedChanges(runtime)) return;
  nav.prev();
  const payload = { path: nav.currentPath(), showCanvas: true, runtimeId: runtime.id };
  events.publish(EventTypes.LOAD_IMAGE_ONTO_CANVAS, payload);
}

export { _initialized as isInitialized };
