// files.js
import { FileBrowser } from '/app-static/js/ui/fileBrowser.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { guardUnsavedChanges } from './canvas/image.js';
import { toast } from '/app-static/js/ui/utils/logging.js';
import { showDeleteImageModal } from '/app-static/js/ui/utils/confirmModal.js';
import { deleteImage, deleteImageDbOnly } from '/app-static/js/shared/api/images.js';

let _initialized = false;
let browser = null;
let _lastBoundaryToast = 0;
/** Image path that was last synced into the file browser (used to detect canvas changes). */
let _lastSyncedImagePath = null;

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

  const imageChanged = imagePath !== _lastSyncedImagePath;
  _lastSyncedImagePath = imagePath;

  if (!imageChanged) {
    // Canvas image unchanged — preserve the user's browsed position.
    browser.rerender();
    return;
  }

  // Canvas image changed — attempt to navigate to its directory.
  if (browser.currentPath !== dir || browser.getEntries().length === 0) {
    // load() resets FileNavigation via setDirectory; re-select afterwards.
    await browser.load(dir);
    runtime.fileNavigation.select(filename);
    if (browser.getEntries().length === 0) {
      toast('Cannot browse to this image — its path is outside the file browser root.', 5000, 'warning');
      return;
    }
  } else {
    // Already in the right directory — just update the selection.
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
    // Import from the user's machine. The Files list is where you already go to
    // find an image, so it is where you would look to add one.
    allowUpload: true,
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

  // If an image is already loaded on the canvas, navigate directly to its
  // directory (avoids a redundant load of '/' that would immediately be
  // replaced). If no image is loaded, fall back to loading root.
  if (runtime.state?.image?.image_path) {
    await revealCurrentFile(runtime);
  } else {
    await browser.init();
  }
}


/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export async function next_image(runtime) {
  const nav = runtime.fileNavigation;
  if (!nav) return; // canvas has no navigation context (e.g. a single-candidate editor modal)
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
  if (!nav) return; // canvas has no navigation context (e.g. a single-candidate editor modal)
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

/**
 * Permanently delete the currently loaded image (DB record + file + sidecar),
 * navigate away, and refresh the file browser. If there is no adjacent image to
 * navigate to, the canvas is cleared rather than left showing the deleted image.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export async function deleteCurrentImage(runtime) {
  const image = runtime.state?.image;
  if (!image?.uuid) return;

  const { uuid: imageUuid, image_path: imagePath } = image;
  const filename = imagePath.split('/').pop();

  const choice = await showDeleteImageModal(filename);
  if (!choice) return;

  try {
    if (choice === 'disk') {
      await deleteImage(imageUuid);
    } else {
      await deleteImageDbOnly(imageUuid);
    }
  } catch (err) {
    toast(`Failed to delete image: ${err.message}`, 4000, 'error');
    return;
  }

  // Navigate away: prefer next, fall back to prev.
  const nav = runtime.fileNavigation;
  let nextPath = null;
  if (nav.hasNext()) {
    nav.next();
    nextPath = nav.currentPath();
  } else if (nav.hasPrev()) {
    nav.prev();
    nextPath = nav.currentPath();
  }

  if (nextPath) {
    events.publish(EventTypes.LOAD_IMAGE_ONTO_CANVAS, { path: nextPath, showCanvas: true, runtimeId: runtime.id });
  } else {
    // Nothing left to navigate to — leaving the just-deleted image on screen would look
    // editable (and its bboxes un-savable against a record that no longer exists), so
    // wipe the canvas back to its empty state to make the loss obvious.
    runtime.clearImage();
  }

  // Refresh the file browser so the deleted entry disappears.
  if (browser) {
    const dir = imagePath.includes('/') ? imagePath.slice(0, imagePath.lastIndexOf('/')) : '/';
    await browser.load(dir);
    if (nextPath) {
      const nextFilename = nextPath.split('/').pop();
      nav.select(nextFilename);
    }
    browser.rerender();
  }

  events.publish(EventTypes.IMAGE_DELETED, { imageUuid, imagePath });
}

export { _initialized as isInitialized };
