// files.js
import { FileBrowser } from '/app-static/js/ui/fileBrowser.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';

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
    filter: entry => entry.type === 'dir' || entry.name.match(/\.(jpg|jpeg|png)$/i),
    onFileClick: (entry, dirPath) => {
      const imageName =
        (dirPath === '/' ? '' : dirPath + '/') + entry.name;

      /**
      * @typedef {import('@events').CanvasImageLoadPayload}
      */
      const payload = { path: imageName, showCanvas: true };
      events.publish(EventTypes.LOAD_IMAGE_ONTO_CANVAS, payload );
    }
  });

  await browser.init();
}


/**
 * @param {import('@image_tagging_types').ImageTaggingState} state
 */
export async function next_image(state) {
  if (state.fileNavigation.next()) {
    const payload = { path: state.fileNavigation.currentPath(), showCanvas: true };
    events.publish(EventTypes.LOAD_IMAGE_ONTO_CANVAS, payload);
  }
}

/**
 * @param {import('@image_tagging_types').ImageTaggingState} state
 */
export async function prev_image(state) {
  if (state.fileNavigation.prev()) {
    const payload = { path: state.fileNavigation.currentPath(), showCanvas: true };
    events.publish(EventTypes.LOAD_IMAGE_ONTO_CANVAS, payload);
  }
}

export { _initialized as isInitialized };
