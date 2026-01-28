// files.js
import { FileBrowser } from '/app-static/js/ui/fileBrowser.js';
import { loadImageAndMetadata } from './canvas/image.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';

let _initialized = false;
let browser = null;

export async function init() {
  if (_initialized) return;
  _initialized = true;

  const container = document.getElementById('tab-files');
  if (!container) return;

  browser = new FileBrowser({
    container,
    initialPath: '/',
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

export { _initialized as isInitialized };
