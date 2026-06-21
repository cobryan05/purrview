import { createModalShell } from './utils/modalShell.js';
import { fetchModelTrainingImages } from '/app-static/js/shared/api/models.js';
import { fetchImagesByUuids } from '/app-static/js/shared/api/images.js';

/**
 * Open a modal showing the training and validation images for a given model.
 * @param {string} modelName
 */
export function openTrainingDataModal(modelName) {
  const { dialog, open, close } = createModalShell();
  dialog.style.cssText = 'width: min(92vw, 900px); max-height: 85vh; display: flex; flex-direction: column; padding: 0;';

  // Header
  const header = document.createElement('div');
  header.style.cssText = 'display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; border-bottom: 1px solid var(--pico-muted-border-color, #333); flex-shrink: 0;';
  const title = document.createElement('h3');
  title.style.cssText = 'margin: 0; font-size: 1em;';
  title.textContent = `Training data — ${modelName}`;
  const closeBtn = document.createElement('button');
  closeBtn.textContent = '✕';
  closeBtn.style.cssText = 'background: none; border: none; cursor: pointer; font-size: 1.1em; color: var(--pico-muted-color, #888); padding: 2px 6px;';
  closeBtn.onclick = close;
  header.appendChild(title);
  header.appendChild(closeBtn);

  // Body
  const body = document.createElement('div');
  body.style.cssText = 'overflow-y: auto; flex: 1; padding: 16px;';

  const status = document.createElement('p');
  status.style.cssText = 'color: var(--pico-muted-color, #888); font-size: 0.85em;';
  status.textContent = 'Loading…';
  body.appendChild(status);

  dialog.appendChild(header);
  dialog.appendChild(body);
  open();

  fetchModelTrainingImages(modelName).then(async ({ train, val }) => {
    if (train.length === 0 && val.length === 0) {
      status.textContent = "No training image UUIDs recorded in this model's sidecar. Re-train to populate.";
      return;
    }

    // Resolve all UUIDs to filenames in one batch call.
    const allUuids = [...train, ...val];
    let images;
    try {
      images = await fetchImagesByUuids(allUuids);
    } catch (err) {
      status.textContent = `Failed to resolve images: ${err.message}`;
      return;
    }

    /** @type {Map<string, string>} uuid → filename */
    const uuidToFilename = new Map(images.map(img => [img.uuid, img.filename]));

    body.innerHTML = '';

    _renderSection(body, 'Training', train, uuidToFilename);
    if (val.length > 0) _renderSection(body, 'Validation', val, uuidToFilename);
  }).catch(err => {
    status.textContent = `Error: ${err.message}`;
  });
}

function _renderSection(container, label, uuids, uuidToFilename) {
  const section = document.createElement('div');
  section.style.cssText = 'margin-bottom: 20px;';

  const heading = document.createElement('div');
  heading.style.cssText = 'font-size: 0.8em; font-weight: 600; color: var(--pico-muted-color, #888); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;';
  heading.textContent = `${label} (${uuids.length})`;
  section.appendChild(heading);

  const grid = document.createElement('div');
  grid.style.cssText = 'display: flex; flex-wrap: wrap; gap: 6px;';

  for (const uuid of uuids) {
    const filename = uuidToFilename.get(uuid);
    const cell = document.createElement('div');
    cell.style.cssText = 'width: 96px; height: 96px; overflow: hidden; border-radius: 4px; background: #1a1a1a; flex-shrink: 0;';
    cell.title = filename ?? uuid;

    if (filename) {
      const img = document.createElement('img');
      img.style.cssText = 'width: 100%; height: 100%; object-fit: cover; display: block;';
      img.src = `/api/images/${encodeURIComponent(filename)}/thumbnail?width=96`;
      img.alt = filename;
      img.onerror = () => { img.style.display = 'none'; };
      cell.appendChild(img);
    } else {
      cell.style.display = 'flex';
      cell.style.alignItems = 'center';
      cell.style.justifyContent = 'center';
      const missing = document.createElement('span');
      missing.style.cssText = 'font-size: 0.65em; color: #555; word-break: break-all; padding: 4px; text-align: center;';
      missing.textContent = uuid.slice(0, 8) + '…';
      cell.appendChild(missing);
    }

    grid.appendChild(cell);
  }

  section.appendChild(grid);
  container.appendChild(section);
}
