/**
 * The Files toolbar's "get images in here" cluster.
 *
 * Three ways in, side by side, because they answer the same question and are
 * only distinguishable by where the images are coming from:
 *
 *   Upload images  — files on this computer.
 *   Import dataset — an annotated dataset, on the server or as an archive.
 *   Collect…       — a folder or camera the collector should keep pulling from.
 *
 * The two wizards are loaded on click. Neither is small, and the Files pane is
 * on the first screen most sessions see; making every one of them pay for code
 * that runs on a deliberate button press would be the wrong trade.
 */
import { createUploadControls } from '/app-static/js/ui/imageUpload.js';
import { Logger, toast } from '/app-static/js/ui/utils/logging.js';

/**
 * @param {Object} opts
 * @param {() => string} opts.getCurrentPath - browser location, read at click time
 * @param {() => (void|Promise<void>)} [opts.onChanged] - refresh the listing
 * @returns {HTMLElement}
 */
export function createImportControls({ getCurrentPath, onChanged = null }) {
  const wrap = document.createElement('span');
  wrap.className = 'fb-import-controls';

  wrap.appendChild(createUploadControls({ getCurrentPath, onUploaded: onChanged }));

  wrap.appendChild(lazyButton({
    label: '📦 Import dataset',
    testid: 'file-import-dataset-button',
    tour: 'files.importDataset',
    title: 'Import a YOLO, COCO or Pascal VOC dataset, or a folder of images',
    load: () => import('/app-static/js/ui/datasetImportModal.js'),
    run: (mod) => mod.showDatasetImportModal({
      initialPath: getCurrentPath(),
      onStarted: onChanged,
    }),
  }));

  wrap.appendChild(lazyButton({
    label: '🎞 Collect…',
    testid: 'file-collect-wizard-button',
    tour: 'files.collectWizard',
    title: 'Set up a collector that pulls training images from a folder, video or camera',
    load: () => import('/app-static/js/ui/collectWizardModal.js'),
    run: (mod) => mod.showCollectWizardModal({
      initialPath: getCurrentPath(),
      onSaved: onChanged,
    }),
  }));

  return wrap;
}

/**
 * A button whose module is fetched on first press.
 *
 * Disabled for the duration: a dynamic import over a cold cache is slow enough
 * to invite a second click, and two wizards would open.
 */
function lazyButton({ label, testid, tour, title, load, run }) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'fb-import-btn secondary';
  btn.dataset.testid = testid;
  btn.dataset.tour = tour;
  btn.textContent = label;
  btn.title = title;
  btn.addEventListener('click', async () => {
    btn.disabled = true;
    try {
      await run(await load());
    } catch (err) {
      Logger.warn(`importControls: ${testid} failed`, err);
      toast(err.message || 'Could not open that', 6000, 'error');
    } finally {
      btn.disabled = false;
    }
  });
  return btn;
}
