/**
 * Import an existing dataset — YOLO, COCO, Pascal VOC, or a pile of images.
 *
 * Three steps, in the order the answers actually depend on each other: where
 * the data is, what turned out to be in it, and where it should land. The
 * middle step cannot be filled in before the first is answered — the class
 * list comes from the dataset — which is why this is a wizard rather than one
 * long form.
 *
 * Labels are created here rather than by the task. The task accepts a complete
 * name→uuid map and creates nothing, so "map this class to an existing label",
 * "make a new label for it" and "skip it" are three choices the user makes
 * per class, instead of one global flag that cannot express the mixture.
 *
 * Nothing is written until Start. Inspecting a folder only reads it, and an
 * uploaded archive is unpacked outside the image library and deleted again if
 * the wizard is closed.
 */
import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';
import { openFileBrowserModal } from '/app-static/js/ui/utils/openFileBrowserModal.js';
import { launchRun } from '/app-static/js/ui/utils/launchRun.js';
import { toast, Logger } from '/app-static/js/ui/utils/logging.js';
import { fetchLabels, createNewLabel, clearLabelsCache } from '/app-static/js/shared/api/labels.js';
import { clearDatasetsCache } from '/app-static/js/shared/api/datasets.js';
import {
  detectDatasetImport,
  discardDatasetStaging,
  fetchDatasetImportFormats,
  stageDatasetArchive,
} from '/app-static/js/shared/api/datasetImport.js';

/** Archive types the staging endpoint accepts; mirrored to pre-filter the picker. */
const ARCHIVE_ACCEPT = '.zip,.tar,.tgz,.tar.gz,.tar.bz2,.tar.xz';

/** Default destination for a copied import, matching the task's own default. */
const DEFAULT_DEST = 'imported';

/** Sentinel select values, kept out of uuid space by the colon. */
const CREATE_NEW = 'new:';
const SKIP_CLASS = 'skip:';

const STEPS = ['Source', 'Contents', 'Destination'];

/** A stable, well-spread colour for a label created from a class name. */
function colorFor(name) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = (hash * 31 + name.charCodeAt(i)) >>> 0;
  return `hsl(${hash % 360} 55% 62%)`;
}

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  for (const child of children) if (child != null) node.append(child);
  return node;
}

function hint(text) {
  const small = el('small', { textContent: text });
  small.style.cssText = 'display:block; opacity:0.75; margin:-0.3em 0 0.8em 0;';
  return small;
}

/** Open the OS picker for one archive. Resolves null if nothing was chosen. */
function pickArchive() {
  return new Promise(resolve => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = ARCHIVE_ACCEPT;
    input.addEventListener('change', () => resolve(input.files?.[0] ?? null), { once: true });
    input.addEventListener('cancel', () => resolve(null), { once: true });
    input.click();
    // Same belt-and-braces as imageUpload.pickFiles: not every browser fires
    // 'cancel', and a promise that never settles leaves the button disabled.
    window.addEventListener('focus', () => {
      setTimeout(() => resolve(input.files?.[0] ?? null), 300);
    }, { once: true });
  });
}

/**
 * Open the Import Dataset wizard.
 *
 * @param {Object} [opts]
 * @param {string} [opts.initialPath] - folder to open the server browser at
 * @param {() => (void|Promise<void>)} [opts.onStarted] - e.g. refresh the file list
 */
export async function showDatasetImportModal({ initialPath = '/', onStarted = null } = {}) {
  const state = {
    step: 0,
    sourcePath: null,
    stagingUuid: null,
    archiveName: '',
    detected: null,
    /** class name → uuid | CREATE_NEW | SKIP_CLASS */
    classChoice: new Map(),
    copyFiles: true,
    markVerified: false,
    destPath: DEFAULT_DEST,
    datasetName: '',
    busy: false,
  };

  let formats = [];
  let labels = new Map();
  try {
    [formats, labels] = await Promise.all([fetchDatasetImportFormats(), fetchLabels()]);
  } catch (err) {
    Logger.warn('datasetImportModal: could not load formats/labels', err);
    // fetchLabels returns a Map; the fallback must be one too, or every
    // `.values()` below throws and the wizard never opens.
    labels = labels instanceof Map ? labels : new Map();
  }

  const { dialog, open, close } = createModalShell({
    onClose: () => {
      // A staged upload the user walked away from is dead weight on the cache
      // volume; the server sweeps it eventually, but not for a day.
      if (state.stagingUuid) discardDatasetStaging(state.stagingUuid).catch(() => {});
    },
  });
  dialog.dataset.testid = 'dataset-import-modal';
  Object.assign(dialog.style, {
    width: '640px',
    maxWidth: '94vw',
    maxHeight: '85vh',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75em',
  });

  const heading = el('strong', { textContent: 'Import dataset' });
  const stepLine = el('small', {});
  stepLine.dataset.testid = 'dataset-import-step';
  stepLine.style.cssText = 'opacity:0.75;';

  const body = el('div', {});
  body.dataset.testid = 'dataset-import-body';
  body.style.cssText = 'flex:1; overflow:auto; min-height:8em;';

  const backBtn = el('button', { type: 'button', className: 'secondary', textContent: 'Back' });
  backBtn.dataset.testid = 'dataset-import-back';
  const nextBtn = el('button', { type: 'button', textContent: 'Next' });
  nextBtn.dataset.testid = 'dataset-import-next';
  const cancelBtn = el('button', { type: 'button', className: 'secondary', textContent: 'Cancel' });
  cancelBtn.addEventListener('click', () => close());

  const footer = el('div', {}, cancelBtn, backBtn, nextBtn);
  footer.style.cssText = 'display:flex; gap:0.5em; justify-content:flex-end; flex-wrap:wrap;';

  dialog.append(heading, stepLine, body, footer);

  // ── rendering ──────────────────────────────────────────────────────────

  function setBusy(busy, label) {
    state.busy = busy;
    nextBtn.disabled = busy;
    backBtn.disabled = busy || state.step === 0;
    if (busy && label) nextBtn.textContent = label;
  }

  function render() {
    stepLine.textContent = `Step ${state.step + 1} of ${STEPS.length} — ${STEPS[state.step]}`;
    body.innerHTML = '';
    backBtn.disabled = state.step === 0 || state.busy;
    nextBtn.textContent = state.step === STEPS.length - 1 ? 'Start import' : 'Next';
    if (state.step === 0) renderSource();
    else if (state.step === 1) renderContents();
    else renderDestination();
  }

  function renderSource() {
    // Next is meaningless here: choosing a source is what advances the step,
    // because inspecting it is the thing that produces the next step's content.
    nextBtn.disabled = true;

    const folderBtn = el('button', { type: 'button', textContent: '📁 Choose a folder on the server' });
    folderBtn.dataset.testid = 'dataset-import-pick-folder';
    folderBtn.style.cssText = 'width:100%; margin-bottom:0.3em;';
    folderBtn.addEventListener('click', onPickFolder);

    const uploadBtn = el('button', { type: 'button', textContent: '⬆ Upload a .zip or .tar' });
    uploadBtn.dataset.testid = 'dataset-import-pick-archive';
    uploadBtn.style.cssText = 'width:100%; margin-bottom:0.3em;';
    uploadBtn.addEventListener('click', onPickArchive);

    body.append(
      folderBtn,
      hint('A folder already visible to the server — a bind mount, or anywhere under the image root. '
        + 'Nothing is copied unless you ask for it on the last step.'),
      uploadBtn,
      hint('Unpacked outside the image library and inspected first. Its images are copied in when you start.'),
    );
  }

  function renderContents() {
    const d = state.detected;
    const formatSelect = el('select', {});
    formatSelect.dataset.testid = 'dataset-import-format';
    for (const fmt of formats) {
      formatSelect.append(el('option', { value: fmt.id, textContent: fmt.name, selected: fmt.id === d.format }));
    }
    formatSelect.addEventListener('change', () => reDetect(formatSelect.value));

    const summary = el('p', {});
    summary.dataset.testid = 'dataset-import-summary';
    summary.textContent = `${d.imageCount} image${d.imageCount === 1 ? '' : 's'}, `
      + `${d.annotationCount} annotation${d.annotationCount === 1 ? '' : 's'}`
      + (d.root ? ` in ${d.root || '(top level)'}` : '');

    body.append(el('label', { textContent: 'Read as' }), formatSelect, summary);

    if (d.warnings.length) {
      const warn = el('details', {}, el('summary', { textContent: `${d.warnings.length} warning(s)` }));
      warn.dataset.testid = 'dataset-import-warnings';
      const list = el('ul', {});
      for (const w of d.warnings) list.append(el('li', { textContent: w }));
      warn.append(list);
      body.append(warn);
    }

    if (!d.classNames.length) {
      body.append(hint(d.annotationCount
        ? 'No class names were found, so the boxes cannot be mapped to labels.'
        : 'No annotations — the images will be imported unlabelled.'));
      return;
    }

    body.append(el('h5', { textContent: 'Classes' }));
    body.append(hint('Each class becomes boxes on the imported images. Whether those count as '
      + 'already reviewed is asked on the next step.'));

    const byName = new Map();
    for (const label of labels.values()) byName.set(label.name.trim().toLowerCase(), label.uuid);

    const table = el('div', {});
    table.dataset.testid = 'dataset-import-classes';
    table.style.cssText = 'display:grid; grid-template-columns:minmax(6em,1fr) minmax(8em,1.4fr); '
      + 'gap:0.4em 0.75em; align-items:center;';

    for (const name of d.classNames) {
      if (!state.classChoice.has(name)) {
        // An existing label of the same name is almost always the intent, and
        // guessing wrong here costs a duplicate label that splits the class.
        state.classChoice.set(name, byName.get(name.trim().toLowerCase()) ?? CREATE_NEW);
      }
      const select = el('select', {});
      select.dataset.testid = `dataset-import-class-${name}`;
      select.append(el('option', { value: CREATE_NEW, textContent: `➕ Create label "${name}"` }));
      select.append(el('option', { value: SKIP_CLASS, textContent: '⊘ Skip this class' }));
      for (const label of [...labels.values()].sort((a, b) => a.name.localeCompare(b.name))) {
        select.append(el('option', { value: label.uuid, textContent: label.name }));
      }
      select.value = state.classChoice.get(name);
      select.addEventListener('change', () => state.classChoice.set(name, select.value));
      table.append(el('span', { textContent: name }), select);
    }
    body.append(table);
  }

  function renderDestination() {
    const d = state.detected;
    const fromUpload = Boolean(state.stagingUuid);

    const copyInput = el('input', { type: 'checkbox', checked: state.copyFiles || fromUpload });
    copyInput.dataset.testid = 'dataset-import-copy';
    copyInput.disabled = fromUpload;
    copyInput.addEventListener('change', () => { state.copyFiles = copyInput.checked; render(); });
    const copyRow = el('label', {}, copyInput, document.createTextNode(' Copy the images into a folder'));
    body.append(copyRow);
    body.append(hint(fromUpload
      ? 'An uploaded archive is always copied in — it was unpacked outside the image library.'
      : 'Off imports the images where they already are, which costs no extra disk. '
        + 'On copies them into the folder below.'));

    if (copyInput.checked) {
      const destInput = el('input', { type: 'text', value: state.destPath });
      destInput.dataset.testid = 'dataset-import-dest';
      destInput.addEventListener('input', () => { state.destPath = destInput.value.trim(); });
      const browse = el('button', { type: 'button', className: 'secondary', textContent: 'Browse…' });
      browse.addEventListener('click', async () => {
        const chosen = await openFileBrowserModal({ initialPath: `/${state.destPath}`, allowCreateDir: true });
        if (chosen === null) return;
        state.destPath = chosen.replace(/^\/+/, '');
        render();
      });
      const row = el('div', {}, destInput, browse);
      row.style.cssText = 'display:flex; gap:0.5em; align-items:center;';
      destInput.style.flex = '1';
      body.append(el('label', { textContent: 'Destination folder' }), row);
      body.append(hint('Sub-folders in the source are kept. Re-importing the same dataset imports it again '
        + 'rather than merging — near-duplicates are reported, not skipped.'));
    }

    if (d.annotationCount) {
      const verifiedInput = el('input', { type: 'checkbox', checked: state.markVerified });
      verifiedInput.dataset.testid = 'dataset-import-verified';
      verifiedInput.addEventListener('change', () => { state.markVerified = verifiedInput.checked; render(); });
      body.append(el('label', {}, verifiedInput, document.createTextNode(' These boxes are already reviewed')));
      body.append(hint('Tick for a dataset a person annotated — the boxes count as confirmed and training '
        + 'uses them straight away. Leave it off for anything auto-labelled or of unknown provenance, and '
        + 'they queue up in review instead. Either way you can change it in bulk later.'));
    }

    const nameInput = el('input', { type: 'text', value: state.datasetName, placeholder: 'Leave empty for none' });
    nameInput.dataset.testid = 'dataset-import-dataset-name';
    nameInput.addEventListener('input', () => { state.datasetName = nameInput.value.trim(); });
    body.append(el('label', { textContent: 'Create a Dataset named' }), nameInput);
    body.append(hint('A Dataset here is a saved filter over a folder, not a fixed list — anything added to '
      + 'that folder later joins it too.'));

    const creating = [...state.classChoice.values()].filter(v => v === CREATE_NEW).length;
    const skipping = [...state.classChoice.values()].filter(v => v === SKIP_CLASS).length;
    const parts = [`${d.imageCount} image${d.imageCount === 1 ? '' : 's'}`];
    if (creating) parts.push(`${creating} new label${creating === 1 ? '' : 's'}`);
    if (skipping) parts.push(`${skipping} class${skipping === 1 ? '' : 'es'} skipped`);
    if (d.annotationCount) parts.push(state.markVerified ? 'boxes marked reviewed' : 'boxes awaiting review');
    const review = el('p', { textContent: `Ready: ${parts.join(', ')}.` });
    review.dataset.testid = 'dataset-import-review';
    body.append(review);
  }

  // ── actions ────────────────────────────────────────────────────────────

  async function inspect(where) {
    setBusy(true, 'Inspecting…');
    try {
      state.detected = await detectDatasetImport(where);
      state.classChoice.clear();
      state.step = 1;
    } catch (err) {
      toast(err.message || 'Could not read that dataset', 7000, 'error');
    } finally {
      setBusy(false);
      render();
    }
  }

  async function reDetect(format) {
    await inspect({ path: state.sourcePath, stagingUuid: state.stagingUuid, format });
  }

  async function onPickFolder() {
    const chosen = await openFileBrowserModal({ initialPath, allowCreateDir: false });
    if (chosen === null) return;
    // Switching source kinds must not leave the previous upload staged.
    if (state.stagingUuid) discardDatasetStaging(state.stagingUuid).catch(() => {});
    state.stagingUuid = null;
    state.sourcePath = chosen.replace(/^\/+/, '');
    state.copyFiles = false;
    await inspect({ path: state.sourcePath });
  }

  async function onPickArchive() {
    const file = await pickArchive();
    if (!file) return;
    setBusy(true, 'Uploading…');
    render();
    try {
      const stagingUuid = await stageDatasetArchive(file);
      if (state.stagingUuid) discardDatasetStaging(state.stagingUuid).catch(() => {});
      state.stagingUuid = stagingUuid;
      state.sourcePath = null;
      state.archiveName = file.name;
      state.copyFiles = true;
      await inspect({ stagingUuid });
    } catch (err) {
      toast(err.message || 'Could not read that archive', 7000, 'error');
      setBusy(false);
      render();
    }
  }

  /** Create the labels the user asked for, and return the full name→uuid map. */
  async function buildClassMap() {
    const map = {};
    const created = [];
    for (const [name, choice] of state.classChoice) {
      if (choice === SKIP_CLASS) continue;
      if (choice !== CREATE_NEW) {
        map[name] = choice;
        continue;
      }
      const label = await createNewLabel(name, colorFor(name));
      if (!label?.uuid) throw new Error(`Could not create a label for "${name}"`);
      map[name] = label.uuid;
      created.push(name);
    }
    // createNewLabel clears the fetcher itself; this keeps the wizard's own
    // snapshot from going stale if it is reopened without a reload.
    if (created.length) clearLabelsCache();
    return map;
  }

  async function start() {
    if (state.copyFiles && !state.destPath) {
      toast('Choose a destination folder, or turn copying off', 5000, 'error');
      return;
    }
    setBusy(true, 'Starting…');
    try {
      const classLabels = await buildClassMap();
      await launchRun({
        typename: 'ImportDatasetTask',
        name: `Import ${state.archiveName || state.detected.root || state.sourcePath || 'dataset'}`,
        params: {
          source_path: state.sourcePath ?? '',
          staging_uuid: state.stagingUuid ?? '',
          format: state.detected.format,
          copy_files: state.copyFiles,
          dest_path: state.destPath,
          class_labels: classLabels,
          // Every class already has an explicit choice, so the task must not
          // invent labels for anything the user chose to skip.
          create_missing_labels: false,
          mark_verified: state.markVerified,
          dataset_name: state.datasetName,
        },
      });
      // The task owns the staged tree from here and deletes it when it finishes;
      // clearing this stops onClose from pulling it out from under the import.
      state.stagingUuid = null;
      if (state.datasetName) clearDatasetsCache();
      toast('Import started — watch it in Jobs', 5000);
      close();
      await onStarted?.();
    } catch (err) {
      Logger.warn('datasetImportModal: import failed to start', err);
      toast(err.message || 'Could not start the import', 7000, 'error');
      setBusy(false);
      render();
    }
  }

  backBtn.addEventListener('click', () => {
    if (state.step > 0) state.step -= 1;
    render();
  });
  nextBtn.addEventListener('click', () => {
    if (state.step === STEPS.length - 1) return void start();
    state.step += 1;
    render();
  });

  render();
  open();
}
