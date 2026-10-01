/**
 * Import images from the user's machine.
 *
 * Two entry points share this: a button in the Files browser toolbar and one on
 * the canvas toolbar. Both end in the same place — files written under
 * FILES_ROOT, registered in the DB, and the first one opened on the canvas.
 *
 * The destination is asked for once and then remembered, because picking a
 * folder for every drop is friction on the common path (everything lands in the
 * same place) while never asking would silently scatter files wherever you
 * happened to be browsing.
 */
import { uploadImages, getUploadTarget, setUploadTarget } from '/app-static/js/shared/api/uploads.js';
import { openFileBrowserModal } from '/app-static/js/ui/utils/openFileBrowserModal.js';
import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { getPrimaryRuntimeId } from '/app-static/js/app/image-tagging/runtime.js';
import { Logger, toast } from '/app-static/js/ui/utils/logging.js';

/** Extensions the server accepts; mirrored here only to pre-filter the picker. */
const ACCEPT = '.jpg,.jpeg,.png,.bmp,.webp';

/**
 * Normalise a browser path ('/', '/a/b') to what the API wants ('', 'a/b').
 * @param {string} path
 * @returns {string}
 */
function toRelative(path) {
  if (!path || path === '/') return '';
  return path.replace(/^\/+/, '');
}

/**
 * Ask where uploads should go, offering to stop asking.
 *
 * @param {string} suggested - directory to start from, browser-style
 * @returns {Promise<?{path: string, remember: boolean}>} null if cancelled
 */
async function promptForTarget(suggested) {
  const chosen = await openFileBrowserModal({
    initialPath: suggested || '/',
    allowCreateDir: true,
  });
  if (chosen === null) return null;

  // A second, deliberately tiny dialog rather than a checkbox wedged into the
  // browser modal: openFileBrowserModal is shared with several callers that
  // have nothing to do with uploading, and this keeps its contract intact.
  return new Promise(resolve => {
    let settled = false;
    const finish = value => {
      if (settled) return;
      settled = true;
      resolve(value);
    };

    const { dialog, open, close } = createModalShell({ onClose: () => finish(null) });
    dialog.dataset.testid = 'upload-remember-dialog';
    Object.assign(dialog.style, { width: '420px', maxWidth: '92vw' });

    const body = document.createElement('div');
    body.className = 'upload-remember';

    const heading = document.createElement('h4');
    heading.textContent = 'Upload to this folder?';
    body.appendChild(heading);

    const where = document.createElement('p');
    where.className = 'upload-remember__path';
    where.textContent = toRelative(chosen) || '(top level)';
    body.appendChild(where);

    const rememberRow = document.createElement('label');
    rememberRow.className = 'upload-remember__check';
    const remember = document.createElement('input');
    remember.type = 'checkbox';
    remember.checked = true;
    remember.dataset.testid = 'upload-remember-checkbox';
    rememberRow.append(remember, document.createTextNode(" Always upload here, don't ask again"));
    body.appendChild(rememberRow);

    const hint = document.createElement('p');
    hint.className = 'upload-remember__hint';
    hint.textContent = 'You can change this later from the Files toolbar.';
    body.appendChild(hint);

    const actions = document.createElement('div');
    actions.className = 'upload-remember__actions';

    const cancel = document.createElement('button');
    cancel.type = 'button';
    cancel.className = 'secondary panel-btn';
    cancel.textContent = 'Cancel';
    // Answer first, close second. `close()` runs `onClose` synchronously, so
    // closing first settles the promise with the dismissal value and the real
    // answer is dropped — which made "Upload here" upload nothing at all.
    cancel.addEventListener('click', () => { finish(null); close(); });

    const confirm = document.createElement('button');
    confirm.type = 'button';
    confirm.className = 'panel-btn';
    confirm.dataset.testid = 'upload-remember-confirm';
    confirm.textContent = 'Upload here';
    confirm.addEventListener('click', () => {
      finish({ path: toRelative(chosen), remember: remember.checked });
      close();
    });

    actions.append(cancel, confirm);
    body.appendChild(actions);
    dialog.appendChild(body);
    open();
    confirm.focus();
  });
}

/**
 * Resolve the destination, prompting only when we have to.
 *
 * @param {Object} opts
 * @param {string} [opts.currentPath] - where the user is browsing, as a default
 * @param {boolean} [opts.forcePrompt] - "Change upload folder…" always asks
 * @returns {Promise<?string>} FILES_ROOT-relative dir, or null if cancelled
 */
async function resolveTarget({ currentPath = '/', forcePrompt = false } = {}) {
  let stored = { path: null, remembered: false };
  try {
    stored = await getUploadTarget();
  } catch (err) {
    // A settings read failure must not block an import; fall through to asking.
    Logger.warn('imageUpload: could not read the stored upload target', err);
  }

  if (!forcePrompt && stored.remembered && stored.path !== null) return stored.path;

  const answer = await promptForTarget(stored.path ?? currentPath);
  if (answer === null) return null;

  try {
    await setUploadTarget(answer.path, answer.remember);
  } catch (err) {
    // Losing the preference is survivable — the upload itself is not.
    Logger.warn('imageUpload: could not save the upload target', err);
  }
  return answer.path;
}

/**
 * Report what happened, in one toast rather than one per file.
 *
 * Failures and duplicates are called out by name: "3 of 20 uploaded" without
 * saying which three is not actionable.
 *
 * @param {Array<object>} results
 */
function reportResults(results) {
  const ok = results.filter(r => r.ok);
  const failed = results.filter(r => !r.ok);
  const duplicates = ok.filter(r => r.duplicateOf);
  const renamed = ok.filter(r => r.renamedFrom);

  if (ok.length) {
    toast(`Imported ${ok.length} image${ok.length === 1 ? '' : 's'}`, 3000);
  }
  if (renamed.length) {
    const names = renamed.slice(0, 3).map(r => `${r.renamedFrom} → ${r.path}`).join(', ');
    toast(`Renamed to avoid overwriting: ${names}${renamed.length > 3 ? '…' : ''}`, 6000, 'warning');
  }
  if (duplicates.length) {
    const names = duplicates.slice(0, 3).map(r => r.filename).join(', ');
    toast(
      `${duplicates.length} look like images already in the database (${names}${duplicates.length > 3 ? '…' : ''}). They were imported anyway.`,
      7000,
      'warning',
    );
  }
  for (const failure of failed.slice(0, 4)) {
    toast(`${failure.filename}: ${failure.message || 'could not be imported'}`, 7000, 'error');
  }
  if (failed.length > 4) {
    toast(`…and ${failed.length - 4} more failed`, 5000, 'error');
  }
}

/**
 * Run an import: pick files, resolve the destination, upload, report, open.
 *
 * @param {Object} opts
 * @param {string} [opts.currentPath] - browser location, used as the default dir
 * @param {boolean} [opts.forcePrompt] - always ask for the destination
 * @param {() => (void|Promise<void>)} [opts.onUploaded] - e.g. refresh the file list
 * @param {boolean} [opts.openFirst] - load the first imported image onto the canvas
 * @returns {Promise<Array<object>>} the per-file results (empty if cancelled)
 */
export async function importImages({
  currentPath = '/',
  forcePrompt = false,
  onUploaded = null,
  openFirst = true,
} = {}) {
  const files = await pickFiles();
  if (!files.length) return [];

  const target = await resolveTarget({ currentPath, forcePrompt });
  if (target === null) return [];

  let results;
  try {
    results = await uploadImages({ files, path: target });
  } catch (err) {
    Logger.warn('imageUpload: upload failed', err);
    toast(err.message || 'Upload failed', 6000, 'error');
    return [];
  }

  reportResults(results);
  await onUploaded?.();

  // Land on the first thing imported: an import you cannot see is hard to
  // believe happened, and annotating it is the reason it was imported.
  //
  // Addressed to the primary runtime explicitly. A canvas-scoped event with no
  // runtimeId is delivered to *every* subscriber (see EventBus.subscribeToCanvas),
  // which would also hijack an open Review editor modal and replace the image
  // the user is part-way through annotating there.
  const first = results.find(r => r.ok && r.path);
  if (openFirst && first) {
    events.publish(EventTypes.LOAD_IMAGE_ONTO_CANVAS, {
      path: first.path,
      showCanvas: true,
      runtimeId: getPrimaryRuntimeId(),
    });
  }
  return results;
}

/**
 * Open the OS file picker.
 *
 * The input is created per call and never attached to the document: a hidden
 * one parked in the DOM keeps its previous selection, so re-picking the same
 * file after a failed upload would fire no change event at all.
 *
 * @returns {Promise<File[]>}
 */
function pickFiles() {
  return new Promise(resolve => {
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = true;
    input.accept = ACCEPT;
    input.addEventListener('change', () => resolve([...(input.files ?? [])]), { once: true });
    // A cancelled picker fires no 'change' in most browsers; 'cancel' is not
    // universal either, so resolve empty on whichever arrives.
    input.addEventListener('cancel', () => resolve([]), { once: true });
    input.click();
    // …and where neither fires (older Safari, and anything else that skips
    // 'cancel'), this promise would never settle: the caller's `finally` never
    // runs, so the Import button stays disabled until the pane is re-rendered.
    // Regaining window focus is the one signal every browser gives, since the
    // picker is modal — but 'change' can land just after it, so give it a beat
    // before concluding nothing was chosen. Resolving twice is a no-op.
    window.addEventListener('focus', () => {
      setTimeout(() => resolve([...(input.files ?? [])]), 300);
    }, { once: true });
  });
}

/**
 * The Files-toolbar control: import, plus a way to change the remembered folder.
 *
 * @param {Object} opts
 * @param {() => string} opts.getCurrentPath
 * @param {() => (void|Promise<void>)} [opts.onUploaded]
 * @returns {HTMLElement}
 */
export function createUploadControls({ getCurrentPath, onUploaded = null }) {
  const wrap = document.createElement('span');
  wrap.className = 'fb-upload-controls';

  const uploadBtn = document.createElement('button');
  uploadBtn.type = 'button';
  uploadBtn.className = 'fb-upload-btn';
  uploadBtn.dataset.testid = 'file-upload-button';
  uploadBtn.dataset.tour = 'files.upload';
  uploadBtn.textContent = '⬆ Upload images';
  uploadBtn.addEventListener('click', async () => {
    uploadBtn.disabled = true;
    try {
      await importImages({ currentPath: getCurrentPath(), onUploaded, openFirst: false });
    } finally {
      uploadBtn.disabled = false;
    }
  });

  const changeBtn = document.createElement('button');
  changeBtn.type = 'button';
  changeBtn.className = 'fb-upload-target-btn secondary';
  changeBtn.dataset.testid = 'file-upload-target-button';
  changeBtn.textContent = '⚙';
  changeBtn.title = 'Change the upload folder';
  changeBtn.setAttribute('aria-label', 'Change the upload folder');
  changeBtn.addEventListener('click', async () => {
    changeBtn.disabled = true;
    try {
      const target = await resolveTarget({ currentPath: getCurrentPath(), forcePrompt: true });
      if (target !== null) toast(`Uploads will go to ${target || '(top level)'}`, 3000);
    } finally {
      changeBtn.disabled = false;
    }
  });

  wrap.append(uploadBtn, changeBtn);
  return wrap;
}
