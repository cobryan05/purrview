/**
 * Importing images from the user's machine.
 *
 * The upload target is remembered as a user setting rather than kept in module
 * state, so it survives a reload — otherwise "remember my choice" would only
 * hold until the next refresh, which is not what remembering means.
 */
import { apiFetch } from './apiUtils.js';
import { getSetting, setSetting } from './settings.js';

/** Where uploads go. Empty string is a legitimate value: FILES_ROOT itself. */
const UPLOAD_TARGET_KEY = 'upload_target_dir';
/** Whether the user has asked not to be prompted again. */
const UPLOAD_TARGET_REMEMBERED_KEY = 'upload_target_remembered';

/**
 * @returns {Promise<{path: string|null, remembered: boolean}>}
 *   `path` is null only when nothing has ever been chosen — distinct from '',
 *   which means "the root, deliberately".
 */
export async function getUploadTarget() {
  const [path, remembered] = await Promise.all([
    getSetting(UPLOAD_TARGET_KEY),
    getSetting(UPLOAD_TARGET_REMEMBERED_KEY),
  ]);
  return { path, remembered: remembered === 'true' };
}

/**
 * @param {string} path - FILES_ROOT-relative destination ('' is the root)
 * @param {boolean} remembered - stop asking on every upload
 */
export async function setUploadTarget(path, remembered) {
  await Promise.all([
    setSetting(UPLOAD_TARGET_KEY, path ?? ''),
    setSetting(UPLOAD_TARGET_REMEMBERED_KEY, remembered ? 'true' : 'false'),
  ]);
}

/**
 * Upload files into a directory under FILES_ROOT.
 *
 * Sent as one multipart request rather than one per file so the whole drop
 * either reaches the server or does not — and so a twenty-image import is one
 * round trip. Per-file outcomes come back in `results`; a rejected file does
 * not fail the request.
 *
 * @param {Object} opts
 * @param {File[]} opts.files
 * @param {string} [opts.path] - destination, FILES_ROOT-relative
 * @param {boolean} [opts.overwrite] - replace rather than uniquify on collision
 * @param {string} [opts.sourceUuid] - camera/stream this image was captured from,
 *   if any (sources table). Omitted for a hand-picked file import, where there
 *   is no camera to attribute it to.
 * @param {boolean} [opts.skipDuplicates] - do not write a file that matches an image
 *   already in the DB; the result carries `skipped: true` and `duplicateOf` instead of
 *   a path. For automated producers only — a hand-picked import is a deliberate act and
 *   gets the advisory `duplicateOf` without losing the file.
 * @returns {Promise<Array<{
 *   filename: string, ok: boolean, path?: string, uuid?: string,
 *   message?: string, duplicateOf?: string, duplicateDistance?: number,
 *   renamedFrom?: string, skipped?: boolean,
 * }>>}
 */
export async function uploadImages({
  files, path = '', overwrite = false, sourceUuid = '', skipDuplicates = false,
}) {
  const form = new FormData();
  for (const file of files) form.append('files', file, file.name);
  // The server reads these as form fields, not JSON — booleans go as strings.
  form.append('path', path ?? '');
  form.append('overwrite', overwrite ? 'true' : 'false');
  form.append('source_uuid', sourceUuid ?? '');
  form.append('skip_duplicates', skipDuplicates ? 'true' : 'false');

  // No Content-Type header: the browser has to set it, because only it knows
  // the multipart boundary it generated.
  const res = await apiFetch('/api/images/upload', { method: 'POST', body: form });
  if (!res.ok) {
    let message = 'Upload failed';
    try {
      message = (await res.json()).message || message;
    } catch {
      // A non-JSON error body (a proxy 502, say) leaves the default.
    }
    throw new Error(message);
  }
  const data = await res.json();
  return data.results ?? [];
}
