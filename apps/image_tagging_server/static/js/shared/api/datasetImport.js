/**
 * Importing an existing dataset — the read side of `datasets.js`'s export.
 *
 * Inspection and import are deliberately separate calls. `detect` only reads,
 * so the wizard can show what a folder holds, let the format be overridden,
 * and be closed again without having written anything; the import itself is a
 * task, started through `launchRun` like every other long job.
 */
import { apiFetch } from './apiUtils.js';

async function postJson(url, body) {
  const res = await apiFetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || `Request to ${url} failed`);
  }
  return data;
}

/** @returns {Promise<Array<{id: string, name: string, description: string}>>} */
export async function fetchDatasetImportFormats() {
  const res = await apiFetch('/api/datasets/import-formats');
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch import formats');
  }
  return data.formats ?? [];
}

/**
 * Unpack an uploaded archive into staging, ready to be inspected.
 *
 * Nothing lands in the image library here: staging is outside FILES_ROOT, so
 * a dataset the user inspects and then abandons never appears in the browser.
 *
 * @param {File} file - a .zip or .tar
 * @returns {Promise<string>} the staging id
 */
export async function stageDatasetArchive(file) {
  const form = new FormData();
  form.append('file', file, file.name);
  const res = await apiFetch('/api/datasets/import/archive', { method: 'POST', body: form });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to upload the archive');
  }
  return data.stagingUuid;
}

/** Delete a staged upload the user backed out of. */
export async function discardDatasetStaging(stagingUuid) {
  if (!stagingUuid) return;
  await postJson('/api/datasets/import/discard', { stagingUuid });
}

/**
 * Describe a folder or staged upload: format, class names, counts, warnings.
 *
 * @param {{path?: string, stagingUuid?: string, format?: string}} where
 * @returns {Promise<{format: string, root: string, classNames: string[],
 *   imageCount: number, annotationCount: number, warnings: string[]}>}
 */
export async function detectDatasetImport({ path, stagingUuid, format } = {}) {
  const data = await postJson('/api/datasets/import/detect', {
    path: path ?? null,
    stagingUuid: stagingUuid ?? null,
    format: format || null,
  });
  return {
    format: data.format,
    root: data.root ?? '',
    classNames: data.classNames ?? [],
    imageCount: data.imageCount ?? 0,
    annotationCount: data.annotationCount ?? 0,
    warnings: data.warnings ?? [],
  };
}
