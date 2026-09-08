import { apiFetch } from './apiUtils.js';

/**
 * Fetch a user setting value by key.
 * Returns null if the key has never been stored.
 * @param {string} key
 * @returns {Promise<string|null>}
 */
export async function getSetting(key) {
  const res = await apiFetch(`/api/settings/${encodeURIComponent(key)}`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Failed to get setting "${key}"`);
  const { value } = await res.json();
  return value ?? null;
}

/**
 * Upsert a user setting.
 * @param {string} key
 * @param {string} value  — must be a string; JSON-encode objects before passing.
 * @returns {Promise<void>}
 */
export async function setSetting(key, value) {
  const res = await apiFetch(`/api/settings/${encodeURIComponent(key)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ value }),
  });
  if (!res.ok) throw new Error(`Failed to save setting "${key}"`);
}

/**
 * Setting keys shared by more than one screen.
 *
 * They live here, in a leaf module, rather than in `app/settings/main.js` with the
 * keys only that screen uses: the training-server modal needs the artifact bound,
 * and importing it from the Settings screen would close the loop
 * settings → serviceStatus → trainingServerModal → settings. A cycle resolves to
 * `undefined` exports silently (see NOTES.md), so the fix is to not have one.
 */

/** Previous generations kept per model. Mirrors `MODEL_VERSIONS_KEEP_SETTING_KEY` in Python. */
export const MODEL_VERSIONS_KEEP_SETTING_KEY = 'model_versions_keep';

/** Finished job folders kept on the training server. Mirrors the Python constant of the same name. */
export const TRAINING_ARTIFACTS_KEEP_SETTING_KEY = 'training_artifacts_keep';
