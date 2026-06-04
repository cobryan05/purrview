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
