// Customizable canvas hotkeys, shared across every canvas (main + modals).
// Bindings are stored in the DB (user_settings) and mirrored to localStorage as a
// fast synchronous fallback.  The { [keyCombo]: actionId } map means a combo maps
// to exactly one action and lookups during keydown are O(1).
//
// actionId forms:
//   'action:<name>'  fixed actions (select/bbox/prevImage/nextImage/delete/undo/redo)
//   'label:<uuid>'   selects that label's draw tool ('bbox:<uuid>')
//   'tag:<uuid>'     selects that tag's apply tool ('tag:<uuid>')

import { getSetting, setSetting } from '/app-static/js/shared/api/settings.js';

const STORAGE_KEY = 'ww_hotkey_bindings';   // localStorage fallback key
const DB_SETTING_KEY = 'hotkey_bindings';   // user_settings DB key

/** Fixed (non-label/tag) actions, in display order for the config modal. */
export const FIXED_ACTIONS = [
  { id: 'action:select', label: 'Select tool' },
  { id: 'action:bbox', label: 'Draw bbox tool' },
  { id: 'action:prevImage', label: 'Previous image' },
  { id: 'action:nextImage', label: 'Next image' },
  { id: 'action:delete', label: 'Delete selected bbox' },
  { id: 'action:deleteImage', label: 'Delete current image' },
  { id: 'action:undo', label: 'Undo' },
  { id: 'action:redo', label: 'Redo' },
];

const DEFAULT_BINDINGS = {
  'v': 'action:select',
  'r': 'action:bbox',
  '[': 'action:prevImage',
  ']': 'action:nextImage',
  'delete': 'action:delete',
  'backspace': 'action:delete',
  'ctrl+z': 'action:undo',
  'ctrl+shift+z': 'action:redo',
  'shift+delete': 'action:deleteImage',
  'shift+backspace': 'action:deleteImage',
};

/** @type {Record<string,string>|null} */
let _bindings = null;

/** Singleton init promise — prevents double-fetching; resolved = done. */
let _initPromise = null;

// ─── Sync localStorage helpers ────────────────────────────────────────────────

function _loadFromLocalStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : { ...DEFAULT_BINDINGS };
  } catch {
    return { ...DEFAULT_BINDINGS };
  }
}

function _saveToLocalStorage() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(_bindings));
  } catch { /* storage unavailable — keep in-memory copy */ }
}

// ─── Async DB persistence ─────────────────────────────────────────────────────

function _persistToDb() {
  const json = JSON.stringify(_bindings);
  setSetting(DB_SETTING_KEY, json).catch(err =>
    console.warn('Failed to save hotkeys to DB:', err)
  );
}

function _persist() {
  _saveToLocalStorage();
  _persistToDb();
}

// ─── Async init ───────────────────────────────────────────────────────────────

/**
 * Load bindings from the DB (with localStorage migration on first run).
 * Safe to call multiple times — only runs once.
 * @returns {Promise<void>}
 */
export async function initHotkeyBindings() {
  if (_initPromise) return _initPromise;

  _initPromise = (async () => {
    try {
      const dbValue = await getSetting(DB_SETTING_KEY);
      if (dbValue !== null) {
        _bindings = JSON.parse(dbValue);
        // Mirror to localStorage so next startup is instant even if DB is slow.
        _saveToLocalStorage();
      } else {
        // DB has no entry yet: load from localStorage (migration) or defaults.
        _bindings = _loadFromLocalStorage();
        // Persist to DB so next load comes from there.
        _persistToDb();
      }
    } catch (err) {
      console.warn('Failed to load hotkeys from DB, using localStorage fallback:', err);
      _bindings = _loadFromLocalStorage();
    }
  })();

  return _initPromise;
}

// ─── Sync helpers (use in-memory cache; call initHotkeyBindings() first) ──────

function _ensureLoaded() {
  if (!_bindings) {
    // Fallback for callers that haven't awaited init (e.g. first keydown before init).
    _bindings = _loadFromLocalStorage();
  }
  return _bindings;
}

// ─── Public API ───────────────────────────────────────────────────────────────

/** Build a normalized combo string from a keyboard event. */
export function eventToCombo(e) {
  const parts = [];
  if (e.ctrlKey || e.metaKey) parts.push('ctrl');
  if (e.altKey) parts.push('alt');
  if (e.shiftKey) parts.push('shift');
  parts.push((e.key || '').toLowerCase());
  return parts.join('+');
}

/** Human-friendly label for a combo, e.g. 'ctrl+shift+z' → 'Ctrl+Shift+Z'. */
export function comboLabel(combo) {
  if (!combo) return '—';
  return combo
    .split('+')
    .map(p => (p.length === 1 ? p.toUpperCase() : p[0].toUpperCase() + p.slice(1)))
    .join('+');
}

/** The action bound to a combo, or null. */
export function actionForCombo(combo) {
  return _ensureLoaded()[combo] ?? null;
}

/** All combos bound to a given action (usually 0 or 1). */
export function combosForAction(actionId) {
  const b = _ensureLoaded();
  return Object.keys(b).filter(c => b[c] === actionId);
}

/**
 * Bind a combo to an action. Any prior combo for that action is removed (one
 * combo per action via the UI), and the combo is detached from any other action.
 */
export function setBinding(actionId, combo) {
  const b = _ensureLoaded();
  for (const c of Object.keys(b)) if (b[c] === actionId) delete b[c];
  delete b[combo];
  if (combo) b[combo] = actionId;
  _persist();
}

/** Remove all combos bound to an action. */
export function clearBinding(actionId) {
  const b = _ensureLoaded();
  for (const c of Object.keys(b)) if (b[c] === actionId) delete b[c];
  _persist();
}

/** Restore the built-in defaults. */
export function resetBindings() {
  _bindings = { ...DEFAULT_BINDINGS };
  _persist();
}
