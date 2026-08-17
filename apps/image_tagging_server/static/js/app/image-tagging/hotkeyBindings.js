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
import { prefs } from '/app-static/js/shared/prefs.js';

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
  { id: 'action:resetZoom', label: 'Reset zoom' },
  { id: 'action:enhance', label: 'Cycle image enhancement' },
  { id: 'action:save', label: 'Save annotations' },
  { id: 'action:runInference', label: 'Run inference' },
  { id: 'action:markAbsent', label: 'Mark targeted labels absent' },
  { id: 'action:resetLabels', label: 'Reset targeted labels' },
  { id: 'action:verifyAll', label: 'Verify all bboxes' },
];

const DEFAULT_BINDINGS = {
  // Select and Draw-bbox sit on Q and B rather than the V and R they used to
  // hold: the label-panel bulk actions below wanted R and V, and those read far
  // better as Reset and Verify than as tool switches.
  'q': 'action:select',
  'b': 'action:bbox',
  '[': 'action:prevImage',
  ']': 'action:nextImage',
  'delete': 'action:delete',
  'backspace': 'action:delete',
  'ctrl+z': 'action:undo',
  'ctrl+shift+z': 'action:redo',
  'shift+delete': 'action:deleteImage',
  'shift+backspace': 'action:deleteImage',
  '0': 'action:resetZoom',
  'e': 'action:enhance',
  's': 'action:save',
  'i': 'action:runInference',
  'a': 'action:markAbsent',
  'r': 'action:resetLabels',
  'v': 'action:verifyAll',
};

/**
 * Add defaults for actions the stored map has never heard of.
 *
 * Bindings are persisted whole, so an action added to DEFAULT_BINDINGS after a
 * user already has a saved map would otherwise arrive unbound for them — the
 * new hotkey silently does nothing until they open the config panel. Only
 * actions with *no* combo at all get one, and only if their default combo is
 * still free, so a rebound or reassigned key is never disturbed.
 *
 * The one cost: an action the user deliberately cleared comes back the next
 * time its default combo is free. Rebinding it to something else — or to a
 * combo they never press — sticks, which is the practical way to opt out.
 *
 * @param {Record<string,string>} stored - mutated in place
 * @returns {boolean} true if anything was added
 */
function _mergeNewDefaults(stored) {
  const alreadyBound = new Set(Object.values(stored));
  let changed = false;
  for (const [combo, action] of Object.entries(DEFAULT_BINDINGS)) {
    if (alreadyBound.has(action)) continue;      // user has a key for it already
    if (stored[combo] !== undefined) continue;   // default combo is taken by something else
    stored[combo] = action;
    changed = true;
  }
  return changed;
}

/** @type {Record<string,string>|null} */
let _bindings = null;

/** Singleton init promise — prevents double-fetching; resolved = done. */
let _initPromise = null;

// ─── Sync localStorage helpers ────────────────────────────────────────────────

function _loadFromLocalStorage() {
  return prefs.read(STORAGE_KEY, { ...DEFAULT_BINDINGS });
}

function _saveToLocalStorage() {
  // A rejected write (quota/storage disabled) is fine — the in-memory copy and
  // the DB mirror both still hold the bindings.
  prefs.write(STORAGE_KEY, _bindings);
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
        const gained = _mergeNewDefaults(_bindings);
        // Mirror to localStorage so next startup is instant even if DB is slow.
        _saveToLocalStorage();
        if (gained) _persistToDb();
      } else {
        // DB has no entry yet: load from localStorage (migration) or defaults.
        _bindings = _loadFromLocalStorage();
        _mergeNewDefaults(_bindings);
        // Persist to DB so next load comes from there.
        _persistToDb();
      }
    } catch (err) {
      console.warn('Failed to load hotkeys from DB, using localStorage fallback:', err);
      _bindings = _loadFromLocalStorage();
      _mergeNewDefaults(_bindings);
    }
  })();

  return _initPromise;
}

// ─── Sync helpers (use in-memory cache; call initHotkeyBindings() first) ──────

function _ensureLoaded() {
  if (!_bindings) {
    // Fallback for callers that haven't awaited init (e.g. first keydown before init).
    _bindings = _loadFromLocalStorage();
    _mergeNewDefaults(_bindings);
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
