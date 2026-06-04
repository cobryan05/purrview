// Customizable canvas hotkeys, shared across every canvas (main + modals).
// Bindings are stored in localStorage as a { [keyCombo]: actionId } map so a
// combo maps to exactly one action and lookups during keydown are O(1).
//
// actionId forms:
//   'action:<name>'  fixed actions (select/bbox/prevImage/nextImage/delete/undo/redo)
//   'label:<uuid>'   selects that label's draw tool ('bbox:<uuid>')
//   'tag:<uuid>'     selects that tag's apply tool ('tag:<uuid>')

const STORAGE_KEY = 'ww_hotkey_bindings';

/** Fixed (non-label/tag) actions, in display order for the config modal. */
export const FIXED_ACTIONS = [
  { id: 'action:select', label: 'Select tool' },
  { id: 'action:bbox', label: 'Draw bbox tool' },
  { id: 'action:prevImage', label: 'Previous image' },
  { id: 'action:nextImage', label: 'Next image' },
  { id: 'action:delete', label: 'Delete selected bbox' },
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
};

/** @type {Record<string,string>|null} */
let _bindings = null;

function _load() {
  if (_bindings) return _bindings;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    _bindings = raw ? JSON.parse(raw) : { ...DEFAULT_BINDINGS };
  } catch {
    _bindings = { ...DEFAULT_BINDINGS };
  }
  return _bindings;
}

function _persist() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(_bindings));
  } catch {
    /* storage unavailable — keep the in-memory copy */
  }
}

/** Build a normalized combo string from a keyboard event. */
export function eventToCombo(e) {
  const parts = [];
  if (e.ctrlKey || e.metaKey) parts.push('ctrl');
  if (e.altKey) parts.push('alt');
  if (e.shiftKey) parts.push('shift');
  parts.push((e.key || '').toLowerCase());
  return parts.join('+');
}

/** Human-friendly label for a combo, e.g. 'ctrl+shift+z' -> 'Ctrl+Shift+Z'. */
export function comboLabel(combo) {
  if (!combo) return '—';
  return combo
    .split('+')
    .map(p => (p.length === 1 ? p.toUpperCase() : p[0].toUpperCase() + p.slice(1)))
    .join('+');
}

/** The action bound to a combo, or null. */
export function actionForCombo(combo) {
  return _load()[combo] ?? null;
}

/** All combos bound to a given action (usually 0 or 1). */
export function combosForAction(actionId) {
  const b = _load();
  return Object.keys(b).filter(c => b[c] === actionId);
}

/**
 * Bind a combo to an action. Any prior combo for that action is removed (one
 * combo per action via the UI), and the combo is detached from any other action.
 */
export function setBinding(actionId, combo) {
  const b = _load();
  for (const c of Object.keys(b)) if (b[c] === actionId) delete b[c];
  delete b[combo];
  if (combo) b[combo] = actionId;
  _persist();
}

/** Remove all combos bound to an action. */
export function clearBinding(actionId) {
  const b = _load();
  for (const c of Object.keys(b)) if (b[c] === actionId) delete b[c];
  _persist();
}

/** Restore the built-in defaults. */
export function resetBindings() {
  _bindings = { ...DEFAULT_BINDINGS };
  _persist();
}
