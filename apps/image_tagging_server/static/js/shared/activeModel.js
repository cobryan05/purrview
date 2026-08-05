/**
 * Tag/label visibility preferences — session-scoped singleton.
 *
 * Holds per-kind tag visibility and the dynamic-labels toggle. (The legacy
 * "active model" filter that used to live here was replaced by
 * shared/activeTrainingConfig.js.)
 *
 * Usage:
 *   import { activeModel } from '/app-static/js/shared/activeModel.js';
 *   activeModel.init();
 *   activeModel.isTagKindVisible('system');
 *   activeModel.setTagKindVisible('system', true);
 */

import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { sessionPrefs } from '/app-static/js/shared/prefs.js';

const SESSION_DYNAMIC_LABELS_KEY = 'wtch_show_dynamic_labels';
const SESSION_TAG_VISIBILITY_KEY = 'wtch_tag_kind_visible';

let _showDynamicLabels = false;
// Per-kind tag visibility: false = hidden from pickers/filters.
// system tags are hidden by default; task_generated tags are visible.
let _tagKindVisible = { system: false, task_generated: true };

/** Initialise from sessionStorage. Call once at app startup before other modules read state. */
export function init() {
  _showDynamicLabels = sessionPrefs.read(SESSION_DYNAMIC_LABELS_KEY, false) === true;
  const savedVis = sessionPrefs.read(SESSION_TAG_VISIBILITY_KEY, null);
  if (savedVis && typeof savedVis === 'object') _tagKindVisible = { ..._tagKindVisible, ...savedVis };
}

/** @returns {boolean} */
export function isDynamicLabelsVisible() {
  return _showDynamicLabels;
}

/** @param {boolean} visible */
export function setDynamicLabelsVisible(visible) {
  _showDynamicLabels = !!visible;
  // JSON booleans serialise to the same "true"/"false" the old String() write
  // produced, so previously-stored values still read back correctly.
  sessionPrefs.write(SESSION_DYNAMIC_LABELS_KEY, _showDynamicLabels);
  events.publish(EventTypes.DYNAMIC_LABELS_VISIBILITY_CHANGED, _showDynamicLabels);
}

/**
 * Returns whether tags of the given kind should appear in UI pickers/filters.
 * Unknown kinds default to visible (true).
 * @param {string} kind  e.g. 'system' | 'task_generated' | 'generic'
 * @returns {boolean}
 */
export function isTagKindVisible(kind) {
  return _tagKindVisible[kind] !== false;
}

/**
 * Returns whether a tag should appear in UI pickers/filters, applying both the
 * global tag-kind visibility toggle and the icon-less-system-tag exclusion.
 * @param {{kind?: string, icon?: string}} tag
 * @returns {boolean}
 */
export function isTagVisible(tag) {
  if (tag.kind === 'system') return !!tag.icon && isTagKindVisible('system');
  if (tag.kind === 'task_generated') return isTagKindVisible('task_generated');
  return true;
}

/**
 * Filter an iterable of tags (e.g. a fetchTags() Map's .values()) down to those
 * that should appear in pickers/filters, per isTagVisible(). Use this instead of
 * `[...tags.values()].filter(isTagVisible)` at every picker call site.
 * @param {Iterable<{kind?: string, icon?: string}>} tags
 * @returns {object[]}
 */
export function filterVisibleTags(tags) {
  return [...tags].filter(isTagVisible);
}

/** Display label for each tag kind, used wherever tags are grouped by kind in the UI. */
export const TAG_KIND_LABELS = { system: 'System', task_generated: 'Task-generated', generic: 'Generic' };

/**
 * Group an iterable of tags by kind, sorting each group by name.
 * @param {Iterable<{kind?: string, name: string}>} tags
 * @param {string[]} order  Kinds in the order groups should appear (unlisted kinds are appended).
 * @returns {Array<{kind: string, label: string, tags: object[]}>}  non-empty groups only.
 */
export function groupTagsByKind(tags, order = ['system', 'task_generated', 'generic']) {
  const byKind = new Map(order.map(k => [k, []]));
  for (const tag of tags) {
    const k = tag.kind ?? 'generic';
    if (!byKind.has(k)) byKind.set(k, []);
    byKind.get(k).push(tag);
  }
  for (const group of byKind.values()) group.sort((a, b) => a.name.localeCompare(b.name));
  return [...byKind.entries()]
    .filter(([, list]) => list.length > 0)
    .map(([kind, list]) => ({ kind, label: TAG_KIND_LABELS[kind] ?? kind, tags: list }));
}

/**
 * Set visibility for a tag kind and notify subscribers.
 * @param {string} kind
 * @param {boolean} visible
 */
export function setTagKindVisible(kind, visible) {
  _tagKindVisible = { ..._tagKindVisible, [kind]: !!visible };
  sessionPrefs.write(SESSION_TAG_VISIBILITY_KEY, _tagKindVisible);
  events.publish(EventTypes.TAG_KIND_VISIBILITY_CHANGED, { kind, visible: !!visible });
}

// Namespace export so callers can do: import { activeModel } from '...'
export const activeModel = {
  init,
  isDynamicLabelsVisible,
  setDynamicLabelsVisible,
  isTagKindVisible,
  setTagKindVisible,
};
