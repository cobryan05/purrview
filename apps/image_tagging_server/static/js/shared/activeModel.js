/**
 * Active Model — session-scoped singleton.
 *
 * Each browser tab has its own active model (sessionStorage), so multiple
 * windows can work on different models independently.
 *
 * Usage:
 *   import { activeModel } from '/app-static/js/shared/activeModel.js';
 *   await activeModel.init();
 *   activeModel.getActiveModel();      // ModelConfigRead | null
 *   activeModel.setActiveModel(cfg);   // publishes ACTIVE_MODEL_CHANGED
 *   activeModel.isLabelFilterEnabled();
 *   activeModel.setLabelFilterEnabled(true);
 */

import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { fetchModelConfigs } from '/app-static/js/shared/api/modelConfigs.js';

const SESSION_UUID_KEY = 'wtch_active_model_uuid';
const SESSION_FILTER_KEY = 'wtch_model_filter_enabled';
const SESSION_DYNAMIC_LABELS_KEY = 'wtch_show_dynamic_labels';

/** @type {import('/app-static/js/shared/api/modelConfigs.js').ModelConfigRead | null} */
let _config = null;
let _filterEnabled = false;
let _showDynamicLabels = false;

/** Initialise from sessionStorage. Call once at app startup before other modules read state. */
export async function init() {
  _filterEnabled = sessionStorage.getItem(SESSION_FILTER_KEY) === 'true';
  _showDynamicLabels = sessionStorage.getItem(SESSION_DYNAMIC_LABELS_KEY) === 'true';

  const savedUuid = sessionStorage.getItem(SESSION_UUID_KEY);
  if (savedUuid) {
    try {
      const configsMap = await fetchModelConfigs();
      _config = configsMap.get(savedUuid) ?? null;
      if (!_config) {
        // Config was deleted since last session — clean up
        sessionStorage.removeItem(SESSION_UUID_KEY);
      }
    } catch {
      // Non-fatal: proceed with no active model
      _config = null;
    }
  }
}

/** @returns {import('/app-static/js/shared/api/modelConfigs.js').ModelConfigRead | null} */
export function getActiveModel() {
  return _config;
}

/**
 * Returns the set of label UUIDs covered by the active model (nodes where labelUuid is non-null).
 * Does NOT expand ancestors — callers that need the full display set should use
 * expandWithAncestors() from labelList.js.
 * @returns {Set<string>}
 */
export function getActiveModelLabelUuids() {
  if (!_config) return new Set();
  const uuids = new Set();
  for (const node of _config.nodes ?? []) {
    if (node.labelUuid) uuids.add(node.labelUuid);
  }
  return uuids;
}

/**
 * Set (or clear) the active model. Pass null to deactivate.
 * @param {import('/app-static/js/shared/api/modelConfigs.js').ModelConfigRead | null} configOrNull
 */
export function setActiveModel(configOrNull) {
  _config = configOrNull ?? null;
  if (_config) {
    sessionStorage.setItem(SESSION_UUID_KEY, _config.uuid);
  } else {
    sessionStorage.removeItem(SESSION_UUID_KEY);
  }
  _publish();
}

/** @returns {boolean} */
export function isLabelFilterEnabled() {
  return _filterEnabled;
}

/** @param {boolean} enabled */
export function setLabelFilterEnabled(enabled) {
  _filterEnabled = !!enabled;
  sessionStorage.setItem(SESSION_FILTER_KEY, String(_filterEnabled));
  _publish();
}

/** @returns {boolean} */
export function isDynamicLabelsVisible() {
  return _showDynamicLabels;
}

/** @param {boolean} visible */
export function setDynamicLabelsVisible(visible) {
  _showDynamicLabels = !!visible;
  sessionStorage.setItem(SESSION_DYNAMIC_LABELS_KEY, String(_showDynamicLabels));
  events.publish(EventTypes.DYNAMIC_LABELS_VISIBILITY_CHANGED, _showDynamicLabels);
}

function _publish() {
  events.publish(EventTypes.ACTIVE_MODEL_CHANGED, {
    config: _config,
    filterEnabled: _filterEnabled,
  });
}

/**
 * Convenience: returns the label UUIDs to display given a full labelsMap,
 * filtering to active model labels (expanded with ancestors) when filtering is on.
 * Always returns the original map when filtering is off or no model is active.
 *
 * @param {Map<string, object>} labelsMap
 * @returns {Map<string, object>}
 */
export function applyLabelFilter(labelsMap) {
  if (!_filterEnabled || !_config) return labelsMap;
  const modelUuids = getActiveModelLabelUuids();
  if (modelUuids.size === 0) return labelsMap;

  // Expand with ancestors so the tree renders correctly
  const display = new Set(modelUuids);
  for (const uuid of modelUuids) {
    let label = labelsMap.get(uuid);
    while (label?.parentUuid) {
      display.add(label.parentUuid);
      label = labelsMap.get(label.parentUuid);
    }
  }

  const filtered = new Map();
  for (const [uuid, label] of labelsMap) {
    if (display.has(uuid)) filtered.set(uuid, label);
  }
  return filtered;
}

// Namespace export so callers can do: import { activeModel } from '...'
export const activeModel = {
  init,
  getActiveModel,
  getActiveModelLabelUuids,
  setActiveModel,
  isLabelFilterEnabled,
  setLabelFilterEnabled,
  isDynamicLabelsVisible,
  setDynamicLabelsVisible,
  applyLabelFilter,
};
