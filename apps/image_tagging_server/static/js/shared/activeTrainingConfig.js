/**
 * Active Training Config — session-scoped singleton.
 *
 * Tracks which training config the user is working toward during a session,
 * driving label-list filtering and the coverage stats popup in the bottom bar.
 *
 * Usage:
 *   import { activeTrainingConfig } from '/app-static/js/shared/activeTrainingConfig.js';
 *   await activeTrainingConfig.init();
 *   activeTrainingConfig.getActive();          // TrainingConfigRead | null
 *   activeTrainingConfig.setActive(tc);        // publishes ACTIVE_TRAINING_CONFIG_CHANGED
 *   activeTrainingConfig.isFilterEnabled();
 *   activeTrainingConfig.setFilterEnabled(true);
 *   activeTrainingConfig.applyLabelFilter(labelsMap);
 *   activeTrainingConfig.getCachedStats();     // { labelStats, totalImages } | null
 *   await activeTrainingConfig.refreshStats(); // fetches fresh stats + re-publishes
 */

import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { fetchTrainingConfigs, fetchTrainingConfigLabelStats } from '/app-static/js/shared/api/trainingConfigs.js';
import { fetchModelConfigs } from '/app-static/js/shared/api/modelConfigs.js';

const SESSION_UUID_KEY = 'wtch_active_tc_uuid';
const SESSION_FILTER_KEY = 'wtch_tc_filter_enabled';

/** @type {object|null} TrainingConfigRead */
let _config = null;
let _filterEnabled = false;
/** @type {Set<string>} Label UUIDs covered by this TC's model node */
let _labelUuids = new Set();
/** @type {{ labelStats: object[], totalImages: number } | null} */
let _stats = null;
let _statsFetching = false;

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Derive label UUIDs from the TC's model config node.
 * Root node (labelUuid=null): returns all non-null label UUIDs in the model config.
 * Specific node: returns just that label UUID.
 */
async function _computeLabelUuids(tc) {
  if (!tc?.modelConfigNodeUuid) return new Set();
  try {
    const modelConfigsMap = await fetchModelConfigs();
    for (const [, mc] of modelConfigsMap) {
      const node = mc.nodes?.find(n => n.uuid === tc.modelConfigNodeUuid);
      if (!node) continue;
      if (node.labelUuid) return new Set([node.labelUuid]);
      // Root node — return all non-null label UUIDs in this model config
      const uuids = new Set();
      for (const n of mc.nodes ?? []) {
        if (n.labelUuid) uuids.add(n.labelUuid);
      }
      return uuids;
    }
  } catch { /* ignore */ }
  return new Set();
}

async function _doFetchStats() {
  if (!_config || _statsFetching) return;
  _statsFetching = true;
  try {
    _stats = await fetchTrainingConfigLabelStats(_config.uuid);
  } catch (err) {
    console.warn('[activeTrainingConfig] Failed to fetch stats:', err);
  } finally {
    _statsFetching = false;
  }
  _publish({ statsUpdated: true });
}

function _publish({ statsUpdated = false } = {}) {
  events.publish(EventTypes.ACTIVE_TRAINING_CONFIG_CHANGED, {
    config: _config,
    filterEnabled: _filterEnabled,
    statsUpdated,
  });
}

// ── Public API ────────────────────────────────────────────────────────────────

/** Initialise from sessionStorage. Call once at app startup before rendering. */
export async function init() {
  _filterEnabled = sessionStorage.getItem(SESSION_FILTER_KEY) === 'true';
  const savedUuid = sessionStorage.getItem(SESSION_UUID_KEY);
  if (savedUuid) {
    try {
      const configsMap = await fetchTrainingConfigs();
      _config = configsMap.get(savedUuid) ?? null;
      if (!_config) {
        sessionStorage.removeItem(SESSION_UUID_KEY);
      } else {
        _labelUuids = await _computeLabelUuids(_config);
        // Kick off stats fetch in the background after init
        _doFetchStats();
      }
    } catch {
      _config = null;
    }
  }
}

/** @returns {object|null} */
export function getActive() { return _config; }

/**
 * Set (or clear) the active training config. Pass null to deactivate.
 * Asynchronously updates label UUIDs and fetches stats, then publishes the event.
 * Returns a Promise that resolves after the event is published.
 *
 * @param {object|null} tcOrNull
 */
export async function setActive(tcOrNull) {
  _config = tcOrNull ?? null;
  _stats = null;
  if (_config) {
    sessionStorage.setItem(SESSION_UUID_KEY, _config.uuid);
    _labelUuids = await _computeLabelUuids(_config);
    _publish();
    _doFetchStats(); // fire-and-forget; will publish again when done
  } else {
    _labelUuids = new Set();
    sessionStorage.removeItem(SESSION_UUID_KEY);
    _publish();
  }
}

/** @returns {boolean} */
export function isFilterEnabled() { return _filterEnabled; }

/** @param {boolean} enabled */
export function setFilterEnabled(enabled) {
  _filterEnabled = !!enabled;
  sessionStorage.setItem(SESSION_FILTER_KEY, String(_filterEnabled));
  _publish();
}

/** Returns cached stats, or null if not yet fetched. */
export function getCachedStats() { return _stats; }

/**
 * Fetch fresh label stats and re-publish. Awaitable.
 * @returns {Promise<void>}
 */
export async function refreshStats() {
  if (!_config) return;
  _stats = null;
  await _doFetchStats();
}

/**
 * Returns the set of label UUIDs covered by the active TC's model node.
 * Synchronous — uses the cached value computed on setActive/init.
 * @returns {Set<string>}
 */
export function getLabelUuids() { return _labelUuids; }

/**
 * Filter a labelsMap to only labels covered by the active TC (plus ancestors).
 * Returns the original map when filtering is off or no TC is active.
 * @param {Map<string, object>} labelsMap
 * @returns {Map<string, object>}
 */
export function applyLabelFilter(labelsMap) {
  if (!_filterEnabled || !_config) return labelsMap;
  if (_labelUuids.size === 0) return labelsMap;

  // Expand with ancestors so the tree renders correctly
  const display = new Set(_labelUuids);
  for (const uuid of _labelUuids) {
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

// Namespace export
export const activeTrainingConfig = {
  init,
  getActive,
  setActive,
  isFilterEnabled,
  setFilterEnabled,
  getLabelUuids,
  applyLabelFilter,
  getCachedStats,
  refreshStats,
};
