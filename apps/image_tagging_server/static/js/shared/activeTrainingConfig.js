/**
 * Active Training Config — app-wide singleton.
 *
 * Tracks which training config the user is working toward, driving label-list
 * filtering and the coverage stats popup in the bottom bar.
 *
 * The active config and the *target* preset (plus the hand-picked selection
 * behind 'custom') persist across browser sessions: together they decide what
 * the bulk "Mark absent" / "Reset" buttons hit, and silently widening that back
 * to every label at the start of each session makes the first bulk op of the
 * day mark more than the user lined up. Label *visibility* stays per-tab —
 * it hides rows, it doesn't change what an action touches.
 *
 * Usage:
 *   import { activeTrainingConfig } from '/app-static/js/shared/activeTrainingConfig.js';
 *   await activeTrainingConfig.init();
 *   activeTrainingConfig.getActive();          // TrainingConfigRead | null
 *   activeTrainingConfig.setActive(tc);        // publishes ACTIVE_TRAINING_CONFIG_CHANGED
 *   activeTrainingConfig.getVisibilityPreset();    // 'all' | 'training-run'
 *   activeTrainingConfig.setVisibilityPreset('training-run');
 *   activeTrainingConfig.getTargetPreset();        // 'all' | 'training-run' | 'object' | 'image' | 'dynamic' | 'none'
 *   activeTrainingConfig.setTargetPreset('object');
 *   activeTrainingConfig.applyLabelFilter(labelsMap);
 *   activeTrainingConfig.getExcludedUuids(labelsMap); // labels outside the active TC
 *   activeTrainingConfig.getCachedStats();      // { labelStats, totalImages } | null
 *   activeTrainingConfig.getCachedReadiness();  // { readyCount, needsReviewCount, notStartedCount, rollUpActive } | null
 *   await activeTrainingConfig.refreshStats();  // fetches fresh stats + readiness, re-publishes
 */

import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { prefs, sessionPrefs } from '/app-static/js/shared/prefs.js';
import {
  fetchTrainingConfigs,
  fetchTrainingConfigLabelStats,
  fetchTrainingConfigReadiness,
} from '/app-static/js/shared/api/trainingConfigs.js';
import { fetchModelConfigs } from '/app-static/js/shared/api/modelConfigs.js';

const SESSION_FILTER_KEY = 'wtch_tc_filter_enabled';   // legacy 3-mode value — read once for migration
const SESSION_VIS_KEY = 'wtch_tc_visibility_preset';
// The active run and targeting live in localStorage, not sessionStorage: what
// "Apply to all" covers is a destructive-op blast radius, and silently
// reverting it to "every label" at the start of each session means the first
// bulk op of the day marks more than the user intends. The 'training-run'
// target preset in particular is only meaningful while its run is active, so
// the two have to persist together. Same key names as the old session-scoped
// ones, which are read once as a migration path.
const TC_UUID_KEY = 'wtch_active_tc_uuid';
const TARGET_KEY = 'wtch_tc_target_preset';
const CUSTOM_KEY = 'wtch_tc_custom_targeted';          // JSON array of hand-picked targeted UUIDs (custom preset)

/**
 * The two independent axes that used to be conflated in a single 3-way toggle:
 *   - Visibility: does a label appear in the list at all?
 *   - Target:     does "Apply to all" (mark absent / reset) affect this label?
 * The old modes map to combinations: all=(all,all), focus=(all,training-run),
 * filtered=(training-run,training-run).
 */
export const VISIBILITY_PRESETS = ['all', 'training-run'];
export const TARGET_PRESETS = ['all', 'training-run', 'object', 'image', 'dynamic', 'none', 'custom'];

/** @type {object|null} TrainingConfigRead */
let _config = null;
/** @type {'all'|'training-run'} */
let _visibilityPreset = 'all';
/** @type {'all'|'training-run'|'object'|'image'|'dynamic'|'none'|'custom'} */
let _targetPreset = 'all';
/**
 * Hand-picked targeted UUIDs backing the 'custom' preset. Only meaningful when
 * _targetPreset === 'custom'. Persisted across remounts (e.g. the remount a
 * save triggers) so a manual selection survives the whole session.
 * @type {Set<string>}
 */
let _customTargeted = new Set();
/** @type {Set<string>} Label UUIDs covered by this TC's model node */
let _labelUuids = new Set();
/** @type {{ labelStats: object[], totalImages: number } | null} */
let _stats = null;
/** @type {{ readyCount: number, needsReviewCount: number, notStartedCount: number, rollUpActive: boolean } | null} */
let _readiness = null;
/** True if the most recent readiness fetch attempt failed (distinct from "not yet fetched"). */
let _readinessFailed = false;
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
  // Fetched independently: a readiness failure shouldn't blank out label
  // stats that loaded fine, or vice versa.
  const [statsResult, readinessResult] = await Promise.allSettled([
    fetchTrainingConfigLabelStats(_config.uuid),
    fetchTrainingConfigReadiness(_config.uuid),
  ]);
  if (statsResult.status === 'fulfilled') _stats = statsResult.value;
  else console.warn('[activeTrainingConfig] Failed to fetch label stats:', statsResult.reason);
  if (readinessResult.status === 'fulfilled') {
    _readiness = readinessResult.value;
    _readinessFailed = false;
  } else {
    console.warn('[activeTrainingConfig] Failed to fetch readiness:', readinessResult.reason);
    _readinessFailed = true;
  }
  _statsFetching = false;
  _publish({ statsUpdated: true });
}

function _publish({ statsUpdated = false } = {}) {
  events.publish(EventTypes.ACTIVE_TRAINING_CONFIG_CHANGED, {
    config: _config,
    visibilityPreset: _visibilityPreset,
    targetPreset: _targetPreset,
    statsUpdated,
  });
}

// ── Public API ────────────────────────────────────────────────────────────────

/** Initialise from sessionStorage. Call once at app startup before rendering. */
export async function init() {
  // Presets are stored as bare strings ('all', 'training-run', …), not JSON.
  // The two axes are read independently — visibility is per-tab, targeting is
  // persistent — so a missing one never discards the other.
  // Legacy 3-mode toggle (all | focus | filtered, plus the even-older
  // true/false boolean), decomposed into the two axes when neither is set.
  const legacy = sessionPrefs.readRaw(SESSION_FILTER_KEY);
  const legacyMode = legacy === 'true' ? 'filtered' : legacy === 'false' ? 'all' : legacy;

  const savedVis = sessionPrefs.readRaw(SESSION_VIS_KEY);
  if (VISIBILITY_PRESETS.includes(savedVis)) _visibilityPreset = savedVis;
  else _visibilityPreset = legacyMode === 'filtered' ? 'training-run' : 'all';

  let savedTarget = prefs.readRaw(TARGET_KEY);
  let savedCustom = prefs.read(CUSTOM_KEY, null);
  if (!TARGET_PRESETS.includes(savedTarget)) {
    // One-time migration from when targeting was session-scoped.
    const priorTarget = sessionPrefs.readRaw(TARGET_KEY);
    if (TARGET_PRESETS.includes(priorTarget)) {
      savedTarget = priorTarget;
      savedCustom = sessionPrefs.read(CUSTOM_KEY, null);
      prefs.writeRaw(TARGET_KEY, savedTarget);
      if (Array.isArray(savedCustom)) prefs.write(CUSTOM_KEY, savedCustom);
    }
  }
  if (TARGET_PRESETS.includes(savedTarget)) {
    _targetPreset = savedTarget;
    if (_targetPreset === 'custom') {
      _customTargeted = Array.isArray(savedCustom) ? new Set(savedCustom) : new Set();
    }
  } else {
    _targetPreset = (legacyMode === 'filtered' || legacyMode === 'focus') ? 'training-run' : 'all';
  }
  // Same one-time migration as the target preset: an older build parked the
  // active run in sessionStorage.
  let savedUuid = prefs.readRaw(TC_UUID_KEY);
  if (!savedUuid) {
    savedUuid = sessionPrefs.readRaw(TC_UUID_KEY);
    if (savedUuid) prefs.writeRaw(TC_UUID_KEY, savedUuid);
  }
  if (savedUuid) {
    try {
      const configsMap = await fetchTrainingConfigs();
      _config = configsMap.get(savedUuid) ?? null;
      if (!_config) {
        // The run was deleted since it was picked — drop the stale pointer so
        // it doesn't get retried on every boot from here on.
        prefs.remove(TC_UUID_KEY);
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
  _readiness = null;
  _readinessFailed = false;
  if (_config) {
    prefs.writeRaw(TC_UUID_KEY, _config.uuid);
    _labelUuids = await _computeLabelUuids(_config);
    _publish();
    _doFetchStats(); // fire-and-forget; will publish again when done
  } else {
    _labelUuids = new Set();
    prefs.remove(TC_UUID_KEY);
    // A run explicitly deactivated must not come back on the next boot via the
    // sessionStorage migration path.
    sessionPrefs.remove(TC_UUID_KEY);
    _publish();
  }
}

/** @returns {'all'|'training-run'} */
export function getVisibilityPreset() { return _visibilityPreset; }

/** @param {'all'|'training-run'} preset */
export function setVisibilityPreset(preset) {
  if (!VISIBILITY_PRESETS.includes(preset)) return;
  _visibilityPreset = preset;
  sessionPrefs.writeRaw(SESSION_VIS_KEY, preset);
  _publish();
}

/** @returns {'all'|'training-run'|'object'|'image'|'dynamic'|'none'} */
export function getTargetPreset() { return _targetPreset; }

/** @param {'all'|'training-run'|'object'|'image'|'dynamic'|'none'|'custom'} preset */
export function setTargetPreset(preset) {
  if (!TARGET_PRESETS.includes(preset)) return;
  _targetPreset = preset;
  prefs.writeRaw(TARGET_KEY, preset);
  // Picking any named preset discards a prior hand-picked selection.
  if (preset !== 'custom') {
    _customTargeted = new Set();
    prefs.remove(CUSTOM_KEY);
  }
  _publish();
}

/** @returns {Set<string>} copy of the hand-picked targeted UUIDs (custom preset) */
export function getCustomTargeted() { return new Set(_customTargeted); }

/**
 * Record a hand-picked targeting selection and switch to the 'custom' preset.
 * Persists to localStorage so it survives both remounts and browser sessions,
 * but intentionally does NOT publish ACTIVE_TRAINING_CONFIG_CHANGED — toggling
 * a single row must not trigger a label-list remount (which would wipe
 * in-progress interaction).
 * @param {Iterable<string>} uuids  UUIDs currently targeted
 */
export function applyCustomTargeting(uuids) {
  _customTargeted = new Set(uuids);
  _targetPreset = 'custom';
  prefs.writeRaw(TARGET_KEY, 'custom');
  prefs.write(CUSTOM_KEY, [..._customTargeted]);
}

/** Returns cached label stats, or null if not yet fetched. */
export function getCachedStats() { return _stats; }

/** Returns cached readiness (ready/needs-review/not-started counts), or null if not yet fetched. */
export function getCachedReadiness() { return _readiness; }

/** True if the most recent readiness fetch failed — distinct from "not yet fetched". */
export function getReadinessFailed() { return _readinessFailed; }

/**
 * Fetch fresh label stats + readiness and re-publish. Awaitable.
 * @returns {Promise<void>}
 */
export async function refreshStats() {
  if (!_config) return;
  _stats = null;
  _readiness = null;
  _readinessFailed = false;
  await _doFetchStats();
}

/**
 * Returns the set of label UUIDs covered by the active TC's model node.
 * Synchronous — uses the cached value computed on setActive/init.
 * @returns {Set<string>}
 */
export function getLabelUuids() { return _labelUuids; }

/**
 * Labels covered by the active TC (plus ancestors, so the tree renders correctly).
 * Returns null when there's nothing to compute against (no TC, or empty coverage).
 * @param {Map<string, object>} labelsMap
 * @returns {Set<string>|null}
 */
function _computeDisplaySet(labelsMap) {
  if (!_config || _labelUuids.size === 0) return null;
  const display = new Set(_labelUuids);
  for (const uuid of _labelUuids) {
    let label = labelsMap.get(uuid);
    while (label?.parentUuid) {
      display.add(label.parentUuid);
      label = labelsMap.get(label.parentUuid);
    }
  }
  return display;
}

/**
 * Filter a labelsMap to only labels covered by the active TC (plus ancestors).
 * Returns the original map unchanged unless the visibility preset is 'training-run'.
 * @param {Map<string, object>} labelsMap
 * @returns {Map<string, object>}
 */
export function applyLabelFilter(labelsMap) {
  if (_visibilityPreset !== 'training-run') return labelsMap;
  const display = _computeDisplaySet(labelsMap);
  if (!display) return labelsMap;

  const filtered = new Map();
  for (const [uuid, label] of labelsMap) {
    if (display.has(uuid)) filtered.set(uuid, label);
  }
  return filtered;
}

/**
 * Labels NOT covered by the active TC — used by the 'training-run' target
 * preset to leave these visible but excluded from "Apply to all" bulk ops.
 * Empty when no TC is active or it covers everything.
 * @param {Map<string, object>} labelsMap
 * @returns {Set<string>}
 */
export function getExcludedUuids(labelsMap) {
  const display = _computeDisplaySet(labelsMap);
  if (!display) return new Set();
  const excluded = new Set();
  for (const uuid of labelsMap.keys()) {
    if (!display.has(uuid)) excluded.add(uuid);
  }
  return excluded;
}

// Namespace export
export const activeTrainingConfig = {
  init,
  getActive,
  setActive,
  getVisibilityPreset,
  setVisibilityPreset,
  getTargetPreset,
  setTargetPreset,
  getCustomTargeted,
  applyCustomTargeting,
  getLabelUuids,
  applyLabelFilter,
  getExcludedUuids,
  getCachedStats,
  getCachedReadiness,
  getReadinessFailed,
  refreshStats,
};
