/**
 * Which tours this workspace has seen, and whether it wants any more.
 *
 * Stored as one server setting rather than in localStorage: "I've done the getting-
 * started tour" is a fact about the *workspace*, and a second browser or a phone
 * opening the same install should not be walked through it again. The prefs cache
 * only covers the moment before the setting has been read (or a db_server that is
 * not answering), so a slow boot never flashes a tour the user already declined.
 */
import { getSetting, setSetting } from '/app-static/js/shared/api/settings.js';
import { prefs } from '/app-static/js/shared/prefs.js';
import { Logger } from '/app-static/js/ui/utils/logging.js';

export const TOUR_STATE_SETTING_KEY = 'tour_state';
const CACHE_KEY = 'ww_tour_state';

/**
 * @typedef {{tourId: string, index: number}} TourReturn
 *   a tour to come back to, and the step it branched away from
 */

/**
 * `paused` and `dismissed` are both "stopped part-way", and the difference is the
 * whole point: pausing says "I'll come back", so landing on that tour's page again
 * picks it up, while dismissing says "leave me alone" and only the ? button
 * reopens it. A tour is in at most one of them.
 *
 * @typedef {Object} TourState
 * @property {string[]} completed - tours walked to the end
 * @property {string[]} dismissed - stopped part-way and not to be offered again
 * @property {string[]} paused - stopped part-way and resumable on returning to its page
 * @property {boolean} neverShow - no tour is ever auto-started
 * @property {Record<string, number>} progress - tourId → step index it stopped on
 * @property {TourReturn[]} stack - tours waiting for a branch to finish, outermost first
 */

/** @returns {TourState} */
function _empty() {
  return { completed: [], dismissed: [], paused: [], neverShow: false, progress: {}, stack: [] };
}

/** @param {unknown} raw @returns {TourState} */
function _normalise(raw) {
  const state = _empty();
  if (!raw || typeof raw !== 'object') return state;
  if (Array.isArray(raw.completed)) state.completed = raw.completed.filter(id => typeof id === 'string');
  if (Array.isArray(raw.dismissed)) state.dismissed = raw.dismissed.filter(id => typeof id === 'string');
  if (Array.isArray(raw.paused)) state.paused = raw.paused.filter(id => typeof id === 'string');
  state.neverShow = raw.neverShow === true;
  if (raw.progress && typeof raw.progress === 'object') {
    for (const [id, i] of Object.entries(raw.progress)) {
      if (Number.isInteger(i) && i >= 0) state.progress[id] = i;
    }
  }
  if (Array.isArray(raw.stack)) {
    state.stack = raw.stack
      .filter(e => e && typeof e.tourId === 'string' && Number.isInteger(e.index) && e.index >= 0)
      .map(e => ({ tourId: e.tourId, index: e.index }));
  }
  return state;
}

/** @returns {Promise<TourState>} the server's answer, or the cache when it has none */
export async function readTourState() {
  try {
    const value = await getSetting(TOUR_STATE_SETTING_KEY);
    if (value != null) {
      const state = _normalise(JSON.parse(value));
      prefs.write(CACHE_KEY, state);
      return state;
    }
  } catch (err) {
    Logger.warn('tour state unreadable, using cache', err);
  }
  return _normalise(prefs.read(CACHE_KEY, null));
}

/** @param {TourState} state */
async function _write(state) {
  prefs.write(CACHE_KEY, state);
  try {
    await setSetting(TOUR_STATE_SETTING_KEY, JSON.stringify(state));
  } catch (err) {
    // The cache already holds it, so this browser will not re-offer the tour; a
    // second browser might, which is the lesser harm.
    Logger.warn('tour state not saved', err);
  }
}

/**
 * Apply a change on top of the current state and persist the result.
 * @param {(state: TourState) => void} mutate
 * @returns {Promise<TourState>}
 */
async function _update(mutate) {
  const state = await readTourState();
  mutate(state);
  await _write(state);
  return state;
}

/**
 * Every write is read-modify-write against one server setting, so two of them in
 * flight at once lose one another's changes. Finishing a branch changes both the
 * completed list and the stack, which is why the stack rides along here rather
 * than going out as a second call.
 *
 * @param {?TourReturn[]} stack @param {TourState} state
 */
function _applyStack(state, stack) {
  if (stack) state.stack = stack.map(e => ({ tourId: e.tourId, index: e.index }));
}

/** @param {string} tourId @param {TourReturn[]} [stack] */
export function markTourCompleted(tourId, stack = null) {
  return _update(state => {
    if (!state.completed.includes(tourId)) state.completed.push(tourId);
    state.dismissed = state.dismissed.filter(id => id !== tourId);
    state.paused = state.paused.filter(id => id !== tourId);
    delete state.progress[tourId];
    _applyStack(state, stack);
  });
}

/**
 * @param {string} tourId
 * @param {number} [stepIndex] - the step the user was on; the help button resumes there
 * @param {TourReturn[]} [stack]
 */
export function markTourDismissed(tourId, stepIndex, stack = null) {
  return _update(state => {
    if (!state.dismissed.includes(tourId)) state.dismissed.push(tourId);
    state.paused = state.paused.filter(id => id !== tourId);
    if (Number.isInteger(stepIndex) && stepIndex > 0) state.progress[tourId] = stepIndex;
    else delete state.progress[tourId];
    _applyStack(state, stack);
  });
}

/**
 * Stopped, but meaning to come back: the tour resumes itself the next time the
 * user navigates to a page it covers.
 *
 * @param {string} tourId
 * @param {number} [stepIndex]
 */
export function markTourPaused(tourId, stepIndex) {
  return _update(state => {
    if (!state.paused.includes(tourId)) state.paused.push(tourId);
    state.dismissed = state.dismissed.filter(id => id !== tourId);
    if (Number.isInteger(stepIndex) && stepIndex > 0) state.progress[tourId] = stepIndex;
    else delete state.progress[tourId];
  });
}

/**
 * Persist where a branch should return to. Written on every push and pop so a
 * reload mid-branch still knows the overview is waiting behind it.
 * @param {TourReturn[]} stack
 */
export function writeTourStack(stack) {
  return _update(state => {
    state.stack = stack.map(e => ({ tourId: e.tourId, index: e.index }));
  });
}

/** @param {boolean} neverShow */
export function setNeverShowTours(neverShow) {
  return _update(state => { state.neverShow = !!neverShow; });
}

/** Forget every tour, so the next eligible one auto-starts again. */
export function resetTourState() {
  return _update(state => {
    state.completed = [];
    state.dismissed = [];
    state.paused = [];
    state.neverShow = false;
    state.progress = {};
    state.stack = [];
  });
}

/**
 * Whether this tour was paused rather than finished or dismissed — the one state
 * that lets it start itself again without the user asking.
 * @param {TourState} state
 * @param {string} tourId
 */
export function tourIsPaused(state, tourId) {
  return !state.neverShow && state.paused.includes(tourId);
}

/**
 * Whether a tour may be auto-started against this state.
 * @param {TourState} state
 * @param {string} tourId
 */
export function tourIsUnseen(state, tourId) {
  return !state.neverShow
    && !state.completed.includes(tourId)
    && !state.dismissed.includes(tourId)
    // A paused tour has been seen; it comes back through its resume point rather
    // than by auto-starting from step one.
    && !state.paused.includes(tourId);
}

/**
 * The step a re-run should open on: where the user walked away from, or the start.
 * A completed tour always restarts from the beginning — its progress is cleared on
 * completion, so re-running it is a re-watch, not a resume.
 *
 * @param {TourState} state
 * @param {string} tourId
 * @returns {number}
 */
export function tourResumeIndex(state, tourId) {
  const i = state.progress?.[tourId];
  return Number.isInteger(i) && i > 0 ? i : 0;
}
