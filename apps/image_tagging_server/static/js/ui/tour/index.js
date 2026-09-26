/**
 * Guided tours: the public surface.
 *
 *   initTours()   — call once after the app has booted; auto-starts the first tour
 *                   whose workspace check passes and that this workspace has not
 *                   already seen or declined.
 *   runTour(id)   — start a tour on demand (the ? button, Settings › Interface),
 *                   regardless of whether it was seen before or tours are off.
 *   toursForPage() — what the ? button offers on the page you are looking at.
 */
import { prefs } from '/app-static/js/shared/prefs.js';
import { Logger } from '/app-static/js/ui/utils/logging.js';
import { startTour, stopTour, isTourRunning } from './engine.js';
import { TOURS, tourById } from './registry.js';
import { readTourState, tourIsUnseen, tourResumeIndex } from './state.js';

export { isTourRunning, stopTour };
export { readTourState, resetTourState, setNeverShowTours } from './state.js';
export { TOURS };

/** Breathing room after boot before a spotlight lands on a freshly painted page. */
const AUTO_START_DELAY_MS = 600;

/**
 * Per-browser kill switch for auto-start: `localStorage.ww_tours_auto = 'off'`.
 * Local rather than a workspace setting on purpose — it is for a browser that must
 * never be interrupted (a wall display, an automated test) whatever the workspace
 * says. `runTour` ignores it.
 */
export const AUTO_START_PREF_KEY = 'ww_tours_auto';

/**
 * @param {string} id
 * @param {{startAt?: number}} [opts] - step index to open on; 0 restarts from the top
 * @returns {?ReturnType<typeof startTour>}
 */
export function runTour(id, { startAt = 0 } = {}) {
  const tour = tourById(id);
  if (!tour) {
    Logger.warn(`no such tour: ${id}`);
    return null;
  }
  return startTour(tour, { startAt });
}

/**
 * @typedef {Object} TourOffer
 * @property {import('./engine.js').Tour} tour
 * @property {number} startAt - the step a re-run opens on
 * @property {boolean} completed - already walked to the end at least once
 */

/**
 * What the help button should offer on a given main tab.
 *
 * A tour with no `pages` is offered everywhere — the getting-started walk crosses
 * every tab, so restricting it to one would hide it exactly where a lost user is.
 * `neverShow` is deliberately ignored: it suppresses tours that pop up on their own,
 * and this list only ever appears because someone asked for it.
 *
 * @param {string} tabName - a main tab id, e.g. 'tab-configuration'
 * @returns {Promise<TourOffer[]>}
 */
export async function toursForPage(tabName) {
  const state = await readTourState();
  return TOURS
    .filter(tour => !tour.pages || tour.pages.includes(tabName))
    .map(tour => ({
      tour,
      startAt: tourResumeIndex(state, tour.id),
      completed: state.completed.includes(tour.id),
    }));
}

/**
 * The first tour that should offer itself on this workspace, or null.
 * @returns {Promise<?import('./engine.js').Tour>}
 */
export async function pickAutoTour() {
  const state = await readTourState();
  if (state.neverShow) return null;
  for (const tour of TOURS) {
    if (!tourIsUnseen(state, tour.id)) continue;
    try {
      if (!tour.eligible || await tour.eligible()) return tour;
    } catch (err) {
      // An eligibility check that cannot answer is a "no": better a missed tour
      // than one that pops over a page that failed to load its data.
      Logger.warn(`tour "${tour.id}" eligibility check failed`, err);
    }
  }
  return null;
}

/** Offer a tour if the workspace wants one. Never throws. */
export async function initTours() {
  try {
    if (prefs.readRaw(AUTO_START_PREF_KEY, '') === 'off') return;
    const tour = await pickAutoTour();
    if (!tour) return;
    setTimeout(() => { if (!isTourRunning()) startTour(tour); }, AUTO_START_DELAY_MS);
  } catch (err) {
    Logger.warn('tours not started', err);
  }
}
