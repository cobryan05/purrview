/**
 * Guided tours: the public surface.
 *
 *   initTours()    — call once after the app has booted; auto-starts the overview
 *                    if this workspace looks new, and from then on resumes a paused
 *                    tour when the user walks back onto its page.
 *   runTour(id)    — start a tour on demand (the ? button, Settings › Interface),
 *                    regardless of whether it was seen before or tours are off.
 *   toursForPage() — what the ? button offers on the page you are looking at.
 *
 * Only a *paused* tour restarts itself. A tour nobody has seen sits behind the ?
 * with a dot on it instead: being walked through a page you opened for some other
 * reason is an ambush, and the difference between "I'll come back to this" and "I
 * have never seen this" is exactly the difference between welcome and unwelcome.
 */
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { prefs } from '/app-static/js/shared/prefs.js';
import { Logger } from '/app-static/js/ui/utils/logging.js';
import { isEngineNavigating, isTourRunning, startTour, stopTour } from './engine.js';
import { TOURS, tourById } from './registry.js';
import { readTourState, tourIsPaused, tourIsUnseen, tourResumeIndex } from './state.js';

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
 * @param {{startAt?: number, stack?: import('./state.js').TourReturn[]}} [opts]
 *   startAt: step index to open on; 0 restarts from the top
 *   stack: tours to return to when this one finishes
 * @returns {?ReturnType<typeof startTour>}
 */
export function runTour(id, { startAt = 0, stack = [] } = {}) {
  const tour = tourById(id);
  if (!tour) {
    Logger.warn(`no such tour: ${id}`);
    return null;
  }
  return startTour(tour, { startAt, stack });
}

/**
 * @typedef {Object} TourOffer
 * @property {import('./engine.js').Tour} tour
 * @property {number} startAt - the step a re-run opens on
 * @property {boolean} completed - already walked to the end at least once
 * @property {boolean} unseen - never started on this workspace
 * @property {import('./state.js').TourReturn[]} stack - what it would return to
 */

/**
 * What the help button should offer on a given main tab.
 *
 * A tour with no `pages` is offered everywhere — the overview crosses every tab, so
 * restricting it to one would hide it exactly where a lost user is. `neverShow` is
 * deliberately ignored: it suppresses tours that pop up on their own, and this list
 * only ever appears because someone asked for it.
 *
 * The most specific tour comes first: on Configuration › Watch Rules, "Watch rules"
 * is the answer to "what is this", Configure is the thing around it, and the overview
 * is the broadest of the three.
 *
 * @param {string} tabName - a main tab id, e.g. 'tab-configuration'
 * @param {string} [subTabName] - the active sub-tab, e.g. 'tab-watcher-management'
 * @returns {Promise<TourOffer[]>}
 */
export async function toursForPage(tabName, subTabName = '') {
  const state = await readTourState();
  return TOURS
    .filter(tour => !tour.pages || tour.pages.some(page => _pageMatches(page, tabName, subTabName)))
    .map(tour => ({
      tour,
      startAt: tourResumeIndex(state, tour.id),
      completed: state.completed.includes(tour.id),
      unseen: tourIsUnseen(state, tour.id),
      // A tour resumed from the ? keeps the branch it was in the middle of, so
      // finishing it still hands the overview back.
      stack: state.stack.filter(frame => frame.tourId !== tour.id),
    }))
    .sort((a, b) => _specificity(b.tour) - _specificity(a.tour));
}

/**
 * Does a `pages` entry cover this page?
 *
 * `tab-configuration` matches the whole tab; `tab-configuration:tab-watcher-management`
 * matches only that sub-page, which is how Configuration's eleven screens can have
 * tours of their own rather than sharing one.
 *
 * @param {string} page @param {string} tabName @param {string} subTabName
 */
function _pageMatches(page, tabName, subTabName) {
  const [main, sub] = page.split(':');
  if (main !== tabName) return false;
  return !sub || sub === subTabName;
}

/** Narrower tours sort first: sub-page, then whole tab, then everywhere. */
function _specificity(tour) {
  if (!tour.pages) return 0;
  return tour.pages.some(page => page.includes(':')) ? 2 : 1;
}

/**
 * Which page the user is looking at, as a (main tab, sub-tab) pair.
 *
 * Read from the DOM rather than tracked from events: the tiers change
 * independently, so a listener remembering only the last `tabName` could not tell
 * "Configuration, Watch Rules" from "Watch Rules" arriving on its own. The first
 * active workspace button in the pane, in DOM order — Configuration's rail items
 * carry that class, and on Annotate, whose outer tier is `.section-tab-button`, it
 * picks the Canvas/Files row, which is the sub-page there anyway.
 *
 * @returns {{main: string, sub: string}}
 */
export function currentTourPage() {
  const main = document.querySelector('.main-tab-button.active')?.dataset.tab ?? '';
  if (!main) return { main: '', sub: '' };
  const sub = document.getElementById(main)
    ?.querySelector('.workspace-tab-button.active')?.dataset.tab ?? '';
  return { main, sub };
}

/**
 * The first tour that should offer itself on this workspace, or null.
 * @returns {Promise<?import('./engine.js').Tour>}
 */
export async function pickAutoTour() {
  const state = await readTourState();
  if (state.neverShow) return null;
  for (const tour of TOURS) {
    if (!tour.autoStart) continue;
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

/**
 * Pick the walk back up where it was left, if the user has just arrived somewhere
 * a paused tour covers.
 *
 * @param {string} tabName
 */
async function _resumePausedTourFor(tabName, subTabName) {
  if (isTourRunning()) return;
  // A tour step navigates, which fires the same event this listens to. Without
  // this the tour would restart itself on every step that changes tab.
  if (isEngineNavigating()) return;
  if (prefs.readRaw(AUTO_START_PREF_KEY, '') === 'off') return;

  const state = await readTourState();
  // Narrowest first, so arriving on Configuration › Watch Rules resumes that
  // screen's tour rather than the broader Configure one paused behind it.
  const candidate = TOURS
    .filter(tour => tour.pages?.some(page => _pageMatches(page, tabName, subTabName)))
    .sort((a, b) => _specificity(b) - _specificity(a))
    .find(tour => tourIsPaused(state, tour.id));
  if (!candidate || isTourRunning()) return;

  startTour(candidate, {
    startAt: tourResumeIndex(state, candidate.id),
    stack: state.stack.filter(frame => frame.tourId !== candidate.id),
  });
}

/** Offer a tour if the workspace wants one. Never throws. */
export async function initTours() {
  try {
    if (prefs.readRaw(AUTO_START_PREF_KEY, '') === 'off') return;

    events.subscribe(EventTypes.TAB_CHANGED, () => {
      const { main, sub } = currentTourPage();
      if (main) void _resumePausedTourFor(main, sub).catch(() => {});
    });

    const tour = await pickAutoTour();
    if (!tour) return;
    setTimeout(() => { if (!isTourRunning()) startTour(tour); }, AUTO_START_DELAY_MS);
  } catch (err) {
    Logger.warn('tours not started', err);
  }
}
