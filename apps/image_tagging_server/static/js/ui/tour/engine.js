/**
 * Runs a tour: one step at a time, each pointed at a `data-tour` anchor.
 *
 * A step names its target by anchor id, never by selector, and says which tab it
 * lives on. The engine does the getting-there — `navigateTabs` for the lazy-init
 * dance, then a wait for the anchor to actually be in the DOM — so a step written
 * as "point at the + New Source button" works from any tab the user happens to be
 * on, including one whose pane has never been mounted.
 *
 * Panes rebuild their subtrees on refresh, which tears out whatever node the
 * spotlight was sized to. The engine watches for that and re-resolves the anchor
 * by id: the highlighted control is the *id*, not the element that carried it a
 * moment ago.
 *
 * A step may `advanceOn` an event — "create a label" moves on when a label is
 * created — which is what makes this a walkthrough rather than a slideshow. Next
 * still works on such a step, relabelled "Skip this step", because a user who
 * already understands does not need to be made to do it.
 *
 * A step whose subject may simply not exist — the per-row controls on a label the
 * user declined to create — sets `requireTarget`, and the engine passes over it
 * rather than talking about a control that is not on screen.
 */
import { navigateTabs } from '/app-static/js/app/utils/tabs.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { Logger } from '/app-static/js/ui/utils/logging.js';
import { createCoachMark } from './coachMark.js';
import { markTourCompleted, markTourDismissed, readTourState, setNeverShowTours } from './state.js';

/** How long a step waits for its anchor after navigating before going on without one. */
const TARGET_TIMEOUT_MS = 4000;
/**
 * The same wait for a `requireTarget` step. Shorter because absence is an expected
 * answer there, not a failure — but not instant: the step still has to outlast its
 * pane's async mount when a resume lands straight on it.
 */
const OPTIONAL_TARGET_TIMEOUT_MS = 1500;
const TARGET_POLL_MS = 50;

/**
 * @typedef {Object} TourStep
 * @property {string} id - unique within the tour
 * @property {?string} target - a `data-tour` anchor id; null for a step with no spotlight
 * @property {[string, string?]} [navigate] - main tab id and optional workspace tab id to land on first
 * @property {string} title
 * @property {string} body
 * @property {{event: string}} [advanceOn] - an EventTypes value that completes the step
 * @property {boolean} [requireTarget] - skip the step entirely when its anchor is absent
 */

/**
 * @typedef {Object} Tour
 * @property {string} id
 * @property {string} title
 * @property {TourStep[]} steps
 * @property {() => Promise<boolean>} [eligible] - whether to auto-start on this workspace
 * @property {string[]} [pages] - main tab ids the help button offers this tour on; all of them when absent
 */

/** @param {string} id */
export function tourAnchor(id) {
  return document.querySelector(`[data-tour="${CSS.escape(id)}"]`);
}

function _visible(node) {
  return !!node && node.isConnected && node.getClientRects().length > 0;
}

/**
 * Resolve an anchor, waiting for it to appear and be laid out.
 * @param {?string} id
 * @param {number} [timeoutMs]
 * @param {boolean} [warnOnMiss] - false for a `requireTarget` step, where absence is an answer
 * @returns {Promise<?Element>}
 */
function _waitForAnchor(id, timeoutMs = TARGET_TIMEOUT_MS, warnOnMiss = true) {
  if (!id) return Promise.resolve(null);
  const found = tourAnchor(id);
  if (_visible(found)) return Promise.resolve(found);
  return new Promise(resolve => {
    const deadline = Date.now() + timeoutMs;
    const poll = () => {
      const node = tourAnchor(id);
      if (_visible(node)) return resolve(node);
      if (Date.now() > deadline) {
        if (warnOnMiss) Logger.warn(`tour anchor "${id}" not found; continuing without a spotlight`);
        return resolve(null);
      }
      setTimeout(poll, TARGET_POLL_MS);
    };
    poll();
  });
}

let _active = null;

/** Whether a tour is on screen. */
export function isTourRunning() {
  return _active != null;
}

/** Stop whatever tour is running, recording nothing about it. */
export function stopTour() {
  _active?.stop();
}

/**
 * Start a tour. Stops any tour already running.
 *
 * @param {Tour} tour
 * @param {Object} [opts]
 * @param {number} [opts.startAt] - step index to open on
 * @returns {{stop: () => void, next: () => void, back: () => void, currentIndex: () => number}}
 */
export function startTour(tour, { startAt = 0 } = {}) {
  _active?.stop();

  const coach = createCoachMark();
  let index = -1;
  let stopped = false;
  let seq = 0;
  /** @type {?() => void} */
  let unsubscribeAdvance = null;
  /** @type {?Element} */
  let target = null;
  let neverShow = false;
  /** @type {?MutationObserver} */
  let rebuildWatch = null;

  readTourState().then(state => { neverShow = state.neverShow; }).catch(() => {});

  function _teardownStep() {
    unsubscribeAdvance?.();
    unsubscribeAdvance = null;
    rebuildWatch?.disconnect();
    rebuildWatch = null;
  }

  function stop() {
    if (stopped) return;
    stopped = true;
    _teardownStep();
    coach.hide();
    if (_active === handle) _active = null;
    events.publish(EventTypes.TOUR_STATE_CHANGED, { running: false, tourId: tour.id });
  }

  function _render(step) {
    coach.show({
      target,
      title: step.title,
      body: step.body,
      index,
      count: tour.steps.length,
      onNext: next,
      onBack: index > 0 ? back : null,
      onSkip: skip,
      onNeverShow: (value) => {
        neverShow = value;
        void setNeverShowTours(value);
      },
      neverShow,
      waiting: !!step.advanceOn,
    });
  }

  /**
   * @param {number} i
   * @param {1|-1} [dir] - which way to keep going if step `i` turns out to be absent
   */
  async function _show(i, dir = 1) {
    const step = tour.steps[i];
    if (!step) return;
    const mySeq = ++seq;
    _teardownStep();
    index = i;

    if (step.navigate) {
      const [mainTab, workspaceTab] = step.navigate;
      if (workspaceTab) navigateTabs(mainTab, workspaceTab);
      else document.querySelector(`.main-tab-button[data-tab="${mainTab}"]`)?.click();
    }

    target = await _waitForAnchor(
      step.target,
      step.requireTarget ? OPTIONAL_TARGET_TIMEOUT_MS : TARGET_TIMEOUT_MS,
      !step.requireTarget,
    );
    if (stopped || mySeq !== seq) return;
    if (step.requireTarget && !target) {
      const onward = i + dir;
      if (onward < 0 || onward >= tour.steps.length) { next(); return; }
      void _show(onward, dir);
      return;
    }
    target?.scrollIntoView?.({ block: 'center', inline: 'nearest' });
    _render(step);

    if (step.advanceOn?.event) {
      unsubscribeAdvance = events.subscribe(step.advanceOn.event, () => {
        if (!stopped && mySeq === seq) next();
      });
    }

    // The anchor's element can be replaced under us by a pane re-render; find its
    // successor by id and re-point. Nothing to do while the same node is still live.
    // Scoped to the anchor's own tab pane, not document.body: the whole-document form
    // would also fire for toasts and unrelated panes re-rendering (see the same rule in
    // listEditorHeight.js) — every anchor lives inside a *-tab-pane, so that's as narrow
    // as this can get without knowing which element the pane's own code will replace.
    // Skipped entirely when the anchor never resolved (target is null): there is no
    // tab pane to scope to, and _waitForAnchor already gave up on finding one.
    if (target && typeof MutationObserver !== 'undefined') {
      const scope = target.closest('.workspace-tab-pane, .main-tab-pane') ?? document.body;
      let pending = 0;
      rebuildWatch = new MutationObserver(() => {
        if (pending) return;
        pending = requestAnimationFrame(() => {
          pending = 0;
          if (stopped || mySeq !== seq) return;
          if (_visible(target)) return;
          const fresh = tourAnchor(step.target);
          if (fresh && fresh !== target) {
            target = fresh;
            _render(step);
          }
        });
      });
      rebuildWatch.observe(scope, { childList: true, subtree: true });
    }
  }

  // stop() announces the tour ended, but the state write that decides where a resume
  // would land lands after it. Announce again once it has, or the help button reads
  // the pre-write state and offers the wrong step.
  function _announceSettled() {
    events.publish(EventTypes.TOUR_STATE_CHANGED, { running: false, tourId: tour.id });
  }

  function next() {
    if (stopped) return;
    if (index >= tour.steps.length - 1) {
      stop();
      void markTourCompleted(tour.id).then(_announceSettled, _announceSettled);
      return;
    }
    void _show(index + 1);
  }

  function back() {
    if (stopped || index <= 0) return;
    void _show(index - 1, -1);
  }

  function skip() {
    if (stopped) return;
    // Read the index before stop() so a resume lands on the step the user walked
    // away from rather than the start.
    const abandonedAt = index;
    stop();
    void markTourDismissed(tour.id, abandonedAt).then(_announceSettled, _announceSettled);
  }

  const handle = { stop, next, back, currentIndex: () => index };
  _active = handle;
  events.publish(EventTypes.TOUR_STATE_CHANGED, { running: true, tourId: tour.id });
  void _show(Math.max(0, Math.min(startAt, tour.steps.length - 1)));
  return handle;
}
