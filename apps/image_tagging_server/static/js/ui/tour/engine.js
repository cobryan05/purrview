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
 *
 * Tours branch by *calling* each other. A step's `branches` offer to run another
 * tour and come back: the current tour and step are pushed on a stack, the branch
 * runs as an ordinary linear tour, and finishing it pops back to the step after
 * the one that branched. That is why the overview can hand off to a page's own
 * tour without losing its place, and why no tour ever jumps into another's step
 * numbering — "3 of 7" stays true because each tour counts only its own steps.
 * Pausing or dismissing a branch does *not* pop: the user asked to stop, not to
 * go back up one level.
 */
import { navigateTabs } from '/app-static/js/app/utils/tabs.js';
import { combosForAction, comboLabel } from '/app-static/js/app/image-tagging/hotkeyBindings.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { Logger } from '/app-static/js/ui/utils/logging.js';
import { createCoachMark } from './coachMark.js';
import { tourById } from './registry.js';
import {
  markTourCompleted,
  markTourDismissed,
  markTourPaused,
  readTourState,
  setNeverShowTours,
  writeTourStack,
} from './state.js';

/** How long a step waits for its anchor after navigating before going on without one. */
const TARGET_TIMEOUT_MS = 4000;
/**
 * The same wait for a `requireTarget` step whose pane may still be mounting — after a
 * navigation, or on a resume that lands straight on it. Absence is an expected answer
 * there, not a failure, so this is far shorter than the timeout above.
 */
const OPTIONAL_TARGET_TIMEOUT_MS = 1500;
/**
 * The wait for an optional anchor once the screen has settled: the step does not
 * navigate and the step before it still had its own anchor on screen, so the pane is
 * painted and the control is either there now or not coming.
 *
 * Long enough to outlast a render scheduled on the next frame, short enough that
 * passing over two such steps in a row is not a visible freeze — Review has two
 * adjacent steps whose controls only exist once a card is ticked.
 */
const SETTLED_TARGET_TIMEOUT_MS = 120;
const TARGET_POLL_MS = 50;

/**
 * @typedef {Object} TourBranch
 * @property {string} label - the button's text, e.g. "Tour this tab →"
 * @property {string} tour - id of the tour to run and return from
 * @property {number} [startAt] - step to open the branch on
 */

/**
 * @typedef {Object} TourStep
 * @property {string} id - unique within the tour
 * @property {?string|string[]} target - a `data-tour` anchor id, or several tried in
 *   order so a step can fall back to a control that is always on screen; null for a
 *   step with no spotlight
 * @property {string[]} [navigate] - a tab path to land on first: the main tab id,
 *   then one id per nested tier (Annotate needs three)
 * @property {string} title
 * @property {string} body
 * @property {{event: string}} [advanceOn] - an EventTypes value that completes the step
 * @property {boolean} [requireTarget] - skip the step entirely when its anchor is absent
 * @property {TourBranch[]} [branches] - other tours this step offers to run and return from
 * @property {string} [nextLabel] - overrides "Next" when the plain word undersells it
 */

/**
 * @typedef {Object} Tour
 * @property {string} id
 * @property {string} title
 * @property {TourStep[]} steps
 * @property {boolean} [autoStart] - may offer itself unasked; page tours never do
 * @property {() => Promise<boolean>} [eligible] - whether to auto-start on this workspace
 * @property {string[]} [pages] - main tab ids the help button offers this tour on; all of them when absent
 */

/**
 * How long after an engine-driven tab switch the guard stays up.
 *
 * A tour step navigates, which fires MAIN_TAB_CHANGED, which is exactly what the
 * auto-resume listener watches for — without this it would restart the tour the
 * user is standing in, every step. `navigateTabs` walks nested tiers on timers,
 * so the guard has to outlast the last of them rather than the first click.
 */
const NAV_GUARD_MS = 1200;

let _engineNavigating = 0;

/** Whether a tab change now is the engine's own doing rather than the user's. */
export function isEngineNavigating() {
  return _engineNavigating > 0;
}

/** Tours waiting for a branch to finish, outermost first. @type {import('./state.js').TourReturn[]} */
let _stack = [];

/** The tour a finished branch would return to, or null. */
export function pendingReturn() {
  return _stack.length ? _stack[_stack.length - 1] : null;
}


function _visible(node) {
  return !!node && node.isConnected && node.getClientRects().length > 0;
}

/**
 * The element a `data-tour` id points at, preferring one that is actually on screen.
 *
 * Anchors are not unique by construction: the list-editor primitive keys its "+ New"
 * on a screen prefix, and three Configuration screens share `mc`. Only one of their
 * panes is displayed at a time, but all of them are in the DOM once mounted, so
 * taking the first match would hand back a hidden twin and the step would wait out
 * its timeout pointing at nothing. Falls back to the first match so a step whose
 * pane has not painted yet still resolves once it does.
 *
 * @param {string} id
 */
export function tourAnchor(id) {
  const matches = document.querySelectorAll(`[data-tour="${CSS.escape(id)}"]`);
  if (matches.length <= 1) return matches[0] ?? null;
  return [...matches].find(_visible) ?? matches[0];
}

/**
 * The first of a step's anchors that is on screen, or null.
 *
 * @param {?string|string[]} ids
 * @returns {?Element}
 */
export function bestAnchor(ids) {
  for (const id of (Array.isArray(ids) ? ids : [ids]).filter(Boolean)) {
    const node = tourAnchor(id);
    if (_visible(node)) return node;
  }
  return null;
}

/**
 * `{key:action:nextImage}` in a step's body → the key that action is bound to now.
 *
 * Copy that prints `[` is documentation until someone rebinds it and wrong afterwards,
 * which is why the event modal's Prev/Next tooltips read the user's own bindings too.
 * Resolved per render, so a rebind shows up in the next step that names the key. An
 * action with no key at all says so rather than naming a default it no longer has.
 *
 * @param {string} text
 * @returns {string}
 */
export function resolveKeyTokens(text) {
  return text.replace(/\{key:([\w:]+)\}/g, (_, action) => {
    const combo = combosForAction(action)[0];
    return combo ? comboLabel(combo) : 'unbound';
  });
}

/**
 * Resolve a step's anchor, waiting for it to appear and be laid out.
 *
 * `ids` may name alternatives, tried in order: the control the step is really about
 * first, then something always on screen that explains why it is missing. That is what
 * keeps "Apply to Selected" — which does not exist until a card is ticked — from being
 * a step the reader never sees.
 *
 * @param {?string|string[]} ids
 * @param {number} [timeoutMs]
 * @param {boolean} [warnOnMiss] - false for a `requireTarget` step, where absence is an answer
 * @returns {Promise<?Element>}
 */
function _waitForAnchor(ids, timeoutMs = TARGET_TIMEOUT_MS, warnOnMiss = true) {
  const list = (Array.isArray(ids) ? ids : [ids]).filter(Boolean);
  if (!list.length) return Promise.resolve(null);
  const found = bestAnchor(list);
  if (found) return Promise.resolve(found);
  return new Promise(resolve => {
    const deadline = Date.now() + timeoutMs;
    const poll = () => {
      const node = bestAnchor(list);
      if (node) return resolve(node);
      if (Date.now() > deadline) {
        if (warnOnMiss) Logger.warn(`tour anchor "${list.join('/')}" not found; continuing without a spotlight`);
        return resolve(null);
      }
      setTimeout(poll, TARGET_POLL_MS);
    };
    poll();
  });
}

let _active = null;
/** Id of the tour `_active` is running, for callers that only care about their own. */
let _activeId = null;

/** Whether a tour is on screen. */
export function isTourRunning() {
  return _active != null;
}

/** The id of the running tour, or null. */
export function runningTourId() {
  return _activeId;
}

/** Stop whatever tour is running, recording nothing about it. */
export function stopTour() {
  _active?.stop();
}

/**
 * Stop the running tour the way the user's own Skip does: the step is remembered, so
 * the next start picks up there. For a surface that can be taken away underneath a
 * tour — a modal being closed — where forgetting the place would restart the walk.
 */
export function pauseTour() {
  _active?.pause();
}

/**
 * Start a tour. Stops any tour already running.
 *
 * @param {Tour} tour
 * @param {Object} [opts]
 * @param {number} [opts.startAt] - step index to open on
 * @param {import('./state.js').TourReturn[]} [opts.stack] - tours to return to when
 *   this one finishes; defaults to clearing the stack, so starting a tour on
 *   purpose is never haunted by a branch someone abandoned an hour ago
 * @returns {{stop: () => void, next: () => void, back: () => void, currentIndex: () => number}}
 */
export function startTour(tour, { startAt = 0, stack = [] } = {}) {
  _active?.stop();
  _stack = stack.slice();

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
    if (_active === handle) {
      _active = null;
      _activeId = null;
    }
    events.publish(EventTypes.TOUR_STATE_CHANGED, { running: false, tourId: tour.id });
  }

  function _render(step) {
    coach.show({
      target,
      title: step.title,
      body: resolveKeyTokens(step.body),
      index,
      count: tour.steps.length,
      tourTitle: tour.title,
      returnTo: _stack.length ? tourById(_stack[_stack.length - 1].tourId)?.title ?? null : null,
      branches: (step.branches ?? []).map(b => ({ label: b.label, onPick: () => _branch(b) })),
      nextLabel: step.nextLabel ?? null,
      onNext: next,
      onBack: index > 0 ? back : null,
      onPause: pause,
      onDismiss: dismiss,
      onNeverShow: (value) => {
        neverShow = value;
        void setNeverShowTours(value);
      },
      neverShow,
      waiting: !!step.advanceOn,
    });
  }

  /**
   * Run another tour and come back to the step after this one.
   * @param {TourBranch} branch
   */
  function _branch(branch) {
    const sub = tourById(branch.tour);
    if (!sub) {
      Logger.warn(`tour "${tour.id}" branches to unknown tour "${branch.tour}"`);
      next();
      return;
    }
    const nextStack = [..._stack, { tourId: tour.id, index }];
    stop();
    void writeTourStack(nextStack);
    startTour(sub, { startAt: branch.startAt ?? 0, stack: nextStack });
  }

  /**
   * @param {number} i
   * @param {1|-1} [dir] - which way to keep going if step `i` turns out to be absent
   * @param {boolean} [settled] - whether the screen is known to be painted, evidenced by
   *   the step being left still having its own anchor on it. Passed along when a step is
   *   skipped: by then `target` is null and the evidence would be lost for the step after.
   */
  async function _show(i, dir = 1, settled = _visible(target)) {
    const step = tour.steps[i];
    if (!step) return;
    const mySeq = ++seq;
    _teardownStep();
    index = i;

    if (step.navigate) {
      const [mainTab, ...rest] = step.navigate;
      _engineNavigating++;
      setTimeout(() => { _engineNavigating--; }, NAV_GUARD_MS);
      if (rest.length) navigateTabs(mainTab, rest);
      else document.querySelector(`.main-tab-button[data-tab="${mainTab}"]`)?.click();
    }

    // An optional step waits out the full timeout only while the screen may still be
    // painting: it navigates itself, or nothing has established that the screen settled.
    // Once the pane is painted, a control that is not there now is not coming, and
    // waiting for it is a freeze rather than a check.
    const optionalWait = (step.navigate || !settled)
      ? OPTIONAL_TARGET_TIMEOUT_MS
      : SETTLED_TARGET_TIMEOUT_MS;
    target = await _waitForAnchor(
      step.target,
      step.requireTarget ? optionalWait : TARGET_TIMEOUT_MS,
      !step.requireTarget,
    );
    if (stopped || mySeq !== seq) return;
    if (step.requireTarget && !target) {
      const onward = i + dir;
      if (onward < 0 || onward >= tour.steps.length) { next(); return; }
      void _show(onward, dir, settled);
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
          const fresh = bestAnchor(step.target);
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
      // Captured before stop(), which leaves _stack alone but may be followed by
      // another startTour that replaces it.
      const frame = pendingReturn();
      const popped = _stack.slice(0, -1);
      stop();
      // The pop rides along with the completion: both are one read-modify-write
      // against the same setting, and two in flight would lose one another.
      void markTourCompleted(tour.id, frame ? popped : null)
        .then(_announceSettled, _announceSettled);
      if (!frame) return;
      const parent = tourById(frame.tourId);
      // The branch came *from* frame.index, so the parent picks up after it.
      if (parent) startTour(parent, { startAt: frame.index + 1, stack: popped });
      else _stack = popped;
      return;
    }
    void _show(index + 1);
  }

  function back() {
    if (stopped || index <= 0) return;
    void _show(index - 1, -1);
  }

  /**
   * Stop, meaning to come back. The stack survives, so a paused branch still
   * knows the overview is waiting behind it.
   */
  function pause() {
    if (stopped) return;
    // Read the index before stop() so a resume lands on the step the user walked
    // away from rather than the start.
    const at = index;
    stop();
    void markTourPaused(tour.id, at).then(_announceSettled, _announceSettled);
  }

  /** Stop for good: this tour never offers itself again, and the chain is dropped. */
  function dismiss() {
    if (stopped) return;
    const at = index;
    stop();
    _stack = [];
    void markTourDismissed(tour.id, at, []).then(_announceSettled, _announceSettled);
  }

  const handle = { stop, next, back, pause, dismiss, currentIndex: () => index };
  _active = handle;
  _activeId = tour.id;
  events.publish(EventTypes.TOUR_STATE_CHANGED, { running: true, tourId: tour.id });
  void _show(Math.max(0, Math.min(startAt, tour.steps.length - 1)));
  return handle;
}
