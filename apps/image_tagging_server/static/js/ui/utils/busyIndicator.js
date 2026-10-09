// One place the app says "I am working on it".
//
// Slow operations here are fire-and-forget event publishes — the Run button
// publishes RUN_INFERENCE_ON_CANVAS and returns, and next/prev publishes
// LOAD_IMAGE_ONTO_CANVAS and returns — so the control that was clicked cannot
// know when the work ends and cannot show its own progress. The indicator is
// therefore driven from the subscriber that owns the work, and is global rather
// than attached to a control: a user who clicked Run may be looking anywhere.

const SHOW_DELAY_MS = 180;
const MIN_VISIBLE_MS = 350;

/** @type {Map<symbol, string>} token -> label, insertion-ordered so the newest wins. */
const _active = new Map();

let _el = null;
let _labelEl = null;
let _showTimer = null;
let _hideTimer = null;
let _shownAt = 0;

function _ensureEl() {
  if (_el) return _el;
  _el = document.createElement('div');
  _el.className = 'busy-indicator';
  _el.dataset.testid = 'busy-indicator';
  // Announced rather than merely drawn: the whole point is "your click was not
  // ignored", which a screen reader user needs told, not shown.
  _el.setAttribute('role', 'status');
  _el.setAttribute('aria-live', 'polite');
  _el.hidden = true;

  const stripe = document.createElement('div');
  stripe.className = 'busy-indicator__stripe';
  _labelEl = document.createElement('span');
  _labelEl.className = 'busy-indicator__label';

  _el.append(stripe, _labelEl);
  document.body.appendChild(_el);
  return _el;
}

/** The label of the most recently started operation, which is the one to name. */
function _currentLabel() {
  let last = '';
  for (const label of _active.values()) last = label;
  return last;
}

function _show() {
  _showTimer = null;
  const el = _ensureEl();
  _labelEl.textContent = _currentLabel();
  if (el.hidden) {
    el.hidden = false;
    _shownAt = Date.now();
  }
}

function _hide() {
  if (!_el || _el.hidden) return;
  _el.hidden = true;
  _labelEl.textContent = '';
}

function _sync() {
  clearTimeout(_hideTimer);
  _hideTimer = null;

  if (_active.size > 0) {
    if (_el && !_el.hidden) {
      _labelEl.textContent = _currentLabel();
      return;
    }
    // Nothing is shown for the first SHOW_DELAY_MS. Most navigations beat it, and
    // an indicator that flashes on every one of them is noise the eye learns to
    // discard — which costs exactly the signal this exists to send.
    //
    // A pending countdown is left alone rather than restarted: it is timed from
    // the oldest unfinished operation, and restarting it per arrival would let a
    // steady stream of overlapping work defer the indicator indefinitely.
    if (!_showTimer) _showTimer = setTimeout(_show, SHOW_DELAY_MS);
    return;
  }

  clearTimeout(_showTimer);
  _showTimer = null;

  // Once visible it stays visible for MIN_VISIBLE_MS, so an operation that lands
  // just past the delay registers as a blink rather than a flicker.
  if (!_el || _el.hidden) {
    _hide();
    return;
  }
  const remaining = MIN_VISIBLE_MS - (Date.now() - _shownAt);
  if (remaining <= 0) _hide();
  else _hideTimer = setTimeout(_hide, remaining);
}

/**
 * Mark the app busy until the returned function is called.
 *
 * Refcounted on purpose: a navigation started while inference is still running
 * must not have the first thing to finish clear the indicator for both. Each
 * caller holds its own token and the indicator drops when the last one ends.
 *
 * @param {string} label - What is happening, e.g. 'Running inference…'.
 * @returns {() => void} end - Idempotent; safe to call from a finally block.
 */
export function beginBusy(label) {
  const token = Symbol('busy');
  _active.set(token, label);
  _sync();
  return function end() {
    if (!_active.delete(token)) return;
    _sync();
  };
}

/**
 * Run `fn` with the app marked busy, releasing the mark however it settles.
 *
 * @template T
 * @param {string} label
 * @param {() => Promise<T>} fn
 * @returns {Promise<T>}
 */
export async function withBusy(label, fn) {
  const end = beginBusy(label);
  try {
    return await fn();
  } finally {
    end();
  }
}

/** Test seam: drop the element and every pending timer. */
export function _resetBusyIndicator() {
  clearTimeout(_showTimer);
  clearTimeout(_hideTimer);
  _showTimer = null;
  _hideTimer = null;
  _active.clear();
  _el?.remove();
  _el = null;
  _labelEl = null;
}
