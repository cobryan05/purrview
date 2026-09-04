/**
 * Shared subscription to the task SSE stream (`/api/tasks/stream`).
 *
 * Every open EventSource against this endpoint costs the server a full
 * image_tagging -> tasks -> db round trip per tick, for as long as it stays open.
 * Three separate call sites used to each open their own — the Operations tab, the
 * task-runner section and the source-preview runner — so a single browser tab
 * sitting on Operations with a preview modal open multiplied that cost by three
 * while receiving three copies of the identical payload.
 *
 * This module keeps exactly one EventSource alive and fans it out to every
 * subscriber. It opens on the first subscriber and closes when the last one
 * leaves, so an idle app still holds no connection at all.
 *
 * Sharing one socket means sharing its failures, so recovery lives here too. An
 * EventSource retries transient drops by itself but a fully CLOSED one never
 * comes back — that is what the browser gives you when the endpoint answers 502
 * or a wrong content type, i.e. whenever the service restarts. Left alone it is
 * a dead handle that this module would hand to every later subscriber as well.
 *
 * A hidden tab holds no connection either. Subscribers stay subscribed — the Jobs
 * badge deliberately outlives tab focus while something is running — but a tab
 * parked in the background for the afternoon should not keep costing a round trip
 * twice a second, which is the whole reason that stream is expensive. Everything
 * downstream re-derives from the full payload on each frame, so the only cost of
 * the gap is that the badge is as old as the tab is stale, and it is correct again
 * on the first frame after the user comes back.
 */

import { suspendWhileHidden } from '/app-static/js/shared/visibility.js';

// Retry backoff for a fatally closed stream. The last entry repeats: a server
// that is down for an hour should be polled every 30s, not abandoned, because
// the Operations badge subscribes once on tab entry and never asks again.
const RETRY_DELAYS_MS = [1000, 2000, 4000, 8000, 15000, 30000];

// Failures to absorb silently before telling subscribers. The task runners await
// a promise that only onError can settle, so they cannot wait forever — but the
// common case (a service restart mid-run) is over within a couple of seconds,
// and failing a running task's modal for that is worse than a few seconds of a
// stale progress bar.
const NOTIFY_AFTER_ATTEMPTS = 3;

let _source = null;
const _subscribers = new Set();
let _retryTimer = null;
let _retryAttempt = 0;
let _outageReported = false;
/** True while the tab has been hidden long enough that we dropped the stream. */
let _suspended = false;

function _fanout(method, arg) {
  // Copy first: a handler may unsubscribe (or subscribe) while we iterate.
  for (const sub of [..._subscribers]) {
    try { sub[method]?.(arg); } catch (err) { console.error('taskStream subscriber failed', err); }
  }
}

function _cancelRetry() {
  if (_retryTimer === null) return;
  clearTimeout(_retryTimer);
  _retryTimer = null;
}

function _teardown() {
  _cancelRetry();
  if (_source === null) return;
  _source.close();
  _source = null;
}

function _closeIfIdle() {
  if (_subscribers.size > 0) return;
  // Also drops any pending reconnect: a closed modal must not leave a socket
  // reopening itself against a server nobody is watching any more.
  _teardown();
  _retryAttempt = 0;
  _outageReported = false;
}

function _onFatalClose(es) {
  // Ignore a handle we have already replaced — its close() fires this too.
  if (es !== _source) return;
  _source.close();
  _source = null;

  if (_subscribers.size === 0) return;

  _retryAttempt += 1;
  const delay = RETRY_DELAYS_MS[Math.min(_retryAttempt - 1, RETRY_DELAYS_MS.length - 1)];
  _cancelRetry();
  _retryTimer = setTimeout(() => { _retryTimer = null; _open(); }, delay);

  // Once per outage, not once per attempt: a subscriber that only wanted to know
  // it should give up gets told, and the rest keep reconnecting behind it.
  if (_retryAttempt >= NOTIFY_AFTER_ATTEMPTS && !_outageReported) {
    _outageReported = true;
    _fanout('onError', new Error('SSE connection closed unexpectedly'));
  }
}

function _open() {
  _cancelRetry();
  if (_source !== null) return;
  // A subscriber arriving while the tab is hidden must not reopen what the
  // visibility gate just closed — it will be opened for them on the way back.
  if (_suspended) return;

  const es = new EventSource('/api/tasks/stream');
  _source = es;

  es.onmessage = (e) => {
    let data;
    try { data = JSON.parse(e.data); } catch { return; }
    // A frame is the only proof the stream actually works — `onopen` fires for a
    // connection that dies immediately, which would keep the backoff pinned at
    // its shortest delay through an outage.
    _retryAttempt = 0;
    _outageReported = false;
    _fanout('onMessage', data);
  };

  es.onerror = () => {
    // readyState CONNECTING means EventSource is already retrying on its own;
    // opening a second socket over the top of that is how you end up with two.
    if (es.readyState !== EventSource.CLOSED) return;
    _onFatalClose(es);
  };
}

// Installed once, for the life of the page. Suspending drops the socket but keeps
// every subscriber, so the set is still there to reopen for on the way back — and
// `_closeIfIdle` stays the only thing that decides nobody wants the stream at all.
suspendWhileHidden({
  close: () => {
    _suspended = true;
    _teardown();
  },
  open: () => {
    _suspended = false;
    if (_subscribers.size > 0) _open();
  },
});

/**
 * Subscribes to task-state frames.
 *
 * @param {object}   handlers
 * @param {function} handlers.onMessage - called with the parsed frame ({ tasks, ... }).
 * @param {function} [handlers.onError] - called once when the stream has stayed
 *   down across several retries. Reconnection continues regardless, so a
 *   long-lived subscriber may simply omit this and wait for frames to resume.
 * @returns {function} unsubscribe — idempotent; closes the stream if it was the last.
 */
export function subscribeTaskStream({ onMessage, onError } = {}) {
  const sub = { onMessage, onError };
  _subscribers.add(sub);
  _open();

  // "Once per outage" is about not repeating yourself to the same subscriber,
  // not about the outage being announced once and then forgotten. A subscriber
  // that arrives mid-outage has been told nothing, and the task runners await a
  // promise that only onError can settle — so with a long-lived subscriber (the
  // Jobs badge) holding the set non-empty, starting a model import while the
  // stream is down left the modal waiting for a frame that was never coming.
  // Deferred so the caller has its unsubscribe handle before onError can run.
  if (_outageReported && onError) {
    setTimeout(() => {
      if (!_subscribers.has(sub) || !_outageReported) return;
      try { sub.onError(new Error('SSE connection closed unexpectedly')); }
      catch (err) { console.error('taskStream subscriber failed', err); }
    }, 0);
  }

  let done = false;
  return () => {
    if (done) return;
    done = true;
    _subscribers.delete(sub);
    _closeIfIdle();
  };
}
