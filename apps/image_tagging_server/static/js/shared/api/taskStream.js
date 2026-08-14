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
 */

let _source = null;
const _subscribers = new Set();

function _closeIfIdle() {
  if (_subscribers.size > 0 || _source === null) return;
  _source.close();
  _source = null;
}

function _open() {
  if (_source !== null) return;

  _source = new EventSource('/api/tasks/stream');

  _source.onmessage = (e) => {
    let data;
    try { data = JSON.parse(e.data); } catch { return; }
    // Copy first: a handler may unsubscribe (or subscribe) while we iterate.
    for (const sub of [..._subscribers]) {
      try { sub.onMessage?.(data); } catch (err) { console.error('taskStream subscriber failed', err); }
    }
  };

  _source.onerror = () => {
    // EventSource auto-reconnects on transient errors (readyState CONNECTING).
    // Only a fully CLOSED connection is fatal and worth telling subscribers about.
    if (_source?.readyState !== EventSource.CLOSED) return;
    const err = new Error('SSE connection closed unexpectedly');
    for (const sub of [..._subscribers]) {
      try { sub.onError?.(err); } catch (e) { console.error('taskStream subscriber failed', e); }
    }
  };
}

/**
 * Subscribes to task-state frames.
 *
 * @param {object}   handlers
 * @param {function} handlers.onMessage - called with the parsed frame ({ tasks, ... }).
 * @param {function} [handlers.onError] - called when the connection closes for good.
 * @returns {function} unsubscribe — idempotent; closes the stream if it was the last.
 */
export function subscribeTaskStream({ onMessage, onError } = {}) {
  const sub = { onMessage, onError };
  _subscribers.add(sub);
  _open();

  let done = false;
  return () => {
    if (done) return;
    done = true;
    _subscribers.delete(sub);
    _closeIfIdle();
  };
}
