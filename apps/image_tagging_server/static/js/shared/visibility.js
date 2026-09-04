/**
 * Stop paying for work nobody is looking at.
 *
 * A backgrounded tab is not a cheap tab. Every open SSE stream against
 * `/api/tasks/stream` costs the server an image_tagging -> tasks -> db round trip
 * per tick for as long as it stays open, and the notices stream is opened by every
 * tab at load and never closed. Leaving half a dozen tabs parked for the afternoon
 * multiplies that by six against a db_server that is also serving the tab actually
 * in front of the user.
 *
 * Two shapes, because the two kinds of background work want different treatment:
 *
 * - `suspendWhileHidden` tears a *connection* down and rebuilds it. That is worth a
 *   grace period: alt-tabbing through windows should not open and close a socket
 *   each way, and a reconnect storm across every tab at once is its own problem.
 * - `skipWhileHidden` gates a *poll*. Skipping a tick costs nothing and needs no
 *   grace, so it is the plain `document.hidden` guard the NVR previews already use.
 *
 * Both assume the consumer is level-triggered — that a frame carries current state
 * rather than a transition. Everything wired up here is: the task badge and the
 * finished-run announcer both re-derive from the whole payload each frame, so a gap
 * costs latency, never an event.
 */

/**
 * How long a tab must stay hidden before its connections are dropped. Long enough
 * that switching windows to check something and coming back is free; short enough
 * that a tab left for a coffee break stops costing anything.
 */
export const HIDDEN_GRACE_MS = 30_000;

/**
 * Close something while the tab is hidden, and reopen it when the tab comes back.
 *
 * `open` is only ever called to undo a `close` this helper performed, so a consumer
 * that was already closed for its own reasons (no subscribers, destroyed) must make
 * `open` a no-op in that state — this does not track why you were open.
 *
 * @param {object}   handlers
 * @param {function} handlers.open   reopen after the tab becomes visible again
 * @param {function} handlers.close  drop the connection once hidden past the grace
 * @param {number}   [handlers.graceMs]
 * @returns {function} stop watching — also cancels a pending suspend
 */
export function suspendWhileHidden({ open, close, graceMs = HIDDEN_GRACE_MS }) {
  let timer = null;
  let suspended = false;

  function _cancelPending() {
    if (timer === null) return;
    clearTimeout(timer);
    timer = null;
  }

  function onVisibility() {
    if (document.hidden) {
      // Already suspended, or already counting down to it — a second
      // visibilitychange must not restart the clock or double-close.
      if (suspended || timer !== null) return;
      timer = setTimeout(() => {
        timer = null;
        suspended = true;
        close();
      }, graceMs);
      return;
    }
    _cancelPending();
    if (!suspended) return;
    suspended = false;
    open();
  }

  document.addEventListener('visibilitychange', onVisibility);
  return () => {
    document.removeEventListener('visibilitychange', onVisibility);
    _cancelPending();
  };
}

/**
 * Wrap a polling callback so its ticks are dropped while the tab is hidden.
 *
 * The timer keeps running — this trades a negligible idle timer for not having to
 * tear down and re-arm one on every visibility flip, and means the caller's
 * `clearInterval` handle stays valid.
 *
 * @param {function} fn
 * @returns {function} fn, gated on visibility
 */
export function skipWhileHidden(fn) {
  return (...args) => {
    if (document.hidden) return undefined;
    return fn(...args);
  };
}
