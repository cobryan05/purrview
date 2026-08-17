// Per-runtime handlers for canvas actions a hotkey can fire but the canvas
// itself cannot implement.
//
// Save and Run-inference both live *outside* the runtime: Save is whatever the
// surrounding editor means by "persist" (the review modal's preview mode swaps
// in an in-memory persist that never touches the DB), and Run-inference needs
// the inference panel's selection/threshold state, which lives in that panel's
// closure. Rather than have hotkeys.js reach into either, each owner registers
// a handler here for the lifetime of its UI and hotkeys.js looks it up by
// runtime id.
//
// Keyed by runtime id (not the runtime object) so the same lookup works from
// event payloads, which carry ids.

/** @type {Map<string, Map<string, Function>>} runtimeId → (actionName → handler) */
const _handlers = new Map();

/**
 * Register a handler for one action on one runtime.
 * @param {string} runtimeId
 * @param {'save'|'runInference'} name
 * @param {Function} fn
 * @returns {() => void} unregister — call it when the owning UI is destroyed.
 */
export function registerCanvasAction(runtimeId, name, fn) {
  let forRuntime = _handlers.get(runtimeId);
  if (!forRuntime) {
    forRuntime = new Map();
    _handlers.set(runtimeId, forRuntime);
  }
  forRuntime.set(name, fn);
  return () => {
    const current = _handlers.get(runtimeId);
    // Only drop it if it is still *this* handler: a modal reopened on a reused
    // runtime id registers before the old one's destroy() runs.
    if (current?.get(name) === fn) {
      current.delete(name);
      if (current.size === 0) _handlers.delete(runtimeId);
    }
  };
}

/**
 * The handler registered for an action, or null.
 * @param {string} runtimeId
 * @param {'save'|'runInference'} name
 * @returns {Function|null}
 */
export function getCanvasAction(runtimeId, name) {
  return _handlers.get(runtimeId)?.get(name) ?? null;
}
