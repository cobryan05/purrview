/**
 * Simple debug logger, prefixes messages with [DEBUG]
 * Can be enhanced to toggle debug on/off via a flag
 */
export function debug(...args) {
  // Toggle this to false to disable debug logs
  const DEBUG_ENABLED = true;

  if (DEBUG_ENABLED) {
    console.log('[DEBUG]', ...args);
  }
}

export function generateUUID() {
  if (window.crypto && typeof window.crypto.randomUUID === 'function') {
    return window.crypto.randomUUID();
  }
  // Fallback: RFC4122 version 4 compliant UUID
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
