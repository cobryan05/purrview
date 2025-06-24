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
