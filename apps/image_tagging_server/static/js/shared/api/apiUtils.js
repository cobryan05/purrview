import { toast } from '/app-static/js/ui/utils/logging.js';

// Maps an API path prefix to the backend service it is proxied to, so a
// connectivity failure can name the service that is actually down.
const SERVICE_BY_PREFIX = [
  ['/api/models', 'Inference'],
  ['/api/inference', 'Inference'],
  ['/api/tasks', 'Tasks'],
  ['/api/training-configs', 'Training'],
  ['/api/training', 'Training'],
];

function serviceNameForUrl(url) {
  const path = String(url);
  for (const [prefix, name] of SERVICE_BY_PREFIX) {
    if (path.startsWith(prefix)) return name;
  }
  // Labels, tags, images, datasets, sources, model-configs all live in db_server.
  return 'Database';
}

// Coalesce bursts of failures (e.g. a page-load fan-out) into one toast per service.
const _lastToastAt = new Map();
const TOAST_THROTTLE_MS = 4000;

/**
 * fetch() wrapper that surfaces a clear "<service> server is not responding"
 * toast on network/connection failures. HTTP error responses (4xx/5xx) still
 * resolve normally and remain the caller's responsibility to handle.
 *
 * @param {string|URL} url
 * @param {RequestInit} [options]
 * @returns {Promise<Response>}
 */
export async function apiFetch(url, options) {
  try {
    return await fetch(url, options);
  } catch (err) {
    const service = serviceNameForUrl(url);
    const now = Date.now();
    if (now - (_lastToastAt.get(service) ?? 0) > TOAST_THROTTLE_MS) {
      _lastToastAt.set(service, now);
      toast(`${service} server is not responding`, 5000, 'error');
    }
    throw err;
  }
}

/**
 * Wraps a function so it accepts single keys or objects containing keys,
 * normalizes keys into arrays, and returns a single result if a single key was given.
 *
 * @param {Function} fn - The async function to wrap.
 * @param {string} [keyArgName] - Optional name of the key property to normalize.
 */
export function wrapSingleKey(fn, keyArgName) {
  return async function (arg) {
    let keys;
    let isSingle = false;
    let callArg = arg;
    let resolvedKeyName = keyArgName;

    if (arg != null) {
      if (typeof arg === 'object' && !Array.isArray(arg)) {
        // Auto-infer key name if only one property is present
        if (!resolvedKeyName) {
          const keysInObject = Object.keys(arg);
          if (keysInObject.length === 1) {
            resolvedKeyName = keysInObject[0];
          }
        }

        if (resolvedKeyName) {
          keys = arg[resolvedKeyName];
          if (keys != null) {
            isSingle = !Array.isArray(keys);
            keys = isSingle ? [keys] : keys;
            callArg = { ...arg, [resolvedKeyName]: keys };
          }
        }
      }

      if (!keys) {
        isSingle = !Array.isArray(arg);
        keys = isSingle ? [arg] : arg;
        callArg = keys;
      }
    }

    // Call original function
    let result = await fn(callArg);

    // Convert plain object results to Map for consistency
    if (!(result instanceof Map) && typeof result === 'object' && result !== null) {
      result = new Map(Object.entries(result));
    }

    // If single key was requested, extract just that value
    if (isSingle && result) {
      const key = keys[0];
      const normalizedKey = typeof key === 'number' ? String(key) : key;
      return result.get(normalizedKey);
    }

    return result;
  };
}


/**
 * Normalize returning a single value. If 'keys' is a single key, return the value directly,
 * otherwise create a map and clone the value to each key.

 @param {Array} keys - The keys for which the value should be returned.
 @param {*} value - The value to return or clone for each key.
 */
export function ignoreKeyReturn(keys, value) {
  const result = new Map();
  for (const key of keys) {
    result.set(key, value);
  }
  return result;
}