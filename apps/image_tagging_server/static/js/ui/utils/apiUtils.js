/**
 * Wraps a function so it accepts single keys or objects containing keys,
 * normalizes keys into arrays, and returns a single result if a single key was given.
 *
 * @param {Function} fn - The async function to wrap.
 * @param {string} [keyArgName] - Optional name of the key property to normalize.
 */
export function wrapSingleKey(fn, keyArgName) {
  return async function (arg) {
    if (arg == null) return await fn(arg);

    let keys;
    let isSingle = false;
    let callArg = arg;
    let resolvedKeyName = keyArgName;

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
        if (keys == null) return await fn(arg);

        isSingle = !Array.isArray(keys);
        keys = isSingle ? [keys] : keys;
        callArg = { ...arg, [resolvedKeyName]: keys };
      }
    }

    if (!keys) {
      isSingle = !Array.isArray(arg);
      keys = isSingle ? [arg] : arg;
      callArg = keys;
    }

    let result = await fn(callArg);

    // Convert plain object results to Map for consistency
    if (!(result instanceof Map) && typeof result === 'object' && result !== null) {
      result = new Map(Object.entries(result));
    }

    // Normalize key to string for Map lookup (prevents number/string mismatch)
    if (isSingle && result) {
      const key = keys[0];
      const normalizedKey = typeof key === 'number' ? String(key) : key;
      return result.get(normalizedKey);
    }

    return result;
  };
}
