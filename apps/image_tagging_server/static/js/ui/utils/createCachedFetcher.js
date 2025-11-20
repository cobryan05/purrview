/**
 * createCachedFetcher(fetchFn)
 *
 * Creates runtime cache for info from server.
 * Only updates a given key from the server once,
 * then always uses the cached version until the
 * cache is cleared.
 *
 * TODO: Support update pass-through to server
 *
 * fetchFn: async (keys: string[]) => {str: any}
 *   - Accepts a list of keys
 *   - Returns an object mapping each key to its data
 */
export function createCachedFetcher(fetchFn) {
  const cache = new Map();     // key -> value
  const inflight = new Map();  // key -> promise for that key

  async function fetchMissingKeys(missingKeys) {
    if (missingKeys.length === 0) return {};

    // Avoid launching duplicate fetches
    const notInFlight = [];
    for (const key of missingKeys) {
      if (!inflight.has(key)) {
        notInFlight.push(key);
      }
    }

    if (notInFlight.length === 0) {
      // All missing keys already have inflights — wait for them
      await Promise.all(missingKeys.map(k => inflight.get(k)));
      return;
    }

    // Create a single inflight promise for all missing keys
    const promise = (async (keys) => {
      try {
        const results = await fetchFn(notInFlight); // returns map: key -> value
        // Fill cache
        for (const [key, value] of results) {
          cache.set(key, value);
        }
      } finally {
        // Clear inflight for ONLY these keys
        for (const key of notInFlight) {
          inflight.delete(key);
        }
      }
    })();

    // Mark all missing keys as inflight
    for (const key of notInFlight) {
      inflight.set(key, promise);
    }

    await promise;
  }

  return {
    async fetch(keys) {
      const isArray = Array.isArray(keys);
      const keyList = isArray ? keys : [keys];

      const missing = keyList.filter(k => !cache.has(k));

      // Fetch only missing ones
      await fetchMissingKeys(missing);

      // All keys should now be either cached
      if (!isArray) {
        return cache.get(keys);
      }
      const result = new Map();
      for (const key of keyList) {
        result.set(key, cache.get(key));
      }

      return result;
    },

    set(key, value) {
      cache.set(key, value);
    },

    delete(key) {
      cache.delete(key);
    },

    clear() {
      cache.clear();
      inflight.clear();
    },
  };
}
