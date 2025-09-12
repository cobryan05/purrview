/**
 * Generic cache with inflight tracking
 */
export function createCachedFetcher(fetchFn) {
  const cache = new Map();
  const inflight = new Map();

  return {
    async fetch(key) {
      if (cache.has(key)) return cache.get(key);
      if (inflight.has(key)) return inflight.get(key);

      const promise = (async () => {
        try {
          const result = await fetchFn(key);
          cache.set(key, result);
          return result;
        } finally {
          inflight.delete(key);
        }
      })();

      inflight.set(key, promise);
      return promise;
    },

    set(key, value) {
      cache.set(key, value);
    },

    delete(key) {
      cache.delete(key);
    },

    clear() {
      cache.clear();
    }
  };
}
