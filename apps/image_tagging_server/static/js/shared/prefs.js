/**
 * Parse-safe wrappers around localStorage / sessionStorage.
 *
 * The repeated hazard these exist to remove is *not* "prefs are scattered" —
 * each pref is genuinely owned by its module and a central registry would have
 * one consumer per key. It is that every module hand-rolled the same
 * `try { JSON.parse(getItem(k) || 'null') } catch {}` dance around a value a
 * previous version of the app may have written in a different shape, and that
 * bare `setItem` throws (quota, Safari private mode) in a way most call sites
 * did not guard at all.
 *
 * So: no central store, no key registry — just one place that swallows the
 * parse/quota errors and hands back the caller's fallback.
 *
 * Two flavours per store:
 *   - `read`/`write`   — JSON-encoded values (objects, arrays, booleans)
 *   - `readRaw`/`writeRaw` — bare strings, for keys whose stored format
 *     predates this helper and must stay byte-compatible (a user's saved
 *     sidebar width is `240`, not `"240"`).
 *
 * Key convention for **new** keys: prefix `ww_`. Existing keys use a mix of
 * `wtch_`, `ww_` and bare names; they are deliberately left alone, since
 * renaming a storage key silently resets that pref for every existing user in
 * exchange for tidiness only.
 */

/**
 * @param {() => Storage} getStore  lazy — accessing `localStorage` itself
 *   throws (not just its methods) when storage is disabled.
 */
function _createPrefStore(getStore) {
  function _store() {
    try {
      return getStore() ?? null;
    } catch {
      return null;
    }
  }

  function _getItem(key) {
    try {
      return _store()?.getItem(key) ?? null;
    } catch {
      return null;
    }
  }

  function _setItem(key, str) {
    try {
      _store()?.setItem(key, str);
      return true;
    } catch {
      return false;
    }
  }

  return {
    /**
     * Read a JSON-encoded pref. Returns `fallback` when the key is absent, the
     * stored value is not valid JSON (an older version wrote a different
     * shape), or it decodes to `null`.
     *
     * Callers keep their own shape checks (`Array.isArray`, `typeof`) — this
     * only guarantees "did not throw", not "is what you expect".
     *
     * @template T
     * @param {string} key
     * @param {T} fallback
     * @returns {T|any}
     */
    read(key, fallback) {
      const raw = _getItem(key);
      if (raw === null) return fallback;
      try {
        const parsed = JSON.parse(raw);
        return parsed === null ? fallback : parsed;
      } catch {
        return fallback;
      }
    },

    /**
     * Write a JSON-encoded pref. Never throws.
     * @param {string} key
     * @param {any} value
     * @returns {boolean} false if storage rejected the write (quota/disabled)
     */
    write(key, value) {
      try {
        return _setItem(key, JSON.stringify(value));
      } catch {
        return false; // circular structure — not storable
      }
    },

    /**
     * Read a bare string pref. Returns `fallback` when absent.
     * @param {string} key
     * @param {string|null} [fallback=null]
     * @returns {string|null}
     */
    readRaw(key, fallback = null) {
      const raw = _getItem(key);
      return raw === null ? fallback : raw;
    },

    /**
     * Write a bare string pref (coerced via String()). Never throws.
     * @param {string} key
     * @param {any} value
     * @returns {boolean} false if storage rejected the write
     */
    writeRaw(key, value) {
      return _setItem(key, String(value));
    },

    /** Remove a pref. Never throws. @param {string} key */
    remove(key) {
      try {
        _store()?.removeItem(key);
      } catch {
        /* storage disabled — nothing to remove */
      }
    },
  };
}

/** Persistent, survives a browser restart. */
export const prefs = _createPrefStore(() => globalThis.localStorage);

/** Per-tab, cleared when the tab closes. */
export const sessionPrefs = _createPrefStore(() => globalThis.sessionStorage);
