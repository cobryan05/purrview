/**
 * Parsing API timestamps.
 *
 * The backend now guarantees an explicit UTC offset on every timestamp it
 * serialises (`UtcDateTime` in `apps/helpers/db/types.py` re-attaches the zone
 * that SQLite drops on read). Before that fix the API emitted offset-less
 * strings like `"2026-08-20T15:47:55.537329"`, and per ES2015 a browser reads
 * an offset-less *date-time* string as **local time** — so a UTC instant
 * rendered shifted by the viewer's offset, with nothing raising an error. The
 * NVR and Review tabs each grew their own private `${value}Z` workaround; the
 * Tasks tab never did, and showed start/end times hours in the future.
 *
 * This module is the one place that knows the rule. It stays defensive about
 * the offset-less form on purpose: rows written before the backend fix are
 * still read through the same code path, and pinning an ambiguous string to UTC
 * is the correct reading for every timestamp this API produces.
 *
 * Note the scope — these are for values *received from the API*. Strings the
 * user typed into a `datetime-local` input are genuinely local and must not be
 * passed through here.
 */

/** Does this ISO string already state its zone? */
const HAS_OFFSET = /(?:Z|[+-]\d{2}:?\d{2})$/;

/**
 * Parse an API timestamp into a `Date`.
 *
 * @param {?string} value ISO timestamp, with or without an explicit offset.
 * @returns {?Date} null if absent or unparseable.
 */
export function parseApiDate(value) {
  if (!value) return null;
  const s = String(value);
  const date = new Date(HAS_OFFSET.test(s) ? s : `${s}Z`);
  return Number.isNaN(date.getTime()) ? null : date;
}

/**
 * Parse an API timestamp into epoch millis — for comparisons and sorting,
 * where allocating a `Date` per element is wasted work.
 *
 * @param {?string} value
 * @returns {?number} epoch millis, or null if absent/unparseable.
 */
export function parseApiDateMs(value) {
  const date = parseApiDate(value);
  return date === null ? null : date.getTime();
}
