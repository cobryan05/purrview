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

/**
 * Stamp a local wall-clock date and time with the viewer's UTC offset.
 *
 * The other direction from `parseApiDate`, and the fix for the case its own header
 * warns about: a `<input type="date">` value is a *local* calendar day, so sending
 * `"2026-08-24T00:00:00"` bare hands the backend a naive datetime it reads as UTC
 * (`UtcDateTime.process_bind_param` assumes UTC for naive input). At UTC−7 "24 Aug"
 * then means 23 Aug 17:00 local — the query window is wrong, and so is anything the
 * client draws from it. Notably `timelineRange`, whose axis walks to local midnight
 * and never reaches it when the range starts at 17:00, leaving the timeline with day
 * gridlines seven hours out.
 *
 * The offset is resolved against the named day itself rather than against "now", so a
 * range spanning a DST boundary stamps each end with the offset actually in force
 * there. Emitting the offset explicitly is what lets `parseApiDate` read the value
 * back unambiguously, so the round trip through the filter state stays lossless.
 *
 * @param {string} dateStr - `YYYY-MM-DD`, as an `<input type="date">` yields it.
 * @param {string} [timeStr] - `HH:MM:SS` local wall clock.
 * @returns {?string} offset-bearing ISO 8601, or null if the date is unusable.
 */
export function localWallClockIso(dateStr, timeStr = '00:00:00') {
  if (!dateStr) return null;
  const [y, m, d] = String(dateStr).split('-').map(Number);
  const [hh = 0, mm = 0, ss = 0] = String(timeStr).split(':').map(Number);
  if (!Number.isFinite(y) || !Number.isFinite(m) || !Number.isFinite(d)) return null;
  const local = new Date(y, (m - 1), d, hh, mm, ss);
  if (Number.isNaN(local.getTime())) return null;
  // `getTimezoneOffset` is minutes *behind* UTC, so its sign is inverted from the one
  // an ISO offset prints: UTC−7 reports +420 and must render as `-07:00`.
  const offsetMin = -local.getTimezoneOffset();
  const sign = offsetMin < 0 ? '-' : '+';
  const abs = Math.abs(offsetMin);
  const pad = n => String(n).padStart(2, '0');
  return `${dateStr}T${pad(hh)}:${pad(mm)}:${pad(ss)}`
    + `${sign}${pad(Math.floor(abs / 60))}:${pad(abs % 60)}`;
}
