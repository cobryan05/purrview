/**
 * Model-confidence persistence and formatting for bboxes.
 *
 * ⚠️ A **leaf** module on purpose — it imports nothing. `mapper.js` already
 * imports `BboxView`, so putting these helpers there and importing them back
 * from `bboxView.js` would close a `mapper -> bboxView -> mapper` cycle, and ES
 * circular imports fail silently as `undefined` exports rather than erroring
 * (see NOTES.md). Keep this file dependency-free.
 */

/**
 * Metadata key holding the model confidence a bbox was created with.
 *
 * ⚠️ `metadata_json` is a raw JSON blob — its inner keys are NOT run through the
 * API's camelCase alias generator, so this stays snake_case on both sides.
 * Written by `ApplyFilterChainTask`'s add_bbox step and by preview-run apply
 * (`Manager.apply_preview_run`); read by the canvas label and the review card.
 */
export const SOURCE_CONFIDENCE_KEY = 'source_confidence';

/**
 * Parse a persisted confidence into a 0–1 number, or `undefined` if absent or junk.
 *
 * It is stored as whatever JSON the writer put there — historically a string —
 * so this parses rather than trusting the type. The `Number.isFinite` guard is
 * the part that matters: `parseFloat(undefined)` is `NaN`, and an unguarded
 * `NaN` renders as "NaN%" on the label.
 *
 * @param {unknown} raw
 * @returns {number|undefined}
 */
export function parseConfidence(raw) {
  if (raw === undefined || raw === null || raw === '') return undefined;
  const n = parseFloat(raw);
  return Number.isFinite(n) ? n : undefined;
}

/**
 * `0.82` → `"82%"`. One implementation for the canvas label, the live-inference
 * label and the review card, which each had their own `Math.round(x * 100)`.
 *
 * @param {number|undefined} confidence
 * @returns {string} the formatted percentage, or `''` when there is nothing to show.
 */
export function formatConfidencePct(confidence) {
  return Number.isFinite(confidence) ? `${Math.round(confidence * 100)}%` : '';
}
