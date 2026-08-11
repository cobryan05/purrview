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
 * Metadata key new writers use for the model confidence a bbox was created with.
 *
 * ⚠️ `metadata_json` is a raw JSON blob — its inner keys are NOT run through the
 * API's camelCase alias generator, so these stay snake_case on both sides.
 */
export const CONFIDENCE_KEY = 'confidence';

/**
 * Every key a confidence has ever been written under, newest-preferred first.
 *
 * ⚠️ There are genuinely two, and readers must handle both — this is data
 * compatibility, not a naming preference:
 *   - `confidence` — CollectTrainingImagesTask (by far the highest-volume
 *     producer) and the canvas save path.
 *   - `source_confidence` — ApplyFilterChainTask's add_bbox step and
 *     preview-run apply.
 * Dropping either silently blanks the confidence on whole classes of bbox.
 */
export const CONFIDENCE_KEYS = [CONFIDENCE_KEY, 'source_confidence'];

/**
 * Pull the confidence out of a bbox's raw metadata, whichever key it was
 * written under.
 *
 * @param {Object<string, any>|null|undefined} metadataJson
 * @returns {number|undefined}
 */
export function readConfidence(metadataJson) {
  if (!metadataJson) return undefined;
  for (const key of CONFIDENCE_KEYS) {
    const parsed = parseConfidence(metadataJson[key]);
    if (parsed !== undefined) return parsed;
  }
  return undefined;
}

/**
 * Parse a persisted confidence into a 0–1 number, or `undefined` if absent or junk.
 *
 * ⚠️ The stored type genuinely varies by writer, so this parses rather than
 * trusting it: the canvas save goes through `extra`, typed `Dict[str, str]`
 * server-side, so it writes a **string**; the Python writers
 * (ApplyFilterChainTask, preview-run apply) write a **float**. The
 * `Number.isFinite` guard is the part that matters — `parseFloat(undefined)` is
 * `NaN`, and an unguarded `NaN` renders as "NaN%" on the label.
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
