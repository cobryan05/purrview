/**
 * Detector-miss metadata (COLLECTOR_UNIFICATION_PLAN.md §4A/§5).
 *
 * ⚠️ A **leaf** module on purpose, same reasoning as confidence.js: mapper.js
 * already imports `BboxView`, so keeping this dependency-free avoids a
 * `mapper -> bboxView -> mapper` cycle.
 */

/**
 * Whether a detector-miss box's interpolated position was ambiguous — another
 * live track was close enough that the tracker could plausibly have picked
 * the wrong identity to interpolate onto (backend: `CollectTrainingImagesAction
 * ._is_association_ambiguous`). Computed at collection time; unrecoverable
 * later, so it is read here rather than re-derived.
 *
 * @param {Object<string, any>|null|undefined} metadataJson
 * @returns {boolean}
 */
export function readDetectorMissAmbiguous(metadataJson) {
  return metadataJson?.detector_miss_ambiguous === 'True';
}
