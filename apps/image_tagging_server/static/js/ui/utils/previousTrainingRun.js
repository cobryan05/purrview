/**
 * "What went into the last training run?" for a training config.
 *
 * There is no training-run history table — the record of a run lives in the
 * trained model's sidecar JSON (`train_image_uuids` / `val_image_uuids`, written
 * by every trainer via `image_manifest_fields`). So a previous run is resolved by
 * finding this config's model and reading that manifest back, then replaying the
 * config's own label-stats query over exactly those images so "then" and "now"
 * are produced by the same counting rules.
 *
 * Per-label counts are therefore a *current* reading of the images that trained
 * the model, not a snapshot frozen at training time: if those images were
 * re-labelled since, the previous-run column moves with them. The image counts
 * themselves are exact.
 */

import { fetchModelTrainingImages } from '/app-static/js/shared/api/models.js';
import {
  fetchTrainingConfigLabelStatsForImages,
  fetchTrainingConfigReadinessForImages,
} from '/app-static/js/shared/api/trainingConfigs.js';
import { parseApiDate } from '/app-static/js/shared/time.js';

/**
 * The model name a training config trains into.
 *
 * Mirrors TrainModelTask._resolve_training_config exactly — hyperparams
 * `model_name` if set, else the config name slugified. Keep the two in sync: if
 * they drift, this silently reads some *other* model's manifest, so the check in
 * loadPreviousRun (node uuid must match) is what stops a wrong answer.
 *
 * @param {{name: string, hyperparamsJson?: object}} config
 */
export function modelNameForConfig(config) {
  const explicit = config?.hyperparamsJson?.model_name;
  if (explicit) return String(explicit);
  return String(config?.name ?? '').replace(/ /g, '_').toLowerCase();
}

/**
 * Resolve the previous training run for `config`, or null when there isn't one
 * (never trained, model deleted, manifest recording turned off, or the model
 * found under this name belongs to a different model-shape node).
 *
 * Never throws: a missing previous run is the normal case for a new config, and
 * it must not take the readiness panel down with it.
 *
 * @param {{uuid: string, name: string, modelConfigNodeUuid?: string|null, hyperparamsJson?: object}} config
 * @param {object} [opts]
 * @param {boolean} [opts.withLabelStats=true] - replay the config's label-stats
 *   query over the manifest. That query is the expensive half of this function
 *   (measured 0.05–0.7s per config against a ~1200-image manifest), and callers
 *   that only want the image totals — the Jobs launcher cards, which show a
 *   single "+N since last run" — should pass false. With false,
 *   `labelStatsByUuid` is empty and `stillInDatasetCount` is 0.
 * @param {boolean} [opts.withReadinessSplit=false] - also read the manifest's
 *   positive/background split, so a caller can say whether the images gained
 *   since the last run carry training signal or are all background. One extra
 *   round trip; off by default because most callers only show image totals.
 *   With false, `baselinePositiveCount`/`baselineNegativeCount` are null.
 * @returns {Promise<null | {
 *   modelName: string,
 *   imageUuids: string[],
 *   trainedAt: string|null,
 *   trainCount: number,
 *   valCount: number,
 *   manifestCount: number,
 *   stillInDatasetCount: number,
 *   labelStatsByUuid: Map<string, object>,
 * }>}
 */
export async function loadPreviousRun(config, { withLabelStats = true, withReadinessSplit = false } = {}) {
  if (!config?.uuid) return null;
  const modelName = modelNameForConfig(config);
  if (!modelName) return null;

  let manifest;
  try {
    manifest = await fetchModelTrainingImages(modelName);
  } catch {
    // No such model — this config has not produced one yet.
    return null;
  }

  // A model whose sidecar names a different shape node is not this config's
  // previous run, it just happens to collide on name. Older sidecars predate the
  // field entirely (null) — those are accepted rather than hidden, since the name
  // match is the only evidence available for them.
  if (
    manifest.modelConfigNodeUuid &&
    config.modelConfigNodeUuid &&
    manifest.modelConfigNodeUuid !== config.modelConfigNodeUuid
  ) {
    return null;
  }

  const imageUuids = [...manifest.train, ...manifest.val];
  if (imageUuids.length === 0) {
    // The model exists but recorded no manifest (record_image_manifest off, or a
    // model imported rather than trained here) — there is nothing to compare.
    return null;
  }

  // The positive/background split over the manifest, read through the SAME
  // endpoint the "ready now" numbers come from — so a re-labelled image moves
  // both sides rather than appearing as growth on one of them.
  let baselinePositiveCount = null;
  let baselineNegativeCount = null;
  if (withReadinessSplit) {
    try {
      const before = await fetchTrainingConfigReadinessForImages(config.uuid, imageUuids);
      baselinePositiveCount = before.readyPositiveCount ?? 0;
      baselineNegativeCount = before.readyNegativeCount ?? 0;
    } catch {
      // null, not 0: a baseline we could not read must render as "unknown", and
      // 0 would report the entire current positive count as newly gained.
      baselinePositiveCount = null;
      baselineNegativeCount = null;
    }
  }

  let labelStats = [];
  let stillInDatasetCount = 0;
  if (withLabelStats) {
    try {
      const stats = await fetchTrainingConfigLabelStatsForImages(config.uuid, imageUuids);
      labelStats = stats.labelStats;
      stillInDatasetCount = stats.totalImages;
    } catch {
      // Counts we can't compute are simply absent — the image totals below are
      // still worth showing on their own.
      labelStats = [];
      stillInDatasetCount = 0;
    }
  }

  return {
    modelName,
    imageUuids,
    trainedAt: manifest.trainedAt,
    trainCount: manifest.train.length,
    valCount: manifest.val.length,
    manifestCount: imageUuids.length,
    stillInDatasetCount,
    baselinePositiveCount,
    baselineNegativeCount,
    labelStatsByUuid: new Map(labelStats.map(s => [s.labelUuid, s])),
  };
}

/**
 * Format an ISO timestamp for the "trained <when>" line, tolerating junk.
 * @param {string|null} iso
 */
export function formatTrainedAt(iso) {
  const d = parseApiDate(iso);
  if (!d) return null;
  return d.toLocaleString();
}
