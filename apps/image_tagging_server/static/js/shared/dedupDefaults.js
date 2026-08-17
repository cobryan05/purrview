// Deduplication defaults — the values used when the matching `dedup_*` DB
// setting is absent (fresh install, or before anyone has opened
// Settings → Deduplication).
//
// Mirrored from `apps/helpers/hashUtils.py`; keep the two in sync. A frontend
// that assumes a different hash size than the backend built its index at does
// not fail loudly — it computes a phash of the wrong width, and comparisons
// against the stored index yield plausible but meaningless Hamming distances.
//
// Thresholds are Hamming distances over a hashSize² -bit hash, so a threshold
// only means anything alongside the size it was chosen for. They are stated
// outright rather than derived as a percentage of the hash range: image dedup
// wants near-identical frames only, while bbox similarity search is
// deliberately looser, and no single ratio expresses both.
export const DEFAULT_IMAGE_HASH_SIZE = 16;
export const DEFAULT_BBOX_HASH_SIZE = 16;
export const DEFAULT_IMAGE_THRESHOLD = 5;
export const DEFAULT_BBOX_THRESHOLD = 75;

/** Hamming-distance ceiling for a given hash grid size: hashSize² bits. */
export function hashMaxFor(hashSize) {
  return hashSize * hashSize;
}

/**
 * The default threshold for a kind, rescaled if the hash size differs from the
 * one its default was chosen for.
 *
 * Used when the user changes hash size in Settings: the stored threshold was
 * picked against the old bit-width and is meaningless at the new one (a
 * distance of 75 is loose over 256 bits and impossible over 64). Scaling by the
 * ratio keeps the *selectivity* the default expresses rather than the number.
 *
 * @param {'image'|'bbox'} kind
 * @param {number} hashSize
 * @returns {number} at least 1
 */
export function defaultThresholdFor(kind, hashSize) {
  const isImage = kind === 'image';
  const base = isImage ? DEFAULT_IMAGE_THRESHOLD : DEFAULT_BBOX_THRESHOLD;
  const baseMax = hashMaxFor(isImage ? DEFAULT_IMAGE_HASH_SIZE : DEFAULT_BBOX_HASH_SIZE);
  const scaled = Math.round(base * (hashMaxFor(hashSize) / baseMax));
  return Math.max(1, scaled);
}
