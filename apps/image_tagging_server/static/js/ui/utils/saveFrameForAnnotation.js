/**
 * "Save this frame for annotation" — one frame, from wherever you are looking at it,
 * into the dataset as an ordinary annotatable image.
 *
 * Deliberately transport-agnostic: it takes a Blob and boxes in pixel coordinates, so
 * the same function serves the NVR live view (a JPEG the watcher encoded) and, later,
 * a paused clip frame captured off a <video> into a canvas. Neither caller needs to
 * know how images are registered.
 *
 * There is no new backend behind this. `/api/images/upload` already writes into
 * FILES_ROOT *and registers the row*, which is what makes the frame immediately
 * findable in Review rather than invisible until something annotates it, and
 * `/api/images/metadata/update` already writes bboxes. This composes the two.
 */
import { uploadImages } from '/app-static/js/shared/api/uploads.js';
import { updateImage } from '/app-static/js/shared/api/images.js';
import { UNVERIFIED_TAG_UUID } from '/app-static/js/shared/api/review.js';
// Not crypto.randomUUID directly: it is absent outside a secure context, and the NVR
// live view is routinely opened over plain http on a LAN address.
import { generateUUID } from '/app-static/js/ui/utils/utils.js';

/**
 * Where captured frames land by default, under FILES_ROOT.
 *
 * Its own directory rather than the user's configured upload target: an upload is a
 * deliberate import of curated files, whereas these arrive one keypress at a time
 * while watching a camera, and mixing them into a hand-managed folder is a mess the
 * user never asked for.
 */
export const DEFAULT_CAPTURE_DIR = 'watch_captures';

/**
 * Filesystem-safe stem built from a source name and the capture time.
 *
 * Second resolution plus the server's own collision handling (upload uniquifies
 * rather than clobbers) is enough — two captures in the same second are the only
 * collision, and the second one is renamed, not lost.
 *
 * @param {string} sourceName
 * @param {Date} [when]
 * @returns {string}
 */
export function captureFilename(sourceName, when = new Date()) {
  const slug = String(sourceName || 'source')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'source';
  const stamp = when.toISOString().replace(/[:.]/g, '-').replace('Z', '');
  return `${slug}_${stamp}.jpg`;
}

/**
 * Convert a pixel-space corner box to the normalised x/y/width/height the DB stores.
 *
 * Clamping is not defensive tidying: bbox columns are validated to [0, 1] server-side,
 * and a detection that runs off the edge of the frame — routine for a cat walking out
 * of shot — would otherwise 422 the *whole* save, losing the frame over a box.
 *
 * @param {[number, number, number, number]} bbox - x1, y1, x2, y2 in pixels
 * @param {number} frameWidth
 * @param {number} frameHeight
 * @returns {?{x: number, y: number, width: number, height: number}} null if degenerate
 */
export function normaliseBox(bbox, frameWidth, frameHeight) {
  if (!frameWidth || !frameHeight) return null;
  const clamp = v => Math.min(1, Math.max(0, v));
  const [x1, y1, x2, y2] = bbox;
  const left = clamp(Math.min(x1, x2) / frameWidth);
  const top = clamp(Math.min(y1, y2) / frameHeight);
  const right = clamp(Math.max(x1, x2) / frameWidth);
  const bottom = clamp(Math.max(y1, y2) / frameHeight);
  const width = right - left;
  const height = bottom - top;
  // A box entirely outside the frame clamps to zero area; saving it would put an
  // invisible degenerate rectangle on the image for someone to puzzle over.
  if (width <= 0 || height <= 0) return null;
  return { x: left, y: top, width, height };
}

/**
 * Save a frame as an annotatable image, optionally seeding it with detected boxes.
 *
 * Boxes are written with the Unverified tag, exactly as the collector task writes
 * them: a model's guess is a starting point for annotation, never a ground truth, and
 * the review workflow is built around that tag.
 *
 * A box whose label is unknown to the DB is skipped rather than failing the save —
 * the frame itself is the thing worth keeping, and an unlabelled detection is not
 * worth losing it over.
 *
 * @param {Object} opts
 * @param {Blob} opts.blob - the encoded frame (JPEG)
 * @param {string} opts.filename
 * @param {string} [opts.path] - FILES_ROOT-relative destination directory
 * @param {Array<{labelUuid: ?string, bbox: [number, number, number, number], confidence?: number}>} [opts.boxes]
 * @param {number} [opts.frameWidth] - required if boxes are given
 * @param {number} [opts.frameHeight] - required if boxes are given
 * @param {Object} [opts.extra] - per-box metadata merged into metadata_json (provenance)
 * @returns {Promise<{path: string, uuid: ?string, boxesSaved: number, renamedFrom: ?string}>}
 * @throws {Error} if the upload itself fails
 */
export async function saveFrameForAnnotation({
  blob,
  filename,
  path = DEFAULT_CAPTURE_DIR,
  boxes = [],
  frameWidth = 0,
  frameHeight = 0,
  extra = {},
}) {
  const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });
  const [result] = await uploadImages({ files: [file], path });
  if (!result?.ok) throw new Error(result?.message || 'Could not save the frame');

  const savedPath = result.path;
  const bboxes = new Map();
  for (const box of boxes) {
    if (!box?.labelUuid) continue;
    const rect = normaliseBox(box.bbox, frameWidth, frameHeight);
    if (!rect) continue;
    const uuid = generateUUID();
    bboxes.set(uuid, {
      uuid,
      ...rect,
      label: { uuid: box.labelUuid },
      tags: [{ uuid: UNVERIFIED_TAG_UUID }],
      // Confidence is stringified to match what the collector writes, so the
      // confidence filter reads both producers' boxes the same way.
      metadataJson: { ...extra, ...(box.confidence != null ? { confidence: String(box.confidence) } : {}) },
    });
  }

  if (bboxes.size) {
    // The image is already saved and registered at this point. A metadata failure
    // costs the boxes, not the frame, so it is reported by the caller rather than
    // unwinding the upload.
    await updateImage({ image_path: savedPath, bboxes });
  }

  return {
    path: savedPath,
    uuid: result.uuid ?? null,
    boxesSaved: bboxes.size,
    renamedFrom: result.renamedFrom ?? null,
  };
}
