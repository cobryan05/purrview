// @ts-check
import { Logger, generateUUID } from '/app-static/js/ui/utils/index.js';
import { BboxView } from '/app-static/js/app/image-tagging/canvas/views/bboxView.js';
import { fetchLabelByUuid } from '/app-static/js/shared/api/labels.js';
import { UNVERIFIED_TAG_UUID, VERIFIED_TAG_UUID, VERIFICATION_TAG_UUIDS } from '/app-static/js/shared/api/review.js';
import { readConfidence } from '/app-static/js/shared/domain/image/confidence.js';

/**
 * @typedef {import('./types.js').UIImage} UIImage
 * @typedef {import('./types.js').UIBbox} UIBbox
 */


/**
 * Create a new empty UI bbox
 *
 * @param {Partial<UIBbox>} overrides
 * @returns {UIBbox}
 */
export function createUIBBbox(overrides = {}) {
  return {
    uuid: generateUUID(),

    label: undefined,
    tags: [],

    x: 0,
    y: 0,
    width: 0,
    height: 0,

    selected: false,
    dirty: true,

    ...overrides,
  };
}

/**
 * Convert a backend ImageRecord → UI Image model
 *
 * @param {import('@web_api').ImageRecordRead} imageRecord - API ImageRecord from backend
 * @param {HTMLImageElement} img - The loaded image element
 * @param {string} imagePath - remote path to the image
 * @returns {UIImage}
 */
export function imageRecordToUI(imageRecord, img, imagePath) {
  /** @type {Map<string, UIBBox>} */
  const bboxMap = new Map();

  const bboxes = imageRecord.bboxes ?? [];

  for (const bbox of bboxes) {
    /** @type {UIBBox} */
    const uiBbox = bboxReadToUI(bbox)
    bboxMap.set(bbox.uuid, uiBbox);
  }

  return {
    uuid: imageRecord.uuid,
    filename: imageRecord.filename,
    image_path: imagePath,
    img: img,
    bboxes: bboxMap,
    dirty: false,
  };
}

/**
 * @param {import('@web_api').BBoxRead} bboxData
 * @returns {import('./types.js').UIBbox}
 */
export function bboxReadToUI(bboxData) {
  const labelUuid = bboxData.label?.uuid;
  const labelText = bboxData.label?.name || labelUuid || "Undefined";
  const labelColor = bboxData.label?.color;

  /** @type {import('./types.js').UILabel|undefined} */
  const label = labelUuid ? { uuid: labelUuid, text: labelText, color: labelColor } : undefined;

  const rawTags = (bboxData.tags ?? []).map(t => ({ uuid: t.uuid, icon: t.icon ?? null }));

  // Assign a default verification tag if the DB row has none.
  // Bboxes with a label default to VERIFIED; bboxes without a label default to UNVERIFIED.
  let tags;
  if (rawTags.some(t => VERIFICATION_TAG_UUIDS.has(t.uuid))) {
    tags = rawTags;
  } else if (label?.uuid) {
    tags = [...rawTags, { uuid: VERIFIED_TAG_UUID, icon: null }];
  } else {
    tags = [...rawTags, { uuid: UNVERIFIED_TAG_UUID, icon: null }];
  }

  // The model confidence this box was created with, so it survives a reload
  // instead of only existing on the freshly-inferred boxes in memory.
  // Surfaced only while still unverified: once a human has confirmed the box,
  // what the model originally thought is noise on the label.
  const isUnverified = tags.some(t => t.uuid === UNVERIFIED_TAG_UUID);
  const confidence = isUnverified ? readConfidence(bboxData.metadataJson) : undefined;

  /** @type {import('./types.js').UIBbox} */
  const uiBbox = createUIBBbox({
    uuid: bboxData.uuid,
    label,
    tags,
    x: bboxData.x,
    y: bboxData.y,
    width: bboxData.width,
    height: bboxData.height,
    selected: false,
    dirty: false,
    confidence,
    // Deliberately NOT carried onto the UI model. The save payload's `extra` is
    // typed Dict[str, str] server-side, so echoing real metadata back (which
    // holds booleans and floats — run_model_added, source_confidence) would
    // 422 the save. Metadata this client didn't author is preserved server-side
    // instead, by omitting `extra` entirely; see Manager.update_image_metadata.
  });

  return uiBbox;
}
