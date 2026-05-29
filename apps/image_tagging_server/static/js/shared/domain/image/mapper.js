// @ts-check
import { Logger, generateUUID } from '/app-static/js/ui/utils/index.js';
import { BboxView } from '/app-static/js/app/image-tagging/canvas/views/bboxView.js';
import { fetchLabelByUuid } from '/app-static/js/shared/api/labels.js';
import { UNVERIFIED_TAG_UUID, VERIFICATION_TAG_UUIDS } from '/app-static/js/shared/api/review.js';

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

  // Default to UNVERIFIED if the bbox has no explicit verification tag and no label.
  // A bbox with a label was human-reviewed (UNVERIFIED was stripped on assignment),
  // so we don't re-add it on load even if the DB row no longer carries any verification tag.
  const tags = (rawTags.some(t => VERIFICATION_TAG_UUIDS.has(t.uuid)) || label?.uuid)
    ? rawTags
    : [...rawTags, { uuid: UNVERIFIED_TAG_UUID, icon: null }];

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
    dirty: false
  });

  return uiBbox;
}
