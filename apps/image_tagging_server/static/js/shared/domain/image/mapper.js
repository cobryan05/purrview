// @ts-check
import { BboxView } from '/app-static/js/app/image-tagging/canvas/views/bboxView.js';
import { fetchLabelByUuid } from '/app-static/js/shared/api/labels.js';

/**
 * @typedef {import('./types.js').UIImage} UIImage
 * @typedef {import('./types.js').UIBbox} UIBBox
 */

/**
 * Convert a backend ImageRecord → UI Image model
 *
 * @param {import('@web_api').ImageRecordRead} imageRecord - API ImageRecord from backend
 * @param {HTMLImageElement} img - The loaded image element
 * @returns {UIImage}
 */
export function imageRecordToUI(imageRecord, img) {
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

  /** @type {import('./types.js').UIBbox} */
  const uiBbox = {
    uuid: bboxData.uuid,
    label,
    x: bboxData.x,
    y: bboxData.y,
    width: bboxData.width,
    height: bboxData.height,
    selected: false,
    dirty: false
  };

  return uiBbox;
}