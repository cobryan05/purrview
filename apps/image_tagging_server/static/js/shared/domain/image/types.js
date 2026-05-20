// @ts-check

/**
 * @typedef {Object} UILabel
 * @property {string} uuid
 * @property {string} [text]
 * @property {string} [color]
 */

/**
 * @typedef {Object} UIBbox
 * @property {string} uuid
 * @property {UILabel} [label]
 * @property {string[]} tagUuids
 * @property {number} x
 * @property {number} y
 * @property {number} width
 * @property {number} height
 * @property {boolean} selected
 * @property {boolean} dirty
 */

/**
 * @typedef {Object} UIImage
 * @property {string} uuid
 * @property {string} image_path
 * @property {string} filename
 * @property {Map<string, UIBbox>} bboxes
 * @property {boolean} dirty
 *
 * @property {HTMLImageElement} img
 */

export {};
