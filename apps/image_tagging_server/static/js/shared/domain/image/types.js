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
 * @property {string} filename
 * @property {Map<string, UIBbox>} bboxes
 * @property {string|null} selectedBBoxId
 * @property {boolean} dirty
 *
 * @property {HTMLImageElement} img
 */

export {};
