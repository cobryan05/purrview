// @ts-check

/**
 * @typedef {Object} UILabel
 * @property {string} uuid
 * @property {string} [text]
 * @property {string} [color]
 */

/**
 * A tag as used within the UI layer — carries uuid and optional icon.
 * @typedef {Object} UITag
 * @property {string} uuid
 * @property {string|null} [icon]
 */

/**
 * @typedef {Object} UIBbox
 * @property {string} uuid
 * @property {UILabel} [label]
 * @property {UITag[]} tags
 * @property {number} x
 * @property {number} y
 * @property {number} width
 * @property {number} height
 * @property {boolean} selected
 * @property {boolean} dirty
 * @property {number} [confidence] - Inference confidence (0-1), persisted until the bbox is verified.
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
