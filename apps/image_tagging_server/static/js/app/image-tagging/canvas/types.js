/**
 * @typedef {import('/app-static/js/app/image-tagging/canvas/groups').CanvasBboxGroup} CanvasBboxGroup
 * @typedef {import('@konva').default.Stage} Stage
 * @typedef {import('@konva').default.Layer} Layer
 * @typedef {import('@konva').default.Transformer} Transformer
 * @typedef {import('@konva').default.Group} Group
 */

/**
 * @typedef {Object} CanvasState
 * @property {Stage|null} stage
 * @property {Layer|null} layer
 * @property {Transformer|null} transformer
 * @property {Map<string,CanvasBboxGroup>} bboxes
 */

/**
 * @typedef {Object} ToolState
 * @property {string} tool
 * @property {string|null} classUuid
 * @property {string|null} selectedBboxUuid
 */

/**
 * @typedef {Object} RuntimeBboxInfo
 * @property {string} uuid
 * @property {number} x
 * @property {number} y
 * @property {number} width
 * @property {number} height
 * @property {number} [confidence]
 * @property {string} [text]
 * @property {string|null} classUuid
 * @property {string[]} [tagUuids]
 */


/**
 * @typedef {Object} RuntimeImage
 * @property {string|null} name
 * @property {HTMLImageElement|null} img
 * @property {Map<string,RuntimeBboxInfo>|null} bboxes
 */

/**
 * @typedef {Object} ImageRuntimeState
 * @property {RuntimeImage|null} image
 */

export { };