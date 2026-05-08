/**
 * @typedef {import('/app-static/js/app/image-tagging/canvas/views/bboxView.js').BboxView} BboxView
 * @typedef {import('@konva').default.Stage} Stage
 * @typedef {import('@konva').default.Layer} Layer
 * @typedef {import('@konva').default.Transformer} Transformer
 * @typedef {import('@konva').default.Group} Group
 */

// /**
//  * @typedef {Object} CanvasState
//  * @property {HTMLElement} container
//  * @property {Stage|null} stage
//  * @property {Layer|null} layer
//  * @property {Transformer|null} transformer
//  */

/**
 * @typedef {Object} ToolState
 * @property {string} tool
 * @property {string|null} labelUuid
 * @property {string|null} selectedBboxUuid
 */

/**
 * @typedef {Object} RuntimeImage
 * @property {string|null} name
 * @property {HTMLImageElement|null} img
 * @property {Map<string,import('@web_api').BoundingBoxMetadataModel>|null} bboxes
 */

/**
 * @typedef {Object} ImageRuntimeState
 * @property {RuntimeImage|null} image
 */

export { };