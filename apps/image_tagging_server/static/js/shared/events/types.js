/**
 * Event: canvas:bbox:clicked
 *
 * @typedef {Object} CanvasBboxClickedPayload
 * @property {string} bboxId
 */

/**
 * Event: canvas:bbox:double_clicked
 *
 * @typedef {Object} CanvasBboxDoubleClickedPayload
 * @property {string} bboxId
 * @property {import('@canvas_types').BboxView} bboxView - The live BboxView instance, used by the inspector to read/update bbox state without going through appState
 */

/**
 * Event: canvas:tool:chnaged
 *
 * @typedef {Object} CanvasToolChangedPayload
 * @property {string} toolName
 */



/**
 * Event: canvas:image:load
 *
 * @typedef {Object} CanvasImageLoadPayload
 * @property {string} path
 * @property {boolean} showCanvas
 */


/**
 * Event: canvas:run-inference
 *
 * @typedef {Object} CanvasRunInferencePayload
 * @property {string} modelName
 */



/**
 * Event: canvas:run-inference
 *
 * @typedef {Object} ShowCanvasPayload
 */


export { };

export const EventTypes = Object.freeze({
  CANVAS_BBOX_DOUBLE_CLICKED: "canvas:bbox:double_clicked",
  CANVAS_BBOX_CLICKED: "canvas:bbox:clicked",
  CANVAS_TOOL_CHANGED: "canvas:tool:changed",
  LOAD_IMAGE_ONTO_CANVAS: "canvas:image:load",
  SHOW_CANVAS: "canvas:show",
  RUN_INFERENCE_ON_CANVAS: "canvas:run-inference",
});
