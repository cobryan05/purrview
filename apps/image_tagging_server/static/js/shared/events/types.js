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
 */

/**
 * Event: canvas:run-inference
 *
 * @typedef {Object} CanvasRunInferencePayload
 * @property {string} modelName
 */


export { };

export const EventTypes = Object.freeze({
  CANVAS_BBOX_DOUBLE_CLICKED: "canvas:bbox:double_clicked",
  CANVAS_BBOX_CLICKED: "canvas:bbox:clicked",
  RUN_INFERENCE_ON_CANVAS: "canvas:run-inference",
});
