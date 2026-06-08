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
  /** Fired after image data is fully loaded into runtime state. */
  IMAGE_LOADED_ONTO_CANVAS: "canvas:image:loaded",
  /** Fired when the bboxes collection changes (add / delete / relabel). */
  CANVAS_BBOXES_CHANGED: "canvas:bboxes:changed",
  /** @deprecated Use SHOW_MAIN_CANVAS */
  SHOW_CANVAS: "canvas:main:show",
  SHOW_MAIN_CANVAS: "canvas:main:show",
  RUN_INFERENCE_ON_CANVAS: "canvas:run-inference",
  TAGS_UPDATED: "tags:updated",
  DATASETS_UPDATED: "datasets:updated",
  ACTIVE_DATASET_CHANGED: "dataset:active:changed",
  /**
   * Fired when a task action requests the Review tab to open with specific filters.
   * Payload: { tagUuids: string[] }
   */
  OPEN_REVIEW_WITH_FILTERS: "review:open_with_filters",
  /**
   * Fired after canvas annotations (bboxes + image labels) have been successfully saved.
   * Payload: { runtimeId: string }
   */
  CANVAS_ANNOTATIONS_SAVED: "canvas:annotations:saved",
  /**
   * Fired when the active model config changes or the label-filter toggle changes.
   * Payload: { config: ModelConfigRead | null, filterEnabled: boolean }
   */
  ACTIVE_MODEL_CHANGED: "model:active:changed",
});
