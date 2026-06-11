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
  /**
   * Fired when the user wants to export review filters to a new dataset.
   * Payload: { conditions: object[], labelUuids: string[] }
   */
  EXPORT_FILTERS_TO_DATASET: "dataset:export_from_review",
  /**
   * Fired when the "Show dynamic labels" toggle changes.
   * Payload: boolean (true = show dynamic labels)
   */
  DYNAMIC_LABELS_VISIBILITY_CHANGED: "labels:dynamic:visibility_changed",
  /**
   * Fired after canvas annotations are saved from any modal (including nested).
   * Payload: { imageUuid: string, imageFilename: string, bboxes: Map<string, UIBbox> }
   */
  IMAGE_ANNOTATIONS_SAVED: "image:annotations:saved",
  /**
   * Fired when bbox(es) on an image are modified AND the change has been
   * committed to the database (e.g., context-menu label/tag actions or bulk
   * operations in a review panel).
   *
   * IMPORTANT: NEVER publish this event for in-memory-only canvas changes.
   * In-canvas edits (inspector, draw, resize) must use CANVAS_BBOXES_CHANGED
   * (scoped to the runtime) until the user clicks "Save & Close", at which
   * point IMAGE_ANNOTATIONS_SAVED is fired instead.
   *
   * All open panels and canvas modals showing bboxes from this image should
   * react — review panels refresh cards, canvas modals check for conflicts.
   * Payload: {
   *   imageUuid: string,
   *   imageFilename: string,
   *   updatedBboxes: Map<string, object>,   // bboxUuid → updated bbox (label + tags updated)
   *   deletedBboxUuids: string[]
   * }
   */
  IMAGE_BBOXES_CHANGED: "image:bboxes:changed",
  /**
   * Fired after any label is created, updated, or deleted.
   * Consumers that display a label list should re-fetch and re-render.
   */
  LABELS_UPDATED: "labels:updated",
});
