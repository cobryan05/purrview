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
 * @property {number} confThresh
 */



/**
 * Event: canvas:run-inference
 *
 * @typedef {Object} ShowCanvasPayload
 */


export { };

export const EventTypes = Object.freeze({
  /**
   * Fired after a main tab is activated (including the initial one).
   * Payload: { tabName: string }  e.g. { tabName: 'tab-configuration' }
   */
  MAIN_TAB_CHANGED: "tabs:main:changed",
  CANVAS_BBOX_DOUBLE_CLICKED: "canvas:bbox:double_clicked",
  CANVAS_BBOX_CLICKED: "canvas:bbox:clicked",
  /**
   * Fired whenever the set of selected bboxes changes, including when it
   * becomes empty. CANVAS_BBOX_CLICKED only ever announces a single box that
   * was just picked, so it cannot describe a multi-selection or a deselect.
   * Payload: { runtimeId: string, bboxIds: string[] }
   */
  CANVAS_SELECTION_CHANGED: "canvas:selection:changed",
  CANVAS_TOOL_CHANGED: "canvas:tool:changed",
  /**
   * Fired when a canvas's display-only image enhancement mode changes.
   * Payload: { runtimeId: string, mode: string }  (mode ids in canvas/enhance.js)
   */
  CANVAS_ENHANCE_CHANGED: "canvas:enhance:changed",
  LOAD_IMAGE_ONTO_CANVAS: "canvas:image:load",
  /**
   * "Open this image on the main canvas" from a tab that is not Annotate.
   *
   * LOAD_IMAGE_ONTO_CANVAS is canvas-scoped and its subscriber is registered by
   * Annotate's *lazy* init, so publishing it from Watch or Jobs before Annotate has
   * ever been opened drops it on the floor. This one is handled in app/main.js,
   * which focuses Annotate, waits for the canvas to exist, and then forwards.
   * Payload: { path: string }  (FILES_ROOT-relative)
   */
  OPEN_IMAGE_ON_MAIN_CANVAS: "canvas:main:open_image",
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
  /**
   * Fired when a task action requests the Review tab to open with specific filters.
   * Payload: { tagUuids: string[] }
   */
  OPEN_REVIEW_WITH_FILTERS: "review:open_with_filters",
  /**
   * "Show me the clip this frame came from, at the moment it came from."
   *
   * Published by the preview panel's harvested-frame cards, where a proposed box has
   * to be judged against something — a cropped still is the ambiguity that caused the
   * mistake in the first place, and only the moving picture settles it.
   *
   * Handled in app/main.js — the composition root — so Review does not have to import
   * NVR's modal directly. It deliberately does *not* switch tabs: the player is an
   * overlay, and taking someone out of a half-triaged run to answer one question about
   * one frame loses their place in it. They glance at the clip, close it, and are still
   * where they were.
   * Payload: { eventUuid: string, atS?: number }
   */
  OPEN_EVENT_CLIP: "nvr:event:open_clip",
  /**
   * "Browse recorded events for this camera." Published by the Live View modal's
   * toolbar button, so a diagnostic glance at the live feed can jump straight to that
   * source's event history without hunting for it in the filter bar.
   *
   * Handled in app/main.js, same reason as OPEN_EVENT_CLIP: NVR's Live view and its
   * Events page are siblings and neither should import the other's module directly.
   * Payload: { sourceUuid: string, sourceName?: string }
   */
  OPEN_NVR_EVENTS_FOR_SOURCE: "nvr:events:open_for_source",
  /**
   * Fired after canvas annotations (bboxes + image labels) have been successfully saved.
   * Payload: { runtimeId: string }
   */
  CANVAS_ANNOTATIONS_SAVED: "canvas:annotations:saved",
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
   * Fired when a tag-kind's UI visibility toggles.
   * Payload: { kind: string, visible: boolean }
   */
  TAG_KIND_VISIBILITY_CHANGED: "tags:kind:visibility_changed",
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
  /**
   * Fired after the set of available models changes — an import, a conversion, a
   * finished training run, or a delete.
   *
   * The model list is cached for the session (`shared/api/models.js`), which is
   * right for a list that is read on every panel build and changes a handful of
   * times a day. The cost was that a model you had just made did not exist as far
   * as any open picker was concerned until you reloaded the page: the canvas
   * inference panel is built once and kept, so its manual ↻ was the only way to
   * see a new model at all.
   *
   * Publish it through `notifyModelsChanged()` rather than directly — the cache
   * has to be cleared before subscribers refetch, or they all re-read the stale
   * list and nothing changes.
   *
   * Payload: none.
   */
  MODELS_UPDATED: "models:updated",
  /**
   * Fired after an image is permanently deleted from the DB and disk.
   * Payload: { imageUuid: string, imagePath: string }
   */
  IMAGE_DELETED: "image:deleted",
  /**
   * Fired after a whole-image label is set (present/absent/uncertain) via the
   * context menu or any other non-canvas path.
   * Payload: { imageUuid: string, imagePath: string }
   */
  IMAGE_LABELS_CHANGED: "image:labels:changed",
  /**
   * Fired when the active training config changes, a label preset (visibility or
   * Apply-to-all target) changes, or coverage stats are freshly fetched.
   * Payload: {
   *   config: TrainingConfigRead | null,
   *   visibilityPreset: 'all'|'training-run',
   *   targetPreset: 'all'|'training-run'|'object'|'image'|'dynamic'|'none'|'custom',
   *   statsUpdated: boolean,
   * }
   */
  ACTIVE_TRAINING_CONFIG_CHANGED: "training-config:active:changed",
});
