import { setCanvasImage, rerenderBboxViews } from './canvas/manager.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { createCanvasState } from './canvas/state.js';
import { Logger } from '/app-static/js/ui/utils/index.js';
import { generateUUID } from '/app-static/js/ui/utils/utils.js';

/** @param {Map<string, import('@image_tagging_types').UIBbox>} bboxes */
function _cloneBboxes(bboxes) {
  return new Map([...bboxes].map(([k, v]) => [k, {
    ...v,
    selected: false,  // selection is driven by the transformer, not persisted in snapshots
    label: v.label ? { ...v.label } : null,
    tags: v.tags.map(t => ({ ...t })),
  }]));
}

/**
 * @returns {import('@image_tagging_types').ImageTaggingRuntime}
 */
export function createImageTaggingRuntime() {
  const _id = generateUUID();

  /** @type {import('@image_tagging_types').ImageTaggingState|null} */
  let _imgTaggingState = { image: null, currentTool: null, currentLabelUuid: null, selectedBoxUuid: null };
  /** @type {import('@image_tagging_types').CanvasState|null} */
  let _canvasState = null;
  /** @type {import('@ui_types').FileNavigation|null} */
  let _fileNavigation = null;
  /** @type {string} */
  let _currentTool = 'select';
  /** @type {string|null} */
  let _currentLabelUuid = null;
  /** @type {HTMLElement | null} */
  let _pane = null;
  /** @type {ViewportTransform|null} */
  let viewport = null;

  // ── Undo / redo ──────────────────────────────────────────────────────────
  // Each entry is a deep-cloned snapshot of image.bboxes. The current state
  // is always at the top of _undoStack. Cap at 101 entries (100 undoable steps
  // + 1 baseline).
  /** @type {Array<Map<string, import('@image_tagging_types').UIBbox>>} */
  let _undoStack = [];
  /** @type {Array<Map<string, import('@image_tagging_types').UIBbox>>} */
  let _redoStack = [];

  /** Push a snapshot of the current bbox state. Clears the redo stack. */
  function pushState() {
    if (!_imgTaggingState?.image) return;
    _undoStack.push(_cloneBboxes(_imgTaggingState.image.bboxes));
    if (_undoStack.length > 101) _undoStack.shift();
    _redoStack = [];
  }

  function undo() {
    if (_undoStack.length <= 1 || !_imgTaggingState?.image || !_canvasState) return;
    _redoStack.push(_undoStack.pop());                       // current → redo
    const prev = _undoStack[_undoStack.length - 1];         // peek new top
    _imgTaggingState.image.bboxes = _cloneBboxes(prev);     // restore (clone keeps stack clean)
    rerenderBboxViews(_canvasState, _imgTaggingState.image.bboxes);
    events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: _id });
  }

  function redo() {
    if (!_redoStack.length || !_imgTaggingState?.image || !_canvasState) return;
    const next = _redoStack.pop();
    _undoStack.push(next);
    _imgTaggingState.image.bboxes = _cloneBboxes(next);
    rerenderBboxViews(_canvasState, _imgTaggingState.image.bboxes);
    events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: _id });
  }

  function clearHistory() {
    _undoStack = [];
    _redoStack = [];
  }
  // ─────────────────────────────────────────────────────────────────────────

  /** @param {HTMLElement | null} pane */
  function setImageTaggingPane(pane) {
    if (!pane) return;
    if (_pane) {
      Logger.warn("Replacing existing pane");
    }
    _canvasState?.resizeObserver?.disconnect();
    _pane = pane;
    const container = _pane.getElementsByClassName("canvas-container")[0];
    _canvasState = createCanvasState(container);
  }

  /**
   * Sets the pane reference only — no canvas creation.
   * Use this when the canvas will be (or already is) initialized via setCanvasContainer / createCanvasInstance.
   * @param {HTMLElement | null} pane
   */
  function setPaneRef(pane) {
    _pane = pane;
  }

  /** @param {HTMLElement} container - The canvas container element directly (no pane wrapper needed). */
  function setCanvasContainer(container) {
    _canvasState?.resizeObserver?.disconnect();
    _canvasState = createCanvasState(container);
  }

  /** Disconnect, destroy and null out the canvas state. Safe to call multiple times. */
  function destroyCanvas() {
    _canvasState?.resizeObserver?.disconnect();
    _canvasState?.stage?.destroy();
    _canvasState = null;
  }

  /** @param {import('@image_tagging_types').UIImage} image */
  function setImage(image) {
    if (!_canvasState) {
      Logger.warn("No canvas available to display image");
      return;
    }
    if (!_imgTaggingState) {
      Logger.warn("No image tagging state available to set image");
      return;
    }
    _imgTaggingState.image = image;
    setCanvasImage(_canvasState, _imgTaggingState.image);
    clearHistory();
    pushState();  // baseline — the "can't undo past this" floor
    events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: _id });
  }

  /** @param {import('@ui_types').FileNavigation|null} navigation */
  function setFileNavigation(navigation) {
    if (_fileNavigation) {
      Logger.warn("Replacing existing file navigation");
      // TODO: Cleanup?
    }
    _fileNavigation = navigation;
  }

  /** @param {string} tool */
  function setTool(tool) {
    _currentTool = tool;
  }

  function clearBboxes() {
    _imgTaggingState?.image?.bboxes.clear();
  }

  /** @param {string|null} uuid */
  function setLabelUuid(uuid) {
    _currentLabelUuid = uuid;
  }

  return {
    get id() { return _id; },
    get state() { return _imgTaggingState; },
    get canvas() { return _canvasState; },
    get imageTaggingPane() { return _pane; },
    get fileNavigation() { return _fileNavigation; },
    get tool() { return _currentTool; },
    get labelUuid() { return _currentLabelUuid; },
    get canUndo() { return _undoStack.length > 1; },
    get canRedo() { return _redoStack.length > 0; },

    setImageTaggingPane,
    setPaneRef,
    setCanvasContainer,
    destroyCanvas,
    setFileNavigation,
    setTool,
    setLabelUuid,
    setImage,
    clearBboxes,
    pushState,
    undo,
    redo,
    clearHistory,
  };
}