import { setCanvasImage, rerenderBboxViews } from './canvas/manager.js';
import { syncBboxDraggable } from './canvas/tools.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { createCanvasState } from './canvas/state.js';
import { Logger } from '/app-static/js/ui/utils/index.js';
import { generateUUID } from '/app-static/js/ui/utils/utils.js';

/**
 * Canonical fingerprint of the annotation state of a bbox map.
 *
 * Covers exactly the fields `updateImage` persists — geometry, label, tags and
 * metadata — so two states with the same fingerprint save identically. Sorted
 * by uuid, with tag lists and metadata keys sorted too, so pure reordering is
 * not mistaken for an edit.
 *
 * ⚠️ If a new persisted bbox field is added, it MUST be added here. A field the
 * fingerprint cannot see is a change that reads as "no changes" — the canvas
 * would close without prompting and the edit would be lost.
 *
 * Unlabeled boxes are included even though `updateImage` filters them out of
 * the payload: drawing one is still work the user would be surprised to lose
 * silently.
 *
 * @param {Map<string, import('@image_tagging_types').UIBbox>|null|undefined} bboxes
 * @returns {string}
 */
function _annotationSignature(bboxes) {
  if (!bboxes) return '';
  const rows = [...bboxes.values()].map(b => [
    b.uuid,
    b.label?.uuid ?? null,
    _roundCoord(b.x), _roundCoord(b.y), _roundCoord(b.width), _roundCoord(b.height),
    [...(b.tags ?? []).map(t => t.uuid)].sort(),
    _canonicalMetadata(b.metadataJson),
  ]);
  rows.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  return JSON.stringify(rows);
}

/**
 * Coordinates are compared at 1e-9, not exactly.
 *
 * Dragging a box away and back returns the Konva node to the same pixel, but
 * the normalized value is recomputed as `(nodeX - offsetX) / scaledWidth` each
 * time, so it comes back as e.g. 0.10000000000000005 rather than 0.1. Compared
 * exactly, a box put back where it started would read as modified forever.
 *
 * 1e-9 of a normalized coordinate is ~1/125000 of a pixel on an 8K-wide image —
 * far below any change a user can make, and far above the ~1e-17 drift.
 */
function _roundCoord(value) {
  return typeof value === 'number' ? Math.round(value * 1e9) / 1e9 : value;
}

/** Metadata as sorted key/value pairs; empty and absent are the same thing
 *  (updateImage omits `extra` entirely when there are no keys). */
function _canonicalMetadata(metadataJson) {
  const keys = Object.keys(metadataJson ?? {}).sort();
  return keys.length ? keys.map(k => [k, metadataJson[k]]) : null;
}

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

  // ── Unsaved-changes tracking ─────────────────────────────────────────────
  // Dirty is DERIVED: the current annotations are compared against a
  // fingerprint of the last loaded or saved state. It used to be a latch that
  // every mutation set and only a save cleared, which meant an edit that was
  // undone still counted as unsaved — the canvas prompted on close and offered
  // to save a state identical to the one on the server.
  //
  // Deriving it also covers a manual revert (drag a box away and back), not
  // just undo, since neither ends up differing from what was loaded.
  /** Fingerprint of the last loaded/saved annotations; null = no image yet. */
  let _cleanSignature = null;
  /**
   * Sticky flag for pending changes that do NOT live in `bboxes` — image-level
   * label and tag presence, set via markDirty by the label list and inspector.
   * No bbox comparison can see those, so they stay dirty until saved.
   */
  let _forcedDirty = false;
  /** Bumped by every markDirty, so a save can tell "I cleared what I sent" from
   *  "a label was toggled after my payload left". */
  let _dirtyEpoch = 0;

  /** Re-sync which bboxes are draggable for the current tool/canvas. */
  function _syncDraggable() {
    syncBboxDraggable({ canvas: _canvasState, tool: _currentTool });
  }

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
    _syncDraggable();
    events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: _id });
  }

  function redo() {
    if (!_redoStack.length || !_imgTaggingState?.image || !_canvasState) return;
    const next = _redoStack.pop();
    _undoStack.push(next);
    _imgTaggingState.image.bboxes = _cloneBboxes(next);
    rerenderBboxViews(_canvasState, _imgTaggingState.image.bboxes);
    _syncDraggable();
    events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: _id });
  }

  /**
   * Force the dirty state on, for pending changes the bbox comparison cannot
   * see (image-level label / tag presence). Sticky until the next save or image
   * load — unlike a bbox edit, there is nothing to compare it back against.
   */
  function markDirty() {
    _forcedDirty = true;
    _dirtyEpoch++;
    events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: _id });
  }

  /**
   * Snapshot what a save is about to send, for `markClean` to adopt later.
   *
   * A save against a busy database takes seconds, and the user keeps drawing
   * through it. Without this the response handler fingerprinted whatever the
   * canvas held *then* — so a box drawn mid-flight was recorded as saved, Save
   * went disabled, and the edit was never written.
   *
   * @returns {{signature: string, dirtyEpoch: number}}
   */
  function beginSave() {
    return {
      signature: _annotationSignature(_imgTaggingState?.image?.bboxes),
      dirtyEpoch: _dirtyEpoch,
    };
  }

  /**
   * Adopt a saved state as the clean baseline (call after a successful save).
   *
   * @param {?{signature: string, dirtyEpoch: number}} [token] - from `beginSave`,
   *   identifying what was actually persisted. Omitted by callers with nothing
   *   in flight (a freshly loaded image), which take the current state as clean.
   */
  function markClean(token = null) {
    _cleanSignature = token ? token.signature : _annotationSignature(_imgTaggingState?.image?.bboxes);
    // Image-level label/tag edits have nothing to diff against, so the sticky
    // flag can only be cleared when none arrived after the payload went out.
    if (!token || token.dirtyEpoch === _dirtyEpoch) _forcedDirty = false;
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
    _syncDraggable();
    clearHistory();
    pushState();  // baseline — the "can't undo past this" floor
    markClean();  // a freshly loaded image starts clean, and sets the comparison baseline
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
    /**
     * True when the annotations differ from the last loaded/saved state, or a
     * non-bbox change is pending. Computed on read rather than latched, so an
     * edit that is undone (or manually reverted) reports clean again. Cheap
     * enough to compute here: every caller reads it at event time — button
     * state, close guards — never in a render loop.
     */
    get isDirty() {
      if (_forcedDirty) return true;
      if (_cleanSignature === null) return false;  // no image loaded yet
      return _annotationSignature(_imgTaggingState?.image?.bboxes) !== _cleanSignature;
    },
    /** The baseline bbox snapshot taken at image load — used for conflict detection. */
    get undoBaseline() { return _undoStack.length > 0 ? _undoStack[0] : null; },

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
    markDirty,
    beginSave,
    markClean,
    clearHistory,
  };
}

// ── Primary runtime id accessor ───────────────────────────────────────────────
// Used by the Review context-menu adapter to address the main canvas.

let _primaryRuntimeId = null;

/** Record the id of the app's main ImageTaggingRuntime. */
export function setPrimaryRuntimeId(id) { _primaryRuntimeId = id; }

/** Retrieve the id of the app's main ImageTaggingRuntime. */
export function getPrimaryRuntimeId() { return _primaryRuntimeId; }
