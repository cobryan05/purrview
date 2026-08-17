import { deleteSelected, saveAnnotations } from './image-tagging/canvas/image.js';
import { getCanvasAction } from './image-tagging/canvasActions.js';
import { resetCanvasZoom } from './image-tagging/canvas/manager.js';
import { cycleCanvasEnhancement } from './image-tagging/canvas/enhance.js';
import { setTool } from './image-tagging/canvas/tools.js';
import { next_image, prev_image, deleteCurrentImage } from './image-tagging/files.js';
import { actionForCombo, eventToCombo } from './image-tagging/hotkeyBindings.js';
import { isTextEntryElement } from '/app-static/js/ui/utils/focus.js';
import { openDialogCount } from '/app-static/js/ui/utils/modalZIndex.js';

/**
 * Each entry pairs a runtime with the dialog-open count captured when its
 * hotkeys were attached. A canvas modal is itself a dialog, so that baseline
 * already includes it — its hotkeys stay live until something opens *on top*
 * (dialogBaseline compared against openDialogCount() on every keydown, not a
 * plain "is anything open" check; see openDialogCount's doc comment).
 * @type {{ runtime: import('@image_tagging_types').ImageTaggingRuntime, dialogBaseline: number }[]}
 */
const _stack = [];

/**
 * Run a bound action against a runtime. Returns true if it was handled.
 * @param {string} action
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
function dispatchAction(action, runtime) {
  if (action.startsWith('label:')) {
    setTool(runtime, 'bbox:' + action.slice('label:'.length));
    return true;
  }
  if (action.startsWith('tag:')) {
    setTool(runtime, 'tag:' + action.slice('tag:'.length));
    return true;
  }
  switch (action) {
    case 'action:select': setTool(runtime, 'select'); return true;
    case 'action:bbox': setTool(runtime, 'bbox'); return true;
    // A runtime with no FileNavigation cannot navigate at all (single-candidate
    // editor modals, embedded read-only previews). prev_image/next_image no-op
    // on it — reporting that as "handled" would preventDefault() a key nothing
    // acted on, which is indistinguishable from the hotkeys being dead.
    case 'action:prevImage': if (!runtime.fileNavigation) return false; prev_image(runtime); return true;
    case 'action:nextImage': if (!runtime.fileNavigation) return false; next_image(runtime); return true;
    case 'action:delete': deleteSelected(runtime); return true;
    case 'action:deleteImage': deleteCurrentImage(runtime); return true;
    case 'action:undo': runtime.undo(); return true;
    case 'action:redo': runtime.redo(); return true;
    case 'action:resetZoom': resetCanvasZoom(runtime.canvas); return true;
    case 'action:enhance': cycleCanvasEnhancement(runtime); return true;
    // Save goes through whatever the surrounding editor means by "persist" when
    // it registered one (the review modal's Save button, which also handles its
    // own button states and broadcast), falling back to the plain DB save the
    // main canvas toolbar uses.
    case 'action:save': (getCanvasAction(runtime.id, 'save') ?? saveAnnotations)(runtime); return true;
    // Only handled where an inference panel is mounted — with no panel there is
    // no model selection to run, so report unhandled rather than preventDefault
    // a key that did nothing (same reasoning as prev/nextImage above).
    case 'action:runInference': {
      const run = getCanvasAction(runtime.id, 'runInference');
      if (!run) return false;
      run(runtime);
      return true;
    }
    default: return false;
  }
}

document.addEventListener('keydown', e => {
  const top = _stack[_stack.length - 1];
  if (!top) return;
  const { runtime, dialogBaseline } = top;

  // Only *text entry* focus swallows hotkeys. Checkboxes, radios and range
  // sliders — the inference panel is full of them — must not.
  if (isTextEntryElement(document.activeElement)) return;

  // A dialog full of buttons (e.g. the "unsaved changes?" confirm) never
  // becomes text-entry focus, so without this a hotkey like ']' kept
  // reaching the canvas underneath it — each press re-ran the guarded
  // navigation and stacked another copy of the same dialog on top. Compared
  // against this context's own baseline, not `> 0`: the canvas modal is
  // itself a dialog, so its own hotkeys must keep working while only it is
  // open — only a dialog stacked *on top of* it should block them.
  if (openDialogCount() > dialogBaseline) return;

  const action = actionForCombo(eventToCombo(e));
  if (!action) return;

  if (dispatchAction(action, runtime)) e.preventDefault();
});

/**
 * The runtime currently receiving hotkeys (top of the stack), or null.
 *
 * Exported for tests and for debugging "my hotkeys stopped working": that
 * symptom is almost always this returning a runtime other than the canvas the
 * user is looking at, because something attached after it. See attachHotkeys.
 *
 * @returns {import('@image_tagging_types').ImageTaggingRuntime|null}
 */
export function activeHotkeyRuntime() {
  return _stack.length ? _stack[_stack.length - 1].runtime : null;
}

/**
 * Registers keyboard shortcuts for a canvas runtime instance.
 * The most recently attached runtime receives all hotkeys, so modal
 * canvases naturally preempt the main canvas while open.
 *
 * ⚠️ "Most recent wins" only behaves for canvases the user is actually driving.
 * A canvas that appears without the user asking for it — an embedded preview
 * inside a polled status card, say — must NOT attach, or it silently takes
 * ownership of every hotkey away from whatever the user is working in, for as
 * long as it lives. Such canvases pass `hotkeys: false` to createCanvasInstance
 * and never reach this function.
 *
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @returns {{ destroy: () => void }}
 */
export function attachHotkeys(runtime) {
  const entry = { runtime, dialogBaseline: openDialogCount() };
  _stack.push(entry);
  return {
    destroy() {
      const idx = _stack.indexOf(entry);
      if (idx !== -1) _stack.splice(idx, 1);
    },
  };
}
