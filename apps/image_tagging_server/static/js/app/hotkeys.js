import { deleteSelected } from './image-tagging/canvas/image.js';
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
    case 'action:prevImage': prev_image(runtime); return true;
    case 'action:nextImage': next_image(runtime); return true;
    case 'action:delete': deleteSelected(runtime); return true;
    case 'action:deleteImage': deleteCurrentImage(runtime); return true;
    case 'action:undo': runtime.undo(); return true;
    case 'action:redo': runtime.redo(); return true;
    case 'action:resetZoom': resetCanvasZoom(runtime.canvas); return true;
    case 'action:enhance': cycleCanvasEnhancement(runtime); return true;
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
 * Registers keyboard shortcuts for a canvas runtime instance.
 * The most recently attached runtime receives all hotkeys, so modal
 * canvases naturally preempt the main canvas while open.
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
