import { deleteSelected } from './image-tagging/canvas/image.js';
import { setTool } from './image-tagging/canvas/tools.js';
import { next_image, prev_image } from './image-tagging/files.js';
import { actionForCombo, eventToCombo } from './image-tagging/hotkeyBindings.js';

/** @type {import('@image_tagging_types').ImageTaggingRuntime[]} */
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
    case 'action:undo': runtime.undo(); return true;
    case 'action:redo': runtime.redo(); return true;
    default: return false;
  }
}

document.addEventListener('keydown', e => {
  const runtime = _stack[_stack.length - 1];
  if (!runtime) return;

  const tag = document.activeElement?.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA') return;

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
  _stack.push(runtime);
  return {
    destroy() {
      const idx = _stack.lastIndexOf(runtime);
      if (idx !== -1) _stack.splice(idx, 1);
    },
  };
}
