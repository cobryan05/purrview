import { deleteSelected } from './image-tagging/canvas/image.js';
import { setTool } from './image-tagging/canvas/tools.js';
import { next_image, prev_image } from './image-tagging/files.js';

/** @type {import('@image_tagging_types').ImageTaggingRuntime[]} */
const _stack = [];

document.addEventListener('keydown', e => {
  const runtime = _stack[_stack.length - 1];
  if (!runtime) return;

  const tag = document.activeElement?.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA') return;

  switch (e.key.toLowerCase()) {
    case 'delete':
    case 'backspace':
      e.preventDefault();
      deleteSelected(runtime);
      break;

    case 'z':
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        e.shiftKey ? runtime.redo() : runtime.undo();
      }
      break;

    case 'v':
      setTool(runtime, 'select');
      break;

    case 'r':
      setTool(runtime, 'bbox:');
      break;

    case '[':
      prev_image(runtime);
      break;

    case ']':
      next_image(runtime);
      break;

    default:
      break;
  }
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
