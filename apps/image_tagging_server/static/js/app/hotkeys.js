import { deleteSelected } from './image-tagging/canvas/image.js';
import { setTool } from './image-tagging/canvas/tools.js';

/**
 * @param {import('@app_types').AppState} appState
 */
export function registerKeyboardShortcuts(appState) {
  document.addEventListener('keydown', e => {
    const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
    const ctrlKey = isMac ? e.metaKey : e.ctrlKey;

    // If typing into an input field, ignore
    const tag = document.activeElement?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;

    switch (e.key.toLowerCase()) {
      case 'delete':
      case 'backspace':
        e.preventDefault();
        deleteSelected(appState.imageTagging);
        break;

      case 'v':
        setTool(appState.imageTagging, 'select');
        break;

      case 'r':
        setTool(appState.imageTagging, 'bbox:');
        break;

      default:
        break;
    }
  });
}
