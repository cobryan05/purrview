import { deleteSelected } from './canvas/io.js';
import { setTool } from './canvas/tools.js';
import { debug } from './canvas/utils.js';

export function registerKeyboardShortcuts() {
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
        deleteSelected();
        break;

      case 'v':
        setTool('select');
        break;

      case 'r':
        setTool('rect');
        break;

      default:
        break;
    }
  });
}
