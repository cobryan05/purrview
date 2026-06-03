import { createLabelList } from '/app-static/js/ui/labelList.js';
import { showPaneError } from '/app-static/js/ui/utils/index.js';

/**
 * Legacy wrapper for the configuration page — prefer createLabelList from labelList.js.
 * Re-calling re-mounts the component (full DOM rebuild).
 * @param {{ target?: string }} options
 */
export async function renderLabelList({ target = 'labels-list' } = {}) {
  const el = document.getElementById(target);
  if (!el) { console.error(`Label container #${target} not found`); return; }
  try {
    await createLabelList(el, { mode: 'edit' });
  } catch (err) {
    showPaneError(el, err.message);
  }
}
