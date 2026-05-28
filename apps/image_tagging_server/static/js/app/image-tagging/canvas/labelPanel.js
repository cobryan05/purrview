import { createLabelList } from '/app-static/js/ui/labelList.js';

/** Thin wrapper for call-site compatibility; prefer createLabelList from ui/labelList.js. */
export async function initLabelPanel(runtime, container) {
  return createLabelList(container, { mode: 'select', runtime });
}
