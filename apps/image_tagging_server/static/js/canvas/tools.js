import { getCurrentLabelUuid, getCurrentTool, getStage, getTransformer, setCurrentTool } from './state.js';

// Clear selection implementation
export function clearSelection() {
  const transformer = getTransformer();
  if (transformer) {
    transformer.nodes([]);
    transformer.getLayer()?.draw();
  }
  getStage().batchDraw();
}

// Set the current tool, update UI and cursor
export function setTool(tool) {
  setCurrentTool(tool);
  clearSelection();
  updateToolbarButtons();
  getStage().container().style.cursor = tool === 'select' ? 'default' : 'crosshair';
}

// Get the current tool
export function getTool() {
  return getCurrentTool();
}

export function parseToolUuid(str) {
  const [tool, uuid] = str.split(':', 2);
  return { tool, uuid };
}

// Highlight toolbar buttons based on current tool
export function updateToolbarButtons() {
  const current = getCurrentTool();
  document.querySelectorAll('#toolbar button[data-tool]').forEach(btn => {
    btn.classList.toggle('selected', btn.dataset.tool === current);
  });
}

export function selectBboxTool() {
  // Switch sidebar to Labels tab
  const labelsTabButton = document.querySelector('[data-tab="tab-labels-tool"]');
  if (labelsTabButton) {
    labelsTabButton.click();
  }

  // Click the active tool
  const selectedLabelUuid = getCurrentLabelUuid();
  const labelRows = document.querySelectorAll('#labels-tool-list .label-row');
  let found = false;
  if (selectedLabelUuid) {
    for (const labelEl of labelRows) {
      if (labelEl.dataset.uuid === selectedLabelUuid) {
        found = true;
        labelEl.click();
        break;
      }
    }
  }
  if (!found && labelRows.length === 0) {
    setTimeout(() => {
      const newLabelRows = document.querySelectorAll('#labels-tool-list .label-row');
      if (newLabelRows.length > 0) {
        newLabelRows[0].click();
      }
    }, 100);
  }
}
