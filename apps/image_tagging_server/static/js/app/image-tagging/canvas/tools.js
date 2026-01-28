import { state } from './state.js';

// Clear selection implementation
export function clearSelection() {
  const transformer = state.canvas.transformer;
  if (transformer) {
    transformer.nodes([]);
    transformer.getLayer()?.draw();
  }
  state.setSelectedBboxUuid(null);
  state.canvas.stage.batchDraw();
}

// Set the current tool, update UI and cursor
export function setTool(tool) {
  state.setTool(tool);
  clearSelection();
  updateToolbarButtons();
  state.canvas.stage.container().style.cursor = tool === 'select' ? 'default' : 'crosshair';
}

// Get the current tool
export function getTool() {
  return state.selectedTool;
}

export function parseToolUuid(str) {
  const [tool, uuid] = str.split(':', 2);
  return { tool, uuid };
}

// Highlight toolbar buttons based on current tool
export function updateToolbarButtons() {
  const current = getTool();
  document.querySelectorAll('#toolbar button[data-tool]').forEach(btn => {
    btn.classList.toggle('selected', btn.dataset.tool === current);
  });
}

export function selectBboxTool() {
  // Switch sidebar to classes tab
  const classesTabButton = document.querySelector('[data-tab="tab-pane-classes-tool"]');
  if (classesTabButton) {
    classesTabButton.click();
  }

  // Click the active tool
  const selectedClassUuid = state.currentClassUuid;
  const classRows = document.querySelectorAll('#classes-tool-list .class-row');
  let found = false;
  if (selectedClassUuid) {
    for (const classEl of classRows) {
      if (classEl.dataset.uuid === selectedClassUuid) {
        found = true;
        classEl.click();
        break;
      }
    }
  }
  if (!found && classRows.length === 0) {
    setTimeout(() => {
      const newClassRows = document.querySelectorAll('#classes-tool-list .class-row');
      if (newClassRows.length > 0) {
        newClassRows[0].click();
      }
    }, 100);
  }
}
