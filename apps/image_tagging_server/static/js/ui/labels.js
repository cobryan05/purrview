// labels.js
import { createButton, createGenericRow } from './utils.js';
import { getCurrentLabelUuid, getLabelList, refreshLabelList } from '/app-static/js/canvas/state.js';
import { getTool } from '/app-static/js/canvas/tools.js';
import { toast } from '/app-static/js/canvas/utils.js';
function createInputRow({
  defaultName = '',
  colorSwatchColor = '#cccccc',
  onSave,
  onCancel,
  editable = false,
  indentLevel = 0,
} = {}) {
  return createGenericRow({
    labelText: defaultName,
    labelPlaceholder: 'New Label Name',
    colorSwatchColor,
    indentLevel,
    editable: editable,
    rightButtons: [
      {
        text: 'Save',
        emoji: '✅',
        onClick: ({ nameInput, colorInput }) => {
          const newName = nameInput.value.trim();
          if (!newName) {
            alert('Name required');
            return;
          }
          onSave?.(newName, colorInput?.value);
        },
      },
      {
        text: 'Cancel',
        emoji: '❌',
        onClick: () => onCancel?.(),
      },
    ],
  });
}

/**
 * Highlights a label visually in the list
 * @param {string|null} selectedUuid
 */
function highlightSelectedLabel(selectedUuid = null) {
  const currentTool = getTool();
  const selected = selectedUuid ?? (currentTool.startsWith('bbox:') ? currentTool.split(':')[1] : null);

  document.querySelectorAll('#labels-tool-list .label-row').forEach(row => {
    row.style.outline = row.dataset.uuid === selected ? '2px solid #ff0033' : '';
  });
}

/**
 * Returns button handlers for a label
 * @param {Object} options
 * @param {Object} options.label - label metadata
 * @param {number} options.indentLevel
 * @param {boolean} options.editable
 * @param {Function} options.renderList - callback to refresh the list
 * @param {HTMLElement} options.labelRow - the DOM element for this row
 */
function createLabelHandlers({ label, indentLevel, editable, renderList }) {
  if (!editable) return { leftButtons: [], rightButtons: [] };

  const { name, color, uuid } = label.metadata;

  const editButton = {
    text: 'Edit',
    emoji: '✏️',
    onClick: ({ nameInput, colorInput, row }) => {
      const inputRow = createInputRow({
        defaultName: name,
        colorSwatchColor: color,
        indentLevel,
        editable: true,
        onSave: async (newName, newColor) => {
          const response = await fetch('/api/labels/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label_uuid: uuid, name: newName, color: newColor }),
          });
          response.ok ? renderList() : alert('Failed to update label');
        },
        onCancel: renderList,
      });
      row.replaceWith(inputRow);
    },
  };

  const deleteButton = {
    text: 'Delete',
    emoji: '🗑️',
    onClick: async ({ row }) => {
      if (!window.confirm(`Delete label "${name}"?`)) return;
      const response = await fetch('/api/labels/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label_uuid: uuid }),
      });
      response.ok ? renderList() : toast((await response.json()).message, 5000, 'error');
    },
  };

  const addChildButton = {
    text: 'Add Child',
    emoji: '➕',
    onClick: ({ row }) => {
      const childRow = createInputRow({
        indentLevel: indentLevel + 1,
        editable: true,
        onSave: async (newName, newColor) => {
          const response = await fetch('/api/labels/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: newName, color: newColor, parent_uuid: uuid }),
          });
          response.ok ? renderList() : alert('Failed to create label');
        },
        onCancel: () => childRow.remove(),
      });
      row.after(childRow);
    },
  };

  return {
    leftButtons: [addChildButton, editButton, deleteButton],
    rightButtons: [],
  };
}


/**
 * Recursively renders a label and its children
 */
function renderLabel(label, container, indentLevel, editable, onSelectCallback, renderList) {
  const { name, color, uuid } = label.metadata;

  // Generate left/right buttons objects from handler
  const { leftButtons, rightButtons } = editable
    ? createLabelHandlers({ label, indentLevel, editable, renderList })
    : { leftButtons: [], rightButtons: [] };

  // Create row once, passing the button specs into createGenericRow
  const labelRow = createGenericRow({
    labelText: name,
    colorSwatchColor: color,
    indentLevel,
    leftButtons,
    rightButtons,
  });
  labelRow.dataset.uuid = uuid;

  // Add selection handling
  if (onSelectCallback) {
    labelRow.style.cursor = 'pointer';
    labelRow.onclick = () => {
      onSelectCallback(uuid);
      highlightSelectedLabel(uuid);
    };
  }

  container.appendChild(labelRow);

  // Render children
  label.children.forEach(child =>
    renderLabel(child, container, indentLevel + 1, editable, onSelectCallback, renderList)
  );
}

/**
 * Renders all labels into a container
 */
export async function renderLabelList({ target = 'labels-list', editable = true, onSelectCallback = null } = {}) {
  try {
    await refreshLabelList();
    const container = document.getElementById(target);
    if (!container) {
      console.error('Label container not found');
      return;
    }
    container.innerHTML = '';

    const labels = getLabelList();
    const renderList = () => renderLabelList({ target, editable, onSelectCallback });

    // Render root labels
    labels
      .filter(l => !l.metadata.parent_uuid)
      .forEach(l => renderLabel(l, container, 0, editable, onSelectCallback, renderList));

    // Add "Add new root label" row
    if (editable) {
      const newRootRow = createGenericRow({
        labelText: '', // No text for this "add" row
        labelPlaceholder: 'New Root Label Name',
        colorSwatchColor: '#cccccc',
        indentLevel: 0,
        editable: true,
        leftButtons: [
          {
            text: 'Refresh',
            emoji: '🔄',
            onClick: renderList,
          },
          {
            text: 'Add label',
            emoji: '➕',
            onClick: async ({ row, colorInput }) => {
              // Try to find an input field in the row
              const input = row.querySelector('input[type="text"]');
              const name = input?.value.trim();
              const color = colorInput?.value || colorSwatchColor;

              if (!name) {
                toast('Error: Name required', 5000, "error");
                return;
              }

              const response = await fetch('/api/labels/add', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, color, parent_uuid: null }),
              });

              response.ok ? renderList() : toast('Failed to create label');
            },
          }
        ],
        rightButtons: [],
      });

      container.appendChild(newRootRow);
    }

    // Highlight selected label if exists
    const selectedLabelUuid = getCurrentLabelUuid();
    if (selectedLabelUuid) highlightSelectedLabel(selectedLabelUuid);
  } catch (err) {
    console.error('Failed to fetch labels:', err);
  }
}
