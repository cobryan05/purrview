// labels.js
import { getCurrentLabelUuid } from '/app-static/js/canvas/state.js';
import { getTool } from '/app-static/js/canvas/tools.js';
import { createGenericRow, createNewLabel, deleteLabel, fetchLabels, toast, updateLabel } from '/app-static/js/ui/utils/index.js';
import { EditableField, TextBoxColorField } from '/app-static/js/ui/utils/fields/index.js';
import { clearLabelCache } from './utils/labelsApi.js';

const DEFAULT_COLOR = '#cccccc';

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

  const deleteButton = {
    text: 'Delete',
    emoji: '🗑️',
    onClick: async () => {
      if (!window.confirm(`Delete label "${name}"?`)) return;
      try {
        await deleteLabel(uuid);
        renderList();
      } catch (err) {
        toast(err.message, 5000, 'error');
      }
    },
  };

  const addChildButton = {
    text: 'Add Child',
    emoji: '➕',
    onClick: ({ row }) => {
      const childRow = createGenericRow({
        field: new EditableField({
          field:
            new TextBoxColorField({ placeholder: "New Label Name", color: DEFAULT_COLOR }),
          editMode: true,
          buttonsLast: true,
          onSave: async ({ text: newText, color: newColor }) => {
            try {
              await createNewLabel(newText, newColor, uuid);
              renderList();
            } catch (err) {
              toast(err.message, 5000, 'error');
            }
          },
          onCancel: () => {
            renderList();
          }
        }),
        indentLevel: indentLevel + 1,
      });
      row.after(childRow);
    },
  };

  return {
    leftButtons: [addChildButton, deleteButton],
    rightButtons: [],
  };
}

/**
 * Recursively renders a label and its children
 */
function renderLabel(label, container, indentLevel, editable, onSelectCallback, rerenderCallback) {
  const { name, color, uuid } = label.metadata;

  // Generate left/right buttons objects from handler
  const { leftButtons, rightButtons } = editable
    ? createLabelHandlers({ label, indentLevel, editable, renderList: rerenderCallback })
    : { leftButtons: [], rightButtons: [] };

  // Create row once, passing the button specs into createGenericRow
  const labelRow = createGenericRow({
    field: new EditableField({
      field: new TextBoxColorField({ text: name, placeholder: "New Label Name", color: color }),
      buttonsLast: true,
      ...(editable && {
        onSave: async ({ text: newText, color: newColor }) => {
          try {
            await updateLabel(uuid, newText, newColor);
            rerenderCallback();
          } catch (err) {
            toast(err.message, 5000, 'error');
          }
        },
        onCancel: () => {
          rerenderCallback();
        },
      })
    }),
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
    renderLabel(child, container, indentLevel + 1, editable, onSelectCallback, rerenderCallback)
  );
}

/**
 * Renders all labels into a container
 */

export function renderLabelList({ target = 'labels-list', editable = true, onSelectCallback = null } = {}) {
  try {
    const container = document.getElementById(target);
    if (!container) {
      console.error('Label container not found');
      return;
    }
    container.innerHTML = '';

    // Render root labels
    const rerender = () => renderLabelList({ target, editable, onSelectCallback });

    fetchLabels().then(labels => {
      [...labels.values()]
        .filter(l => !l.metadata?.parent_uuid)
        .forEach(label => {
          renderLabel(label, container, 0, editable, onSelectCallback, rerender);
        });

      // Add "Add new root label" row
      if (editable) {
        const divider = document.createElement('hr');
        container.appendChild(divider);

        const newRootRow = createGenericRow({
          field: new EditableField({
            field: new TextBoxColorField({ text: '', placeholder: "New Root Label Name", color: DEFAULT_COLOR }),
            editMode: true,
          }),
          indentLevel: 0,
          leftButtons: [
            {
              text: 'Refresh',
              emoji: '🔄',
              onClick: async () => {
                clearLabelCache();
                rerender();
              },
            },
            {
              text: 'Add label',
              emoji: '➕',
              onClick: async ({ field }) => {
                const name = field.getValue().text?.trim();
                const color = field.getValue().color || DEFAULT_COLOR;

                if (!name) {
                  toast('Error: Name required', 5000, "error");
                  return;
                }

                try {
                  await createNewLabel(name, color);
                  rerender();
                } catch (err) {
                  toast(err.message, 5000, 'error');
                }
              },
            },
          ],
          rightButtons: [],
        });

        container.appendChild(newRootRow);
      }

      // Highlight selected label if exists
      const selectedLabelUuid = getCurrentLabelUuid();
      if (selectedLabelUuid) highlightSelectedLabel(selectedLabelUuid);
    });
  } catch (err) {
    console.error('Failed to fetch labels:', err);
  }
}
