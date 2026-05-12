import { getTool } from '../app/image-tagging/canvas/tools.js';
import { clearLabelsCache, createNewLabel, deleteLabel, fetchLabels, updateLabel, } from '/app-static/js/shared/api/labels.js';
import { EditableField, TextBoxColorField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, toast } from '/app-static/js/ui/utils/index.js';

const DEFAULT_COLOR = '#cccccc';

/**
 * Highlights a label visually in the list
 * @param {string|null} selectedUuid
 */
function highlightLabelUuid(selectedUuid = null) {
  document.querySelectorAll('#labels-tool-list .label-row').forEach(row => {
    row.style.outline = row.dataset.uuid === selectedUuid ? '2px solid #ff0033' : '';
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

  const { name, color, uuid } = label;

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
    onClick: ({ field, row }) => {
      const childRow = createGenericRow({
        field: new EditableField({
          buttonsLast: true,
          field:
            new TextBoxColorField({ placeholder: "New Label Name", color: DEFAULT_COLOR }),
          editMode: true,
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
function renderLabel(label, childrenByParent, container, indentLevel, editable, onSelectCallback, rerenderCallback) {
  const { name, color, uuid } = label;

  // Generate left/right buttons objects from handler
  const { leftButtons, rightButtons } = editable
    ? createLabelHandlers({ label, indentLevel, editable, renderList: rerenderCallback })
    : { leftButtons: [], rightButtons: [] };

  // Create row once, passing the button specs into createGenericRow
  const labelRow = createGenericRow({
    field: new EditableField({
      buttonsLast: true,
      field: new TextBoxColorField({ text: name, uuid: uuid, placeholder: "New Label Name", color: color }),
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
    alignLeftButtons: true,
    leftButtons,
    rightButtons,
  });

  labelRow.dataset.uuid = uuid;

  // Add selection handling
  if (onSelectCallback) {
    labelRow.style.cursor = 'pointer';
    labelRow.onclick = (e) => {
      // Prevent selection if clicking buttons/inputs
      if (e.target.closest('button, input')) return;
      onSelectCallback(uuid);
      highlightLabelUuid(uuid);
    };
  }

  container.appendChild(labelRow);

  // Render children from the pre-computed lookup
  const children = childrenByParent[uuid] || [];
  children.sort((a, b) => a.name.localeCompare(b.name));

  children.forEach(child =>
    renderLabel(child, childrenByParent, container, indentLevel + 1, editable, onSelectCallback, rerenderCallback)
  );
}

/**
 * Renders all labels into a container
 */
export async function renderLabelList({ target = 'labels-list', selectedLabelUuid = null, editable = true, onSelectCallback = null } = {}) {
  try {
    const targetElement = document.getElementById(target);
    if (!targetElement) {
      console.error('Label container not found');
      return;
    }

    targetElement.innerHTML = '';
    const container = document.createElement('div');

    // Explicit styles as requested
    container.style.display = 'inline-block';
    container.style.verticalAlign = 'top';
    container.style.width = 'max-content';
    container.style.minWidth = '0';
    targetElement.appendChild(container);

    const rerender = () => renderLabelList({ target, selectedLabelUuid, editable, onSelectCallback });

    // Fetch and Build Hierarchy Lookup
    const labelsMap = await fetchLabels();
    const allLabels = [...labelsMap.values()];

    const childrenByParent = {};
    const rootLabels = [];

    allLabels.forEach(l => {
      if (!l.parent_uuid) {
        rootLabels.push(l);
      } else {
        if (!childrenByParent[l.parent_uuid]) childrenByParent[l.parent_uuid] = [];
        childrenByParent[l.parent_uuid].push(l);
      }
    });

    // Sort and Render Roots
    rootLabels.sort((a, b) => a.name.localeCompare(b.name));
    rootLabels.forEach(label => {
      renderLabel(label, childrenByParent, container, 0, editable, onSelectCallback, rerender);
    });

    // Add "Add new root label" row
    if (editable) {
      const divider = document.createElement('hr');
      divider.style.margin = '10px 0';
      container.appendChild(divider);

      const newRootRow = createGenericRow({
        field: new EditableField({
          buttonsLast: true,
          field: new TextBoxColorField({ text: '', placeholder: "New Root Label Name", color: DEFAULT_COLOR }),
          editMode: true,
        }),
        indentLevel: 0,
        leftButtons: [
          {
            text: 'Add label',
            emoji: '➕',
            onClick: async ({ field }) => {
              const val = field.getValue();
              const name = val.text?.trim();
              const color = val.color || DEFAULT_COLOR;

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
        rightButtons: [
          {
            text: 'Refresh',
            emoji: '🔄',
            onClick: async () => {
              clearLabelsCache();
              rerender();
            },
          },
        ],
      });

      container.appendChild(newRootRow);
    }

    // Highlight selected label if exists
    if (selectedLabelUuid) highlightLabelUuid(selectedLabelUuid);

  } catch (err) {
    console.error('Failed to render labels:', err);
  }
}
