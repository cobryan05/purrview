// classes.js
import { appState } from '../app/state.js';
import { getTool } from '../app/image-tagging/canvas/tools.js';
import { clearClassCache, createNewClass, deleteClass, fetchClasses, updateClass, } from '/app-static/js/shared/api/classes.js';
import { EditableField, TextBoxColorField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, toast } from '/app-static/js/ui/utils/index.js';

const DEFAULT_COLOR = '#cccccc';

/**
 * Highlights a class visually in the list
 * @param {string|null} selectedUuid
 */
function highlightSelectedClass(selectedUuid = null) {
  const currentTool = getTool(appState.imageTagging);
  const selected = selectedUuid ?? (currentTool.startsWith('bbox:') ? currentTool.split(':')[1] : null);

  document.querySelectorAll('#classes-tool-list .class-row').forEach(row => {
    row.style.outline = row.dataset.uuid === selected ? '2px solid #ff0033' : '';
  });
}

/**
 * Returns button handlers for a class
 * @param {Object} options
 * @param {Object} options.label - label metadata
 * @param {number} options.indentLevel
 * @param {boolean} options.editable
 * @param {Function} options.renderList - callback to refresh the list
 * @param {HTMLElement} options.labelRow - the DOM element for this row
 */
function createClassHandlers({ cls, indentLevel, editable, renderList }) {
  if (!editable) return { leftButtons: [], rightButtons: [] };

  const { name, color, uuid } = cls.metadata;

  const deleteButton = {
    text: 'Delete',
    emoji: '🗑️',
    onClick: async () => {
      if (!window.confirm(`Delete class "${name}"?`)) return;
      try {
        await deleteClass(uuid);
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
            new TextBoxColorField({ placeholder: "New Class Name", color: DEFAULT_COLOR }),
          editMode: true,
          onSave: async ({ text: newText, color: newColor }) => {
            try {
              await createNewClass(newText, newColor, uuid);
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
 * Recursively renders a class and its children
 */
function renderClass(cls, container, indentLevel, editable, onSelectCallback, rerenderCallback) {
  const { name, color, uuid } = cls.metadata;

  // Generate left/right buttons objects from handler
  const { leftButtons, rightButtons } = editable
    ? createClassHandlers({ cls, indentLevel, editable, renderList: rerenderCallback })
    : { leftButtons: [], rightButtons: [] };

  // Create row once, passing the button specs into createGenericRow
  const classRow = createGenericRow({
    field: new EditableField({
      buttonsLast: true,
      field: new TextBoxColorField({ text: name, uuid: uuid, placeholder: "New Class Name", color: color }),
      ...(editable && {
        onSave: async ({ text: newText, color: newColor }) => {
          try {
            await updateClass(uuid, newText, newColor);
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
  classRow.dataset.uuid = uuid;

  // Add selection handling
  if (onSelectCallback) {
    classRow.style.cursor = 'pointer';
    classRow.onclick = () => {
      onSelectCallback(uuid);
      highlightSelectedClass(uuid);
    };
  }

  container.appendChild(classRow);

  // Render children
  cls.children.forEach(child =>
    renderClass(child, container, indentLevel + 1, editable, onSelectCallback, rerenderCallback)
  );
}

/**
 * Renders all classes into a container
 */

export function renderClassList({ target = 'classes-list', editable = true, onSelectCallback = null } = {}) {
  try {
    const targetElement = document.getElementById(target);
    if (!targetElement) {
      console.error('Class container not found');
      return;
    }
    targetElement.innerHTML = ''
    const container = document.createElement('div');
    container.style.display = 'inline-block';      // shrink-wrap width
    container.style.verticalAlign = 'top';         // optional, align with top of parent
    container.style.width = 'max-content';         // shrink to longest content
    container.style.minWidth = '0';                // prevent overflow issues
    targetElement.appendChild(container);
    // Render root classes
    const rerender = () => renderClassList({ target, editable, onSelectCallback });

    fetchClasses().then(classes => {
      [...classes.values()]
        .filter(c => !c.metadata?.parent_uuid)
        .forEach(cls => {
          renderClass(cls, container, 0, editable, onSelectCallback, rerender);
        });

      // Add "Add new root class" row
      if (editable) {
        const divider = document.createElement('hr');
        container.appendChild(divider);

        const newRootRow = createGenericRow({
          field: new EditableField({
            buttonsLast: true,
            field: new TextBoxColorField({ text: '', placeholder: "New Root Class Name", color: DEFAULT_COLOR }),
            editMode: true,
          }),
          indentLevel: 0,
          leftButtons: [
            {
              text: 'Add class',
              emoji: '➕',
              onClick: async ({ field }) => {
                const name = field.getValue().text?.trim();
                const color = field.getValue().color || DEFAULT_COLOR;

                if (!name) {
                  toast('Error: Name required', 5000, "error");
                  return;
                }

                try {
                  await createNewClass(name, color);
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
                clearClassCache();
                rerender();
              },
            },
          ],
        });

        container.appendChild(newRootRow);
      }

      // Highlight selected class if exists
      const selectedClassUuid = appState.imageTagging.currentClassUuid;
      if (selectedClassUuid) highlightSelectedClass(selectedClassUuid);
    });
  } catch (err) {
    console.error('Failed to fetch classes:', err);
  }
}
