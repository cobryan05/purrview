// tags.js
import { clearTagCache } from './utils/tagsApi.js';
import { state } from '/app-static/js/canvas/state.js';
import { getTool } from '/app-static/js/canvas/tools.js';
import { EditableField, TextBoxColorField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, createNewTag, deleteTag, fetchTags, toast, updateTag } from '/app-static/js/ui/utils/index.js';

const DEFAULT_COLOR = '#cccccc';

/**
 * Returns button handlers for a tag
 * @param {Object} options
 * @param {Object} options.label - label metadata
 * @param {boolean} options.editable
 * @param {Function} options.renderList - callback to refresh the list
 * @param {HTMLElement} options.labelRow - the DOM element for this row
 */
function createTagHandlers({ tag, editable, renderList }) {
  if (!editable) return { leftButtons: [], rightButtons: [] };

  const { name, color, uuid } = tag;

  const deleteButton = {
    text: 'Delete',
    emoji: '🗑️',
    onClick: async () => {
      if (!window.confirm(`Delete tag "${name}"?`)) return;
      try {
        await deleteTag(uuid);
        renderList();
      } catch (err) {
        toast(err.message, 5000, 'error');
      }
    },
  };

  return {
    leftButtons: [deleteButton],
    rightButtons: [],
  };
}

/**
 * Renders a tag
 */
function renderTag(tag, container, editable, onSelectCallback, rerenderCallback) {
  const { name, color, uuid } = tag;

  // Generate left/right buttons objects from handler
  const { leftButtons, rightButtons } = editable
    ? createTagHandlers({ tag, editable, renderList: rerenderCallback })
    : { leftButtons: [], rightButtons: [] };

  // Create row once, passing the button specs into createGenericRow
  const tagRow = createGenericRow({
    field: new EditableField({
      buttonsLast: true,
      field: new TextBoxColorField({ text: name, uuid: uuid, placeholder: "New Tag Name", color: color }),
      ...(editable && {
        onSave: async ({ text: newText, color: newColor }) => {
          try {
            await updateTag(uuid, newText, newColor);
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
    alignLeftButtons: true,
    leftButtons,
    rightButtons,
  });
  tagRow.dataset.uuid = uuid;

  // Add selection handling
  if (onSelectCallback) {
    tagRow.style.cursor = 'pointer';
    tagRow.onclick = () => {
      onSelectCallback(uuid);
    };
  }

  container.appendChild(tagRow);
}

/**
 * Renders all tags into a container
 */

export function renderTagList({ target = 'tags-list', editable = true, onSelectCallback = null } = {}) {
  try {
    const targetElement = document.getElementById(target);
    if (!targetElement) {
      console.error('Tag container not found');
      return;
    }
    targetElement.innerHTML = ''
    const container = document.createElement('div');
    container.style.display = 'inline-block';      // shrink-wrap width
    container.style.verticalAlign = 'top';         // optional, align with top of parent
    container.style.width = 'max-content';         // shrink to longest content
    container.style.minWidth = '0';                // prevent overflow issues
    targetElement.appendChild(container);
    // Render root tags
    const rerender = () => renderTagList({ target, editable, onSelectCallback });

    fetchTags().then(tags => {
      [...tags.values()]
        .forEach(tag => {
          renderTag(tag, container, editable, onSelectCallback, rerender);
        });

      // Add "Add new tag" row
      if (editable) {
        const divider = document.createElement('hr');
        container.appendChild(divider);

        const newRootRow = createGenericRow({
          field: new EditableField({
            buttonsLast: true,
            field: new TextBoxColorField({ text: '', placeholder: "New Tag Name", color: DEFAULT_COLOR }),
            editMode: true,
          }),
          indentLevel: 0,
          leftButtons: [
            {
              text: 'Add tag',
              emoji: '➕',
              onClick: async ({ field }) => {
                const name = field.getValue().text?.trim();
                const color = field.getValue().color || DEFAULT_COLOR;

                if (!name) {
                  toast('Error: Name required', 5000, "error");
                  return;
                }

                try {
                  await createNewTag(name, color);
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
                clearTagCache();
                rerender();
              },
            },
          ],
        });

        container.appendChild(newRootRow);
      }
    });
  } catch (err) {
    console.error('Failed to fetch tags:', err);
  }
}
