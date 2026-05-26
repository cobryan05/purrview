// tags.js
import { clearTagCache, createNewTag, deleteTag, fetchTags, updateTag } from '/app-static/js/shared/api/tags.js';
import { EditableField, TagField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, toast } from '/app-static/js/ui/utils/index.js';
const DEFAULT_COLOR = '#cccccc';

/**
 * Returns button handlers for a tag
 * @param {Object} options
 * @param {Object} options.tag - tag metadata
 * @param {boolean} options.editable
 * @param {Function} options.renderList - callback to refresh the list
 */
function createTagHandlers({ tag, editable, renderList }) {
  if (!editable) return { leftButtons: [], rightButtons: [] };

  const { name, uuid } = tag;

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
 * Renders a tag row
 */
function renderTag(tag, container, editable, onSelectCallback, rerenderCallback) {
  const { name, color, icon, uuid } = tag;

  const { leftButtons, rightButtons } = editable
    ? createTagHandlers({ tag, editable, renderList: rerenderCallback })
    : { leftButtons: [], rightButtons: [] };

  const tagRow = createGenericRow({
    field: new EditableField({
      buttonsLast: true,
      field: new TagField({ text: name, uuid, placeholder: 'New Tag Name', color, icon: icon ?? null }),
      ...(editable && {
        onSave: async ({ text: newText, color: newColor, icon: newIcon }) => {
          try {
            await updateTag(uuid, newText, newColor, newIcon);
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
    targetElement.innerHTML = '';
    const container = document.createElement('div');
    container.style.display = 'inline-block';
    container.style.verticalAlign = 'top';
    container.style.width = 'max-content';
    container.style.minWidth = '0';
    targetElement.appendChild(container);

    const rerender = () => renderTagList({ target, editable, onSelectCallback });

    fetchTags().then(tags => {
      [...tags.values()]
        .forEach(tag => {
          renderTag(tag, container, editable, onSelectCallback, rerender);
        });

      if (editable) {
        const divider = document.createElement('hr');
        container.appendChild(divider);

        const newRootRow = createGenericRow({
          field: new EditableField({
            buttonsLast: true,
            field: new TagField({ text: '', placeholder: 'New Tag Name', color: DEFAULT_COLOR, icon: null }),
            editMode: true,
          }),
          indentLevel: 0,
          leftButtons: [
            {
              text: 'Add tag',
              emoji: '➕',
              onClick: async ({ field }) => {
                const { text: name, color, icon } = field.getValue();

                if (!name?.trim()) {
                  toast('Error: Name required', 5000, 'error');
                  return;
                }

                try {
                  await createNewTag(name.trim(), color || DEFAULT_COLOR, icon);
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
