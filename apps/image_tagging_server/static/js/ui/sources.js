// ================= Sources Manager =================
import { EditableField, SourceConfigField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, createSource, deleteSources, fetchImageProviderSchema, fetchSourceList, runPreviewSourceTest, toast, updateSource } from '/app-static/js/ui/utils/index.js';

/**
 * Creates a row for a single source with editable buttons
 */
function createSourceRow({ source, editable = false, renderList, onEdit, onDelete }) {
  const { name, typename, uuid, params } = source;
  const rowDiv = document.createElement('div');
  const deleteButton = {
    text: 'Delete',
    emoji: '🗑️',
    onClick: async () => {
      if (!window.confirm(`Are you sure you want to delete "${name}"?`)) return;
      try {
        await deleteSources({ uuids: [uuid] });
        onDelete?.();
      } catch (error) {
        toast(error.message || 'Failed to delete source', 5000, 'error');
      }
    }
  };

  const testBtnInfo = {
    text: 'Test Source',
    emoji: '🧪',
    onClick: async ({ field }) => {
      const { text: name, typename, schema } = field.getValue();
      await runPreviewSourceTest({ providerName: typename, params: schema });
    },
  };


  SourceConfigField.create({ value: { source_name: name, typename, uuid, schema: params } }).then(fieldInstance => {
    const row = createGenericRow({
      field: new EditableField({
        field: fieldInstance,
        onSave: async ({ source_name, typename, schema: filled_schema }) => {
          await updateSource({ uuid, name: source_name, providerName: typename, params: filled_schema });
          renderList();
        },
        onCancel: () => {
          renderList();
        },
      }),
      indentLevel: 0,
      editable: false,
      leftButtons: editable ? [deleteButton] : [],
      rightButtons: [testBtnInfo]
    });

    rowDiv.appendChild(row);
  });

  return rowDiv;
}

/**
 * Renders the list of sources
 */
export function renderSourceList({ parent, editable = false, onEdit, onDelete }) {
  parent.innerHTML = '';
  fetchSourceList().then(sources => {
    sources.forEach(src => {
      const row = createSourceRow({
        source: src,
        editable,
        renderList: () => renderSourceList({ parent, editable, onEdit, onDelete }),
        onEdit,
        onDelete,
      });
      parent.appendChild(row);
    });
  });
}


/**
 * Renders the UI for managing sources (list, delete, create).
 * @param {Object} options
 * @param {string} options.target - ID of the container to render into.
 */
/**
 * Renders the UI for managing sources.
 * @param {Object} options
 * @param {string} options.target - ID of the container to render into.
 */
export function renderSourceManager({ target = 'sources-box' }) {
  const container = document.getElementById(target);
  container.innerHTML = '';

  const header = document.createElement('h3');
  header.textContent = 'Sources';
  container.appendChild(header);

  const list = document.createElement('div');
  container.appendChild(list);

  const refresh = () =>
    renderSourceList({ parent: list, editable: true, onDelete: refresh });

  refresh();

  container.appendChild(document.createElement('hr'));
  const createHeader = document.createElement('h3');
  createHeader.textContent = 'Create Source';
  container.appendChild(createHeader);

  let newSourceRow; // keep a reference so we can replace it later

  function makeNewSourceRow() {
    return SourceConfigField.create({
      value: {
        name: '',
        typename: '',
        schema: {}
      },
      editMode: true,
    }).then(fieldInstance => {
      return createGenericRow({
        field: fieldInstance,
        rightButtons: [saveBtnInfo, testBtnInfo],
        editMode: true,
      });
    });
  }

  const saveBtnInfo = {
    text: 'Save',
    emoji: '✅',
    onClick: async ({ field }) => {
      const { source_name, typename, schema } = field.getValue();
      try {
        await createSource({ name: source_name, providerName: typename, params: schema });
        await refresh();

        // Replace the row with a fresh blank one
        makeNewSourceRow().then(freshRow => {
          container.replaceChild(freshRow, newSourceRow);
          newSourceRow = freshRow;
        });
      } catch (error) {
        console.error('Error saving source:', error);
      }
    },
  };

  const testBtnInfo = {
    text: 'Test Source',
    emoji: '🧪',
    onClick: async ({ field }) => {
      const { text: name, typename, schema } = field.getValue();
      await runPreviewSourceTest({ providerName: typename, params: schema });
    },
  };

  // Create the initial row
  makeNewSourceRow().then(row => {
    newSourceRow = row;
    container.appendChild(newSourceRow);
  });
}
