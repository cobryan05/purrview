// ================= Sources Manager =================
import { EditableField, SourceConfigField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, createSource, deleteSources, fetchImageProviderSchema, getAllSources, refreshImageProvidersCache, refreshSourceCache, runPreviewSourceTest, toast, updateSource } from '/app-static/js/ui/utils/index.js';

/**
 * Creates a row for a single source with editable buttons
 */
async function createSourceRow({ source, editable = false, renderList, onEdit, onDelete }) {
  const { name, typename, uuid, params } = source;

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

  let schema = null;
  try {
    schema = await fetchImageProviderSchema(typename);
  } catch (error) {
    toast(error.message || 'Failed to fetch provider schema', 5000, 'error');
  }
  return createGenericRow({
    field: new EditableField({
      field: new SourceConfigField({ name, typename, schema, value: params }),
      onSave: async ({ text: name, typename, schema: filled_schema }) => {
        // Save changes
        await updateSource({ uuid, name, providerName: typename, params: filled_schema });
        await renderList();
      },
      onCancel: async () => {
        await renderList();
      },
    }),
    indentLevel: 0,
    editable: false,
    leftButtons: editable ? [deleteButton] : [],
    rightButtons: [testBtnInfo]
  });
}

/**
 * Renders the list of sources
 */
export async function renderSourceList({ parent, editable = false, onEdit, onDelete }) {
  await refreshSourceCache();
  await refreshImageProvidersCache();
  parent.innerHTML = '';
  const sources = getAllSources();

  sources.forEach(async src => {
    const row = await createSourceRow({
      source: src,
      editable,
      renderList: () => renderSourceList({ parent, editable, onEdit, onDelete }),
      onEdit,
      onDelete,
    });
    parent.appendChild(row);
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
export async function renderSourceManager({ target = 'sources-box' }) {
  const container = document.getElementById(target);
  container.innerHTML = '';

  const header = document.createElement('h3');
  header.textContent = 'Sources';
  container.appendChild(header);

  const list = document.createElement('div');
  container.appendChild(list);

  const refresh = () =>
    renderSourceList({ parent: list, editable: true, onDelete: refresh });

  await refresh();

  container.appendChild(document.createElement('hr'));
  const createHeader = document.createElement('h3');
  createHeader.textContent = 'Create Source';
  container.appendChild(createHeader);

  let newSourceRow; // keep a reference so we can replace it later

  const makeNewSourceRow = () =>
    createGenericRow({
      field: new SourceConfigField({
        name: '',
        typename: '',
        schema: {},
        editMode: true,
      }),
      rightButtons: [saveBtnInfo, testBtnInfo],
      editMode: true,
    });

  const saveBtnInfo = {
    text: 'Save',
    emoji: '✅',
    onClick: async ({ field }) => {
      const { text, typename, schema } = field.getValue();
      try {
        await createSource({ name: text, providerName: typename, params: schema });
        await refresh();

        // Replace the row with a fresh blank one
        const freshRow = makeNewSourceRow();
        container.replaceChild(freshRow, newSourceRow);
        newSourceRow = freshRow;
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

  newSourceRow = makeNewSourceRow();
  container.appendChild(newSourceRow);
}
