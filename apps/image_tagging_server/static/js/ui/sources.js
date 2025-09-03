// ================= Sources Manager =================
import { EditableField, SourceConfigField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, createSource, deleteSources, fetchImageProviderSchema, refreshImageProvidersCache, refreshSourceCache, toast, updateSource } from '/app-static/js/ui/utils/index.js';

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
  });
}

/**
 * Renders the list of sources
 */
export async function renderSourceList({ parent, editable = false, onEdit, onDelete }) {
  await refreshSourceCache()
  await refreshImageProvidersCache();
  parent.innerHTML = '';
  const res = await fetch('/api/sources/get');
  const data = await res.json();

  Object.values(data.sources).forEach(async src => {
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


// const testBtn = createButton('emoji-button', 'Test', '🧪');
// testBtn.type = 'button';
// btnWrapper.appendChild(testBtn);

// testBtn.onclick = async () => {
//   try {
//     const provider = providerSelect.value;
//     if (!provider) {
//       toast("Please fill in provider before testing", 3000, "warning");
//       return;
//     }

//     const name = nameInput.value;
//     const providerParams = getParamsFromForm(paramsContainer);
//     const params = {
//       provider: provider,
//       provider_params: providerParams,
//     };

//     testBtn.disabled = true;
//     const testResultBox = renderTestResultsBox(form);
//     testResultBox.hidden = false;
//     testResultBox.textContent = 'Starting test task...';
//     const taskUuid = await startPreviewSourceTest({
//       params
//     });

//     testResultBox.textContent = 'Running test...';
//     pollTaskStatus(taskUuid, testResultBox, async (finalResult) => {
//       testResultBox.textContent += `\nTest ${finalResult.status}: ${finalResult.status_message || ''}`;
//       testBtn.disabled = false;

//       // Check for image
//       if (finalResult.data?.image) {
//         const img = document.createElement('img');
//         img.src = `data:image/png;base64,${finalResult.data.image}`;
//         img.style.maxWidth = '100%';
//         img.alt = 'Test result image';
//         testResultBox.appendChild(document.createElement('br'));
//         testResultBox.appendChild(img);
//       }
//     });

//   } catch (err) {
//     toast(`Test failed: ${err.message}`, 5000, "error");
//     testBtn.disabled = false;
//   }
// };


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
      // TODO: implement test
    },
  };

  newSourceRow = makeNewSourceRow();
  container.appendChild(newSourceRow);
}


function renderTestResultsBox(parent) {
  const box = document.createElement('div');
  box.style.border = '1px solid #ccc';
  box.style.padding = '0.5em';
  box.style.marginTop = '1em';
  box.style.fontSize = '0.9em';
  box.style.whiteSpace = 'pre-wrap';
  box.textContent = 'No test started.';
  parent.appendChild(box);
  return box;
}


async function startPreviewSourceTest({ params = {} }) {
  const configRes = await fetch('/api/tasks/configs/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      typename: 'PreviewSourceTask',
      params: params,
      persistent: false
    }),
  });
  const configResult = await configRes.json();
  if (configResult.status !== 'success') throw new Error(configResult.message);

  const taskConfigUuid = configResult.config_uuid;

  const res = await fetch(`/api/tasks/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      config_uuid: taskConfigUuid
    }),
  });
  const result = await res.json();
  if (result.status !== 'success') throw new Error(result.message);
  return result.task_id;
}
