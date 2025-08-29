// ================= Sources Manager =================
import { fetchImageProviderList, getParamsFromForm, pollTaskStatus, renderSchemaForm } from './utils.js';
import { refreshImageProvidersCache } from './utils/sourcesApi.js';
import { createButton, createGenericRow, deleteSources, EditableField, fetchImageProviderSchema, refreshSourceCache, SourceConfigField, TextField, toast } from '/app-static/js/ui/utils/index.js';

/**
 * Creates a row for a single source with editable buttons
 */
async function createSourceRow({ source, editable = false, renderList, onEdit, onDelete }) {
  const { name, typename, uuid } = source;

  const deleteButton = {
    text: 'Delete',
    emoji: '🗑️',
    onClick: async () => {
      if (!window.confirm(`Are you sure you want to delete "${name}"?`)) return;
      try {
        await deleteSources(uuid);
        onDelete?.();
      } catch (error) {
        toast(error.message || 'Failed to delete source', 5000, 'error');
      }
    }
  };

  let schema = null;
  try {
    ({ schema } = await fetchImageProviderSchema(typename))
  } catch (error) {
    toast(error.message || 'Failed to fetch provider schema', 5000, 'error');
  }
  return createGenericRow({
    field: new EditableField({
      field: new SourceConfigField({ name, typename, schema }),
      onSave: async () => {
        // // Replace row with inline form
        // const formContainer = document.createElement('div');
        // row.replaceWith(formContainer);

        // await renderSourceForm({
        //   parent: formContainer,
        //   existingSource: source,
        //   onCreate: renderList,
        //   onCancel: renderList,
        // });
      },
      onCancel: async () => {
        // Handle cancel action
        //field.setValue(`${typename}: ${name}`);
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

/**
 * Renders a form for creating or editing a source
 */
export async function renderSourceForm({ parent, existingSource = null, onCreate, onCancel }) {
  parent.innerHTML = '';

  const formTitle = document.createElement('h4');
  formTitle.textContent = existingSource ? 'Edit Source' : 'Create New Source';
  parent.appendChild(formTitle);

  const newSourceField = new SourceConfigField({ name: '', typename: '', schema: {} });
  parent.appendChild(newSourceField.renderEdit());
  // const providerSelect = document.createElement('select');
  // providerSelect.required = true;
  // providerSelect.style.width = '100%';
  // topRow.appendChild(providerSelect);

  // const nameInput = document.createElement('input');
  // nameInput.type = 'text';
  // nameInput.placeholder = 'Source name';
  // nameInput.required = true;
  // nameInput.style.width = '100%';
  // topRow.appendChild(nameInput);

  // const btnWrapper = document.createElement('div');
  // btnWrapper.style.display = 'flex';
  // btnWrapper.style.flexDirection = 'column';

  // const submitBtn = createButton('emoji-button', 'Save', '✅');
  // submitBtn.type = 'submit';
  // btnWrapper.appendChild(submitBtn);

  // if (existingSource && onCancel) {
  //   const cancelBtn = createButton('emoji-button', 'Cancel', '❌');
  //   cancelBtn.onclick = () => onCancel();
  //   btnWrapper.appendChild(cancelBtn);
  // }

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

  // topRow.appendChild(btnWrapper);
  // form.appendChild(topRow);

  // const paramsContainer = document.createElement('div');
  // form.appendChild(paramsContainer);

  // parent.appendChild(form);

  // // Load providers
  // const providers = await fetchImageProviderList();
  // providers.forEach(p => {
  //   const opt = document.createElement('option');
  //   opt.value = p;
  //   opt.textContent = p;
  //   providerSelect.appendChild(opt);
  // });

  // providerSelect.onchange = async () => {
  //   const provider = providerSelect.value;
  //   if (!provider) return;

  //   const schemaRes = await fetch('/api/sources/image-providers/schema', {
  //     method: 'POST',
  //     headers: { 'Content-Type': 'application/json' },
  //     body: JSON.stringify({ image_provider: provider }),
  //   });
  //   const schema = await schemaRes.json();
  //   renderSchemaForm(schema.schema, paramsContainer);
  // };

  // if (existingSource) {
  //   nameInput.value = existingSource.name;
  //   providerSelect.value = existingSource.typename;

  //   const schemaRes = await fetch('/api/sources/image-providers/schema', {
  //     method: 'POST',
  //     headers: { 'Content-Type': 'application/json' },
  //     body: JSON.stringify({ image_provider: existingSource.typename }),
  //   });
  //   const schema = await schemaRes.json();
  //   renderSchemaForm(schema.schema, paramsContainer);

  //   setTimeout(() => {
  //     Object.entries(existingSource.params || {}).forEach(([key, value]) => {
  //       const el = paramsContainer.querySelector(`[name="${key}"]`);
  //       if (!el) return;
  //       if (el.type === 'checkbox') el.checked = !!value;
  //       else el.value = value;
  //     });
  //   }, 0);
  // }

  // form.onsubmit = async e => {
  //   e.preventDefault();
  //   const provider = providerSelect.value;
  //   const name = nameInput.value;
  //   const params = getParamsFromForm(paramsContainer);

  //   const payload = {
  //     source_name: name,
  //     image_provider: provider,
  //     params,
  //   };

  //   if (existingSource?.uuid) payload.source_uuid = existingSource.uuid;

  //   const url = existingSource ? '/api/sources/update' : '/api/sources/create';
  //   const res = await fetch(url, {
  //     method: 'POST',
  //     headers: { 'Content-Type': 'application/json' },
  //     body: JSON.stringify(payload),
  //   });
  //   const result = await res.json();

  //   if (result.status === 'success') {
  //     toast('Saved successfully', 3000, 'success');
  //     if (onCreate) onCreate();
  //     parent.innerHTML = '';
  //   } else {
  //     toast(result.message || 'Unknown error', 5000, 'error');
  //   }
  // };
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

  const refresh = () => renderSourceList({ parent: list, editable: true, onDelete: refresh });

  await refresh();

  container.appendChild(document.createElement('hr'));

  const newFormContainer = document.createElement('div');
  await renderSourceForm({
    parent: newFormContainer,
    onCreate: refresh
  });
  container.appendChild(newFormContainer);
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
