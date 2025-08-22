import { createButton, createGenericRow } from './utils.js'
import { getCurrentLabelUuid, refreshLabelList, getLabelList } from '/app-static/js/canvas/state.js';
import { getTool } from '/app-static/js/canvas/tools.js';
import { toast } from '/app-static/js/canvas/utils.js';


// ================= Source Manager =================

export async function renderSourceList({ parent, editable = false, onEdit = () => { }, onDelete = () => { } }) {
  parent.innerHTML = '';
  const res = await fetch('/api/sources/get');
  const data = await res.json();

  Object.values(data.sources).forEach(src => {
    const row = document.createElement('div');
    row.style.display = 'flex';
    row.style.justifyContent = 'space-between';
    row.style.alignItems = 'center';
    row.style.marginBottom = '0.5em';
    row.style.flexWrap = 'wrap';

    const label = document.createElement('span');
    label.textContent = `${src.typename}: ${src.name}`;
    row.appendChild(label);

    const actions = document.createElement('div');
    actions.style.display = 'flex';
    actions.style.gap = '0.5em';

    if (editable) {
      const editBtn = createButton('emoji-button', 'Edit', '🖉');
      editBtn.onclick = async () => {
        // Re-render the entire list first
        await renderSourceList({
          parent,
          editable,
          onEdit,
          onDelete
        });

        // Then find the correct row again (since DOM has changed)
        const row = [...parent.children].find(child =>
          child.textContent?.includes(`${src.typename}: ${src.name}`)
        );

        if (!row) return;

        // Now inject the inline edit form
        const editContainer = document.createElement('div');
        editContainer.className = 'inline-edit-form';
        editContainer.style.margin = '1em 0';
        editContainer.style.padding = '0.5em';
        editContainer.style.border = '1px solid #ccc';
        editContainer.style.borderRadius = '0.5em';

        row.innerHTML = '';
        row.appendChild(editContainer);

        await renderSourceForm({
          parent: editContainer,
          existingSource: src,
          onCreate: () => renderSourceList({ parent, editable, onEdit, onDelete }),
          onCancel: () => renderSourceList({ parent, editable, onEdit, onDelete }),
        });
      };
      actions.appendChild(editBtn);

      const deleteBtn = createButton('emoji-button', 'Delete', '🗑️');
      deleteBtn.onclick = async () => {
        const confirmed = confirm(`Are you sure you want to delete "${src.name}"?`);
        if (!confirmed) return;

        await fetch('/api/sources/delete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source_uuids: [src.uuid] })
        });

        onDelete();
      };
      actions.appendChild(deleteBtn);
    }

    row.appendChild(actions);
    parent.appendChild(row);
  });
}

export async function renderSourceForm({ parent, onCreate, existingSource = null, onCancel = null }) {
  const formTitle = document.createElement('h4');
  formTitle.textContent = existingSource ? 'Edit Source' : 'Create New Source';
  parent.appendChild(formTitle);

  const form = document.createElement('form');
  form.style.display = 'flex';
  form.style.flexDirection = 'column';
  form.style.gap = '0.5em';

  const topRow = document.createElement('div');
  topRow.style.display = 'grid';
  topRow.style.gridTemplateColumns = '1fr 1fr auto';
  topRow.style.gap = '0.5em';
  topRow.style.alignItems = 'center';

  const providerSelect = document.createElement('select');
  providerSelect.required = true;
  providerSelect.style.width = '100%';

  const defaultOpt = document.createElement('option');
  defaultOpt.disabled = true;
  defaultOpt.selected = true;
  defaultOpt.textContent = 'Select provider';
  providerSelect.appendChild(defaultOpt);
  topRow.appendChild(providerSelect);

  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.placeholder = 'Source name';
  nameInput.required = true;
  nameInput.style.width = '100%';
  topRow.appendChild(nameInput);

  const btnWrapper = document.createElement('div');
  btnWrapper.style.display = 'flex';
  btnWrapper.style.flexDirection = 'column';
  btnWrapper.style.marginBottom = '1em';

  const submitBtn = createButton('emoji-button', 'Save', '✅');
  submitBtn.type = 'submit';
  submitBtn.textContent = existingSource ? '✅' : '➕';
  submitBtn.title = existingSource ? 'Update source' : 'Create source';
  btnWrapper.appendChild(submitBtn);
  if (existingSource && onCancel) {
    const discardBtn = createButton('emoji-button', 'Cancel', '❌');
    discardBtn.onclick = () => onCancel();
    btnWrapper.appendChild(discardBtn);
  }

  const testBtn = createButton('emoji-button', 'Test', '🧪');
  testBtn.type = 'button';
  testBtn.title = 'Run source test';
  btnWrapper.appendChild(testBtn);

  topRow.appendChild(btnWrapper);

  form.appendChild(topRow);
  const testResultBox = renderTestResultsBox(form);
  testResultBox.hidden = true;

  if (existingSource?.uuid) {
    const uuidLabel = document.createElement('div');
    uuidLabel.textContent = `UUID: ${existingSource.uuid}`;
    uuidLabel.style.fontSize = '0.75em';
    uuidLabel.style.opacity = '0.6';
    uuidLabel.style.userSelect = 'text';
    uuidLabel.style.marginBottom = '0.5em';
    parent.appendChild(uuidLabel);
  }

  const paramsContainer = document.createElement('div');
  paramsContainer.style.marginTop = '1em';
  form.appendChild(paramsContainer);

  parent.appendChild(form);

  const providers = await loadImageProviders();
  providers.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p;
    opt.textContent = p;
    providerSelect.appendChild(opt);
  });

  providerSelect.onchange = async () => {
    const provider = providerSelect.value;
    if (!provider) return;
    testResultBox.hidden = true;

    const schemaRes = await fetch('/api/sources/image-providers/schema', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_provider: provider }),
    });
    const schema = await schemaRes.json();
    renderSchemaForm(schema.schema, paramsContainer);
  };

  if (existingSource) {
    nameInput.value = existingSource.name;
    providerSelect.value = existingSource.typename;

    const schemaRes = await fetch('/api/sources/image-providers/schema', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_provider: existingSource.typename }),
    });
    const schema = await schemaRes.json();
    renderSchemaForm(schema.schema, paramsContainer);

    setTimeout(() => {
      Object.entries(existingSource.params || {}).forEach(([key, value]) => {
        const el = paramsContainer.querySelector(`[name="${key}"]`);
        if (!el) return;
        if (el.type === 'checkbox') el.checked = !!value;
        else el.value = value;
      });
    }, 0);
  }

  testBtn.onclick = async () => {
    try {
      const provider = providerSelect.value;
      if (!provider) {
        toast("Please fill in provider before testing", 3000, "warning");
        return;
      }

      const name = nameInput.value;
      const providerParams = getParamsFromForm(paramsContainer);
      const params = {
        provider: provider,
        provider_params: providerParams,
      };

      testBtn.disabled = true;
      testResultBox.hidden = false;
      testResultBox.textContent = 'Starting test task...';
      const taskUuid = await startPreviewSourceTest({
        params
      });

      testResultBox.textContent = 'Running test...';
      pollTaskStatus(taskUuid, testResultBox, async (finalResult) => {
        testResultBox.textContent += `\nTest ${finalResult.status}: ${finalResult.status_message || ''}`;
        testBtn.disabled = false;

        // Check for image
        if (finalResult.data?.image) {
          const img = document.createElement('img');
          img.src = `data:image/png;base64,${finalResult.data.image}`;
          img.style.maxWidth = '100%';
          img.alt = 'Test result image';
          testResultBox.appendChild(document.createElement('br'));
          testResultBox.appendChild(img);
        }
      });

    } catch (err) {
      toast(`Test failed: ${err.message}`, 5000, "error");
      testBtn.disabled = false;
    }
  };

  form.onsubmit = async e => {
    e.preventDefault();
    const provider = providerSelect.value;
    const name = nameInput.value;
    const params = getParamsFromForm(paramsContainer);

    const payload = {
      source_name: name,
      params,
      image_provider: provider
    };

    if (existingSource?.uuid) {
      payload.source_uuid = existingSource.uuid;
    }

    const url = existingSource
      ? `/api/sources/update`
      : '/api/sources/create';

    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const result = await res.json();

    if (result.status === 'success') {
      if (onCreate) onCreate();
      parent.innerHTML = '';  // remove the form
      renderSourceForm({ parent, onCreate, onCancel });
    } else {
      toast(result.message || 'Unknown error', 5000, "error");
    }
  };
}

export async function loadImageProviders() {
  const res = await fetch('/api/sources/image-providers/list');
  const data = await res.json();
  return data.providers;
}

export function getParamsFromForm(container) {
  const params = {};
  container.querySelectorAll('input, select, textarea').forEach(el => {
    if (!el.name) return;
    if (el.type === 'checkbox') {
      params[el.name] = el.checked;
    } else if (el.type === 'number') {
      params[el.name] = parseFloat(el.value);
    } else {
      params[el.name] = el.value;
    }
  });
  return params;
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

/**
 * Renders a form dynamically based on JSON schema input.
 * Supports string, boolean, and array-of-strings input types.
 * @param {Object} schema - JSON schema to render.
 * @param {HTMLElement} container - Container to append form inputs to.
 */
function renderSchemaForm(schema, container) {
  container.innerHTML = '';

  Object.entries(schema).forEach(([name, field]) => {
    const wrapper = document.createElement('div');
    wrapper.style.display = 'flex';
    wrapper.style.flexDirection = 'column';
    wrapper.style.marginBottom = '1em';

    const label = document.createElement('label');
    label.textContent = field.title || name;
    label.htmlFor = name;
    if (field.required) {
      label.innerHTML += ' <span style="color: red">*</span>';
    }
    wrapper.appendChild(label);

    let input;

    if (field.type === 'boolean') {
      const checkboxWrapper = document.createElement('label');
      checkboxWrapper.style.display = 'flex';
      checkboxWrapper.style.alignItems = 'center';
      checkboxWrapper.style.gap = '0.5em';
      checkboxWrapper.style.cursor = 'pointer';

      input = document.createElement('input');
      input.type = 'checkbox';
      input.name = name;
      input.checked = field.default === true;
      input.style.width = '1.2em';
      input.style.height = '1.2em';
      input.style.cursor = 'pointer';

      const checkboxLabel = document.createElement('span');
      checkboxLabel.textContent = field.description || field.title || name;

      checkboxWrapper.appendChild(input);
      checkboxWrapper.appendChild(checkboxLabel);
      wrapper.appendChild(checkboxWrapper);
    } else if (field.type === 'array' && field.items?.type === 'string') {
      input = document.createElement('textarea');
      input.name = name;
      input.placeholder = (field.items.description || field.description || '') + ' (one per line)';
      input.rows = 3;
    } else {
      input = document.createElement('input');
      input.type = 'text';
      input.name = name;
      input.placeholder = field.description || '';
      if (field.default !== undefined) {
        input.value = field.default;
      }
    }

    if (field.required) {
      input.required = true;
    }

    wrapper.appendChild(input);
    container.appendChild(wrapper);
  });
}

// Global cache for labels metadata
let _cachedLabels = null;

export async function renderModelLabelAssignments({ target = "model-labels-box", editable = true, preselectedModel = null } = {}) {
  const container = document.getElementById(target);
  container.innerHTML = '';

  // Load and cache labels metadata only once
  if (!_cachedLabels) {
    const labelRes = await fetch('/api/labels/list');
    const { labels } = await labelRes.json();
    // Create a map from UUID to label metadata for quick lookup
    _cachedLabels = new Map(labels.map(label => [label.metadata.uuid, label.metadata]));
  }

  const headerRow = document.createElement('div');
  headerRow.style.display = 'flex';
  headerRow.style.alignItems = 'center';
  headerRow.style.gap = '0.5em';
  headerRow.style.marginBottom = '1em';

  const modelSelect = document.createElement('select');

  if (!preselectedModel) {
    const placeholderOption = document.createElement('option');
    placeholderOption.value = '';
    placeholderOption.textContent = 'Select a model';
    placeholderOption.disabled = true;
    placeholderOption.selected = true;
    modelSelect.appendChild(placeholderOption);
  }

  const refreshBtn = createButton('emoji-button', 'Refresh', '🔄');
  refreshBtn.onclick = () => {
    _cachedLabels = null; // Clear cache on refresh so new labels are fetched
    renderModelLabelAssignments({ target, editable });
  };

  headerRow.appendChild(modelSelect);
  headerRow.appendChild(refreshBtn);
  container.appendChild(headerRow);

  // Load models
  const modelRes = await fetch('/api/models/list');
  const { models } = await modelRes.json();
  models.forEach(model => {
    const opt = document.createElement('option');
    opt.value = model;
    opt.textContent = model;
    modelSelect.appendChild(opt);
  });

  if (preselectedModel) {
    modelSelect.value = preselectedModel;
  }

  modelSelect.onchange = () => {
    const selectedModel = modelSelect.value;
    if (!selectedModel) return;

    setTimeout(() => {
      renderModelLabelAssignments({ target, editable, preselectedModel: selectedModel });
    }, 0);
  };

  // Exit early if no model is selected
  if (!preselectedModel) return;

  const res = await fetch('/api/models/labels/get', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_name: preselectedModel }),
  });

  const { labels: classMap } = await res.json();

  Object.entries(classMap).forEach(([cls, uuid]) => {
    const row = document.createElement('div');
    row.style.marginBottom = '0.5em';

    const labelSpan = document.createElement('span');
    labelSpan.style.marginRight = '0.5em';

    const labelAssigned = (uuid && _cachedLabels.has(uuid))
    if (labelAssigned) {
      const labelMeta = _cachedLabels.get(uuid);
      labelSpan.textContent = cls + ' ';
      // Create span for assigned label name with color
      const assignedLabelSpan = document.createElement('span');
      assignedLabelSpan.textContent = labelMeta.name;
      assignedLabelSpan.style.color = labelMeta.color || 'inherit';
      assignedLabelSpan.style.fontWeight = 'bold';

      labelSpan.appendChild(assignedLabelSpan);
    } else {
      labelSpan.textContent = cls + ' (unassigned)';
    }

    const labelRow = document.createElement('div');
    labelRow.style.display = 'flex';
    labelRow.style.alignItems = 'center';
    labelRow.style.gap = '0.5em';

    labelRow.appendChild(labelSpan);

    if (editable) {
      const editBtn = createButton('emoji-button', 'Edit', '✏️');
      labelRow.appendChild(editBtn);

      // --- Add Clear button ---
      if (labelAssigned) {
        const clearBtn = createButton('emoji-button', 'Clear', '🗑️');
        clearBtn.style.marginLeft = '0.3em';
        clearBtn.onclick = async () => {
          await fetch('/api/models/labels/associate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model_name: preselectedModel,
              model_class: cls,
              label_uuid: null
            }),
          });
          renderModelLabelAssignments({ target, editable, preselectedModel });
        };
        labelRow.appendChild(clearBtn);
      }

      const labelListContainer = document.createElement('div');
      labelListContainer.style.marginTop = '0.5em';

      editBtn.onclick = async () => {
        labelListContainer.innerHTML = '';

        const labels = Array.from(_cachedLabels.values());

        const labelList = document.createElement('div');
        labelList.style.marginTop = '0.5em';

        const helpContainer = document.createElement('div');
        helpContainer.style.display = 'flex';
        helpContainer.style.alignItems = 'center';
        helpContainer.style.marginBottom = '0.5em';

        const closeBtn = document.createElement('button');
        closeBtn.textContent = 'X';
        closeBtn.style.cursor = 'pointer';
        closeBtn.style.background = 'transparent';
        closeBtn.style.border = 'none';
        closeBtn.style.color = 'red';
        closeBtn.style.fontWeight = 'bold';
        closeBtn.style.fontSize = '1em';
        closeBtn.style.padding = '0 0.3em';
        closeBtn.style.lineHeight = '1';
        closeBtn.style.marginRight = '0.3em';

        closeBtn.onclick = () => {
          labelListContainer.innerHTML = '';
        };

        const helpText = document.createElement('div');
        helpText.textContent = 'Click a label to assign it to ' + cls;
        helpText.style.fontStyle = 'italic';
        helpText.style.fontSize = '0.9em';

        helpContainer.appendChild(closeBtn);
        helpContainer.appendChild(helpText);
        labelList.appendChild(helpContainer);

        const labelTextContainer = document.createElement('span');
        labelTextContainer.style.display = 'flex';
        labelTextContainer.style.flexWrap = 'wrap';
        labelTextContainer.style.gap = '0.5em';
        labelTextContainer.style.alignItems = 'center';
        labelTextContainer.style.marginTop = '0.3em';

        labels.forEach((label, index) => {
          const labelSpan = document.createElement('span');
          labelSpan.style.cursor = 'pointer';
          labelSpan.style.display = 'inline-block';

          if (label.color) {
            labelSpan.style.color = label.color;
          }

          labelSpan.textContent = label.name;

          labelSpan.onclick = async () => {
            await fetch('/api/models/labels/associate', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                model_name: preselectedModel,
                model_class: cls,
                label_uuid: label.uuid,
              }),
            });
            renderModelLabelAssignments({ target, editable, preselectedModel });
          };

          labelTextContainer.appendChild(labelSpan);

          if (index < labels.length - 1) {
            const comma = document.createElement('span');
            comma.textContent = ',';
            labelTextContainer.appendChild(comma);
          }
        });

        labelList.appendChild(labelTextContainer);
        labelListContainer.appendChild(labelList);
      };

      row.appendChild(labelListContainer);
    }

    row.appendChild(labelRow);
    container.appendChild(row);
  });
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

function pollTaskStatus(taskUuid, resultBox, onComplete, timeoutMs = 30000) {
  const interval = setInterval(async () => {
    try {
      const res = await fetch(`/api/tasks/status`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_ids: [taskUuid] }),
      });
      const result = await res.json();

      if (result.status === "success") {
        const taskStatus = result.tasks[taskUuid];
        resultBox.textContent = taskStatus.message;

        if (taskStatus.status === 'completed' || taskStatus.status === 'error') {
          clearInterval(interval);
          clearTimeout(timeout);
          const taskRes = await fetch(`/api/tasks/result`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: taskUuid }),
          });
          const taskResult = await taskRes.json();
          onComplete(taskResult.result);
        }
      }
    } catch (err) {
      console.error("Polling error:", err);
    }
  }, 1000);

  const timeout = setTimeout(() => {
    clearInterval(interval);
    resultBox.textContent = "Task timed out.";
    onComplete({ status: 'timeout', message: 'The task did not complete in time.' });
  }, timeoutMs);
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


export async function renderTaskList({ parent, editable = false, onEdit = () => { }, onDelete = () => { } }) {
  parent.innerHTML = '';

  const res = await fetch('/api/tasks/list-avail');
  const data = await res.json();

  if (data.status !== 'success') {
    parent.textContent = `Error loading tasks: ${data.message || 'Unknown error'}`;
    return;
  }

  data.tasks.forEach(task => {
    const row = document.createElement('div');
    row.style.display = 'flex';
    row.style.justifyContent = 'space-between';
    row.style.alignItems = 'center';
    row.style.marginBottom = '0.5em';

    const label = document.createElement('span');
    label.textContent = `${task.typename}: ${task.name || '(no name)'}`;
    row.appendChild(label);

    if (editable) {
      const actions = document.createElement('div');
      actions.style.display = 'flex';
      actions.style.gap = '0.5em';

      const editBtn = document.createElement('button');
      editBtn.textContent = 'Edit';
      editBtn.onclick = () => onEdit(task);
      actions.appendChild(editBtn);

      const deleteBtn = document.createElement('button');
      deleteBtn.textContent = 'Delete';
      deleteBtn.onclick = async () => {
        if (!confirm(`Are you sure you want to delete "${task.name}"?`)) return;
        await fetch('/api/tasks/delete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ task_ids: [task.id || task.uuid] }),
        });
        onDelete();
      };
      actions.appendChild(deleteBtn);

      row.appendChild(actions);
    }

    parent.appendChild(row);
  });
}


export async function renderTaskForm({ parent, onCreate, existingTask = null, onCancel = null }) {
  const formTitle = document.createElement('h4');
  formTitle.textContent = existingTask ? 'Edit Task' : 'Create New Task';
  parent.appendChild(formTitle);

  const form = document.createElement('form');
  form.style.display = 'flex';
  form.style.flexDirection = 'column';
  form.style.gap = '0.5em';

  const row = document.createElement('div');
  row.style.display = 'grid';
  row.style.gridTemplateColumns = '1fr 1fr auto';
  row.style.gap = '0.5em';

  const typeSelect = document.createElement('select');
  typeSelect.required = true;
  row.appendChild(typeSelect);

  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.placeholder = 'Task name';
  nameInput.required = true;
  row.appendChild(nameInput);

  const btnWrapper = document.createElement('div');
  btnWrapper.style.display = 'flex';
  btnWrapper.style.flexDirection = 'column';

  const saveBtn = createButton('emoji-button', 'Save', existingTask ? '✅' : '➕');
  saveBtn.type = 'submit';
  btnWrapper.appendChild(saveBtn);

  if (existingTask && onCancel) {
    const cancelBtn = createButton('emoji-button', 'Cancel', '❌');
    cancelBtn.onclick = () => onCancel();
    btnWrapper.appendChild(cancelBtn);
  }

  row.appendChild(btnWrapper);
  form.appendChild(row);

  const paramContainer = document.createElement('div');
  paramContainer.style.marginTop = '1em';
  form.appendChild(paramContainer);

  parent.appendChild(form);

  // Load available task types
  const res = await fetch('/api/tasks/list-avail');
  const { types } = await res.json();

  types.forEach(type => {
    const opt = document.createElement('option');
    opt.value = type;
    opt.textContent = type;
    typeSelect.appendChild(opt);
  });

  typeSelect.onchange = async () => {
    const selectedType = typeSelect.value;
    const schemaRes = await fetch('/api/tasks/schema', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ typename: selectedType }),
    });
    const { schema } = await schemaRes.json();
    renderSchemaForm(schema, paramContainer);
  };

  // Pre-fill values if editing
  if (existingTask) {
    nameInput.value = existingTask.name;
    typeSelect.value = existingTask.typename;
    typeSelect.dispatchEvent(new Event('change'));

    setTimeout(() => {
      Object.entries(existingTask.params || {}).forEach(([key, value]) => {
        const el = paramContainer.querySelector(`[name="${key}"]`);
        if (!el) return;
        if (el.type === 'checkbox') el.checked = !!value;
        else el.value = value;
      });
    }, 100);
  }

  form.onsubmit = async e => {
    e.preventDefault();

    const name = nameInput.value;
    const typename = typeSelect.value;
    const params = getParamsFromForm(paramContainer);

    const payload = {
      name,
      typename,
      params,
    };

    if (existingTask?.uuid) {
      payload.task_uuid = existingTask.uuid;
    }

    const url = existingTask ? '/api/tasks/configured/update' : '/api/tasks/configured/create';

    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const result = await res.json();

    if (result.status === 'success') {
      if (onCreate) onCreate();
      parent.innerHTML = '';
      await renderTaskForm({ parent, onCreate });
    } else {
      toast(result.message || 'Unknown error', 5000, 'error');
    }
  };
}


// ================= Task Management Tab =================

/**
 * Renders the task configs (left column).
 * @param {object} params
 * @param {string} params.target - ID of the container
 */
export async function renderTaskConfigs({ target = 'task-config-list' }) {
  const container = document.getElementById(target);
  if (!container) return;
  container.innerHTML = '';

  const header = document.createElement('h3');
  header.textContent = 'Task Configs';
  container.appendChild(header);

  const list = document.createElement('div');
  container.appendChild(list);

  const refresh = async () => {
    list.innerHTML = '';
    const res = await fetch('/api/tasks/configs/list');
    const data = await res.json();
    if (data.status !== 'success') {
      list.textContent = `Error loading configs: ${data.message || 'Unknown'}`;
      return;
    }

    data.configs.forEach(task => {
      const row = document.createElement('div');
      row.style.display = 'flex';
      row.style.justifyContent = 'space-between';
      row.style.marginBottom = '0.5em';

      const label = document.createElement('span');
      label.textContent = `${task.typename}: ${task.name || '(no name)'}`;
      row.appendChild(label);

      const actions = document.createElement('div');
      actions.style.display = 'flex';
      actions.style.gap = '0.5em';

      const editBtn = createButton('emoji-button', 'Edit', '✏️');
      editBtn.onclick = async () => {
        await renderTaskForm({
          parent: container,
          existingTask: task,
          onCreate: refresh,
          onCancel: refresh
        });
      };
      actions.appendChild(editBtn);

      const deleteBtn = createButton('emoji-button', 'Delete', '🗑️');
      deleteBtn.onclick = async () => {
        if (!confirm(`Delete task config "${task.name}"?`)) return;
        await fetch('/api/tasks/configs/delete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ config_uuids: [task.uuid] }),
        });
        await refresh();
      };
      actions.appendChild(deleteBtn);

      row.appendChild(actions);
      list.appendChild(row);
    });

    // "New Task Config" button
    const newBtn = createButton('emoji-button', 'New Task Config', '➕');
    newBtn.onclick = () => renderTaskForm({ parent: container, onCreate: refresh });
    list.appendChild(newBtn);
  };

  await refresh();
}

/**
 * Renders the active tasks (right column).
 * @param {object} params
 * @param {string} params.target - ID of the container
 */
export async function renderActiveTasks({ target = 'active-tasks-list' }) {
  const container = document.getElementById(target);
  if (!container) return;
  container.innerHTML = '';

  const header = document.createElement('h3');
  header.textContent = 'Active Tasks';
  container.appendChild(header);

  const list = document.createElement('div');
  container.appendChild(list);

  const refresh = async () => {
    list.innerHTML = '';
    const res = await fetch('/api/tasks/status', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ uuids: null }),
    });
    const data = await res.json();
    if (data.status !== 'success') {
      list.textContent = `Error loading tasks: ${data.message || 'Unknown'}`;
      return;
    }

    data.tasks.forEach(task => {
      const row = document.createElement('div');
      row.style.display = 'flex';
      row.style.justifyContent = 'space-between';
      row.style.marginBottom = '0.5em';

      const label = document.createElement('span');
      label.textContent = `${task.typename}: ${task.name || '(no name)'}`;
      row.appendChild(label);

      const stopBtn = createButton('emoji-button', 'Stop', '🛑');
      stopBtn.onclick = async () => {
        if (!confirm(`Stop task "${task.name}"?`)) return;
        await fetch('/api/tasks/stop', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ uuids: [task.uuid] }),
        });
        await refresh();
      };
      row.appendChild(stopBtn);

      list.appendChild(row);
    });
  };

  await refresh();
}
