import { createButton, createGenericRow, getParamsFromForm } from './utils.js'
import { getCurrentLabelUuid, refreshLabelList, getLabelList } from '/app-static/js/canvas/state.js';
import { getTool } from '/app-static/js/canvas/tools.js';
import { toast } from '/app-static/js/canvas/utils.js';

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
