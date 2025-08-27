import { getParamsFromForm } from './utils.js';
import { createButton, toast } from '/app-static/js/ui/utils/index.js';


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
