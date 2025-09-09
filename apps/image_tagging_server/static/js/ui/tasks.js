import { EditableField, TaskConfigField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow } from '/app-static/js/ui/utils/createGenericRow.js'
import { error, fetchTaskTypeList, fetchTaskConfigs } from '/app-static/js/ui/utils/index.js'

/**
 * Renders the task configs (left column).
 * @param {object} params
 * @param {string} params.target - ID of the container
 */
export function renderTaskConfigs({ target = 'task-config-list' }) {
  const container = document.getElementById(target);
  if (!container) return;
  container.innerHTML = '';

  const configListDiv = document.createElement('div');
  container.appendChild(configListDiv);

  fetchTaskConfigs().then(configs => {
    configs.forEach((config, uuid) => {
      TaskConfigField.create({ value: config.params, ...config }).then(fieldInstance => {
        const row = createGenericRow({
          field: new EditableField({
            field: fieldInstance,
            onSave: async ({ text: name, typename, schema: filled_schema }) => {
              //await updateSource({ uuid, name, providerName: typename, params: filled_schema });
              renderTaskConfigs({ target });
            },
            onCancel: () => {
              renderTaskConfigs({ target });
            },
          }),
        })
        configListDiv.appendChild(row);
      });
    });
  });


  container.appendChild(document.createElement('hr'));
  const createHeader = document.createElement('h3');
  createHeader.textContent = 'Create New Task Config';
  container.appendChild(createHeader);

  const newConfigDiv = document.createElement('div');
  container.appendChild(newConfigDiv);
  TaskConfigField.create({ editMode: true }).then(fieldInstance => {
    const row = createGenericRow({
      field: fieldInstance,
      editMode: true,
    })
    newConfigDiv.appendChild(row);
  });
}

/**
 * Renders the active tasks (right column).
 * @param {object} params
 * @param {string} params.target - ID of the container
 */
export function renderActiveTasks({ target = 'active-tasks-list' }) {
  const container = document.getElementById(target);
  if (!container) return;
  container.innerHTML = '';
}
