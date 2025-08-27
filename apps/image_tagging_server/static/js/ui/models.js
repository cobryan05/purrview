import { createButton, createGenericRow, LabelDropDownField, DropDownField, getLabelByUuid, getAllLabelNames, refreshLabelCache } from './utils/index.js'


export async function renderModelLabelAssignments({
  target = "model-labels-box",
  editable = false,
  preselectedModel = null
} = {}) {
  const container = document.getElementById(target);
  container.innerHTML = '';

  // --- Header row: model selection dropdown ---
  const modelRes = await fetch('/api/models/list');
  const { models } = await modelRes.json();

  const modelDropdownRow = createGenericRow({
    field: new DropDownField({
      value: preselectedModel,
      options: models,
      placeholder: 'Choose a model...',
      onChange: value => {
        if (!value) return;
        setTimeout(() => {
          renderModelLabelAssignments({ target, editable, preselectedModel: value });
        }, 0);
      }
    }),
    editable: true,
    rightButtons: [
      {
        text: 'Refresh',
        emoji: '🔄',
        onClick: () => {
          refreshLabelCache();
          renderModelLabelAssignments({ target, editable });
        }
      }
    ]
  });

  container.appendChild(modelDropdownRow);

  if (!preselectedModel) return;

  // --- Load model-label assignments ---
  const res = await fetch('/api/models/labels/get', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_name: preselectedModel }),
  });
  const { labels: classMap } = await res.json();

  Object.entries(classMap).forEach(async ([cls, uuid]) => {
    const assignedLabel = await getLabelByUuid(uuid);

    const dropdownOptions = await getAllLabelNames();
    const assignmentField = new LabelDropDownField({
      labelText: cls,
      value: assignedLabel?.name || '',
      options: dropdownOptions,
      placeholder: '(unassigned)',
      onChange: async newLabelName => {
        const selectedLabel = await getLabelByName(newLabelName);
        const labelUuid = selectedLabel?.uuid || null;

        await fetch('/api/models/labels/associate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model_name: preselectedModel,
            model_class: cls,
            label_uuid: labelUuid,
          }),
        });

        renderModelLabelAssignments({ target, editable, preselectedModel });
      },
      onEdit: () => {
        renderModelLabelAssignments({ target, editable: true, preselectedModel });
      }
    });
  });
}
