import { createButton, createGenericRow, LabelDropDownField, DropDownField } from './utils/index.js'

let _cachedLabels = null;

export async function renderModelLabelAssignments({
  target = "model-labels-box",
  editable = false,
  preselectedModel = null
} = {}) {
  const container = document.getElementById(target);
  container.innerHTML = '';

  // Load and cache labels metadata
  if (!_cachedLabels) {
    const labelRes = await fetch('/api/labels/list');
    const { labels } = await labelRes.json();
    _cachedLabels = new Map(labels.map(l => [l.metadata.uuid, l.metadata]));
  }

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
          _cachedLabels = null; // clear cache to reload
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

  Object.entries(classMap).forEach(([cls, uuid]) => {
    const assignedLabel = uuid && _cachedLabels.has(uuid) ? _cachedLabels.get(uuid) : null;

    const dropdownOptions = Array.from(_cachedLabels.values()).map(l => l.name);
    const assignmentField = new LabelDropDownField({
      labelText: cls,
      value: assignedLabel?.name || '',
      options: dropdownOptions,
      placeholder: '(unassigned)',
      onChange: async newLabelName => {
        const selectedLabel = Array.from(_cachedLabels.values()).find(l => l.name === newLabelName);
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

    const row = createGenericRow({
      field: assignmentField,
      editable,
      rightButtons: assignedLabel ? [
        {
          text: 'Clear',
          emoji: '🗑️',
          onClick: async () => {
            await fetch('/api/models/labels/associate', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                model_name: preselectedModel,
                model_class: cls,
                label_uuid: null,
              }),
            });
            renderModelLabelAssignments({ target, editable, preselectedModel });
          }
        }
      ] : [],
    });

    container.appendChild(row);
  });
}
