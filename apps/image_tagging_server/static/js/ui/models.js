import { refreshModelCache, getAllModelNames, getModelMappings, associateLabel, } from './utils/modelCache.js';
import { refreshLabelCache, getLabelByUuid, getLabelByName, getAllLabelNames, } from './utils/labelCache.js';
import { createGenericRow, DropDownField, LabelDropDownField, } from './utils/index.js';

export async function renderModelLabelAssignments({
  target = "model-labels-box",
  editable = false,
  preselectedModel = null
} = {}) {
  const container = document.getElementById(target);
  container.innerHTML = '';

  // --- Header row: model selection dropdown ---
  await refreshModelCache();
  const models = getAllModelNames();

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
  const classMap = await getModelMappings(preselectedModel);
  const dropdownOptions = await getAllLabelNames();

  Object.entries(classMap).forEach(async ([cls, uuid]) => {
    const assignedLabel = await getLabelByUuid(uuid);

    const assignmentRow = createGenericRow({
      field: new LabelDropDownField({
        labelText: cls,
        value: assignedLabel?.name || '',
        options: dropdownOptions,
        placeholder: '(unassigned)',
        onChange: async newLabelName => {
          const selectedLabel = await getLabelByName(newLabelName);
          const labelUuid = selectedLabel?.uuid || null;

          await associateLabel(preselectedModel, cls, labelUuid);

          renderModelLabelAssignments({ target, editable, preselectedModel });
        },
        onEdit: () => {
          renderModelLabelAssignments({ target, editable: true, preselectedModel });
        }
      }),
      editable
    });

    container.appendChild(assignmentRow);
  });
}
