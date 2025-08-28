import { associateLabel, createGenericRow, DropDownField, getAllLabels, getAllModelNames, getLabelByUuid, getModelMappings, LabelDropDownField, refreshLabelCache, refreshModelCache } from './utils/index.js';

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
  const allLabels = getAllLabels(); // Use the label cache to get all labels

  // Convert labels to the format required by DropDownField
  const labelItems = Array.from(allLabels.values()).map(label => ({
    key: label.metadata.uuid,
    text: label.metadata.name,
    color: label.metadata.color || '#cccccc'
  }));

  Object.entries(classMap).forEach(async ([cls, uuid]) => {
    const assignedLabel = await getLabelByUuid(uuid);

    const assignmentRow = createGenericRow({
      field: new LabelDropDownField({
        options: labelItems,
        labelText: cls,
        value: assignedLabel?.metadata.uuid || null,
        editable,
        onChange: async selectedUuid => {
          const selectedLabel = labelItems.find(item => item.key === selectedUuid);
          const labelUuid = selectedLabel?.key || null;

          await associateLabel(preselectedModel, cls, labelUuid);

          renderModelLabelAssignments({ target, editable, preselectedModel });
        }
      }),
      editable
    });

    container.appendChild(assignmentRow);
  });
}
