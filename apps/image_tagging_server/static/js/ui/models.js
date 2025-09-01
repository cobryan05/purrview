import { DropDownField, EditableField, LabelDropDownField } from '/app-static/js/ui/utils/fields/index.js';
import { associateLabel, createGenericRow, getAllLabels, getAllModelNames, getLabelByUuid, getModelMappings, refreshLabelCache, refreshModelCache } from '/app-static/js/ui/utils/index.js';


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
      editMode: true,
      options: models,
      placeholder: 'Choose a model...',
      onChange: value => {
        if (!value) return;
        setTimeout(() => {
          renderModelLabelAssignments({ target, editable, preselectedModel: value });
        }, 0);
      }
    }),
    rightButtons: [
      {
        text: 'Refresh',
        emoji: '🔄',
        onClick: () => {
          refreshLabelCache();
          renderModelLabelAssignments({ target, editable, preselectedModel: preselectedModel });
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

  Object.entries(classMap).forEach(async ([modelClass, assignedUuid]) => {
    const assignedLabel = await getLabelByUuid(assignedUuid);
    const labelKey = assignedLabel ? { key: assignedLabel.metadata.uuid, text: assignedLabel.metadata.name } : assignedUuid
    const deleteButton = {
      text: 'Delete',
      emoji: '🗑️',
      onClick: async () => {
        try {
          await associateLabel(preselectedModel, modelClass, null);
        } catch (err) {
          toast(err.message, 5000, 'error');
        }
        renderModelLabelAssignments({ target, editable, preselectedModel: preselectedModel });
      },
    };
    const assignmentRow = createGenericRow({
      field: new EditableField({
        field: new LabelDropDownField({
          options: labelItems,
          labelText: modelClass,
          value: labelKey
        }),
        editable: true,
        editMode: false,
        buttonsLast: true,
        onSave: async ({ option }) => {
          // Save the selected label
          if (option?.key) {
            await associateLabel(preselectedModel, modelClass, option.key);
          }
          renderModelLabelAssignments({ target, editable, preselectedModel: preselectedModel });
        },
        onCancel: async () => {
          renderModelLabelAssignments({ target, editable, preselectedModel: preselectedModel });
        },
      }),
      rightButtons: assignedUuid ? [deleteButton] : []
    });

    container.appendChild(assignmentRow);
  });
}
