import { DropDownField, EditableField, LabelDropDownField } from '/app-static/js/ui/utils/fields/index.js';
import { associateLabel, clearModelsCache, createGenericRow, error, fetchLabels, fetchModelLabelMappings, fetchModelsList } from '/app-static/js/ui/utils/index.js';


export function renderModelLabelAssignments({
  target = "model-labels-box",
  editable = false,
  preselectedModel = null
} = {}) {
  const container = document.getElementById(target);
  container.innerHTML = '';

  const modelSelectDiv = document.createElement('div');
  container.appendChild(modelSelectDiv);
  // --- Header row: model selection dropdown ---
  fetchModelsList().then(models => {
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
            clearModelsCache();
            renderModelLabelAssignments({ target, editable, preselectedModel: preselectedModel });
          }
        }
      ]
    });
    modelSelectDiv.appendChild(modelDropdownRow);
  });

  if (!preselectedModel) return;

  const labelAssignmentDiv = document.createElement('div');
  container.appendChild(labelAssignmentDiv);

  // --- Load model-label assignments ---
  Promise.all([
    fetchModelLabelMappings(preselectedModel),
    fetchLabels()
  ]).then(([classMap, allLabels]) => {
    // Convert labels to the format required by DropDownField
    const labelItems = Array.from(allLabels.values()).map(label => ({
      key: label.metadata.uuid,
      text: label.metadata.name,
      color: label.metadata.color || '#cccccc'
    }));

    classMap.forEach((assignedUuid, modelClass) => {
      const assignedLabel = allLabels.get(assignedUuid);
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
          onCancel: () => {
            renderModelLabelAssignments({ target, editable, preselectedModel: preselectedModel });
          },
        }),
        rightButtons: assignedUuid ? [deleteButton] : []
      });

      labelAssignmentDiv.appendChild(assignmentRow);
    });
  }).catch(err => error(err));
}
