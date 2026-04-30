import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { associateLabel, clearModelsCache, fetchModelsLabelMappings, fetchModelsList } from '/app-static/js/shared/api/models.js';
import { DropDownField, EditableField, LabelDropDownField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, Logger, toast } from '/app-static/js/ui/utils/index.js';

export function renderModelLabelAssignments({
  target = "model-labels-box",
  editable = false,
  preselectedModel = null
} = {}) {
  const container = document.getElementById(target);
  container.innerHTML = '';

  const modelSelectDiv = document.createElement('div');
  modelSelectDiv.style.display = 'inline-block';
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
    fetchModelsLabelMappings({modelNames: preselectedModel}),
    fetchLabels()
  ]).then(([labelMap, allLabels]) => {
    // Convert labels to the format required by DropDownField
    const labelItems = Array.from(allLabels.values()).map(label => ({
      key: label.uuid,
      text: label.name,
      color: label.color || '#cccccc'
    }));

    labelMap.forEach((assignedUuid, modelClass) => {
      const assignedLabel = allLabels.get(assignedUuid);
      const classKey = assignedLabel ? { key: assignedLabel.uuid, text: assignedLabel.name } : assignedUuid
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
            value: classKey
          }),
          editable: true,
          editMode: false,
          buttonsLast: true,
          onSave: async label_uuid => {
            // Save the selected class
            if (label_uuid) {
              await associateLabel(preselectedModel, modelClass, label_uuid);
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
  }).catch(err => Logger.error(err));
}
