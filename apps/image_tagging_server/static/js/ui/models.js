import { fetchClasses } from '/app-static/js/shared/api/classes.js';
import { associateClass, clearModelsCache, fetchModelsClassMappings, fetchModelsList } from '/app-static/js/shared/api/models.js';
import { DropDownField, EditableField, LabelDropDownField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, Logger, toast } from '/app-static/js/ui/utils/index.js';

export function renderModelClassAssignments({
  target = "model-class-box",
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
            renderModelClassAssignments({ target, editable, preselectedModel: value });
          }, 0);
        }
      }),
      rightButtons: [
        {
          text: 'Refresh',
          emoji: '🔄',
          onClick: () => {
            clearModelsCache();
            renderModelClassAssignments({ target, editable, preselectedModel: preselectedModel });
          }
        }
      ]
    });
    modelSelectDiv.appendChild(modelDropdownRow);
  });

  if (!preselectedModel) return;

  const classAssignmentDiv = document.createElement('div');
  container.appendChild(classAssignmentDiv);

  // --- Load model-class assignments ---
  Promise.all([
    fetchModelsClassMappings({modelNames: preselectedModel}),
    fetchClasses()
  ]).then(([classMap, allClasses]) => {
    // Convert classes to the format required by DropDownField
    const classItems = Array.from(allClasses.values()).map(cls => ({
      key: cls.metadata.uuid,
      text: cls.metadata.name,
      color: cls.metadata.color || '#cccccc'
    }));

    classMap.forEach((assignedUuid, modelClass) => {
      const assignedClass = allClasses.get(assignedUuid);
      const classKey = assignedClass ? { key: assignedClass.metadata.uuid, text: assignedClass.metadata.name } : assignedUuid
      const deleteButton = {
        text: 'Delete',
        emoji: '🗑️',
        onClick: async () => {
          try {
            await associateClass(preselectedModel, modelClass, null);
          } catch (err) {
            toast(err.message, 5000, 'error');
          }
          renderModelClassAssignments({ target, editable, preselectedModel: preselectedModel });
        },
      };
      const assignmentRow = createGenericRow({
        field: new EditableField({
          field: new LabelDropDownField({
            options: classItems,
            labelText: modelClass,
            value: classKey
          }),
          editable: true,
          editMode: false,
          buttonsLast: true,
          onSave: async uuid => {
            // Save the selected class
            if (uuid) {
              await associateClass(preselectedModel, modelClass, uuid);
            }
            renderModelClassAssignments({ target, editable, preselectedModel: preselectedModel });
          },
          onCancel: () => {
            renderModelClassAssignments({ target, editable, preselectedModel: preselectedModel });
          },
        }),
        rightButtons: assignedUuid ? [deleteButton] : []
      });

      classAssignmentDiv.appendChild(assignmentRow);
    });
  }).catch(err => Logger.error(err));
}
