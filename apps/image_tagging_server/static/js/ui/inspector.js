import { state } from '/app-static/js/canvas/state.js';
import { getTool } from '/app-static/js/canvas/tools.js';
import { createGenericRow, createNewClass, deleteClass, Logger, fetchClasses, toast, updateClass } from '/app-static/js/ui/utils/index.js';
import { EditableField, BboxInfoField } from '/app-static/js/ui/utils/fields/index.js';
import { clearClassCache } from './utils/classesApi.js';
import { refreshCanvas } from '../canvas/image.js' // TODO: Better way?


/**
 * Opens the inspector tab in the sidebar and highlights it briefly.
 */
export function openInspectorTab() {
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;

  const tabs = sidebar.querySelectorAll('.tab-button');
  const panes = sidebar.querySelectorAll('.tab-pane');

  tabs.forEach(tab => {
    const tabName = tab.getAttribute('data-tab');
    tab.classList.toggle('active', tabName === 'tab-pane-inspector');
  });

  panes.forEach(pane => {
    pane.classList.toggle('active', pane.id === 'tab-pane-inspector');
  });

  document.getElementById('classInput')?.focus();

  const inspector = document.getElementById('tab-pane-inspector');
  if (inspector) {
    inspector.style.outline = '2px solid #ff0033';
    setTimeout(() => inspector.style.outline = '', 1000);
  }
}


export function renderBboxInspector({ bboxUuid = null, target = 'tab-pane-inspector', editable = true, onSelectCallback = null } = {}) {
  bboxUuid = bboxUuid ?? state.selectedBboxUuid;
  const bbox = state.getBboxGroup(bboxUuid)
  try {
    const targetElement = document.getElementById(target);
    if (!targetElement) {
      console.error('Class container not found');
      return;
    }
    targetElement.innerHTML = 'Inspector Loading...'
    const container = document.createElement('div');
    container.style.display = 'inline-block';      // shrink-wrap width
    container.style.verticalAlign = 'top';         // optional, align with top of parent
    container.style.width = 'max-content';         // shrink to longest content
    container.style.minWidth = '0';                // prevent overflow issues
    targetElement.appendChild(container);

    BboxInfoField.create({ bbox: bbox }).then(bboxInfoFieldInstance => {
      const row = createGenericRow({
        field: new EditableField({
          field: bboxInfoFieldInstance,
          editMode: false,
          buttonsLast: true,
          onChange: (val) => {
            Logger.warn(val);
          },
          onSave: (val) => {
            const group = state.getBboxGroup(bboxUuid);
            if (!group) {
              Logger.error(`No bbox found for UUID: ${bboxUuid}`);
              return;
            }

            group.metadata = {
              ...group.metadata, // preserve other metadata fields
              classUuid: val.bbox_info.classUuid,
              tagUuids: val.bbox_info.tagUuids || [],
            };
            state.setBboxGroup(bboxUuid, group);
            refreshCanvas();
          },
          onCancel: () => {
            Logger.warn("Canceled");
          }
        }),
      });
      container.appendChild(row);
      targetElement.innerHTML = '';
      targetElement.appendChild(container);
    });
  } catch (err) {
    console.error('Failed to fetch classes:', err);
  }

}
