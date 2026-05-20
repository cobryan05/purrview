import { BboxInfoField, EditableField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, Logger } from '/app-static/js/ui/utils/index.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { fetchLabelByUuid } from '/app-static/js/shared/api/labels.js';

/**
 * Opens the inspector tab in the sidebar and highlights it briefly.
 */
export function openInspectorTab() {
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;

  const tabs = sidebar.querySelectorAll('.sidebar-tab-button');
  const panes = sidebar.querySelectorAll('.sidebar-tab-pane');

  tabs.forEach(tab => {
    const tabName = tab.getAttribute('data-tab');
    tab.classList.toggle('active', tabName === 'tab-inspector');
  });

  panes.forEach(pane => {
    pane.classList.toggle('active', pane.id === 'tab-inspector');
  });

  document.getElementById('classInput')?.focus();

  const inspector = document.getElementById('tab-inspector');
  if (inspector) {
    inspector.style.outline = '2px solid #ff0033';
    setTimeout(() => inspector.style.outline = '', 1000);
  }
}


export function renderBboxInspector({ bboxView = null, target = 'tab-inspector', editable = true, onSelectCallback = null } = {}) {
  if (!bboxView) return;
  try {
    const targetElement = document.getElementById(target);
    if (!targetElement) {
      console.error('Class container not found');
      return;
    }
    targetElement.innerHTML = 'Inspector Loading...'
    const container = document.createElement('div');
    container.style.display = 'inline-block';
    container.style.verticalAlign = 'top';
    container.style.width = 'max-content';
    container.style.minWidth = '0';
    targetElement.appendChild(container);

    BboxInfoField.create({ bboxGroup: bboxView }).then(bboxInfoFieldInstance => {
      const row = createGenericRow({
        field: new EditableField({
          field: bboxInfoFieldInstance,
          editMode: false,
          buttonsLast: true,
          onChange: (val) => {
            Logger.warn(val);
          },
          onSave: async (val) => {
            const labelUuid = val.bbox_info.labelUuid;
            if (labelUuid) {
              const labelData = await fetchLabelByUuid(labelUuid);
              if (labelData) {
                bboxView.uiBBox.label = { uuid: labelUuid, text: labelData.name, color: labelData.color };
                bboxView.updateAppearance(labelData.name, labelData.color);
              }
            } else {
              bboxView.uiBBox.label = undefined;
              bboxView.updateAppearance('', 'grey');
            }
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
    console.error('Failed to render bbox inspector:', err);
  }

}


events.subscribe(
  EventTypes.CANVAS_BBOX_DOUBLE_CLICKED,
  onCanvasBboxSelected
);

/**
 * @param {import('@events').CanvasBboxDoubleClickedPayload} payload
 */
function onCanvasBboxSelected(payload) {
  openInspectorTab();
  renderBboxInspector({ bboxView: payload.bboxView });
}
