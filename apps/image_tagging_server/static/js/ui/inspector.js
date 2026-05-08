import { refreshCanvas } from '../app/image-tagging/canvas/image.js'; // TODO: Better way?
import { BboxInfoField, EditableField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, Logger } from '/app-static/js/ui/utils/index.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';

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


export function renderBboxInspector({ bboxUuid = null, target = 'tab-inspector', editable = true, onSelectCallback = null } = {}) {
  bboxUuid = bboxUuid ?? appState.imageTagging.selectedBboxUuid;
  const bbox = appState.imageTagging.getBboxGroup(bboxUuid)
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

    BboxInfoField.create({ bboxGroup: bbox }).then(bboxInfoFieldInstance => {
      const row = createGenericRow({
        field: new EditableField({
          field: bboxInfoFieldInstance,
          editMode: false,
          buttonsLast: true,
          onChange: (val) => {
            Logger.warn(val);
          },
          onSave: async (val) => {
            const bboxGroup = appState.imageTagging.getBboxGroup(bboxUuid);
            if (!bboxGroup) {
              Logger.error(`No bbox found for UUID: ${bboxUuid}`);
              return;
            }
            bboxGroup.updateMetadata({ class_uuid: val.bbox_info.class_uuid, tagUuids: val.bbox_info.tagUuids });
            appState.imageTagging.updateCanvasBboxView(bboxGroup);
            await refreshCanvas(appState.imageTagging);
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


events.subscribe(
  EventTypes.CANVAS_BBOX_DOUBLE_CLICKED,
  onCanvasBboxSelected
);

/**
 * @param {import('@events').CanvasBboxDoubleClickedPayload} payload
 */
function onCanvasBboxSelected(payload) {
  openInspectorTab();
  renderBboxInspector({ bboxUuid: payload.bboxId });
}
