import { BboxInfoField, EditableField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, Logger } from '/app-static/js/ui/utils/index.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { fetchLabelByUuid } from '/app-static/js/shared/api/labels.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import { UNVERIFIED_TAG_UUID } from '/app-static/js/shared/api/review.js';

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


/**
 * @param {{ bboxView?: any, target?: string|HTMLElement, editable?: boolean, onSelectCallback?: any }} [opts]
 *   target – a string element ID *or* an HTMLElement; defaults to 'tab-inspector'
 */
export function renderBboxInspector({ bboxView = null, target = 'tab-inspector', editable = true, onSelectCallback = null, runtimeId = null } = {}) {
  if (!bboxView) return;
  try {
    const targetElement = typeof target === 'string' ? document.getElementById(target) : target;
    if (!targetElement) {
      console.error('Class container not found');
      return;
    }
    targetElement.innerHTML = 'Inspector Loading...'
    const container = document.createElement('div');
    container.style.display = 'block';
    container.style.width = '100%';
    container.style.minWidth = '0';
    container.style.overflow = 'hidden';
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
            const originalLabelUuid = bboxView.uiBBox.label?.uuid ?? null;
            const labelUuid = val.bbox_info.labelUuid ?? null;
            const labelChanged = labelUuid !== originalLabelUuid;

            // Build new tag list. Only strip UNVERIFIED when the user is actively
            // assigning/changing a label — not when they're just editing tags on a
            // bbox that already has a label.
            let savedTagUuids = val.bbox_info.tagUuids ?? [];
            if (labelChanged && labelUuid) {
              savedTagUuids = savedTagUuids.filter(id => id !== UNVERIFIED_TAG_UUID);
            } else if (labelChanged && !labelUuid) {
              // Removing a label: re-mark as unverified so the bbox returns to the review queue.
              if (!savedTagUuids.includes(UNVERIFIED_TAG_UUID)) {
                savedTagUuids = [...savedTagUuids, UNVERIFIED_TAG_UUID];
              }
            }

            // Immediate optimistic update using any icon data we already have.
            const existingIconsMap = new Map((bboxView.uiBBox.tags ?? []).map(t => [t.uuid, t.icon]));
            bboxView.uiBBox.tags = savedTagUuids.map(id => ({
              uuid: id,
              icon: existingIconsMap.get(id) ?? null,
            }));
            bboxView.refreshIcons(bboxView.uiBBox.tags);
            events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId });

            // Async: hydrate tag icons from cache/network and do a second pass.
            const tagsMap = await fetchTags();
            // Guard: if the inspector was re-rendered for a different bbox while we
            // were awaiting, bail out to avoid corrupting the new bbox's state.
            if (!container.isConnected) return;
            bboxView.uiBBox.tags = savedTagUuids.map(id => {
              const tag = tagsMap.get(id);
              return { uuid: id, icon: tag?.icon ?? null };
            });
            bboxView.refreshIcons(bboxView.uiBBox.tags);
            // Always publish after hydration so subscribers see the final icon state,
            // not the placeholder icons from the optimistic update above.
            events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId });

            // Async: update label appearance only if it changed.
            if (labelChanged) {
              if (labelUuid) {
                const labelData = await fetchLabelByUuid(labelUuid);
                if (!container.isConnected) return;
                if (labelData) {
                  bboxView.uiBBox.label = { uuid: labelUuid, text: labelData.name, color: labelData.color };
                  bboxView.updateAppearance(labelData.name, labelData.color);
                }
              } else {
                bboxView.uiBBox.label = undefined;
                bboxView.updateAppearance('', 'grey');
              }
              events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId });
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


/**
 * Bind the inspector to a specific canvas runtime so it only responds to
 * double-clicks from that canvas.
 *
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {{ container?: HTMLElement, openTab?: boolean }} [opts]
 *   container – render into this element instead of the default sidebar tab
 *   openTab   – switch the sidebar to the inspector tab on double-click (default true)
 * @returns {() => void} unsubscribe function
 */
export function initInspector(runtime, { container = null, openTab = true } = {}) {
  return events.subscribeToCanvas(
    runtime.id,
    EventTypes.CANVAS_BBOX_DOUBLE_CLICKED,
    ({ bboxView }) => {
      if (openTab) openInspectorTab();
      renderBboxInspector({ bboxView, target: container ?? 'tab-inspector', runtimeId: runtime.id });
    }
  );
}
