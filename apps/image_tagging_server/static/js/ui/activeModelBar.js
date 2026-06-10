/**
 * Active Model Bar — a fixed bottom bar visible on all tabs.
 *
 * Shows the current session's active model and a toggle for filtering label
 * lists to only model-relevant labels.
 *
 * Mount once at app startup:
 *   import { mountActiveModelBar } from '/app-static/js/ui/activeModelBar.js';
 *   mountActiveModelBar('active-model-bar');
 */

import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { activeModel, isDynamicLabelsVisible, setDynamicLabelsVisible } from '/app-static/js/shared/activeModel.js';
import { navigateTabs } from '/app-static/js/app/utils/tabs.js';

/**
 * @param {string} containerId  id of the mount <div>
 */
export function mountActiveModelBar(containerId = 'active-model-bar') {
  const container = document.getElementById(containerId);
  if (!container) return;

  function render() {
    const config = activeModel.getActiveModel();
    const filterOn = activeModel.isLabelFilterEnabled();

    container.innerHTML = '';

    const bar = document.createElement('div');
    bar.className = 'active-model-bar';
    container.appendChild(bar);

    // ── Left: model info ─────────────────────────────────────────────────
    const left = document.createElement('span');
    left.className = 'active-model-bar__info';

    if (config) {
      const label = document.createElement('span');
      label.className = 'active-model-bar__label';
      label.textContent = 'Active model:';
      left.appendChild(label);

      const nameBtn = document.createElement('button');
      nameBtn.type = 'button';
      nameBtn.className = 'active-model-bar__name';
      nameBtn.textContent = config.name;
      nameBtn.title = 'Go to model configuration';
      nameBtn.addEventListener('click', () => {
        navigateTabs('tab-configuration', 'tab-model-management');
      });
      left.appendChild(nameBtn);

      const deactivateBtn = document.createElement('button');
      deactivateBtn.type = 'button';
      deactivateBtn.className = 'active-model-bar__deactivate';
      deactivateBtn.textContent = '✕';
      deactivateBtn.title = 'Deactivate model';
      deactivateBtn.addEventListener('click', () => activeModel.setActiveModel(null));
      left.appendChild(deactivateBtn);
    } else {
      const none = document.createElement('span');
      none.className = 'active-model-bar__none';
      none.textContent = 'No active model';
      left.appendChild(none);
    }

    bar.appendChild(left);

    // ── Right: filter toggle ──────────────────────────────────────────────
    if (config) {
      const right = document.createElement('label');
      right.className = 'active-model-bar__filter-label';

      const toggle = document.createElement('input');
      toggle.type = 'checkbox';
      toggle.checked = filterOn;
      toggle.addEventListener('change', () => {
        activeModel.setLabelFilterEnabled(toggle.checked);
      });

      right.appendChild(toggle);
      right.append(' Filter labels to model');
      bar.appendChild(right);
    }

    // ── Always visible: dynamic labels toggle ─────────────────────────────
    const dynLabel = document.createElement('label');
    dynLabel.className = 'active-model-bar__filter-label';

    const dynToggle = document.createElement('input');
    dynToggle.type = 'checkbox';
    dynToggle.checked = isDynamicLabelsVisible();
    dynToggle.addEventListener('change', () => {
      setDynamicLabelsVisible(dynToggle.checked);
    });

    dynLabel.appendChild(dynToggle);
    dynLabel.append(' Show dynamic labels');
    bar.appendChild(dynLabel);
  }

  render();

  const unsubModel = events.subscribe(EventTypes.ACTIVE_MODEL_CHANGED, render);
  const unsubDyn = events.subscribe(EventTypes.DYNAMIC_LABELS_VISIBILITY_CHANGED, render);

  // Return a destroy handle in case the bar ever needs to be torn down
  return {
    destroy() {
      unsubModel();
      unsubDyn();
    },
  };
}
