/**
 * Shared dataset-selection checklist used by the Training Configs form and the
 * Model Configs "Train all → bundle" section, so the two Configuration-tab
 * dataset pickers can't drift in markup or behavior.
 *
 * Synchronous: callers hand in an already-fetched datasetsMap
 * (`fetchDatasets()` returns a Map).
 */
import { navigateTabs } from '/app-static/js/app/utils/tabs.js';

/** Muted "no datasets yet" hint linking to the Datasets workspace tab. */
function makeEmptyHint() {
  const hint = document.createElement('p');
  hint.style.cssText = 'font-size: 0.85em; color: var(--pico-muted-color, #888); margin: 0; padding: 2px 4px;';
  hint.append('No datasets defined yet. ');
  const link = document.createElement('a');
  link.href = '#';
  link.textContent = 'Create one in the Datasets tab →';
  link.addEventListener('click', (e) => { e.preventDefault(); navigateTabs('tab-configuration', 'tab-dataset-management'); });
  hint.appendChild(link);
  return hint;
}

/**
 * @param {Object} opts
 * @param {Map} opts.datasetsMap - uuid → dataset, as returned by fetchDatasets()
 * @param {string[]} [opts.selectedUuids] - initially-checked dataset uuids
 * @param {Function} [opts.onChange] - called on any checkbox toggle
 * @returns {{ el: HTMLElement, getSelectedUuids: () => string[] }}
 */
export function createDatasetChecklist({ datasetsMap, selectedUuids = [], onChange } = {}) {
  const box = document.createElement('div');
  box.className = 'panel-tree-box';
  const selected = new Set(selectedUuids);
  const checkboxes = new Map();

  if (datasetsMap.size === 0) {
    box.appendChild(makeEmptyHint());
  }
  for (const [uuid, ds] of datasetsMap) {
    const row = document.createElement('label');
    row.style.cssText = 'display: flex; align-items: center; gap: 0.4em; padding: 2px 4px; cursor: pointer; font-size: 0.88em; user-select: none;';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = uuid;
    cb.checked = selected.has(uuid);
    cb.style.cssText = 'margin: 0; flex-shrink: 0;';
    if (onChange) cb.addEventListener('change', onChange);
    checkboxes.set(uuid, cb);
    row.appendChild(cb);
    const nameSpan = document.createElement('span');
    nameSpan.textContent = ds.name;
    row.appendChild(nameSpan);
    box.appendChild(row);
  }

  return {
    el: box,
    getSelectedUuids() {
      return [...checkboxes.entries()].filter(([, cb]) => cb.checked).map(([uuid]) => uuid);
    },
  };
}
