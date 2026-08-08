import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import { filterVisibleTags } from '/app-static/js/shared/activeModel.js';
import {
  FIXED_ACTIONS,
  initHotkeyBindings,
  combosForAction,
  setBinding,
  clearBinding,
  resetBindings,
  eventToCombo,
  comboLabel,
} from './hotkeyBindings.js';

/**
 * Render the hotkey config UI into `container`.
 * Attaches a document-level keydown listener (capture phase) while a key is
 * being captured; the listener is inert otherwise so it is safe to leave
 * attached for long-lived panels like the Settings tab.
 *
 * @param {HTMLElement} container
 * @returns {Promise<{ destroy: () => void, reset: () => void }>}
 */
export async function renderHotkeyPanel(container) {
  const [labels, tags] = await Promise.all([fetchLabels(), fetchTags(), initHotkeyBindings()]);

  let capturingActionId = null;

  /** @param {{id:string,label:string,icon?:string,color?:string}} a */
  function renderRow(a) {
    const row = document.createElement('div');
    row.className = 'hotkey-row';

    const name = document.createElement('span');
    name.className = 'hotkey-row__name';
    if (a.color) {
      const sw = document.createElement('span');
      sw.className = 'hotkey-row__swatch';
      sw.style.background = a.color;
      name.appendChild(sw);
    }
    name.appendChild(document.createTextNode(`${a.icon ? a.icon + ' ' : ''}${a.label}`));
    name.title = a.label;

    const keyBtn = document.createElement('button');
    keyBtn.className = 'panel-btn hotkey-row__key';
    if (capturingActionId === a.id) {
      keyBtn.textContent = 'Press a key…';
      keyBtn.classList.add('hotkey-row__key--capturing');
    } else {
      const combos = combosForAction(a.id);
      if (combos.length) {
        keyBtn.textContent = combos.map(comboLabel).join(', ');
      } else {
        keyBtn.textContent = 'Set key';
        keyBtn.classList.add('hotkey-row__key--unset');
      }
    }
    keyBtn.onclick = () => {
      capturingActionId = capturingActionId === a.id ? null : a.id;
      renderRows();
    };

    const clearBtn = document.createElement('button');
    clearBtn.textContent = '✕';
    clearBtn.className = 'secondary panel-btn hotkey-row__clear';
    clearBtn.title = 'Clear binding';
    clearBtn.onclick = () => {
      clearBinding(a.id);
      if (capturingActionId === a.id) capturingActionId = null;
      renderRows();
    };

    row.append(name, keyBtn, clearBtn);
    return row;
  }

  function section(heading, rows) {
    const wrap = document.createElement('div');
    const h = document.createElement('div');
    h.className = 'panel-section-label';
    h.textContent = heading;

    // Rows flow into as many columns as the panel width allows — one row per
    // label *and* per visible tag makes a single column arbitrarily tall.
    const grid = document.createElement('div');
    grid.className = 'hotkey-grid';
    rows.forEach(r => grid.appendChild(r));

    wrap.append(h, grid);
    return wrap;
  }

  function renderRows() {
    container.innerHTML = '';
    container.appendChild(section('Actions', FIXED_ACTIONS.map(renderRow)));

    const labelRows = [...labels.values()].map(l =>
      renderRow({ id: `label:${l.uuid}`, label: l.name ?? l.uuid, color: l.color })
    );
    if (labelRows.length) container.appendChild(section('Label tools', labelRows));

    const tagRows = filterVisibleTags(tags.values())
      .map(t => renderRow({ id: `tag:${t.uuid}`, label: t.name ?? t.uuid, icon: t.icon ?? '', color: t.color }));
    if (tagRows.length) container.appendChild(section('Tag tools', tagRows));
  }

  const onKeyCapture = (e) => {
    if (!capturingActionId) return;
    e.preventDefault();
    e.stopPropagation();
    if (e.key === 'Escape') {
      capturingActionId = null;
      renderRows();
      return;
    }
    if (['Control', 'Shift', 'Alt', 'Meta'].includes(e.key)) return;
    setBinding(capturingActionId, eventToCombo(e));
    capturingActionId = null;
    renderRows();
  };
  document.addEventListener('keydown', onKeyCapture, true);

  renderRows();

  return {
    destroy() {
      document.removeEventListener('keydown', onKeyCapture, true);
    },
    reset() {
      resetBindings();
      capturingActionId = null;
      renderRows();
    },
  };
}

