import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import { isTagVisible } from '/app-static/js/shared/activeModel.js';
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

  container.style.cssText = 'display:flex; flex-direction:column; gap:1em;';

  let capturingActionId = null;

  /** @param {{id:string,label:string,icon?:string,color?:string}} a */
  function renderRow(a) {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex; align-items:center; gap:0.5em;';

    const name = document.createElement('span');
    name.style.cssText = 'flex:1; display:flex; align-items:center; gap:0.4em;';
    if (a.color) {
      const sw = document.createElement('span');
      sw.style.cssText = `display:inline-block; width:0.8em; height:0.8em; border-radius:2px; background:${a.color};`;
      name.appendChild(sw);
    }
    name.appendChild(document.createTextNode(`${a.icon ? a.icon + ' ' : ''}${a.label}`));

    const keyBtn = document.createElement('button');
    keyBtn.style.cssText = 'min-width:120px;';
    if (capturingActionId === a.id) {
      keyBtn.textContent = 'Press a key…';
      keyBtn.classList.add('selected');
    } else {
      const combos = combosForAction(a.id);
      keyBtn.textContent = combos.length ? combos.map(comboLabel).join(', ') : 'Set key';
    }
    keyBtn.onclick = () => {
      capturingActionId = capturingActionId === a.id ? null : a.id;
      renderRows();
    };

    const clearBtn = document.createElement('button');
    clearBtn.textContent = '✕';
    clearBtn.className = 'secondary';
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
    h.style.cssText = 'font-weight:600; opacity:0.7; margin-bottom:0.3em;';
    h.textContent = heading;
    wrap.appendChild(h);
    rows.forEach(r => wrap.appendChild(r));
    return wrap;
  }

  function renderRows() {
    container.innerHTML = '';
    container.appendChild(section('Actions', FIXED_ACTIONS.map(renderRow)));

    const labelRows = [...labels.values()].map(l =>
      renderRow({ id: `label:${l.uuid}`, label: l.name ?? l.uuid, color: l.color })
    );
    if (labelRows.length) container.appendChild(section('Label tools', labelRows));

    const tagRows = [...tags.values()]
      .filter(isTagVisible)
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

