import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import {
  FIXED_ACTIONS,
  combosForAction,
  setBinding,
  clearBinding,
  resetBindings,
  eventToCombo,
  comboLabel,
} from './hotkeyBindings.js';

/**
 * Open the (shared) hotkey configuration modal. Bindings apply to every canvas.
 * @returns {Promise<void>}
 */
export async function openHotkeyModal() {
  const [labels, tags] = await Promise.all([fetchLabels(), fetchTags()]);

  const overlay = document.createElement('div');
  overlay.style.cssText = `
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.5);
    display: flex; align-items: center; justify-content: center;
    z-index: 9999;
  `;

  const dialog = document.createElement('div');
  dialog.style.cssText = `
    background: var(--pico-background-color, #fff);
    border-radius: 8px;
    padding: 1.5em;
    min-width: 420px;
    max-width: 560px;
    max-height: 80vh;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 1em;
  `;

  const title = document.createElement('strong');
  title.textContent = 'Configure Hotkeys';

  const body = document.createElement('div');
  body.style.cssText = 'display:flex; flex-direction:column; gap:1em;';

  // capturingActionId is set while we wait for the user to press a key.
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
    body.innerHTML = '';
    body.appendChild(section('Actions', FIXED_ACTIONS.map(renderRow)));

    const labelRows = [...labels.values()].map(l =>
      renderRow({ id: `label:${l.uuid}`, label: l.name ?? l.uuid, color: l.color })
    );
    if (labelRows.length) body.appendChild(section('Label tools', labelRows));

    const tagRows = [...tags.values()].map(t =>
      renderRow({ id: `tag:${t.uuid}`, label: t.name ?? t.uuid, icon: t.icon ?? '', color: t.color })
    );
    if (tagRows.length) body.appendChild(section('Tag tools', tagRows));
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
    // Ignore lone modifier presses; wait for the actual key.
    if (['Control', 'Shift', 'Alt', 'Meta'].includes(e.key)) return;
    setBinding(capturingActionId, eventToCombo(e));
    capturingActionId = null;
    renderRows();
  };
  document.addEventListener('keydown', onKeyCapture, true);

  const footer = document.createElement('div');
  footer.style.cssText = 'display:flex; gap:0.5em; justify-content:space-between; align-items:center;';

  const resetBtn = document.createElement('button');
  resetBtn.textContent = 'Reset to defaults';
  resetBtn.className = 'secondary';
  resetBtn.onclick = () => { resetBindings(); capturingActionId = null; renderRows(); };

  const closeBtn = document.createElement('button');
  closeBtn.textContent = 'Done';

  const close = () => {
    document.removeEventListener('keydown', onKeyCapture, true);
    if (overlay.parentNode) document.body.removeChild(overlay);
  };
  closeBtn.onclick = close;
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

  footer.append(resetBtn, closeBtn);
  dialog.append(title, body, footer);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);

  renderRows();
}
