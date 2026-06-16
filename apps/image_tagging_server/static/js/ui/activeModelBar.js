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

  // ── Visibility dropdown (flyout above the bar) ──────────────────────────
  let _visPopup = null;
  let _outsideListener = null;

  function _closeVisPopup() {
    if (_visPopup) { _visPopup.remove(); _visPopup = null; }
    if (_outsideListener) { document.removeEventListener('mousedown', _outsideListener, true); _outsideListener = null; }
  }

  function _openVisPopup(anchorEl) {
    _closeVisPopup();
    const config = activeModel.getActiveModel();

    const popup = document.createElement('div');
    popup.className = 'vis-popup';
    popup.style.cssText = [
      'position:fixed', 'z-index:9999',
      'background:var(--pico-card-background-color,#1e1e2e)',
      'border:1px solid var(--pico-muted-border-color,#444)',
      'border-radius:6px', 'padding:6px 0', 'min-width:220px',
      'box-shadow:0 -4px 16px rgba(0,0,0,.4)',
    ].join(';');

    function _makeRow(labelText, checked, onChange) {
      const lbl = document.createElement('label');
      lbl.style.cssText = 'display:flex;align-items:center;gap:8px;padding:6px 14px;cursor:pointer;font-size:0.875em;user-select:none;';
      lbl.addEventListener('mouseenter', () => lbl.style.background = 'var(--pico-primary-background,rgba(255,255,255,.07))');
      lbl.addEventListener('mouseleave', () => lbl.style.background = '');
      lbl.addEventListener('mousedown', e => e.stopPropagation());
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = checked;
      cb.addEventListener('change', () => onChange(cb.checked));
      lbl.appendChild(cb);
      lbl.appendChild(document.createTextNode(labelText));
      return lbl;
    }

    function _makeSep(text) {
      const s = document.createElement('div');
      s.style.cssText = 'padding:4px 14px 2px;font-size:0.75em;color:var(--pico-muted-color,#888);text-transform:uppercase;letter-spacing:.05em;';
      s.textContent = text;
      return s;
    }

    // ── Tags section ──────────────────────────────────────────────────────
    popup.appendChild(_makeSep('Tags'));
    popup.appendChild(_makeRow('Show system tags', activeModel.isTagKindVisible('system'),
      v => activeModel.setTagKindVisible('system', v)));
    popup.appendChild(_makeRow('Show task-generated tags', activeModel.isTagKindVisible('task_generated'),
      v => activeModel.setTagKindVisible('task_generated', v)));

    // ── Labels section ────────────────────────────────────────────────────
    popup.appendChild(_makeSep('Labels'));
    popup.appendChild(_makeRow('Show dynamic labels', isDynamicLabelsVisible(),
      v => setDynamicLabelsVisible(v)));
    if (config) {
      popup.appendChild(_makeRow('Filter labels to model', activeModel.isLabelFilterEnabled(),
        v => activeModel.setLabelFilterEnabled(v)));
    }

    // Position above the anchor button, right-aligned to button's right edge.
    document.body.appendChild(popup);
    const rect = anchorEl.getBoundingClientRect();
    const popupRect = popup.getBoundingClientRect();
    const popupH = popupRect.height;
    const popupW = popupRect.width;
    popup.style.left = Math.max(4, rect.right - popupW) + 'px';
    popup.style.top = (rect.top - popupH - 4) + 'px';

    _visPopup = popup;
    _outsideListener = (e) => {
      if (!popup.contains(e.target) && e.target !== anchorEl) _closeVisPopup();
    };
    document.addEventListener('mousedown', _outsideListener, true);
  }

  function render() {
    container.innerHTML = '';

    const bar = document.createElement('div');
    bar.className = 'active-model-bar';
    container.appendChild(bar);

    // ── Left: model info ─────────────────────────────────────────────────
    const left = document.createElement('span');
    left.className = 'active-model-bar__info';
    const config = activeModel.getActiveModel();

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

    // ── Right: visibility button ─────────────────────────────────────────
    const visBtn = document.createElement('button');
    visBtn.type = 'button';
    visBtn.className = 'active-model-bar__vis-btn';
    visBtn.title = 'Visibility options';
    visBtn.textContent = '👁';
    visBtn.addEventListener('click', () => {
      if (_visPopup) { _closeVisPopup(); } else { _openVisPopup(visBtn); }
    });
    bar.appendChild(visBtn);
  }

  render();

  const unsubModel = events.subscribe(EventTypes.ACTIVE_MODEL_CHANGED, () => { render(); });
  const unsubDyn = events.subscribe(EventTypes.DYNAMIC_LABELS_VISIBILITY_CHANGED, () => {});
  const unsubTag = events.subscribe(EventTypes.TAG_KIND_VISIBILITY_CHANGED, () => {});

  return {
    destroy() {
      _closeVisPopup();
      unsubModel();
      unsubDyn();
      unsubTag();
    },
  };
}
