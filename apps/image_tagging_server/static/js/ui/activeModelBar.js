/**
 * Active Training Config Bar — a fixed bottom bar visible on all tabs.
 *
 * Shows the current session's active training config. Clicking the config name
 * opens a compact coverage-stats popup; there is also a visibility-options flyout.
 *
 * Mount once at app startup:
 *   import { mountActiveModelBar } from '/app-static/js/ui/activeModelBar.js';
 *   mountActiveModelBar('active-model-bar');
 */

import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { activeTrainingConfig } from '/app-static/js/shared/activeTrainingConfig.js';
import { activeModel, isDynamicLabelsVisible, setDynamicLabelsVisible } from '/app-static/js/shared/activeModel.js';
import { navigateTabs } from '/app-static/js/app/utils/tabs.js';
import { openAnchoredPopup } from '/app-static/js/ui/utils/anchoredPopup.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchTrainingConfigs } from '/app-static/js/shared/api/trainingConfigs.js';
import { renderCoverageStats, renderReadinessStats } from '/app-static/js/ui/utils/coverageStatsTable.js';

/**
 * @param {string} containerId  id of the mount <div>
 */
export function mountActiveModelBar(containerId = 'active-model-bar') {
  const container = document.getElementById(containerId);
  if (!container) return;

  // ── Popup state ─────────────────────────────────────────────────────────────
  let _visPopup = null;
  let _statsPopup = null;
  let _runPickerPopup = null;

  function _closeVisPopup() { _visPopup?.close(); _visPopup = null; }
  function _closeStatsPopup() { _statsPopup?.close(); _statsPopup = null; }
  function _closeRunPickerPopup() { _runPickerPopup?.close(); _runPickerPopup = null; }

  // ── Visibility options popup (tags / labels toggles) ─────────────────────
  function _openVisPopup(anchorEl) {
    _closeVisPopup();
    _closeStatsPopup();
    _closeRunPickerPopup();
    const tc = activeTrainingConfig.getActive();

    const popup = document.createElement('div');
    popup.className = 'vis-popup';
    popup.style.cssText = [
      'z-index:10001',
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

    // Two independent axes (each with a longer explanation than the compact
    // menus in the label-list header): visibility (is a label shown) and target
    // (does "Apply to all" affect it).
    const VISIBILITY_PRESET_INFO = [
      { value: 'all', title: 'Show all labels', desc: 'Every label is visible.' },
      { value: 'training-run', title: 'Hide labels outside training run', desc: 'Only labels in this training run appear in the list.' },
    ];
    const TARGET_PRESET_INFO = [
      { value: 'all', title: 'Everything', desc: 'Mark absent / Reset affect every visible label.' },
      { value: 'training-run', title: 'Only this training run', desc: 'Only labels in this training run are affected; others stay visible but untargeted.' },
      { value: 'object', title: 'Only object labels', desc: 'Only bounding-box labels are affected.' },
      { value: 'image', title: 'Only image classification', desc: 'Only whole-image labels are affected.' },
      { value: 'dynamic', title: 'Only dynamic labels', desc: 'Only dynamic labels are affected.' },
      { value: 'none', title: 'Nothing', desc: 'Nothing is affected by default — target labels individually.' },
    ];

    function _makeRadioRow(name, checked, title, desc, onChange) {
      const lbl = document.createElement('label');
      lbl.style.cssText = 'display:flex;align-items:flex-start;gap:8px;padding:6px 14px;cursor:pointer;font-size:0.875em;user-select:none;';
      lbl.addEventListener('mouseenter', () => lbl.style.background = 'var(--pico-primary-background,rgba(255,255,255,.07))');
      lbl.addEventListener('mouseleave', () => lbl.style.background = '');
      lbl.addEventListener('mousedown', e => e.stopPropagation());
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = name;
      radio.checked = checked;
      radio.style.marginTop = '2px';
      radio.addEventListener('change', () => { if (radio.checked) onChange(); });
      const textWrap = document.createElement('span');
      const titleEl = document.createElement('div');
      titleEl.textContent = title;
      const descEl = document.createElement('div');
      descEl.style.cssText = 'font-size:0.85em;color:var(--pico-muted-color,#999);margin-top:1px;';
      descEl.textContent = desc;
      textWrap.appendChild(titleEl);
      textWrap.appendChild(descEl);
      lbl.appendChild(radio);
      lbl.appendChild(textWrap);
      return lbl;
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
    if (tc) {
      popup.appendChild(_makeSep('Label visibility'));
      const curVis = activeTrainingConfig.getVisibilityPreset();
      for (const { value, title, desc } of VISIBILITY_PRESET_INFO) {
        popup.appendChild(_makeRadioRow('tc-visibility-preset', curVis === value, title, desc,
          () => activeTrainingConfig.setVisibilityPreset(value)));
      }
      popup.appendChild(_makeSep('Apply-to-all target'));
      const curTgt = activeTrainingConfig.getTargetPreset();
      // 'custom' is entered implicitly by toggling rows in the label list; show
      // it here (as an active, selectable option) only once it's in effect.
      const targetInfo = curTgt === 'custom'
        ? [...TARGET_PRESET_INFO, { value: 'custom', title: 'Custom (pick individually)', desc: 'Your hand-picked selection from the label list.' }]
        : TARGET_PRESET_INFO;
      for (const { value, title, desc } of targetInfo) {
        popup.appendChild(_makeRadioRow('tc-target-preset', curTgt === value, title, desc,
          () => activeTrainingConfig.setTargetPreset(value)));
      }
    }

    _visPopup = openAnchoredPopup(anchorEl, popup);
  }

  // ── Run picker popup (switch/clear active training run) ──────────────────
  // Uses openAnchoredPopup (caller-owned z-index) rather than the shared
  // ui/contextMenu/menu.js engine, because that engine's z-index (2000) sits
  // below open modals (~9999) — this bar must stay usable while a modal is up.
  function _openRunPicker(anchorEl) {
    _closeRunPickerPopup();
    _closeStatsPopup();
    _closeVisPopup();

    const popup = document.createElement('div');
    popup.className = 'tc-run-picker';
    popup.style.cssText = [
      'z-index:10001',
      'background:var(--pico-card-background-color,#1e1e2e)',
      'border:1px solid var(--pico-muted-border-color,#444)',
      'border-radius:6px', 'padding:6px 0', 'min-width:220px', 'max-width:320px',
      'max-height:60vh', 'overflow-y:auto',
      'box-shadow:0 -4px 16px rgba(0,0,0,.4)',
    ].join(';');

    function _makeSep(text) {
      const s = document.createElement('div');
      s.style.cssText = 'padding:4px 14px 2px;font-size:0.75em;color:var(--pico-muted-color,#888);text-transform:uppercase;letter-spacing:.05em;';
      s.textContent = text;
      return s;
    }

    function _makeRunRow(labelText, active, onSelect) {
      const row = document.createElement('div');
      row.className = 'tc-run-picker__row';
      row.style.cssText = 'display:flex;align-items:center;gap:8px;padding:6px 14px;cursor:pointer;font-size:0.875em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;';
      if (active) row.style.color = 'var(--pico-primary, #ff6b6b)';
      row.addEventListener('mouseenter', () => row.style.background = 'var(--pico-primary-background,rgba(255,255,255,.07))');
      row.addEventListener('mouseleave', () => row.style.background = '');
      const check = document.createElement('span');
      check.style.cssText = 'width:1em;flex-shrink:0;';
      check.textContent = active ? '✓' : '';
      row.appendChild(check);
      const lbl = document.createElement('span');
      lbl.textContent = labelText;
      lbl.style.cssText = 'overflow:hidden;text-overflow:ellipsis;';
      row.appendChild(lbl);
      row.addEventListener('click', async () => {
        _closeRunPickerPopup();
        await onSelect();
      });
      return row;
    }

    const loading = document.createElement('div');
    loading.style.cssText = 'padding:6px 14px;color:var(--pico-muted-color,#888);font-size:0.875em;';
    loading.textContent = 'Loading…';
    popup.appendChild(loading);

    const manageBtn = document.createElement('div');
    manageBtn.className = 'tc-run-picker__manage-row';
    manageBtn.style.cssText = 'padding:6px 14px;cursor:pointer;font-size:0.875em;color:var(--pico-muted-color,#aaa);border-top:1px solid var(--pico-muted-border-color,#444);margin-top:4px;';
    manageBtn.textContent = 'Manage training runs →';
    manageBtn.addEventListener('mouseenter', () => manageBtn.style.background = 'var(--pico-primary-background,rgba(255,255,255,.07))');
    manageBtn.addEventListener('mouseleave', () => manageBtn.style.background = '');
    manageBtn.addEventListener('click', () => {
      _closeRunPickerPopup();
      navigateTabs('tab-configuration', 'tab-training-management');
    });

    _runPickerPopup = openAnchoredPopup(anchorEl, popup);

    // Content size changes once data loads (Loading… → rows) — the anchored
    // popup only measures/positions once at open time, so re-anchor now that
    // final size is known rather than leaving it mis-positioned.
    function _reanchor() {
      if (_runPickerPopup) _runPickerPopup.close();
      _runPickerPopup = openAnchoredPopup(anchorEl, popup);
    }

    fetchTrainingConfigs().then(configsMap => {
      popup.innerHTML = '';
      const activeUuid = activeTrainingConfig.getActive()?.uuid ?? null;
      const configs = [...configsMap.values()].sort((a, b) => a.name.localeCompare(b.name));

      if (configs.length === 0) {
        const none = document.createElement('div');
        none.style.cssText = 'padding:6px 14px;color:var(--pico-muted-color,#888);font-style:italic;font-size:0.875em;';
        none.textContent = 'No training runs yet.';
        popup.appendChild(none);
      } else {
        popup.appendChild(_makeSep('Training Runs'));
        for (const tc of configs) {
          popup.appendChild(_makeRunRow(tc.name, tc.uuid === activeUuid, () => activeTrainingConfig.setActive(tc)));
        }
        if (activeUuid !== null) {
          popup.appendChild(_makeRunRow('No active training run', false, () => activeTrainingConfig.setActive(null)));
        }
      }
      popup.appendChild(manageBtn);
      _reanchor();
    }).catch(err => {
      popup.innerHTML = '';
      const errDiv = document.createElement('div');
      errDiv.style.cssText = 'padding:6px 14px;color:#f66;font-size:0.875em;';
      errDiv.textContent = `Could not load training runs: ${err.message}`;
      popup.appendChild(errDiv);
      popup.appendChild(manageBtn);
      _reanchor();
    });
  }

  // ── Stats panel (full-height left-side overlay) ──────────────────────────
  function _openStatsPopup(anchorEl) {
    _closeStatsPopup();
    _closeVisPopup();
    _closeRunPickerPopup();
    const tc = activeTrainingConfig.getActive();
    if (!tc) return;

    // Measure the bottom-bars height so the panel ends right above them
    const bottomBarsEl = document.getElementById('bottom-bars');
    const bottomOffset = bottomBarsEl ? bottomBarsEl.offsetHeight : 44;

    const popup = document.createElement('div');
    popup.className = 'tc-stats-panel';
    popup.style.cssText = [
      'position:fixed',
      'left:0',
      'top:0',
      `bottom:${bottomOffset}px`,
      'width:440px',
      'min-width:300px',
      'max-width:90vw',
      'resize:horizontal',
      'z-index:10001',
      'background:var(--pico-card-background-color,#1e1e2e)',
      'border-right:1px solid var(--pico-muted-border-color,#444)',
      'box-shadow:4px 0 24px rgba(0,0,0,.55)',
      'display:flex',
      'flex-direction:column',
      'overflow:hidden',
    ].join(';');

    // Fixed header (not scrolled)
    const hdr = document.createElement('div');
    hdr.style.cssText = 'display:flex;align-items:center;gap:6px;padding:10px 12px 8px;border-bottom:1px solid var(--pico-muted-border-color,#444);flex-shrink:0;';
    const hdrTitle = document.createElement('span');
    hdrTitle.style.cssText = 'font-weight:600;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;';
    hdrTitle.textContent = `Coverage — ${tc.name}`;
    hdrTitle.title = tc.name;
    hdr.appendChild(hdrTitle);

    const refreshBtn = document.createElement('button');
    refreshBtn.type = 'button';
    refreshBtn.textContent = '↻';
    refreshBtn.title = 'Refresh';
    refreshBtn.style.cssText = 'background:none;border:none;cursor:pointer;padding:0 3px;font-size:1em;color:var(--pico-muted-color,#888);margin:0;';
    hdr.appendChild(refreshBtn);

    const goBtn = document.createElement('button');
    goBtn.type = 'button';
    goBtn.textContent = '→ Run';
    goBtn.title = 'Go to training run';
    goBtn.style.cssText = 'background:none;border:1px solid var(--pico-muted-border-color,#555);border-radius:3px;cursor:pointer;padding:1px 6px;font-size:0.85em;color:var(--pico-muted-color,#aaa);margin:0;white-space:nowrap;';
    goBtn.addEventListener('click', () => {
      _closeStatsPopup();
      navigateTabs('tab-configuration', 'tab-training-management');
    });
    hdr.appendChild(goBtn);

    // Scrollable content area
    const content = document.createElement('div');
    content.style.cssText = 'flex:1;overflow:auto;padding:8px 12px;font-size:0.82em;';
    popup.appendChild(hdr);
    popup.appendChild(content);

    async function _renderStats() {
      content.innerHTML = '';
      refreshBtn.disabled = true;

      const loading = document.createElement('div');
      loading.style.cssText = 'color:var(--pico-muted-color,#888);padding:4px 0;';
      loading.textContent = 'Loading…';
      content.appendChild(loading);

      try {
        // Use cached stats/readiness if available, otherwise fetch both
        // (refreshStats() fetches label-stats + readiness together).
        let cachedStats = activeTrainingConfig.getCachedStats();
        let cachedReadiness = activeTrainingConfig.getCachedReadiness();
        if (!cachedStats || !cachedReadiness) {
          await activeTrainingConfig.refreshStats();
          cachedStats = activeTrainingConfig.getCachedStats();
          cachedReadiness = activeTrainingConfig.getCachedReadiness();
        }
        if (!cachedStats) throw new Error('No stats available');

        const { labelStats, totalImages } = cachedStats;
        content.innerHTML = '';

        // Fetch labels map for names / colors
        const labelsMap = await fetchLabels().catch(() => new Map());

        // Readiness + coverage-table cells open the review modal (much
        // taller content, no side panel of its own) — close this floating
        // popup first so it doesn't sit on top of the modal it just triggered.
        if (cachedReadiness) {
          const readinessTitle = document.createElement('div');
          readinessTitle.style.cssText = 'font-weight:600;margin-bottom:0.3em;';
          readinessTitle.textContent = 'Training readiness';
          content.appendChild(readinessTitle);
          renderReadinessStats(content, { config: tc, readiness: cachedReadiness, onNavigate: _closeStatsPopup });

          const coverageTitle = document.createElement('div');
          coverageTitle.style.cssText = 'font-weight:600;margin:0.75em 0 0.3em;';
          coverageTitle.textContent = 'Dataset coverage by label';
          content.appendChild(coverageTitle);
        }

        renderCoverageStats(content, { config: tc, labelStats, totalImages, labelsMap, onNavigate: _closeStatsPopup });
      } catch (err) {
        content.innerHTML = '';
        const errDiv = document.createElement('div');
        errDiv.style.cssText = 'color:#f66;';
        errDiv.textContent = `Could not load stats: ${err.message}`;
        content.appendChild(errDiv);
      } finally {
        refreshBtn.disabled = false;
      }
    }

    refreshBtn.addEventListener('click', async () => {
      await activeTrainingConfig.refreshStats();
      await _renderStats();
    });

    // Mount panel and wire outside-click dismissal
    document.body.appendChild(popup);

    function _closePanel() {
      popup.remove();
      document.removeEventListener('mousedown', _outsideListener, true);
      _statsPopup = null;
    }
    function _outsideListener(e) {
      if (!popup.contains(e.target) && e.target !== anchorEl) _closePanel();
    }
    document.addEventListener('mousedown', _outsideListener, true);

    _statsPopup = { close: _closePanel };
    _renderStats().catch(err => console.warn('[activeModelBar] _renderStats failed:', err));
  }

  // ── Main render ──────────────────────────────────────────────────────────
  function render() {
    container.innerHTML = '';

    const bar = document.createElement('div');
    bar.className = 'active-model-bar';
    container.appendChild(bar);

    // ── Left: training config info ───────────────────────────────────────
    const left = document.createElement('span');
    left.className = 'active-model-bar__info';
    const tc = activeTrainingConfig.getActive();

    if (tc) {
      const label = document.createElement('span');
      label.className = 'active-model-bar__label';
      label.textContent = 'Training run:';
      left.appendChild(label);

      const nameBtn = document.createElement('button');
      nameBtn.type = 'button';
      nameBtn.className = 'active-model-bar__name';
      nameBtn.textContent = tc.name;
      nameBtn.title = 'View coverage stats';
      nameBtn.addEventListener('click', () => {
        if (_statsPopup) { _closeStatsPopup(); } else { _openStatsPopup(nameBtn); }
      });
      left.appendChild(nameBtn);

      const pickerBtn = document.createElement('button');
      pickerBtn.type = 'button';
      pickerBtn.className = 'active-model-bar__picker-btn';
      pickerBtn.textContent = '▾';
      pickerBtn.title = 'Switch training run';
      pickerBtn.addEventListener('click', () => {
        if (_runPickerPopup) { _closeRunPickerPopup(); } else { _openRunPicker(pickerBtn); }
      });
      left.appendChild(pickerBtn);

      // Preset indicator: 'filtered' when the list hides labels, else 'focus'
      // when Apply-to-all is narrowed to a subset.
      const visP = activeTrainingConfig.getVisibilityPreset();
      const tgtP = activeTrainingConfig.getTargetPreset();
      if (visP === 'training-run' || tgtP !== 'all') {
        const filterBadge = document.createElement('span');
        filterBadge.className = 'active-model-bar__filter-badge';
        filterBadge.textContent = visP === 'training-run' ? 'filtered' : 'focus';
        filterBadge.title = visP === 'training-run'
          ? 'Label list is filtered to this training run'
          : 'Apply-to-all is limited to a subset of labels';
        left.appendChild(filterBadge);
      }

      const deactivateBtn = document.createElement('button');
      deactivateBtn.type = 'button';
      deactivateBtn.className = 'active-model-bar__deactivate';
      deactivateBtn.textContent = '✕';
      deactivateBtn.title = 'Deactivate training run';
      deactivateBtn.addEventListener('click', () => {
        _closeStatsPopup();
        activeTrainingConfig.setActive(null);
      });
      left.appendChild(deactivateBtn);
    } else {
      const none = document.createElement('span');
      none.className = 'active-model-bar__none';
      none.textContent = 'No active training run';
      left.appendChild(none);

      // Picker also covers this case — lists available runs plus a link to
      // the training run page, so it works the same whether or not modals
      // are currently covering the page.
      const setBtn = document.createElement('button');
      setBtn.type = 'button';
      setBtn.className = 'active-model-bar__set-tc-btn';
      setBtn.textContent = 'Set one ▾';
      setBtn.title = 'Select a training run';
      setBtn.addEventListener('click', () => {
        if (_runPickerPopup) { _closeRunPickerPopup(); } else { _openRunPicker(setBtn); }
      });
      left.appendChild(setBtn);
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

  const unsubTC = events.subscribe(EventTypes.ACTIVE_TRAINING_CONFIG_CHANGED, ({ statsUpdated }) => {
    if (!statsUpdated) render(); // only re-render bar structure on config/filter changes
  });
  return {
    destroy() {
      _closeVisPopup();
      _closeStatsPopup();
      _closeRunPickerPopup();
      unsubTC();
    },
  };
}
