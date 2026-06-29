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

/**
 * @param {string} containerId  id of the mount <div>
 */
export function mountActiveModelBar(containerId = 'active-model-bar') {
  const container = document.getElementById(containerId);
  if (!container) return;

  // ── Popup state ─────────────────────────────────────────────────────────────
  let _visPopup = null;
  let _statsPopup = null;

  function _closeVisPopup() { _visPopup?.close(); _visPopup = null; }
  function _closeStatsPopup() { _statsPopup?.close(); _statsPopup = null; }

  // ── Visibility options popup (tags / labels toggles) ─────────────────────
  function _openVisPopup(anchorEl) {
    _closeVisPopup();
    _closeStatsPopup();
    const tc = activeTrainingConfig.getActive();

    const popup = document.createElement('div');
    popup.className = 'vis-popup';
    popup.style.cssText = [
      'z-index:9999',
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
    if (tc) {
      popup.appendChild(_makeRow('Filter labels to training config', activeTrainingConfig.isFilterEnabled(),
        v => activeTrainingConfig.setFilterEnabled(v)));
    }

    _visPopup = openAnchoredPopup(anchorEl, popup);
  }

  // ── Stats panel (full-height left-side overlay) ──────────────────────────
  function _openStatsPopup(anchorEl) {
    _closeStatsPopup();
    _closeVisPopup();
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
      'width:360px',
      'z-index:9999',
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
    goBtn.textContent = '→ Config';
    goBtn.title = 'Go to training config';
    goBtn.style.cssText = 'background:none;border:1px solid var(--pico-muted-border-color,#555);border-radius:3px;cursor:pointer;padding:1px 6px;font-size:0.85em;color:var(--pico-muted-color,#aaa);margin:0;white-space:nowrap;';
    goBtn.addEventListener('click', () => {
      _closeStatsPopup();
      navigateTabs('tab-configuration', 'tab-training-management');
    });
    hdr.appendChild(goBtn);

    // Scrollable content area
    const content = document.createElement('div');
    content.style.cssText = 'flex:1;overflow-y:auto;padding:8px 12px;font-size:0.82em;';
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
        // Use cached stats if available, otherwise fetch
        let cached = activeTrainingConfig.getCachedStats();
        if (!cached) {
          await activeTrainingConfig.refreshStats();
          cached = activeTrainingConfig.getCachedStats();
        }
        if (!cached) throw new Error('No stats available');

        const { labelStats, totalImages } = cached;
        content.innerHTML = '';

        // Total images line
        const totLine = document.createElement('div');
        totLine.style.cssText = 'color:var(--pico-muted-color,#aaa);margin-bottom:6px;';
        totLine.textContent = `${totalImages.toLocaleString()} image${totalImages === 1 ? '' : 's'} in dataset${(tc.datasetUuids?.length ?? 0) === 1 ? '' : 's'}`;
        content.appendChild(totLine);

        if (labelStats.length === 0) {
          const none = document.createElement('div');
          none.style.cssText = 'color:var(--pico-muted-color,#888);font-style:italic;';
          none.textContent = 'No label stats available.';
          content.appendChild(none);
          return;
        }

        // Fetch labels map for names / colors
        const labelsMap = await fetchLabels().catch(() => new Map());

        function _labelName(uuid) {
          return labelsMap?.get(uuid)?.name ?? uuid;
        }

        const imageLabels = labelStats.filter(s => s.labelScope === 'image');
        const objectLabels = labelStats.filter(s => s.labelScope === 'object');

        function _renderSection(title, stats, cols) {
          if (stats.length === 0) return;
          const secHdr = document.createElement('div');
          secHdr.style.cssText = 'font-weight:600;font-size:0.88em;text-transform:uppercase;letter-spacing:.04em;color:var(--pico-muted-color,#999);margin:6px 0 3px;';
          secHdr.textContent = title;
          content.appendChild(secHdr);

          const table = document.createElement('table');
          table.style.cssText = 'width:100%;border-collapse:collapse;';

          for (const stat of stats) {
            const tr = document.createElement('tr');
            tr.style.cssText = 'border-bottom:1px solid rgba(255,255,255,0.05);';

            // Label name cell
            const nameTd = document.createElement('td');
            nameTd.style.cssText = 'padding:3px 6px 3px 0;';
            const dot = labelsMap?.get(stat.labelUuid)?.color;
            if (dot) {
              const dotEl = document.createElement('span');
              dotEl.style.cssText = `display:inline-block;width:7px;height:7px;border-radius:50%;background:${dot};margin-right:4px;vertical-align:middle;flex-shrink:0;`;
              nameTd.appendChild(dotEl);
            }
            nameTd.appendChild(document.createTextNode(_labelName(stat.labelUuid)));
            if (stat.hasRollup) {
              const badge = document.createElement('span');
              badge.style.cssText = 'display:inline-block;font-size:0.75em;background:#2a4a6a;color:#8cf;border-radius:3px;padding:0 3px;margin-left:4px;';
              badge.textContent = '⟳';
              badge.title = 'Roll-up active';
              nameTd.appendChild(badge);
            }
            tr.appendChild(nameTd);

            for (const col of cols) {
              const td = document.createElement('td');
              const val = stat[col.key] ?? 0;
              td.textContent = val.toLocaleString();
              td.style.cssText = 'text-align:right;padding:3px 0 3px 8px;white-space:nowrap;';
              td.title = col.title ?? '';
              if (col.warnIf?.(stat)) td.style.color = '#f90';
              else if (col.goodIf?.(stat)) td.style.color = '#4c4';
              tr.appendChild(td);
            }
            table.appendChild(tr);
          }
          content.appendChild(table);
        }

        _renderSection('Image Labels', imageLabels, [
          { key: 'presentImageCount', title: 'Positive examples', goodIf: s => s.presentImageCount > 0 },
          { key: 'absentImageCount', title: 'Negative examples', goodIf: s => s.absentImageCount > 0 },
          { key: 'unverifiedImageCount', title: 'Unreviewed — click \'Review unverified\' to address', warnIf: s => s.unverifiedImageCount > 0 },
        ]);

        _renderSection('Object Labels', objectLabels, [
          { key: 'verifiedBboxCount', title: 'Verified bounding boxes', goodIf: s => s.verifiedBboxCount > 0 && s.unverifiedBboxCount === 0 },
          { key: 'unverifiedBboxCount', title: 'Unverified bounding boxes', warnIf: s => s.unverifiedBboxCount > 0 },
          { key: 'coLabelGapsCount', title: 'Images with co-label gaps', warnIf: s => s.coLabelGapsCount > 0 },
        ]);

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
    _renderStats();
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
      label.textContent = 'Training config:';
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

      // Filter toggle indicator
      if (activeTrainingConfig.isFilterEnabled()) {
        const filterBadge = document.createElement('span');
        filterBadge.className = 'active-model-bar__filter-badge';
        filterBadge.textContent = 'filtered';
        filterBadge.title = 'Label list is filtered to this training config';
        left.appendChild(filterBadge);
      }

      const deactivateBtn = document.createElement('button');
      deactivateBtn.type = 'button';
      deactivateBtn.className = 'active-model-bar__deactivate';
      deactivateBtn.textContent = '✕';
      deactivateBtn.title = 'Deactivate training config';
      deactivateBtn.addEventListener('click', () => {
        _closeStatsPopup();
        activeTrainingConfig.setActive(null);
      });
      left.appendChild(deactivateBtn);
    } else {
      const none = document.createElement('span');
      none.className = 'active-model-bar__none';
      none.textContent = 'No active training config';
      left.appendChild(none);

      // Link to set one
      const setBtn = document.createElement('button');
      setBtn.type = 'button';
      setBtn.className = 'active-model-bar__set-tc-btn';
      setBtn.textContent = 'Set one →';
      setBtn.title = 'Go to Training configuration';
      setBtn.addEventListener('click', () => navigateTabs('tab-configuration', 'tab-training-management'));
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
  const unsubModel = events.subscribe(EventTypes.ACTIVE_MODEL_CHANGED, () => { render(); });

  return {
    destroy() {
      _closeVisPopup();
      _closeStatsPopup();
      unsubTC();
      unsubModel();
    },
  };
}
