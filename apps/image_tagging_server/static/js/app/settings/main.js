import { renderHotkeyPanel } from '/app-static/js/app/image-tagging/hotkeyModal.js';
import { apiFetch } from '/app-static/js/shared/api/apiUtils.js';
import { getSetting, setSetting } from '/app-static/js/shared/api/settings.js';
import { showConfirmModal } from '/app-static/js/ui/utils/confirmModal.js';
import { setConfiguredHoverDelay } from '/app-static/js/ui/thumbnailHoverPreview.js';
import { showTrainingServerModal } from '/app-static/js/ui/trainingServerModal.js';

// ── Service Status ────────────────────────────────────────────────────────────

const SERVICE_DISPLAY = {
  db_server:           { label: 'DB Server',          icon: '🗄' },
  inference_server:    { label: 'Inference Server',   icon: '🤖' },
  tasks_server:        { label: 'Tasks Server',       icon: '⚙' },
  training_server:     { label: 'Training Server',    icon: '🏋' },
  relay_buffer_server: { label: 'Relay Buffer Server',icon: '📡' },
};

function _statusColor(status) {
  if (status === 'ok')          return 'var(--pico-ins-color,#4b8)';
  if (status === 'unreachable') return 'var(--pico-muted-color,#888)';
  return 'var(--pico-del-color,#e55)';
}

const SERVICE_CLICK_HANDLERS = {
  training_server: () => showTrainingServerModal(),
};

function _renderServiceRow(svc) {
  const meta = SERVICE_DISPLAY[svc.service] ?? { label: svc.service, icon: '●' };
  const onClick = SERVICE_CLICK_HANDLERS[svc.service];

  const row = document.createElement('div');
  row.className = 'settings-service-status-row';
  const baseStyle = 'display:flex;gap:6px;align-items:center;padding:6px 0;';
  row.style.cssText = onClick
    ? baseStyle + 'cursor:pointer;border-radius:4px;transition:background 0.1s;'
    : baseStyle;
  if (onClick) {
    row.addEventListener('click', onClick);
    row.addEventListener('mouseenter', () => { row.style.background = 'var(--pico-card-background-color,rgba(255,255,255,0.05))'; });
    row.addEventListener('mouseleave', () => { row.style.background = ''; });
  }

  const indicator = document.createElement('span');
  indicator.style.cssText = `color:${_statusColor(svc.status)};font-size:1.1em;flex-shrink:0;`;
  indicator.textContent = svc.status === 'ok' ? '●' : svc.status === 'unreachable' ? '○' : '✕';
  indicator.title = svc.status;

  const name = document.createElement('span');
  name.style.cssText = 'font-size:0.85em;font-weight:600;flex-shrink:0;min-width:140px;';
  name.textContent = `${meta.icon}  ${meta.label}`;

  const detail = document.createElement('span');
  detail.style.cssText = 'font-size:0.78em;color:var(--pico-muted-color,#888);flex:1;';

  if (svc.status === 'ok') {
    const parts = [];
    if (svc.tasks)      parts.push(`running: ${svc.tasks.running}`);
    if (svc.models)     parts.push(`loaded: ${svc.models.loaded}/${svc.models.available ?? '?'}`);
    if (svc.streams)    parts.push(`streams: ${svc.streams.active}`);
    if (svc.hash_index) {
      const hi = svc.hash_index;
      parts.push(`index: ${hi.image_valid ?? 0} imgs${hi.needs_rebuild ? ' ⚠rebuild' : ''}`);
    }
    detail.textContent = parts.join('  ·  ') || 'ok';
  } else {
    detail.textContent = svc.message ?? svc.status;
    detail.style.color = _statusColor(svc.status);
  }

  const chevron = document.createElement('span');
  chevron.style.cssText = `font-size:0.8em;color:var(--pico-muted-color,#888);flex-shrink:0;visibility:${onClick ? 'visible' : 'hidden'};`;
  chevron.textContent = '›';

  row.append(indicator, name, detail, chevron);
  return row;
}

function _buildServiceStatusSection() {
  const section = document.createElement('section');
  section.className = 'settings-section';

  const header = document.createElement('div');
  header.className = 'settings-section-header';
  header.style.cssText = 'display:flex;align-items:center;gap:8px;';

  const title = document.createElement('h4');
  title.textContent = 'Service Status';
  title.style.margin = '0';

  const refreshBtn = document.createElement('button');
  refreshBtn.textContent = '↻ Refresh';
  refreshBtn.className = 'secondary';
  refreshBtn.style.cssText = 'font-size:0.8em;padding:0.2em 0.6em;margin-left:auto;';

  header.append(title, refreshBtn);
  section.appendChild(header);

  const body = document.createElement('div');
  body.style.cssText = 'margin-top:6px;';
  section.appendChild(body);

  async function refresh() {
    refreshBtn.disabled = true;
    refreshBtn.textContent = '↻ Refreshing…';
    body.innerHTML = '';
    try {
      const res = await apiFetch('/api/service-status');
      const data = await res.json();
      const services = data.services ?? [];
      for (const svc of services) {
        body.appendChild(_renderServiceRow(svc));
      }
      if (!services.length) {
        body.textContent = 'No services found.';
        body.style.color = 'var(--pico-muted-color,#888)';
      }
    } catch (e) {
      body.textContent = `Error: ${e.message}`;
      body.style.color = 'var(--pico-del-color,#e55)';
    } finally {
      refreshBtn.disabled = false;
      refreshBtn.textContent = '↻ Refresh';
    }
  }

  refreshBtn.addEventListener('click', refresh);
  // Fire-and-forget so the rest of the settings panel renders immediately.
  refresh();
  return section;
}

let _initialized = false;
let _hotkeyHandle = null;

/**
 * Initialise the Settings tab. Safe to call multiple times (no-op after first call).
 */
export async function init() {
  if (_initialized) return;
  _initialized = true;

  const container = document.getElementById('tab-settings');
  if (!container) return;

  container.innerHTML = '';

  const wrap = document.createElement('div');
  wrap.className = 'settings-wrap';

  // ── Service Status section ────────────────────────────────────────────
  wrap.appendChild(_buildServiceStatusSection());

  // ── Server Configuration section ─────────────────────────────────────
  const serverSection = document.createElement('section');
  serverSection.className = 'settings-section';
  serverSection.id = 'settings-section-server';

  const serverHeader = document.createElement('div');
  serverHeader.className = 'settings-section-header';

  const serverTitle = document.createElement('h4');
  serverTitle.textContent = 'Server Configuration';

  serverHeader.appendChild(serverTitle);
  serverSection.appendChild(serverHeader);

  // Load current server config for defaults/overrides
  let serverConfig = {};
  try {
    const res = await fetch('/server-config');
    if (res.ok) {
      const data = await res.json();
      serverConfig = data.config ?? {};
    }
  } catch (_) {}

  serverSection.appendChild(
    _buildServerRow({
      label: 'Inference Server',
      serverType: 'inference_server',
      settingKey: 'inference_server_url',
      defaultUrl: serverConfig.inference_server ?? 'http://localhost:8001',
      currentOverride: serverConfig.inference_server_url_override ?? null,
    })
  );

  serverSection.appendChild(
    _buildServerRow({
      label: 'Training Server',
      serverType: 'training_server',
      settingKey: 'training_server_url',
      defaultUrl: serverConfig.training_server ?? 'http://localhost:8005',
      currentOverride: serverConfig.training_server_url_override ?? null,
    })
  );

  wrap.appendChild(serverSection);

  // ── Deduplication Settings section ───────────────────────────────────
  wrap.appendChild(await _buildDedupSection());

  // ── UI Settings section ───────────────────────────────────────────────
  wrap.appendChild(await _buildUiSettingsSection());

  // ── Keyboard Shortcuts section (collapsible) ──────────────────────────
  const hotkeySection = document.createElement('section');
  hotkeySection.className = 'settings-section';

  const hotkeyHeader = document.createElement('div');
  hotkeyHeader.className = 'settings-section-header settings-section-header--collapsible';

  const hotkeyTitleRow = document.createElement('div');
  hotkeyTitleRow.style.cssText = 'display:flex; align-items:center; gap:0.5em; cursor:pointer; flex:1;';

  const collapseToggle = document.createElement('span');
  collapseToggle.className = 'settings-collapse-chevron';
  collapseToggle.textContent = '▶';

  const hotkeyTitle = document.createElement('h4');
  hotkeyTitle.textContent = 'Keyboard Shortcuts';

  hotkeyTitleRow.append(collapseToggle, hotkeyTitle);

  const resetBtn = document.createElement('button');
  resetBtn.textContent = 'Reset to defaults';
  resetBtn.className = 'secondary';
  resetBtn.style.cssText = 'font-size:0.85em; padding:0.2em 0.6em;';
  resetBtn.onclick = (e) => {
    e.stopPropagation();
    _hotkeyHandle?.reset();
  };

  hotkeyHeader.append(hotkeyTitleRow, resetBtn);

  // Outer div controls show/hide. Inner div is passed to renderHotkeyPanel,
  // which sets its own inline display:flex — keeping that separate prevents
  // the inline style from overriding the outer div's display:none on collapse.
  const hotkeyBody = document.createElement('div');
  hotkeyBody.style.display = 'none';

  const hotkeyInner = document.createElement('div');
  hotkeyInner.className = 'settings-hotkey-body';
  hotkeyBody.appendChild(hotkeyInner);

  let hotkeyLoaded = false;

  async function toggleHotkeys() {
    const isCollapsed = hotkeyBody.style.display === 'none';
    hotkeyBody.style.display = isCollapsed ? '' : 'none';
    collapseToggle.textContent = isCollapsed ? '▼' : '▶';
    if (isCollapsed && !hotkeyLoaded) {
      hotkeyLoaded = true;
      _hotkeyHandle = await renderHotkeyPanel(hotkeyInner);
    }
  }

  hotkeyTitleRow.addEventListener('click', toggleHotkeys);

  hotkeySection.append(hotkeyHeader, hotkeyBody);
  wrap.appendChild(hotkeySection);
  container.appendChild(wrap);
}

async function _buildDedupSection() {
  const section = document.createElement('section');
  section.className = 'settings-section';

  const header = document.createElement('div');
  header.className = 'settings-section-header';
  const title = document.createElement('h4');
  title.textContent = 'Deduplication';
  header.appendChild(title);
  section.appendChild(header);

  const warning = document.createElement('p');
  warning.className = 'settings-dedup-warning';
  warning.textContent =
    'Hash sizes must match the index. Saving a new hash size rebuilds the index immediately. ' +
    'Existing images keep their old hashes until reprocessed — run Database Maintenance (recompute hashes) to update them all at once.';
  section.appendChild(warning);

  // Load all four settings in parallel
  const [imgHashStored, bboxHashStored, imgThreshStored, bboxThreshStored] = await Promise.all([
    getSetting('dedup_image_hash_size').catch(() => null),
    getSetting('dedup_bbox_hash_size').catch(() => null),
    getSetting('dedup_image_threshold').catch(() => null),
    getSetting('dedup_bbox_threshold').catch(() => null),
  ]);

  const imgN    = parseInt(imgHashStored, 10) || 8;
  const bboxN   = parseInt(bboxHashStored, 10) || 16;
  const imgMax  = imgN * imgN;
  const bboxMax = bboxN * bboxN;

  const _threshVal = (stored, max) => {
    if (stored !== null && stored !== '') {
      const n = parseInt(stored, 10);
      if (!isNaN(n) && n >= 1) return Math.min(n, max);
    }
    return Math.round(max * 0.2);
  };

  const { element: imgHashRow,   select: imgHashSelect  } = _buildHashSizeRow({
    key: 'dedup_image_hash_size',
    label: 'Image Hash Size',
    description: 'Grid dimension n for whole-image perceptual hash (n² bits). Default: 8 → 64-bit.',
    currentN: imgN,
  });
  const { element: imgThreshRow, input:  imgThreshInput  } = _buildThresholdRow({
    key: 'dedup_image_threshold',
    label: 'Image Similarity Threshold',
    description: 'Max Hamming distance for "find similar images" searches. Auto-set to 20% of hash range when hash size changes.',
    currentVal: _threshVal(imgThreshStored, imgMax),
    max: imgMax,
  });

  const { element: bboxHashRow,   select: bboxHashSelect  } = _buildHashSizeRow({
    key: 'dedup_bbox_hash_size',
    label: 'BBox Hash Size',
    description: 'Grid dimension n for bounding-box crop perceptual hash (n² bits). Default: 16 → 256-bit.',
    currentN: bboxN,
  });
  const { element: bboxThreshRow, input:  bboxThreshInput  } = _buildThresholdRow({
    key: 'dedup_bbox_threshold',
    label: 'BBox Similarity Threshold',
    description: 'Max Hamming distance for "find similar bboxes" searches. Auto-set to 20% of hash range when hash size changes.',
    currentVal: _threshVal(bboxThreshStored, bboxMax),
    max: bboxMax,
  });

  // When hash size changes, auto-update the paired threshold input to the 20% default
  function _wireHashToThreshold(hashSelect, threshInput) {
    hashSelect.addEventListener('change', () => {
      const max = parseInt(hashSelect.value, 10) ** 2;
      threshInput.max = String(max);
      threshInput.value = String(Math.round(max * 0.2));
    });
  }
  _wireHashToThreshold(imgHashSelect, imgThreshInput);
  _wireHashToThreshold(bboxHashSelect, bboxThreshInput);

  section.append(imgHashRow, imgThreshRow, bboxHashRow, bboxThreshRow);
  return section;
}

function _buildHashSizeRow({ key, label, description, currentN }) {
  const HASH_SIZES = [4, 8, 16, 32, 64];

  const row = document.createElement('div');
  row.className = 'settings-server-row';

  const rowLabel = document.createElement('div');
  rowLabel.className = 'settings-server-label';
  rowLabel.textContent = label;

  const desc = document.createElement('div');
  desc.className = 'settings-server-desc';
  desc.textContent = description;

  const inputRow = document.createElement('div');
  inputRow.className = 'settings-server-input-row';

  const select = document.createElement('select');
  select.className = 'settings-server-input settings-server-input--narrow';
  for (const n of HASH_SIZES) {
    const opt = document.createElement('option');
    opt.value = String(n);
    opt.textContent = `${n} (${n * n}-bit)`;
    if (n === currentN) opt.selected = true;
    select.appendChild(opt);
  }

  const saveBtn = document.createElement('button');
  saveBtn.textContent = 'Save';
  saveBtn.className = 'settings-server-btn';

  inputRow.append(select, saveBtn);

  const statusLine = document.createElement('div');
  statusLine.className = 'settings-server-status';
  statusLine.dataset.state = 'muted';
  statusLine.textContent = `Current: ${currentN} (${currentN * currentN}-bit)`;

  saveBtn.addEventListener('click', async () => {
    const val = parseInt(select.value, 10);
    const confirmed = await showConfirmModal({
      title: 'Change hash size?',
      message:
        `Changing the hash size will save the new setting and immediately rebuild the similarity index. ` +
        `Existing images will get new-size hashes the next time they are processed, or you can run ` +
        `Database Maintenance (recompute hashes) to update them all at once. Continue?`,
      confirmText: 'Change & Rebuild',
      cancelText: 'Cancel',
      danger: true,
    });
    if (!confirmed) return;
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving…';
    try {
      await setSetting(key, String(val));
      statusLine.textContent = `Saved: ${val} (${val * val}-bit) — rebuilding index…`;
      statusLine.dataset.state = 'info';
      saveBtn.textContent = 'Rebuilding…';
      try {
        await apiFetch('/api/images/phash/clear-and-rebuild', { method: 'POST' });
        statusLine.textContent = `Saved: ${val} (${val * val}-bit) — index rebuilt.`;
      } catch (rebuildErr) {
        statusLine.textContent = `Saved: ${val} (${val * val}-bit) — index rebuild failed: ${rebuildErr.message}`;
        statusLine.dataset.state = 'error';
      }
      saveBtn.textContent = '✓ Done';
      setTimeout(() => {
        saveBtn.textContent = 'Save';
        saveBtn.disabled = false;
      }, 1500);
    } catch (e) {
      statusLine.textContent = `✗ ${e.message}`;
      statusLine.dataset.state = 'error';
      saveBtn.textContent = 'Save';
      saveBtn.disabled = false;
    }
  });

  row.append(rowLabel, desc, inputRow, statusLine);
  return { element: row, select };
}

function _buildThresholdRow({ key, label, description, currentVal, max }) {
  const row = document.createElement('div');
  row.className = 'settings-server-row';

  const rowLabel = document.createElement('div');
  rowLabel.className = 'settings-server-label';
  rowLabel.textContent = label;

  const desc = document.createElement('div');
  desc.className = 'settings-server-desc';
  desc.textContent = description;

  const inputRow = document.createElement('div');
  inputRow.className = 'settings-server-input-row';

  const input = document.createElement('input');
  input.type = 'number';
  input.min = '1';
  input.max = String(max);
  input.step = '1';
  input.className = 'settings-server-input settings-server-input--narrow';
  input.value = String(currentVal);

  const saveBtn = document.createElement('button');
  saveBtn.textContent = 'Save';
  saveBtn.className = 'settings-server-btn';

  inputRow.append(input, saveBtn);

  const statusLine = document.createElement('div');
  statusLine.className = 'settings-server-status';
  statusLine.dataset.state = 'muted';
  statusLine.textContent = `Current: ${currentVal}`;

  saveBtn.addEventListener('click', async () => {
    const curMax = parseInt(input.max, 10);
    const val = parseInt(input.value, 10);
    if (isNaN(val) || val < 1 || val > curMax) {
      statusLine.textContent = `Value must be between 1 and ${curMax}.`;
      statusLine.dataset.state = 'error';
      return;
    }
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving…';
    try {
      await setSetting(key, String(val));
      statusLine.textContent = `Saved: ${val}`;
      statusLine.dataset.state = 'info';
      saveBtn.textContent = '✓ Saved';
      setTimeout(() => {
        saveBtn.textContent = 'Save';
        saveBtn.disabled = false;
      }, 1500);
    } catch (e) {
      statusLine.textContent = `✗ ${e.message}`;
      statusLine.dataset.state = 'error';
      saveBtn.textContent = 'Save';
      saveBtn.disabled = false;
    }
  });

  row.append(rowLabel, desc, inputRow, statusLine);
  return { element: row, input };
}

/**
 * Build a server URL configuration row.
 * @param {{label:string, serverType:string, settingKey:string, defaultUrl:string, currentOverride:string|null}} opts
 * @returns {HTMLElement}
 */
async function _buildUiSettingsSection() {
  const section = document.createElement('section');
  section.className = 'settings-section';

  const header = document.createElement('div');
  header.className = 'settings-section-header';
  const title = document.createElement('h4');
  title.textContent = 'UI Settings';
  header.appendChild(title);
  section.appendChild(header);

  // ── Hover thumbnail delay ────────────────────────────────────────────
  const HOVER_DELAY_KEY = 'ui_hover_delay_ms';
  const HOVER_DELAY_DEFAULT = 500;

  let currentDelay = HOVER_DELAY_DEFAULT;
  try {
    const stored = await getSetting(HOVER_DELAY_KEY);
    if (stored !== null) currentDelay = parseInt(stored, 10) || HOVER_DELAY_DEFAULT;
  } catch (_) {}

  const row = document.createElement('div');
  row.className = 'settings-server-row';

  const rowLabel = document.createElement('div');
  rowLabel.className = 'settings-server-label';
  rowLabel.textContent = 'Hover thumbnail delay';

  const desc = document.createElement('div');
  desc.className = 'settings-server-desc';
  desc.textContent = 'Milliseconds to wait before showing the thumbnail popup on hover. Default: 500.';

  const inputRow = document.createElement('div');
  inputRow.className = 'settings-server-input-row';

  const input = document.createElement('input');
  input.type = 'number';
  input.min = '0';
  input.max = '5000';
  input.step = '50';
  input.className = 'settings-server-input settings-server-input--narrow';
  input.value = String(currentDelay);

  const saveBtn = document.createElement('button');
  saveBtn.textContent = 'Save';
  saveBtn.className = 'settings-server-btn';

  inputRow.append(input, saveBtn);

  const statusLine = document.createElement('div');
  statusLine.className = 'settings-server-status';
  statusLine.dataset.state = 'muted';
  statusLine.textContent = `Current: ${currentDelay} ms`;

  saveBtn.addEventListener('click', async () => {
    const val = parseInt(input.value, 10);
    if (!Number.isFinite(val) || val < 0 || val > 5000) {
      statusLine.textContent = 'Value must be between 0 and 5000 ms.';
      statusLine.dataset.state = 'error';
      return;
    }
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving…';
    try {
      await setSetting(HOVER_DELAY_KEY, String(val));
      setConfiguredHoverDelay(val);
      statusLine.textContent = `Saved: ${val} ms`;
      statusLine.dataset.state = 'info';
      saveBtn.textContent = '✓ Saved';
      setTimeout(() => {
        saveBtn.textContent = 'Save';
        saveBtn.disabled = false;
      }, 1500);
    } catch (e) {
      statusLine.textContent = `✗ ${e.message}`;
      statusLine.dataset.state = 'error';
      saveBtn.textContent = 'Save';
      saveBtn.disabled = false;
    }
  });

  row.append(rowLabel, desc, inputRow, statusLine);
  section.appendChild(row);

  return section;
}

function _buildServerRow({ label, serverType, settingKey, defaultUrl, currentOverride }) {
  const row = document.createElement('div');
  row.className = 'settings-server-row';

  const rowLabel = document.createElement('div');
  rowLabel.className = 'settings-server-label';
  rowLabel.textContent = label;

  const inputRow = document.createElement('div');
  inputRow.className = 'settings-server-input-row';

  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'settings-server-input';
  input.placeholder = `${defaultUrl}  (default)`;
  input.value = currentOverride ?? '';

  const testBtn = document.createElement('button');
  testBtn.textContent = 'Test';
  testBtn.className = 'secondary settings-server-btn';

  const saveBtn = document.createElement('button');
  saveBtn.textContent = 'Save';
  saveBtn.className = 'settings-server-btn';

  inputRow.append(input, testBtn, saveBtn);

  const statusLine = document.createElement('div');
  statusLine.className = 'settings-server-status';
  if (currentOverride) {
    statusLine.textContent = `Using: ${currentOverride}`;
    statusLine.dataset.state = 'info';
  } else {
    statusLine.textContent = `Using default: ${defaultUrl}`;
    statusLine.dataset.state = 'muted';
  }

  testBtn.addEventListener('click', async () => {
    const url = input.value.trim() || defaultUrl;
    testBtn.disabled = true;
    testBtn.textContent = 'Testing…';
    statusLine.dataset.state = 'muted';
    statusLine.textContent = `Testing ${url}…`;
    try {
      const res = await apiFetch('/api/test-server', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, server_type: serverType }),
      });
      const data = await res.json();
      if (data.success) {
        statusLine.textContent = `✓ ${data.message}`;
        statusLine.dataset.state = 'ok';
      } else {
        statusLine.textContent = `✗ ${data.message}`;
        statusLine.dataset.state = 'error';
      }
    } catch (e) {
      statusLine.textContent = `✗ ${e.message}`;
      statusLine.dataset.state = 'error';
    } finally {
      testBtn.disabled = false;
      testBtn.textContent = 'Test';
    }
  });

  saveBtn.addEventListener('click', async () => {
    const url = input.value.trim();
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving…';
    try {
      if (url) {
        await setSetting(settingKey, url);
      } else {
        // Clear override — delete by saving empty string; backend treats '' as "no override"
        await setSetting(settingKey, '');
      }
      // Tell backend to reload proxies
      const res = await apiFetch('/api/reload-server-config', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        const effective = url || defaultUrl;
        statusLine.textContent = url ? `Using: ${url}` : `Using default: ${defaultUrl}`;
        statusLine.dataset.state = url ? 'info' : 'muted';
        saveBtn.textContent = '✓ Saved';
        setTimeout(() => {
          saveBtn.textContent = 'Save';
          saveBtn.disabled = false;
        }, 1500);
      } else {
        statusLine.textContent = `✗ Save failed: ${data.message ?? 'unknown error'}`;
        statusLine.dataset.state = 'error';
        saveBtn.textContent = 'Save';
        saveBtn.disabled = false;
      }
    } catch (e) {
      statusLine.textContent = `✗ ${e.message}`;
      statusLine.dataset.state = 'error';
      saveBtn.textContent = 'Save';
      saveBtn.disabled = false;
    }
  });

  row.append(rowLabel, inputRow, statusLine);
  return row;
}
