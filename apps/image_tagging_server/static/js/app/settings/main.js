import { renderHotkeyPanel } from '/app-static/js/app/image-tagging/hotkeyModal.js';
import { apiFetch } from '/app-static/js/shared/api/apiUtils.js';
import { getSetting, setSetting } from '/app-static/js/shared/api/settings.js';

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
    'Hash sizes must match the index. Changing them requires recomputing all stored image hashes ' +
    '(use the "Rebuild Hash Index" action on the Images tab).';
  section.appendChild(warning);

  const fields = [
    {
      key: 'dedup_image_hash_size',
      label: 'Image Hash Size',
      description: 'Grid dimension for whole-image perceptual hash (8 → 64-bit). Default: 8.',
      defaultVal: 8,
    },
    {
      key: 'dedup_bbox_hash_size',
      label: 'BBox Hash Size',
      description: 'Grid dimension for bounding-box crop perceptual hash (16 → 256-bit). Default: 16.',
      defaultVal: 16,
    },
  ];

  for (const field of fields) {
    let currentVal = field.defaultVal;
    try {
      const stored = await getSetting(field.key);
      if (stored !== null) currentVal = parseInt(stored, 10) || field.defaultVal;
    } catch (_) {}

    section.appendChild(_buildDedupRow({ ...field, currentVal }));
  }

  return section;
}

function _buildDedupRow({ key, label, description, defaultVal, currentVal }) {
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
  input.min = '4';
  input.max = '64';
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
    const val = parseInt(input.value, 10);
    if (!val || val < 4 || val > 64) {
      statusLine.textContent = 'Value must be between 4 and 64.';
      statusLine.dataset.state = 'error';
      return;
    }
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving…';
    try {
      await setSetting(key, String(val));
      statusLine.textContent = `Saved: ${val} — rebuild the hash index for this to take effect.`;
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
  return row;
}

/**
 * Build a server URL configuration row.
 * @param {{label:string, serverType:string, settingKey:string, defaultUrl:string, currentOverride:string|null}} opts
 * @returns {HTMLElement}
 */
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
