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

  const serverHeader = document.createElement('div');
  serverHeader.className = 'settings-section-header';

  const serverTitle = document.createElement('h4');
  serverTitle.textContent = 'Server Configuration';
  serverTitle.style.margin = '0';

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
  hotkeyTitle.style.margin = '0';

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
