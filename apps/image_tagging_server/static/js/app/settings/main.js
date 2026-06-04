import { renderHotkeyPanel } from '/app-static/js/app/image-tagging/hotkeyModal.js';

let _initialized = false;
let _handle = null;

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

  // ── Keyboard Shortcuts section ────────────────────────────────────
  const hotkeySection = document.createElement('section');
  hotkeySection.className = 'settings-section';

  const hotkeyHeader = document.createElement('div');
  hotkeyHeader.className = 'settings-section-header';

  const hotkeyTitle = document.createElement('h4');
  hotkeyTitle.textContent = 'Keyboard Shortcuts';
  hotkeyTitle.style.margin = '0';

  const resetBtn = document.createElement('button');
  resetBtn.textContent = 'Reset to defaults';
  resetBtn.className = 'secondary';
  resetBtn.style.cssText = 'font-size:0.85em; padding:0.2em 0.6em;';
  resetBtn.onclick = () => _handle?.reset();

  hotkeyHeader.append(hotkeyTitle, resetBtn);

  const hotkeyBody = document.createElement('div');
  hotkeyBody.className = 'settings-hotkey-body';

  hotkeySection.append(hotkeyHeader, hotkeyBody);
  wrap.appendChild(hotkeySection);
  container.appendChild(wrap);

  _handle = await renderHotkeyPanel(hotkeyBody);
}
