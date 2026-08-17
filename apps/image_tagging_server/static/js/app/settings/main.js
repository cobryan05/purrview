import { renderHotkeyPanel } from '/app-static/js/app/image-tagging/hotkeyModal.js';
import { apiFetch } from '/app-static/js/shared/api/apiUtils.js';
import { getSetting, setSetting } from '/app-static/js/shared/api/settings.js';
import { showConfirmModal } from '/app-static/js/ui/utils/confirmModal.js';
import { setConfiguredHoverDelay } from '/app-static/js/ui/thumbnailHoverPreview.js';
import {
  DEFAULT_BBOX_HASH_SIZE,
  DEFAULT_BBOX_THRESHOLD,
  DEFAULT_IMAGE_HASH_SIZE,
  DEFAULT_IMAGE_THRESHOLD,
  defaultThresholdFor,
  hashMaxFor,
} from '/app-static/js/shared/dedupDefaults.js';

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

  // The three settings panels load their values concurrently — each awaits its
  // own settings reads, so building them in sequence would serialise round-trips
  // that have nothing to do with each other.
  const [serverPanel, dedupPanel, uiPanel] = await Promise.all([
    _buildServerSection(),
    _buildDedupSection(),
    _buildUiSettingsSection(),
  ]);

  wrap.append(serverPanel, dedupPanel, uiPanel, _buildHotkeySection());
  container.appendChild(wrap);
}

/* ── Panel scaffolding ──────────────────────────────────────────────────────
 * Every section is a `.panel-box` (the same bordered bar+body component the
 * Configuration screens use), so Settings picks up the app's panel styling for
 * free instead of carrying its own section chrome. */

/**
 * @param {string} title
 * @param {{id?: string, extraClass?: string}} [opts]
 * @returns {{section: HTMLElement, bar: HTMLElement, body: HTMLElement, heading: HTMLElement}}
 */
function _panel(title, { id, extraClass } = {}) {
  const section = document.createElement('section');
  section.className = 'panel-box settings-panel';
  if (extraClass) section.classList.add(extraClass);
  if (id) section.id = id;

  const bar = document.createElement('div');
  bar.className = 'panel-box__bar settings-section-header';

  const heading = document.createElement('div');
  heading.className = 'panel-box__title';
  heading.textContent = title;
  bar.appendChild(heading);

  const body = document.createElement('div');
  body.className = 'panel-box__body settings-panel__body';

  section.append(bar, body);
  return { section, bar, body, heading };
}

/**
 * Build one settings row: label on the left, controls on the right, with an
 * optional hint underneath and a status line that only takes vertical space
 * once it has something to say.
 *
 * @param {{label: string, description?: string, controls: HTMLElement[], stacked?: boolean}} opts
 * @returns {{row: HTMLElement, status: HTMLElement, setStatus: (text: string, state?: string) => void}}
 */
function _buildRow({ label, description, controls, stacked = false }) {
  const row = document.createElement('div');
  row.className = 'settings-server-row';
  if (stacked) row.classList.add('settings-server-row--stacked');

  const head = document.createElement('div');
  head.className = 'settings-row-head';

  const rowLabel = document.createElement('div');
  rowLabel.className = 'settings-server-label';
  rowLabel.textContent = label;

  const inputRow = document.createElement('div');
  inputRow.className = 'settings-server-input-row';
  inputRow.append(...controls);

  head.append(rowLabel, inputRow);
  row.appendChild(head);

  if (description) {
    const desc = document.createElement('div');
    desc.className = 'settings-server-desc';
    desc.textContent = description;
    row.appendChild(desc);
  }

  const status = document.createElement('div');
  status.className = 'settings-server-status';
  status.dataset.state = 'muted';
  row.appendChild(status);

  const setStatus = (text, state = 'muted') => {
    status.textContent = text;
    status.dataset.state = state;
  };

  return { row, status, setStatus };
}

/** A compact Save/Test button for a settings row. */
function _rowButton(text, { secondary = false } = {}) {
  const btn = document.createElement('button');
  btn.textContent = text;
  btn.className = secondary ? 'secondary panel-btn settings-server-btn' : 'panel-btn settings-server-btn';
  return btn;
}

/**
 * Run a save handler with the usual button feedback: disable → "Saving…" →
 * "✓ Saved" → back to the original label. Every row shared this block verbatim.
 *
 * @param {HTMLButtonElement} btn
 * @param {() => Promise<void>} work
 * @param {(msg: string) => void} onError
 */
async function _withSaveFeedback(btn, work, onError) {
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Saving…';
  try {
    await work();
    btn.textContent = '✓ Saved';
    setTimeout(() => {
      btn.textContent = original;
      btn.disabled = false;
    }, 1500);
  } catch (e) {
    onError(e.message);
    btn.textContent = original;
    btn.disabled = false;
  }
}

/* ── Server configuration ──────────────────────────────────────────────────── */

async function _buildServerSection() {
  const { section, body } = _panel('Servers', { id: 'settings-section-server' });

  // Load current server config for defaults/overrides
  let serverConfig = {};
  try {
    const res = await fetch('/server-config');
    if (res.ok) {
      const data = await res.json();
      serverConfig = data.config ?? {};
    }
  } catch (_) {}

  body.append(
    _buildServerRow({
      label: 'Inference Server',
      serverType: 'inference_server',
      settingKey: 'inference_server_url',
      description:
        'Runs ONNX detection and classification. Takes images over HTTP and needs no database, so it can run on ' +
        'another host — but the models directory must live with it, and trained models are currently written to the ' +
        'models directory of whichever host runs the tasks server. Point this elsewhere only if both share that directory.',
      defaultUrl: serverConfig.inference_server ?? 'http://localhost:8001',
      currentOverride: serverConfig.inference_server_url_override ?? null,
    }),
    _buildServerRow({
      label: 'Training Server',
      serverType: 'training_server',
      settingKey: 'training_server_url',
      description:
        'Runs training jobs. Fully network-isolated — it keeps its own artifacts directory and hands finished models ' +
        'back over HTTP, so it needs no shared filesystem and is safe to run on a separate GPU host.',
      defaultUrl: serverConfig.training_server ?? 'http://localhost:8005',
      currentOverride: serverConfig.training_server_url_override ?? null,
    })
  );

  return section;
}

/**
 * Build a server URL configuration row. The URL input needs the full row width,
 * so this row stacks its label above the controls rather than sitting beside them.
 *
 * @param {{label:string, serverType:string, settingKey:string, description?:string, defaultUrl:string, currentOverride:string|null}} opts
 * @returns {HTMLElement}
 */
function _buildServerRow({ label, serverType, settingKey, description, defaultUrl, currentOverride }) {
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'settings-server-input';
  input.placeholder = `${defaultUrl}  (default)`;
  input.value = currentOverride ?? '';

  const testBtn = _rowButton('Test', { secondary: true });
  const saveBtn = _rowButton('Save');

  const { row, setStatus } = _buildRow({
    label,
    description,
    controls: [input, testBtn, saveBtn],
    stacked: true,
  });

  if (currentOverride) {
    setStatus(`Using: ${currentOverride}`, 'info');
  } else {
    setStatus(`Using default: ${defaultUrl}`, 'muted');
  }

  testBtn.addEventListener('click', async () => {
    const url = input.value.trim() || defaultUrl;
    testBtn.disabled = true;
    testBtn.textContent = 'Testing…';
    setStatus(`Testing ${url}…`, 'muted');
    try {
      const res = await apiFetch('/api/test-server', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, server_type: serverType }),
      });
      const data = await res.json();
      setStatus(data.success ? `✓ ${data.message}` : `✗ ${data.message}`, data.success ? 'ok' : 'error');
    } catch (e) {
      setStatus(`✗ ${e.message}`, 'error');
    } finally {
      testBtn.disabled = false;
      testBtn.textContent = 'Test';
    }
  });

  saveBtn.addEventListener('click', async () => {
    const url = input.value.trim();
    await _withSaveFeedback(
      saveBtn,
      async () => {
        // Clear override — save empty string; backend treats '' as "no override"
        await setSetting(settingKey, url);
        // Tell backend to reload proxies
        const res = await apiFetch('/api/reload-server-config', { method: 'POST' });
        const data = await res.json();
        if (data.status !== 'success') {
          throw new Error(`Save failed: ${data.message ?? 'unknown error'}`);
        }
        setStatus(url ? `Using: ${url}` : `Using default: ${defaultUrl}`, url ? 'info' : 'muted');
      },
      (msg) => setStatus(`✗ ${msg}`, 'error')
    );
  });

  return row;
}

/* ── Deduplication ─────────────────────────────────────────────────────────── */

async function _buildDedupSection() {
  const { section, body } = _panel('Deduplication', { extraClass: 'settings-panel--tall' });

  const warning = document.createElement('p');
  warning.className = 'settings-dedup-warning';
  warning.textContent =
    'Hash sizes must match the index. Saving a new hash size clears all stored hashes and rebuilds an empty index immediately — ' +
    'existing images will not match in similarity search until reprocessed. Each image can be reprocessed individually, or you ' +
    'can run a Database Maintenance task (Configuration → Tasks) with "Recompute ALL perceptual hashes" enabled to update them all at once.';
  body.appendChild(warning);

  // Load all four settings in parallel
  const [imgHashStored, bboxHashStored, imgThreshStored, bboxThreshStored] = await Promise.all([
    getSetting('dedup_image_hash_size').catch(() => null),
    getSetting('dedup_bbox_hash_size').catch(() => null),
    getSetting('dedup_image_threshold').catch(() => null),
    getSetting('dedup_bbox_threshold').catch(() => null),
  ]);

  const imgN    = parseInt(imgHashStored, 10) || DEFAULT_IMAGE_HASH_SIZE;
  const bboxN   = parseInt(bboxHashStored, 10) || DEFAULT_BBOX_HASH_SIZE;
  const imgMax  = hashMaxFor(imgN);
  const bboxMax = hashMaxFor(bboxN);

  const _threshVal = (stored, max, kind) => {
    if (stored !== null && stored !== '') {
      const n = parseInt(stored, 10);
      if (!isNaN(n) && n >= 1) return Math.min(n, max);
    }
    return Math.min(defaultThresholdFor(kind, Math.round(Math.sqrt(max))), max);
  };

  const { element: imgHashRow,   select: imgHashSelect  } = _buildHashSizeRow({
    key: 'dedup_image_hash_size',
    label: 'Image hash size',
    description: `Grid dimension n for whole-image perceptual hash (n² bits). Default: ${DEFAULT_IMAGE_HASH_SIZE} → ${hashMaxFor(DEFAULT_IMAGE_HASH_SIZE)}-bit.`,
    currentN: imgN,
  });
  const { element: imgThreshRow, input:  imgThreshInput  } = _buildThresholdRow({
    key: 'dedup_image_threshold',
    label: 'Image threshold',
    description: `Max Hamming distance for "find similar images" searches. Default: ${DEFAULT_IMAGE_THRESHOLD} — rescaled to keep the same selectivity when hash size changes.`,
    currentVal: _threshVal(imgThreshStored, imgMax, 'image'),
    max: imgMax,
  });

  const { element: bboxHashRow,   select: bboxHashSelect  } = _buildHashSizeRow({
    key: 'dedup_bbox_hash_size',
    label: 'BBox hash size',
    description: `Grid dimension n for bounding-box crop perceptual hash (n² bits). Default: ${DEFAULT_BBOX_HASH_SIZE} → ${hashMaxFor(DEFAULT_BBOX_HASH_SIZE)}-bit.`,
    currentN: bboxN,
  });
  const { element: bboxThreshRow, input:  bboxThreshInput  } = _buildThresholdRow({
    key: 'dedup_bbox_threshold',
    label: 'BBox threshold',
    description: `Max Hamming distance for "find similar bboxes" searches. Default: ${DEFAULT_BBOX_THRESHOLD} — rescaled to keep the same selectivity when hash size changes.`,
    currentVal: _threshVal(bboxThreshStored, bboxMax, 'bbox'),
    max: bboxMax,
  });

  // Changing hash size changes the bit-width the threshold is measured over, so
  // the number in the box stops meaning what it did (75 is loose over 256 bits
  // and impossible over 64). Reset it to this kind's default, rescaled to the
  // new width, rather than leaving a stale figure the user has to reason about.
  function _wireHashToThreshold(hashSelect, threshInput, kind) {
    hashSelect.addEventListener('change', () => {
      const n = parseInt(hashSelect.value, 10);
      threshInput.max = String(hashMaxFor(n));
      threshInput.value = String(defaultThresholdFor(kind, n));
    });
  }
  _wireHashToThreshold(imgHashSelect, imgThreshInput, 'image');
  _wireHashToThreshold(bboxHashSelect, bboxThreshInput, 'bbox');

  body.append(imgHashRow, imgThreshRow, bboxHashRow, bboxThreshRow);
  return section;
}

function _buildHashSizeRow({ key, label, description, currentN }) {
  const HASH_SIZES = [4, 8, 16, 32, 64];

  const select = document.createElement('select');
  select.className = 'settings-server-input settings-server-input--select';
  for (const n of HASH_SIZES) {
    const opt = document.createElement('option');
    opt.value = String(n);
    opt.textContent = `${n} (${n * n}-bit)`;
    if (n === currentN) opt.selected = true;
    select.appendChild(opt);
  }

  const saveBtn = _rowButton('Save');

  const { row, setStatus } = _buildRow({ label, description, controls: [select, saveBtn] });

  saveBtn.addEventListener('click', async () => {
    const val = parseInt(select.value, 10);
    const confirmed = await showConfirmModal({
      title: 'Change hash size?',
      message:
        `Changing the hash size will save the new setting and immediately clear all stored hashes, rebuilding an ` +
        `empty similarity index. Existing images will have no hash — and won't match in similarity search — until ` +
        `each is reprocessed individually, or you run a Database Maintenance task (Configuration → Tasks) with ` +
        `"Recompute ALL perceptual hashes" enabled to update them all at once. Continue?`,
      confirmText: 'Change & Rebuild',
      cancelText: 'Cancel',
      danger: true,
    });
    if (!confirmed) return;
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving…';
    try {
      await setSetting(key, String(val));
      setStatus(`Saved: ${val} (${val * val}-bit) — rebuilding index…`, 'info');
      saveBtn.textContent = 'Rebuilding…';
      try {
        await apiFetch('/api/images/phash/clear-and-rebuild', { method: 'POST' });
        setStatus(`Saved: ${val} (${val * val}-bit) — index rebuilt.`, 'info');
      } catch (rebuildErr) {
        setStatus(`Saved: ${val} (${val * val}-bit) — index rebuild failed: ${rebuildErr.message}`, 'error');
      }
      saveBtn.textContent = '✓ Done';
      setTimeout(() => {
        saveBtn.textContent = 'Save';
        saveBtn.disabled = false;
      }, 1500);
    } catch (e) {
      setStatus(`✗ ${e.message}`, 'error');
      saveBtn.textContent = 'Save';
      saveBtn.disabled = false;
    }
  });

  return { element: row, select };
}

function _buildThresholdRow({ key, label, description, currentVal, max }) {
  const input = document.createElement('input');
  input.type = 'number';
  input.min = '1';
  input.max = String(max);
  input.step = '1';
  input.className = 'settings-server-input settings-server-input--narrow';
  input.value = String(currentVal);

  const saveBtn = _rowButton('Save');

  const { row, setStatus } = _buildRow({ label, description, controls: [input, saveBtn] });

  saveBtn.addEventListener('click', async () => {
    const curMax = parseInt(input.max, 10);
    const val = parseInt(input.value, 10);
    if (isNaN(val) || val < 1 || val > curMax) {
      setStatus(`Value must be between 1 and ${curMax}.`, 'error');
      return;
    }
    await _withSaveFeedback(
      saveBtn,
      async () => {
        await setSetting(key, String(val));
        setStatus(`Saved: ${val}`, 'info');
      },
      (msg) => setStatus(`✗ ${msg}`, 'error')
    );
  });

  return { element: row, input };
}

/* ── UI settings ───────────────────────────────────────────────────────────── */

async function _buildUiSettingsSection() {
  const { section, body } = _panel('Interface');

  // ── Hover thumbnail delay ────────────────────────────────────────────
  const HOVER_DELAY_KEY = 'ui_hover_delay_ms';
  const HOVER_DELAY_DEFAULT = 500;

  let currentDelay = HOVER_DELAY_DEFAULT;
  try {
    const stored = await getSetting(HOVER_DELAY_KEY);
    if (stored !== null) currentDelay = parseInt(stored, 10) || HOVER_DELAY_DEFAULT;
  } catch (_) {}

  const input = document.createElement('input');
  input.type = 'number';
  input.min = '0';
  input.max = '5000';
  input.step = '50';
  input.className = 'settings-server-input settings-server-input--narrow';
  input.value = String(currentDelay);

  const unit = document.createElement('span');
  unit.className = 'settings-input-unit';
  unit.textContent = 'ms';

  const saveBtn = _rowButton('Save');

  const { row, setStatus } = _buildRow({
    label: 'Hover thumbnail delay',
    description: 'How long to wait before showing the thumbnail popup on hover. Default: 500 ms.',
    controls: [input, unit, saveBtn],
  });

  saveBtn.addEventListener('click', async () => {
    const val = parseInt(input.value, 10);
    if (!Number.isFinite(val) || val < 0 || val > 5000) {
      setStatus('Value must be between 0 and 5000 ms.', 'error');
      return;
    }
    await _withSaveFeedback(
      saveBtn,
      async () => {
        await setSetting(HOVER_DELAY_KEY, String(val));
        setConfiguredHoverDelay(val);
        setStatus(`Saved: ${val} ms`, 'info');
      },
      (msg) => setStatus(`✗ ${msg}`, 'error')
    );
  });

  body.appendChild(row);
  return section;
}

/* ── Keyboard shortcuts (collapsible, full width) ──────────────────────────── */

function _buildHotkeySection() {
  const { section, bar, body, heading } = _panel('Keyboard Shortcuts', { extraClass: 'settings-panel--full' });
  bar.classList.add('settings-section-header--collapsible');

  // The chevron and the title move into one wrapper so the whole heading — not
  // just the word — is the toggle target.
  const toggleRow = document.createElement('div');
  toggleRow.className = 'settings-collapse-toggle';

  const collapseToggle = document.createElement('span');
  collapseToggle.className = 'settings-collapse-chevron';
  collapseToggle.textContent = '▶';

  bar.insertBefore(toggleRow, heading);
  toggleRow.append(collapseToggle, heading);

  const resetBtn = _rowButton('Reset to defaults', { secondary: true });
  resetBtn.onclick = (e) => {
    e.stopPropagation();
    _hotkeyHandle?.reset();
  };

  bar.appendChild(resetBtn);

  // The panel body carries the collapse (inline display:none); the inner div is
  // what renderHotkeyPanel fills. Keeping them separate means the panel's own
  // `display: flex` never has to fight the collapse.
  body.style.display = 'none';

  const hotkeyInner = document.createElement('div');
  hotkeyInner.className = 'settings-hotkey-body';
  body.appendChild(hotkeyInner);

  let hotkeyLoaded = false;

  async function toggleHotkeys() {
    const isCollapsed = body.style.display === 'none';
    body.style.display = isCollapsed ? '' : 'none';
    collapseToggle.textContent = isCollapsed ? '▼' : '▶';
    if (isCollapsed && !hotkeyLoaded) {
      hotkeyLoaded = true;
      _hotkeyHandle = await renderHotkeyPanel(hotkeyInner);
    }
  }

  toggleRow.addEventListener('click', toggleHotkeys);

  return section;
}
