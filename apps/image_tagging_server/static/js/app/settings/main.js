import { renderHotkeyPanel } from '/app-static/js/app/image-tagging/hotkeyModal.js';
import { setupWorkspaceTabs } from '/app-static/js/app/utils/tabs.js';
import { apiFetch } from '/app-static/js/shared/api/apiUtils.js';
import {
  getSetting,
  setSetting,
  MODEL_VERSIONS_KEEP_SETTING_KEY,
  TRAINING_ARTIFACTS_KEEP_SETTING_KEY,
} from '/app-static/js/shared/api/settings.js';
import {
  clearThumbnailCache,
  fetchStorageUsage,
  purgeTimelapseFrames,
  runStorageSweep,
} from '/app-static/js/shared/api/storage.js';
import { createServiceStatusPanel } from '/app-static/js/ui/serviceStatus.js';
import { showConfirmModal } from '/app-static/js/ui/utils/confirmModal.js';
import { promptRestartWatchersUsingMqtt } from '/app-static/js/ui/utils/restartAffectedTasks.js';
import { setConfiguredHoverDelay } from '/app-static/js/ui/thumbnailHoverPreview.js';
import { readTourState, resetTourState, runTour, setNeverShowTours, TOURS } from '/app-static/js/ui/tour/index.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import {
  DEFAULT_BBOX_HASH_SIZE,
  DEFAULT_BBOX_THRESHOLD,
  DEFAULT_IMAGE_HASH_SIZE,
  DEFAULT_IMAGE_THRESHOLD,
  defaultThresholdFor,
  hashMaxFor,
} from '/app-static/js/shared/dedupDefaults.js';

let _initialized = false;

/**
 * Settings sections, in rail order and grouped the way Configuration's rail
 * groups its own destinations (a heading per group, an ungrouped leading item).
 * Each `build` returns (or resolves to) the panel element that goes inside its
 * pane — sync where a section has nothing to await, async otherwise.
 */
function _settingsRailGroups() {
  return [
    {
      items: [
        { tab: 'tab-settings-status', label: 'Service Status', build: () => _buildServiceStatusSection() },
      ],
    },
    {
      heading: 'Connections',
      items: [
        { tab: 'tab-settings-servers', label: 'Servers', build: () => _buildServerSection() },
        { tab: 'tab-settings-mqtt', label: 'MQTT Broker', build: () => _buildMqttSection() },
      ],
    },
    {
      heading: 'Storage',
      items: [
        { tab: 'tab-settings-retention', label: 'Recording Retention', build: () => _buildRetentionSection() },
        { tab: 'tab-settings-timelapse', label: 'Timelapse Retention', build: () => _buildTimelapseRetentionSection() },
        { tab: 'tab-settings-models', label: 'Model History', build: () => _buildModelHistorySection() },
        // Not awaited by its own tab either: the usage figures come from a directory
        // walk, and the pane renders its own placeholder and fills in when it lands.
        { tab: 'tab-settings-storage', label: 'Storage Usage', build: () => _buildStorageSection() },
      ],
    },
    {
      heading: 'System',
      items: [
        { tab: 'tab-settings-security', label: 'Security', build: () => _buildSecuritySection() },
        { tab: 'tab-settings-dedup', label: 'Deduplication', build: () => _buildDedupSection() },
        { tab: 'tab-settings-interface', label: 'Interface', build: () => _buildUiSettingsSection() },
        { tab: 'tab-settings-hotkeys', label: 'Keyboard Shortcuts', build: () => _buildHotkeySection() },
      ],
    },
  ];
}

/**
 * Initialise the Settings tab. Safe to call multiple times (no-op after first call).
 *
 * Mirrors Configuration's left-rail layout (`.config-rail-layout`): a nav of
 * destinations grouped by purpose, and one pane visible at a time, rather than
 * every panel stacked into one long scrolling grid. Built in JS, not the
 * template, because this tab always has been — Configuration's rail markup
 * lives in index.html only because its panes' *contents* also mostly do.
 */
export async function init() {
  if (_initialized) return;
  _initialized = true;

  const container = document.getElementById('tab-settings');
  if (!container) return;

  container.innerHTML = '';

  const layout = document.createElement('div');
  layout.className = 'config-rail-layout';

  const nav = document.createElement('nav');
  nav.className = 'config-rail';
  nav.setAttribute('aria-label', 'Settings sections');

  const body = document.createElement('div');
  body.className = 'config-rail-body';

  const groups = _settingsRailGroups();
  const panes = new Map();

  for (const group of groups) {
    const groupEl = document.createElement('div');
    groupEl.className = 'config-rail__group';

    if (group.heading) {
      const heading = document.createElement('span');
      heading.className = 'config-rail__heading';
      heading.textContent = group.heading;
      groupEl.appendChild(heading);
    }

    for (const item of group.items) {
      const button = document.createElement('span');
      button.className = 'workspace-tab-button config-rail__item';
      button.dataset.tab = item.tab;
      // Tour anchor, keyed on the screen so every Settings destination is
      // addressable without each one opting in (`settings.hotkeys`, …).
      button.dataset.tour = `settings.${item.tab.replace('tab-settings-', '')}`;
      button.textContent = item.label;
      groupEl.appendChild(button);

      const pane = document.createElement('div');
      pane.id = item.tab;
      pane.className = 'workspace-tab-pane';
      body.appendChild(pane);

      panes.set(item.tab, { pane, build: item.build });
    }

    nav.appendChild(groupEl);
  }

  layout.append(nav, body);
  container.appendChild(layout);

  // Wired before any pane content loads, same as Configuration's rail — the
  // buttons must be clickable the instant they're on screen. Waiting for
  // Promise.all below to attach listeners left a window where the rail looked
  // ready but a click silently did nothing, because no listener existed yet.
  setupWorkspaceTabs('#tab-settings', () => {});

  // Every pane's content loads concurrently — each section awaits its own
  // settings reads, so building them in sequence would serialise round-trips
  // that have nothing to do with each other. All panes are built up front
  // (not lazily on first activation) so switching sections is instant and
  // every field is reachable without waiting on a fetch mid-click.
  await Promise.all([...panes.values()].map(async ({ pane, build }) => {
    pane.appendChild(await build());
  }));
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
        'another host with its own models directory — newly trained models are pushed here over HTTP as soon as ' +
        'training finishes, so no shared filesystem is required.',
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

/* ── Recording retention ───────────────────────────────────────────────────── */

/**
 * How many days of event clips and card art to keep.
 * Mirrors `RECORDING_RETENTION_SETTING_KEY` in `apps/db_server/manager.py`.
 */
export const RETENTION_SETTING_KEY = 'recording_retention_days';

/**
 * Ceiling on total recording bytes, in MB.
 * Mirrors `RECORDING_MAX_TOTAL_MB_SETTING_KEY` in `apps/db_server/manager.py`.
 */
export const RETENTION_MAX_MB_SETTING_KEY = 'recording_max_total_mb';

/** Mirror `DEFAULT_RECORDING_*` in `apps/db_server/manager.py`. */
const RETENTION_DEFAULT_DAYS = 30;
const RETENTION_DEFAULT_MAX_MB = 10240;

/**
 * Read one numeric setting, falling back when it is absent or unreadable.
 *
 * `Number('')`, `Number(null)` and `Number(false)` are all a finite 0, which here would
 * read as the deliberate "no bound" value rather than as "unset" — so emptiness is
 * checked before the parse is trusted, not after.
 */
async function _numericSetting(key, fallback) {
  let stored = null;
  try {
    stored = await getSetting(key);
  } catch (_) {}
  const parsed = stored != null && String(stored).trim() !== '' ? Number(stored) : NaN;
  return Number.isFinite(parsed) ? Math.trunc(parsed) : fallback;
}

/**
 * One "bound" row: a number input, a Save button, and a live description.
 *
 * @param {{label:string, description:string, testid:string, settingKey:string,
 *          value:number, describe:(n:number)=>[string,string], unit:string}} opts
 */
function _boundRow({ label, description, testid, settingKey, value, describe, unit }) {
  const input = document.createElement('input');
  input.type = 'number';
  input.min = '0';
  input.className = 'settings-server-input settings-server-input--narrow';
  input.dataset.testid = testid;
  input.value = String(value);

  const saveBtn = _rowButton('Save');
  saveBtn.dataset.testid = `${testid}-save`;

  const { row, setStatus } = _buildRow({ label, description, controls: [input, saveBtn] });

  const show = (n) => {
    const [text, tone] = describe(n);
    setStatus(text, tone);
  };
  show(value);

  saveBtn.addEventListener('click', async () => {
    const parsed = Number(input.value);
    if (input.value.trim() === '' || !Number.isFinite(parsed) || parsed < 0) {
      setStatus(`Enter a whole number of ${unit}, or 0 for no limit.`, 'error');
      return;
    }
    const next = Math.trunc(parsed);
    await _withSaveFeedback(
      saveBtn,
      async () => {
        await setSetting(settingKey, String(next));
        input.value = String(next);
        show(next);
      },
      (msg) => setStatus(`✗ ${msg}`, 'error')
    );
  });

  return row;
}

async function _buildRetentionSection() {
  const { section, body } = _panel('Recording Retention', { id: 'settings-section-retention' });

  const [days, maxMb] = await Promise.all([
    _numericSetting(RETENTION_SETTING_KEY, RETENTION_DEFAULT_DAYS),
    _numericSetting(RETENTION_MAX_MB_SETTING_KEY, RETENTION_DEFAULT_MAX_MB),
  ]);

  const note = document.createElement('p');
  note.className = 'settings-dedup-warning';
  note.textContent =
    'Event clips and the snapshots shown on event cards are removed once they pass either bound — '
    + 'whichever is reached first. Both are independent, so switching one off with 0 leaves the other '
    + 'working. Files nothing points at any more are cleaned up on the same pass. Frames you saved for '
    + 'annotation are never touched: those live under the image datasets and are yours to curate. The '
    + 'sweep runs about once an hour and measures age from when the event started, not from when the '
    + 'clip was written.';
  body.appendChild(note);

  body.appendChild(_boundRow({
    label: 'Keep recordings for',
    description: 'Days. Set to 0 to apply no age limit.',
    testid: 'settings-retention-days',
    settingKey: RETENTION_SETTING_KEY,
    value: days,
    unit: 'days',
    describe: (n) => (n > 0
      ? [`Deleting events older than ${n} day${n === 1 ? '' : 's'}.`, 'info']
      : ['No age limit — events are kept until the size limit evicts them.', 'muted']),
  }));

  body.appendChild(_boundRow({
    label: 'Use at most',
    description: 'MB of disk for clips and snapshots. Set to 0 to apply no size limit.',
    testid: 'settings-retention-max-mb',
    settingKey: RETENTION_MAX_MB_SETTING_KEY,
    value: maxMb,
    unit: 'MB',
    describe: (n) => (n > 0
      ? [`Evicting the oldest events once recordings exceed ${n} MB.`, 'info']
      : ['No size limit — disk use grows until the age limit removes something.', 'muted']),
  }));

  const caveat = document.createElement('p');
  caveat.className = 'settings-dedup-warning';
  caveat.dataset.testid = 'settings-retention-caveat';
  caveat.textContent =
    'Only the default clip and snapshot folders are managed. A "Record clip" action pointed at a '
    + 'custom output folder writes outside them, so those files are neither counted toward the size '
    + 'limit nor deleted.';
  body.appendChild(caveat);

  return section;
}

/** Mirror `DEFAULT_VERSIONS_KEEP` / `DEFAULT_TRAINING_ARTIFACTS_KEEP` in Python. */
const MODEL_VERSIONS_KEEP_DEFAULT = 3;
const TRAINING_ARTIFACTS_KEEP_DEFAULT = 3;

/**
 * The two bounds on what training leaves behind.
 *
 * They are separate numbers because they buy different things. Model versions are
 * the rollback path — a retrain that made the model worse is undone by restoring
 * one, from a model's History button. Artifact folders are the training server's
 * own working output (weights plus every checkpoint of the run, around a gigabyte
 * for a detector); they are what a restart or a re-download reads, and they used
 * to accumulate with no bound at all.
 */
async function _buildModelHistorySection() {
  const { section, body } = _panel('Model History', { id: 'settings-section-model-history' });

  const [versionsKeep, artifactsKeep] = await Promise.all([
    _numericSetting(MODEL_VERSIONS_KEEP_SETTING_KEY, MODEL_VERSIONS_KEEP_DEFAULT),
    _numericSetting(TRAINING_ARTIFACTS_KEEP_SETTING_KEY, TRAINING_ARTIFACTS_KEEP_DEFAULT),
  ]);

  const note = document.createElement('p');
  note.className = 'settings-dedup-warning';
  note.textContent =
    'Every training run archives the weights it replaces, so a retrain that made a model worse can be '
    + 'rolled back from that model\'s History button in Configuration → Models. Versions are full copies: '
    + 'each one costs about what the model itself costs, and the oldest is deleted once the limit is passed. '
    + 'Deleting a model deletes its history too.';
  body.appendChild(note);

  body.appendChild(_boundRow({
    label: 'Keep previous model versions',
    description: 'Per model. Set to 0 to keep no history — a retrain then overwrites with no way back.',
    testid: 'settings-model-versions-keep',
    settingKey: MODEL_VERSIONS_KEEP_SETTING_KEY,
    value: versionsKeep,
    unit: 'versions',
    describe: (n) => (n > 0
      ? [`Keeping the ${n} generation${n === 1 ? '' : 's'} before the current one, per model.`, 'info']
      : ['No history — each retrain replaces the model with no rollback.', 'muted']),
  }));

  body.appendChild(_boundRow({
    label: 'Keep training job folders',
    description: 'On the training server. Each holds one finished run\'s weights and checkpoints.',
    testid: 'settings-training-artifacts-keep',
    settingKey: TRAINING_ARTIFACTS_KEEP_SETTING_KEY,
    value: artifactsKeep,
    unit: 'folders',
    describe: (n) => (n > 0
      ? [`Keeping the ${n} most recent finished run${n === 1 ? '' : 's'}; older folders are removed `
        + 'after the next run completes.', 'info']
      : ['Removing every finished run\'s folder after the next run completes.', 'muted']),
  }));

  const caveat = document.createElement('p');
  caveat.className = 'settings-dedup-warning';
  caveat.dataset.testid = 'settings-model-history-caveat';
  caveat.textContent =
    'A job folder is only removed once a later run has finished and its weights are safely saved. '
    + 'Queued, running and interrupted jobs are never touched — an interrupted run can still be '
    + 'restarted from the checkpoint in its folder.';
  body.appendChild(caveat);

  return section;
}

/**
 * Timelapse bounds, mirroring `TIMELAPSE_*` in `apps/db_server/manager.py`.
 *
 * A separate panel from Recording Retention rather than two more rows inside it,
 * because the numbers are not comparable: a clip is kept because something happened and
 * a month of them is a modest folder, while a timelapse keeps shooting when nothing
 * happens and a month at 30-second intervals is hundreds of GB per camera. Sharing one
 * pair of bounds would mean whichever material is bulkier silently evicts the other.
 */
export const TIMELAPSE_RETENTION_SETTING_KEY = 'timelapse_retention_days';
export const TIMELAPSE_MAX_MB_SETTING_KEY = 'timelapse_max_total_mb';

/** Mirror `DEFAULT_TIMELAPSE_*` in `apps/db_server/manager.py`. */
const TIMELAPSE_DEFAULT_DAYS = 14;
const TIMELAPSE_DEFAULT_MAX_MB = 20480;

async function _buildTimelapseRetentionSection() {
  const { section, body } = _panel('Timelapse Retention', { id: 'settings-section-timelapse-retention' });

  const [days, maxMb] = await Promise.all([
    _numericSetting(TIMELAPSE_RETENTION_SETTING_KEY, TIMELAPSE_DEFAULT_DAYS),
    _numericSetting(TIMELAPSE_MAX_MB_SETTING_KEY, TIMELAPSE_DEFAULT_MAX_MB),
  ]);

  const note = document.createElement('p');
  note.className = 'settings-dedup-warning';
  note.textContent =
    'Frames written by a "Save timelapse" action are removed once they pass either bound — whichever '
    + 'is reached first, and either can be switched off with 0 without disabling the other. A camera '
    + 'shooting every 30 seconds writes about 2,900 frames a day, so the size limit is usually the one '
    + 'that binds. Age is measured from the day folder the frame was filmed in, not from when the file '
    + 'was last touched, and the sweep runs about once an hour.';
  body.appendChild(note);

  body.appendChild(_boundRow({
    label: 'Keep timelapse frames for',
    description: 'Days. Set to 0 to apply no age limit.',
    testid: 'settings-timelapse-days',
    settingKey: TIMELAPSE_RETENTION_SETTING_KEY,
    value: days,
    unit: 'days',
    describe: (n) => (n > 0
      ? [`Deleting frames filmed more than ${n} day${n === 1 ? '' : 's'} ago.`, 'info']
      : ['No age limit — frames are kept until the size limit evicts them.', 'muted']),
  }));

  body.appendChild(_boundRow({
    label: 'Use at most',
    description: 'MB of disk for timelapse frames. Set to 0 to apply no size limit.',
    testid: 'settings-timelapse-max-mb',
    settingKey: TIMELAPSE_MAX_MB_SETTING_KEY,
    value: maxMb,
    unit: 'MB',
    describe: (n) => (n > 0
      ? [`Evicting the oldest frames once timelapses exceed ${n} MB.`, 'info']
      : ['No size limit — disk use grows until the age limit removes something.', 'muted']),
  }));

  const caveat = document.createElement('p');
  caveat.className = 'settings-dedup-warning';
  caveat.dataset.testid = 'settings-timelapse-caveat';
  caveat.textContent =
    'Only the output folders your "Save timelapse" actions name are swept, and only files that match '
    + 'what the action writes — a day folder named YYYY-MM-DD holding <camera>_HH-MM-SS.jpg frames. '
    + 'Anything else in those folders is left alone, including any image with a JSON sidecar, so a '
    + 'folder you also annotate in is safe. Deleting the action itself stops its old frames being '
    + 'managed, because nothing left names the folder — clear them out before removing it.';
  body.appendChild(caveat);

  return section;
}

/* ── Storage usage ──────────────────────────────────────────────────────────
 *
 * The two panels above set the bounds; this one says where you actually are
 * against them. Without it the settings were unanswerable in the only way that
 * matters — "is 20 GB the right cap?" cannot be judged without knowing that the
 * timelapses are already at 9.9 GB and the clips at 2.4.
 *
 * It sits directly under the retention panels rather than in Operations → service
 * status, where a first instinct might put it. Service status is monitoring: it
 * answers "is everything up", refreshes on a timer, and must stay cheap. These
 * numbers cost a walk of three directory trees, and the decision they inform — how
 * long to keep footage — is made right here, next to the inputs that set it.
 */

/** Human-readable bytes. Binary units, matching what the server logs. */
function _fmtBytes(n) {
  if (!Number.isFinite(n) || n <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return `${v >= 100 || i === 0 ? Math.round(v) : v.toFixed(1)} ${units[i]}`;
}

/**
 * One bucket's line: what it holds, and how close that is to its cap.
 *
 * The percentage is only shown when there *is* a cap. A bucket with the size
 * limit switched off has no "% full" — printing one against an invented ceiling
 * would be the most misleading thing on the panel.
 */
function _storageRow({ label, bucket, note }) {
  const row = document.createElement('div');
  row.className = 'settings-storage-row';

  const head = document.createElement('div');
  head.className = 'settings-storage-head';

  const name = document.createElement('span');
  name.className = 'settings-storage-label';
  name.textContent = label;

  const value = document.createElement('span');
  value.className = 'settings-storage-value';
  const max = bucket?.maxBytes ?? 0;
  const used = bucket?.bytes ?? 0;
  const pct = max > 0 ? Math.min(100, Math.round((used / max) * 100)) : null;
  value.textContent = max > 0
    ? `${_fmtBytes(used)} of ${_fmtBytes(max)} (${pct}%)`
    : `${_fmtBytes(used)} — no size limit set`;

  head.append(name, value);
  row.appendChild(head);

  if (max > 0) {
    const meter = document.createElement('div');
    meter.className = 'settings-storage-meter';
    const fill = document.createElement('div');
    fill.className = 'settings-storage-meter__fill';
    fill.style.width = `${pct}%`;
    // Over 90% the next thing that happens is eviction, which is a normal, working
    // outcome — but it is the moment where "why did my old clips vanish?" gets asked,
    // so the bar says it is about to happen rather than leaving it to be discovered.
    if (pct >= 90) fill.dataset.state = 'full';
    meter.appendChild(fill);
    row.appendChild(meter);
  }

  const detail = document.createElement('div');
  detail.className = 'settings-storage-detail';
  const parts = [`${(bucket?.files ?? 0).toLocaleString()} file(s)`];
  if (bucket?.retentionDays) parts.push(`kept ${bucket.retentionDays} day(s)`);
  else if (bucket?.retentionDays === 0) parts.push('no age limit');
  if (note) parts.push(note);
  detail.textContent = parts.join(' · ');
  row.appendChild(detail);

  return row;
}

function _buildStorageSection() {
  const { section, bar, body } = _panel('Storage', { id: 'settings-section-storage' });

  const refreshBtn = _rowButton('↻ Refresh', { secondary: true });
  refreshBtn.dataset.testid = 'settings-storage-refresh';
  bar.appendChild(refreshBtn);

  const note = document.createElement('p');
  note.className = 'settings-dedup-warning';
  note.textContent =
    'Measured the same way the sweeps that delete this material measure it, so these '
    + 'numbers and the limits above always agree about what counts. Annotated images are '
    + 'not included: they are your dataset, not managed storage, and nothing here can '
    + 'reach them.';
  body.appendChild(note);

  const rows = document.createElement('div');
  rows.className = 'settings-storage-rows';
  rows.dataset.testid = 'settings-storage-rows';
  rows.textContent = 'Measuring…';
  body.appendChild(rows);

  const actions = document.createElement('div');
  actions.className = 'settings-storage-actions';

  const status = document.createElement('div');
  status.className = 'settings-server-status';
  status.dataset.state = 'muted';
  status.dataset.testid = 'settings-storage-status';

  /**
   * Wire one reclaim button: confirm if it asks to, run it, say what it freed,
   * then re-measure. Re-measuring is not optional — the whole panel's credibility
   * is that the number under the button reflects the button having been pressed.
   */
  function _action({ label, testid, title, confirm, run }) {
    const btn = _rowButton(label, { secondary: true });
    btn.dataset.testid = testid;
    btn.title = title;
    btn.addEventListener('click', async () => {
      if (confirm && !window.confirm(confirm)) return;
      const original = btn.textContent;
      btn.disabled = true;
      btn.textContent = 'Working…';
      status.dataset.state = 'muted';
      status.textContent = '';
      try {
        const result = await run();
        const bits = [];
        if (result.events) bits.push(`${result.events} event(s)`);
        if (result.recordings) bits.push(`${result.recordings} clip(s)`);
        if (result.files) bits.push(`${result.files} file(s)`);
        status.textContent = bits.length
          ? `Removed ${bits.join(', ')}, freeing ${_fmtBytes(result.bytes)}`
          : 'Nothing to remove — everything is inside its limits.';
        status.dataset.state = bits.length ? 'ok' : 'muted';
        await refresh();
      } catch (e) {
        status.textContent = `✗ ${e.message}`;
        status.dataset.state = 'error';
      } finally {
        btn.disabled = false;
        btn.textContent = original;
      }
    });
    return btn;
  }

  actions.append(
    _action({
      label: 'Clean up now',
      testid: 'settings-storage-sweep',
      title: 'Apply the age and size limits above immediately, instead of waiting for the hourly sweep',
      run: runStorageSweep,
    }),
    _action({
      label: 'Delete all timelapse frames',
      testid: 'settings-storage-purge-timelapse',
      title: 'Delete every timelapse frame regardless of the limits above',
      confirm:
        'Delete every timelapse frame?\n\n'
        + 'This ignores the retention limits and removes the lot. Only files a "Save timelapse" '
        + 'action wrote are affected — anything with a JSON sidecar beside it is left alone, so '
        + 'images you have annotated are safe. This cannot be undone.',
      run: purgeTimelapseFrames,
    }),
    _action({
      label: 'Clear thumbnail cache',
      testid: 'settings-storage-clear-thumbs',
      title: 'Delete cached thumbnails; they are rebuilt from the originals on next view',
      run: clearThumbnailCache,
    }),
  );

  body.append(actions, status);

  const caveat = document.createElement('p');
  caveat.className = 'settings-dedup-warning';
  caveat.textContent =
    'There is deliberately no "delete all recordings" here. An event is the record that '
    + 'something happened, and losing the lot to one click is not recoverable — set the '
    + 'limits above to shed history, or delete individual events from Watch.';
  body.appendChild(caveat);

  async function refresh() {
    refreshBtn.disabled = true;
    try {
      const usage = await fetchStorageUsage();
      rows.innerHTML = '';
      rows.append(
        _storageRow({
          label: 'Event clips & snapshots',
          bucket: usage.recordings,
          note: 'bounded by Recording Retention',
        }),
        _storageRow({
          label: 'Timelapse frames',
          bucket: usage.timelapse,
          note: 'bounded by Timelapse Retention',
        }),
        _storageRow({
          label: 'Thumbnail cache',
          bucket: usage.thumbnailCache,
          note: 'rebuilt on demand — safe to clear',
        }),
      );
    } catch (e) {
      rows.textContent = `Could not measure storage: ${e.message}`;
    } finally {
      refreshBtn.disabled = false;
    }
  }

  refreshBtn.addEventListener('click', () => { void refresh(); });
  void refresh();

  return section;
}

/* ── MQTT broker ───────────────────────────────────────────────────────────── */

/**
 * The settings key the watcher reads its broker connection from.
 * Mirrors `_load_mqtt_settings` in `apps/tasks_server/tasks/StreamWatcherTask.py`.
 */
export const MQTT_SETTING_KEY = 'mqtt_broker';

/** Defaults mirror `MqttSettings` in `apps/helpers/actions/mqttPublisher.py`. */
const MQTT_DEFAULTS = {
  host: '',
  port: 1883,
  prefix: 'purrview',
  username: '',
  password: '',
  tls: false,
  client_id: 'purrview',
};

/**
 * Read the stored broker config. The settings table holds strings, so this key is
 * JSON — a value that fails to parse is treated as unset rather than thrown, so a
 * hand-edited row can't make the whole Settings tab fail to render.
 * @returns {Promise<Record<string, any>>}
 */
async function _loadMqttConfig() {
  let stored = null;
  try {
    stored = await getSetting(MQTT_SETTING_KEY);
  } catch (_) {}
  if (!stored) return { ...MQTT_DEFAULTS };
  try {
    const parsed = JSON.parse(stored);
    if (parsed && typeof parsed === 'object') return { ...MQTT_DEFAULTS, ...parsed };
  } catch (_) {}
  return { ...MQTT_DEFAULTS };
}

/** One labelled input inside the MQTT panel. */
function _mqttField({ label, description, testid, type = 'text', value, placeholder, narrow = false }) {
  const input = document.createElement('input');
  input.type = type;
  input.className = narrow
    ? 'settings-server-input settings-server-input--narrow'
    : 'settings-server-input';
  input.dataset.testid = testid;
  if (placeholder) input.placeholder = placeholder;
  if (type === 'checkbox') {
    input.className = '';
    input.checked = Boolean(value);
  } else {
    input.value = value == null ? '' : String(value);
  }
  const { row } = _buildRow({ label, description, controls: [input] });
  return { row, input };
}

async function _buildMqttSection() {
  const { section, body } = _panel('MQTT Broker', { id: 'settings-section-mqtt' });

  const config = await _loadMqttConfig();

  const note = document.createElement('p');
  note.className = 'settings-dedup-warning';
  note.textContent =
    'One broker per deployment, shared by every watcher: the "Publish MQTT" and "Publish MQTT image" actions ' +
    '(Configuration → Watchers) connect here rather than carrying their own connection details. A watcher reads ' +
    'these settings when it starts, so restart any running watcher after changing them. The password is stored ' +
    'unencrypted in the settings database.';
  body.appendChild(note);

  const host = _mqttField({
    label: 'Broker host',
    description: 'Hostname or IP of the MQTT broker. Leave blank to disable MQTT publishing entirely.',
    testid: 'settings-mqtt-host',
    value: config.host,
    placeholder: 'e.g. 192.168.1.10 or mqtt.local',
  });
  const port = _mqttField({
    label: 'Port',
    description: 'Default: 1883 (8883 is the usual TLS port).',
    testid: 'settings-mqtt-port',
    type: 'number',
    value: config.port,
    narrow: true,
  });
  const prefix = _mqttField({
    label: 'Topic prefix',
    description:
      'Prepended to every topic, including the online/offline state topic (<prefix>/state). Each action\'s ' +
      'topic template is appended to this.',
    testid: 'settings-mqtt-prefix',
    value: config.prefix,
    placeholder: MQTT_DEFAULTS.prefix,
  });
  const username = _mqttField({
    label: 'Username',
    description: 'Leave blank for an anonymous broker.',
    testid: 'settings-mqtt-username',
    value: config.username,
  });
  const password = _mqttField({
    label: 'Password',
    testid: 'settings-mqtt-password',
    type: 'password',
    value: config.password,
  });
  const tls = _mqttField({
    label: 'Use TLS',
    description: 'Connect with TLS using the system CA bundle.',
    testid: 'settings-mqtt-tls',
    type: 'checkbox',
    value: config.tls,
  });
  const clientId = _mqttField({
    label: 'Client ID',
    description:
      'Identifies this deployment to the broker. Brokers disconnect an existing session when a second client ' +
      'connects with the same id, so make it unique per deployment.',
    testid: 'settings-mqtt-client-id',
    value: config.client_id,
    placeholder: MQTT_DEFAULTS.client_id,
  });

  const testBtn = _rowButton('Test', { secondary: true });
  testBtn.dataset.testid = 'settings-mqtt-test';
  const saveBtn = _rowButton('Save');
  saveBtn.dataset.testid = 'settings-mqtt-save';

  const { row: actionRow, setStatus } = _buildRow({
    label: 'Connection',
    controls: [testBtn, saveBtn],
  });
  actionRow.dataset.testid = 'settings-mqtt-actions';

  setStatus(
    config.host ? `Configured: ${config.host}:${config.port || MQTT_DEFAULTS.port}` : 'No broker configured.',
    config.host ? 'info' : 'muted'
  );

  /** Current form values, normalised the same way for Test and for Save. */
  const _values = () => ({
    host: host.input.value.trim(),
    port: parseInt(port.input.value, 10) || MQTT_DEFAULTS.port,
    prefix: prefix.input.value.trim() || MQTT_DEFAULTS.prefix,
    username: username.input.value.trim(),
    password: password.input.value,
    tls: tls.input.checked,
    client_id: clientId.input.value.trim() || MQTT_DEFAULTS.client_id,
  });

  testBtn.addEventListener('click', async () => {
    const values = _values();
    if (!values.host) {
      setStatus('Enter a broker host first.', 'error');
      return;
    }
    testBtn.disabled = true;
    testBtn.textContent = 'Testing…';
    setStatus(`Testing ${values.host}:${values.port}…`, 'muted');
    try {
      // Tests what is in the form, not what is saved — otherwise the button can only
      // confirm settings you have already committed, which is the wrong order.
      const res = await apiFetch('/api/test-mqtt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          host: values.host,
          port: values.port,
          username: values.username || null,
          password: values.password || null,
          tls: values.tls,
        }),
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
    const values = _values();
    const portVal = parseInt(port.input.value, 10);
    if (port.input.value.trim() !== '' && (!Number.isFinite(portVal) || portVal < 1 || portVal > 65535)) {
      setStatus('Port must be between 1 and 65535.', 'error');
      return;
    }
    await _withSaveFeedback(
      saveBtn,
      async () => {
        // A blank host means "no broker": store an empty string so the watcher's
        // `if not value` check disables MQTT, rather than a JSON object that would
        // parse into a connection attempt against localhost.
        await setSetting(MQTT_SETTING_KEY, values.host ? JSON.stringify(values) : '');
        setStatus(
          values.host
            ? `Saved: ${values.host}:${values.port} — restart running watchers to apply.`
            : 'Saved: MQTT publishing disabled.',
          values.host ? 'info' : 'muted'
        );
        // A watcher opens its broker connection once, in _build_actions, so both
        // directions need the restart: a new host, and a cleared one that should
        // stop publishing. No-op when no live watcher publishes over MQTT.
        await promptRestartWatchersUsingMqtt();
      },
      (msg) => setStatus(`✗ ${msg}`, 'error')
    );
  });

  body.append(
    host.row,
    port.row,
    prefix.row,
    username.row,
    password.row,
    tls.row,
    clientId.row,
    actionRow
  );
  return section;
}

/* ── Security ─────────────────────────────────────────────────────────────── */

/**
 * Two credentials, one gate. The API token authenticates machines — Home Assistant,
 * and `<img>`/`<video>` URLs that cannot attach a header — as `Authorization: Bearer`
 * or `?token=`. The optional login password authenticates a person at `/login`, and
 * is the only credential that form accepts once it is on, so a token pasted into a
 * Home Assistant config entry never doubles as a browser session.
 *
 * This panel does not write the `auth_config` setting itself: it posts to
 * `/api/auth-config`, because the password has to be hashed server-side and is never
 * stored or read back in the clear. See `apps/helpers/authSettings.py`.
 */
const AUTH_CONFIG_ENDPOINT = '/api/auth-config';

const SECURITY_DEFAULTS = {
  requireAuth: false,
  token: '',
  passwordEnabled: false,
  passwordSet: false,
};

async function _loadSecurityConfig() {
  try {
    const res = await apiFetch(AUTH_CONFIG_ENDPOINT);
    const data = await res.json();
    if (data && data.status === 'success') return { ...SECURITY_DEFAULTS, ...data };
  } catch (_) {}
  return { ...SECURITY_DEFAULTS };
}

/**
 * 256 bits from the platform CSPRNG, base64url-encoded. Plenty for a token that's
 * typed by hand exactly once, into Home Assistant's config flow.
 */
function _generateToken() {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  let binary = '';
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function _securityStatusText(config) {
  if (!config.requireAuth) return 'Open — no credentials required.';
  return config.passwordEnabled
    ? 'Token required for the API; password required to sign in.'
    : 'Token required — it also signs you in.';
}

async function _buildSecuritySection() {
  const { section, body } = _panel('Security', { id: 'settings-section-security' });

  const config = await _loadSecurityConfig();

  const warning = document.createElement('p');
  warning.className = 'settings-dedup-warning';
  warning.textContent =
    'Off by default. Once access control is on, the API token is required for every ' +
    'request (as an Authorization: Bearer header, or ?token= for things like <video> ' +
    'tags that cannot send a header) — paste it into the PurrView integration\'s setup ' +
    'form in Home Assistant. The login password is separate and optional: turn it on ' +
    'and the sign-in page stops accepting the token, so the integration\'s copy of it ' +
    'cannot open a browser session. The token is stored unencrypted in the settings ' +
    'database, the same as the MQTT broker password above; the login password is only ' +
    'ever stored as a hash — which means a forgotten one cannot be recovered, only ' +
    'cleared, by running scripts/reset_login_password.py on the host.';
  body.appendChild(warning);

  const requireInput = document.createElement('input');
  requireInput.type = 'checkbox';
  requireInput.checked = Boolean(config.requireAuth);
  requireInput.dataset.testid = 'settings-security-require';
  const { row: requireRow } = _buildRow({
    label: 'Require access control',
    controls: [requireInput],
  });

  const tokenInput = document.createElement('input');
  tokenInput.type = 'password';
  tokenInput.className = 'settings-server-input';
  tokenInput.placeholder = 'Generate one, or paste your own';
  tokenInput.value = config.token;
  tokenInput.dataset.testid = 'settings-security-token';

  const generateBtn = _rowButton('Generate', { secondary: true });
  generateBtn.dataset.testid = 'settings-security-generate';
  generateBtn.addEventListener('click', () => {
    tokenInput.type = 'text';
    tokenInput.value = _generateToken();
  });

  const { row: tokenRow } = _buildRow({
    label: 'API token',
    description: 'For Home Assistant and other API callers.',
    controls: [tokenInput, generateBtn],
  });

  const passwordToggle = document.createElement('input');
  passwordToggle.type = 'checkbox';
  passwordToggle.checked = Boolean(config.passwordEnabled);
  passwordToggle.dataset.testid = 'settings-security-password-enabled';
  const { row: passwordToggleRow } = _buildRow({
    label: 'Require a login password',
    controls: [passwordToggle],
  });

  const passwordInput = document.createElement('input');
  passwordInput.type = 'password';
  passwordInput.className = 'settings-server-input';
  passwordInput.autocomplete = 'new-password';
  // Never prefilled: the server only reports *whether* a password exists, so the
  // field is the one place a plaintext password lives, and only until Save.
  passwordInput.placeholder = config.passwordSet
    ? 'Leave blank to keep the current password'
    : 'Choose a password';
  passwordInput.dataset.testid = 'settings-security-password';

  const { row: passwordRow } = _buildRow({
    label: 'Login password',
    description: 'For signing in to this web interface.',
    controls: [passwordInput],
  });

  const saveBtn = _rowButton('Save');
  saveBtn.dataset.testid = 'settings-security-save';
  const logoutBtn = _rowButton('Log out of this browser', { secondary: true });
  logoutBtn.dataset.testid = 'settings-security-logout';
  logoutBtn.addEventListener('click', () => {
    window.location.href = '/logout';
  });

  const { row: actionRow, setStatus } = _buildRow({
    label: 'Access control',
    controls: [saveBtn, logoutBtn],
  });
  actionRow.dataset.testid = 'settings-security-actions';

  setStatus(_securityStatusText(config), config.requireAuth ? 'info' : 'muted');

  saveBtn.addEventListener('click', async () => {
    const values = {
      requireAuth: requireInput.checked,
      token: tokenInput.value.trim(),
      passwordEnabled: passwordToggle.checked,
      password: passwordInput.value,
    };
    if (values.requireAuth && !values.token) {
      setStatus('Enter or generate an API token before requiring access control.', 'error');
      return;
    }
    if (values.passwordEnabled && !values.password && !config.passwordSet) {
      setStatus('Choose a login password before requiring one.', 'error');
      return;
    }
    await _withSaveFeedback(
      saveBtn,
      async () => {
        const res = await apiFetch(AUTH_CONFIG_ENDPOINT, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(values),
        });
        const data = await res.json();
        if (data.status !== 'success') {
          throw new Error(`Save failed: ${data.message ?? 'unknown error'}`);
        }
        config.requireAuth = data.requireAuth;
        config.passwordEnabled = data.passwordEnabled;
        config.passwordSet = data.passwordSet;
        passwordInput.value = '';
        passwordInput.placeholder = config.passwordSet
          ? 'Leave blank to keep the current password'
          : 'Choose a password';
        setStatus(_securityStatusText(config), config.requireAuth ? 'info' : 'muted');
      },
      (msg) => setStatus(`✗ ${msg}`, 'error')
    );
  });

  body.append(requireRow, tokenRow, passwordToggleRow, passwordRow, actionRow);
  return section;
}

/* ── Deduplication ─────────────────────────────────────────────────────────── */

async function _buildDedupSection() {
  const { section, body } = _panel('Deduplication', { id: 'settings-section-dedup' });

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
  const { section, body } = _panel('Interface', { id: 'settings-section-interface' });

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
  body.appendChild(await _buildToursRow());
  return section;
}

/**
 * Guided tours: whether they offer themselves, and a way to run one on purpose.
 *
 * Running is always available, switched off or not — "don't pop up" and "never
 * let me see it again" are different wishes, and the checkbox only means the
 * first. The reset exists for the case the eligibility check cannot see: someone
 * who skipped the tour on an empty workspace, filled it in, and wants the walk.
 */
async function _buildToursRow() {
  let state = { neverShow: false, completed: [], dismissed: [] };
  try {
    state = await readTourState();
  } catch (_) {}

  const autoLabel = document.createElement('label');
  autoLabel.className = 'settings-inline-check';
  const autoBox = document.createElement('input');
  autoBox.type = 'checkbox';
  autoBox.checked = !state.neverShow;
  autoBox.dataset.testid = 'settings-tours-auto';
  autoLabel.append(autoBox, document.createTextNode(' Offer tours automatically'));

  const runBtn = _rowButton('Run getting started');
  runBtn.dataset.testid = 'settings-tours-run';
  const resetBtn = _rowButton('Forget seen tours', { secondary: true });
  resetBtn.dataset.testid = 'settings-tours-reset';

  const { row, setStatus } = _buildRow({
    label: 'Guided tours',
    description: 'Each tab has a tour, and an overview walks all of them — stop at any tab to go '
      + 'deeper, and it picks up where it left off. The overview offers itself on an empty '
      + 'workspace; the ? in the bottom left opens the tour for the page you are on.',
    controls: [autoLabel, runBtn, resetBtn],
  });

  autoBox.addEventListener('change', async () => {
    try {
      await setNeverShowTours(!autoBox.checked);
      setStatus(autoBox.checked ? 'Tours will offer themselves.' : 'Tours will not pop up on their own.', 'info');
    } catch (e) {
      setStatus(`✗ ${e.message}`, 'error');
    }
  });
  runBtn.addEventListener('click', () => {
    runTour(TOURS[0].id);
  });
  resetBtn.addEventListener('click', async () => {
    try {
      await resetTourState();
      autoBox.checked = true;
      // The help button caches what a resume would open on; forgetting the state
      // moves every tour back to step one, so tell it to re-read.
      events.publish(EventTypes.TOUR_STATE_CHANGED, { running: false, tourId: '' });
      setStatus('Seen tours forgotten — the next eligible one will offer itself on reload.', 'info');
    } catch (e) {
      setStatus(`✗ ${e.message}`, 'error');
    }
  });

  return row;
}

/* ── Service status ───────────────────────────────────────────────────────── */

/**
 * The live service-health strip, hosted in its own rail destination. Builds
 * and refreshes itself; collapsed it is a one-line summary, expanded it is a
 * row per service — that own internal collapse (see serviceStatus.js) is
 * unrelated to which rail item is active.
 */
function _buildServiceStatusSection() {
  return createServiceStatusPanel();
}

/* ── Keyboard shortcuts ───────────────────────────────────────────────────── */

/**
 * No longer collapsible — it used to hide behind a chevron because it sat
 * partway down a long scrolling grid; now it's its own rail destination, so
 * hiding its own content behind a second click would just be redundant.
 */
async function _buildHotkeySection() {
  const { section, bar, body } = _panel('Keyboard Shortcuts', { id: 'settings-section-hotkeys' });

  const resetBtn = _rowButton('Reset to defaults', { secondary: true });
  bar.appendChild(resetBtn);

  const hotkeyInner = document.createElement('div');
  hotkeyInner.className = 'settings-hotkey-body';
  body.appendChild(hotkeyInner);

  const handle = await renderHotkeyPanel(hotkeyInner);
  resetBtn.addEventListener('click', () => handle?.reset());

  return section;
}
