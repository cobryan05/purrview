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

  // The settings panels load their values concurrently — each awaits its
  // own settings reads, so building them in sequence would serialise round-trips
  // that have nothing to do with each other.
  const [serverPanel, mqttPanel, retentionPanel, timelapsePanel, dedupPanel, uiPanel] = await Promise.all([
    _buildServerSection(),
    _buildMqttSection(),
    _buildRetentionSection(),
    _buildTimelapseRetentionSection(),
    _buildDedupSection(),
    _buildUiSettingsSection(),
  ]);

  wrap.append(
    serverPanel, mqttPanel, retentionPanel, timelapsePanel, dedupPanel, uiPanel, _buildHotkeySection(),
  );
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
  prefix: 'whiskerwatcher',
  username: '',
  password: '',
  tls: false,
  client_id: 'whiskerwatcher',
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
