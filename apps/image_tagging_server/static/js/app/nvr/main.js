/**
 * NVR — the surveillance half of the app.
 *
 * Deliberately placeholders for now. The structure exists so the two surfaces
 * have a home and a stable id to grow into; neither invents data it does not
 * have.
 *
 *  - **Live** lists the watch profiles you have defined and whether each is
 *    running. It reads the same task instances Runs shows, over the same shared
 *    SSE stream, so the two views cannot disagree — Runs answers "what is
 *    holding my GPU?", this answers "is the house being watched?".
 *  - **Events** is the media browser from STREAM_WATCHER_PLAN §7.5 (Phase 8),
 *    which is not started. `watch_events` and `recordings` exist with the
 *    foreign keys it needs, so this shows a real (currently empty) count rather
 *    than pretending.
 */
import { setupWorkspaceTabs } from '/app-static/js/app/utils/tabs.js';
import { fetchWatchProfiles } from '/app-static/js/shared/api/watchProfiles.js';
import { fetchTasksStatus } from '/app-static/js/shared/api/tasks.js';
import { navigateTabs } from '/app-static/js/app/utils/tabs.js';
import { Logger, showPaneError } from '/app-static/js/ui/utils/index.js';
import { openWatcherLiveView } from '/app-static/js/app/nvr/liveViewModal.js';

/**
 * Small local button helper.
 *
 * `createButton` in ui/utils takes positional (className, title, text) and has
 * no click handler; `createEmojiButton` hard-codes its own inline styling. This
 * pane wants a plain panel button with an onClick, which is neither.
 *
 * @param {string} text
 * @param {() => void} onClick
 * @returns {HTMLButtonElement}
 */
function _panelButton(text, onClick) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'panel-btn';
  btn.textContent = text;
  btn.addEventListener('click', onClick);
  return btn;
}

/** Task typename that runs a watch profile. */
const WATCHER_TYPENAME = 'StreamWatcherTask';

/**
 * Instance statuses that mean "this watcher is still a live thing".
 *
 * `pending` counts: a watcher that is starting up is on its way to watching, and
 * showing it as stopped would make the pane flicker through a lie on every start.
 */
const LIVE_STATUSES = new Set(['running', 'pending', 'paused']);

/**
 * Map watch-profile uuid → live task info, for profiles that are running.
 *
 * A watcher task carries its profile uuid in the config params rather than in a
 * column, so this reads params rather than joining. A profile with no entry is
 * simply not running.
 *
 * @returns {Promise<Map<string, object>>}
 */
async function fetchRunningWatchers() {
  const running = new Map();
  try {
    const { tasks } = await fetchTasksStatus();
    for (const [taskUuid, task] of Object.entries(tasks ?? {})) {
      if (task?.config?.typename !== WATCHER_TYPENAME) continue;
      // The instance list includes *finished* instances, which are retained after a
      // watcher is stopped. Without this filter a stopped watcher kept its "running"
      // pill for as long as its instance row survived — the single most misleading
      // thing this pane could say.
      if (!LIVE_STATUSES.has(task?.instance?.status)) continue;
      const uuid = task?.config?.params?.watch_profile_uuid;
      // The map key is the instance uuid, and the live view needs it; `instance.uuid`
      // agrees, but reading the key means a payload shape change cannot silently
      // produce an undefined task uuid.
      if (uuid) running.set(uuid, { ...task, taskUuid });
    }
  } catch (err) {
    // A watcher that cannot be resolved is reported as not running rather than
    // blanking the pane — the profile list is still worth showing.
    Logger.warn('nvr: could not read task status', err);
  }
  return running;
}

/**
 * Live — one card per watch profile, showing whether it is currently running.
 */
async function renderLive({ target = 'nvr-live-box' } = {}) {
  const container = document.getElementById(target);
  if (!container) return;
  container.innerHTML = '';

  let profiles;
  try {
    // ⚠️ A Map keyed by uuid, not an array (same convention as fetchLabels —
    // see CLAUDE.md "Key Gotchas"). Iterate with .values().
    profiles = await fetchWatchProfiles();
  } catch (err) {
    Logger.warn('nvr: could not load watch profiles', err);
    showPaneError(container, 'Could not load watch profiles');
    return;
  }

  const running = await fetchRunningWatchers();
  const rows = [...(profiles?.values?.() ?? [])];

  if (!rows.length) {
    container.appendChild(_emptyState(
      'No watch profiles yet',
      'A watch profile is a set of interaction rules — "cat is inside litter box" — plus the sources to watch and the model stack to detect with.',
      { label: 'Define one in Configuration', onClick: () => navigateTabs('tab-configuration', 'tab-watcher-management') },
    ));
    return;
  }

  const list = document.createElement('div');
  list.className = 'nvr-live-list';

  for (const profile of rows) {
    const task = running.get(profile.uuid) ?? null;
    list.appendChild(_profileCard(profile, task));
  }

  container.appendChild(list);
}

/**
 * The analysis rate this profile actually runs at, as display text.
 *
 * `defaultAnalysisFps` is only the fallback for a source that sets no rate of its
 * own — showing it alone reported 5 fps for a watcher whose single source was
 * configured at 1. Sources are what get analysed, so they are what this reports;
 * the default only appears via the sources that inherit it. Mixed rates are shown
 * as a range rather than being collapsed to one misleading number.
 *
 * A running watcher's `analysis_fps_cap` param is applied for the same reason — while
 * it is running, that cap is the rate, and reporting the uncapped config would be the
 * same lie in a different place.
 *
 * @param {object} profile
 * @param {?object} task - live task info, or null when not running
 * @returns {?string} e.g. "1 fps", "1–5 fps", or null when there are no sources
 */
function _analysisFpsText(profile, task) {
  const sources = (profile.sources ?? []).filter(s => s.enabled !== false);
  if (!sources.length) return null;
  const cap = Number(task?.config?.params?.analysis_fps_cap) || null;
  const rates = sources
    .map(s => s.analysisFps ?? profile.defaultAnalysisFps)
    .filter(v => v != null)
    .map(v => (cap ? Math.min(v, cap) : v));
  if (!rates.length) return null;
  const low = Math.min(...rates);
  const high = Math.max(...rates);
  return low === high ? `${low} fps` : `${low}–${high} fps`;
}

/**
 * @param {object} profile
 * @param {?object} task - live task info, or null when not running
 * @returns {HTMLElement}
 */
function _profileCard(profile, task) {
  const card = document.createElement('section');
  card.className = 'panel-box nvr-profile-card';
  card.dataset.tour = 'nvr.profileCard';
  card.dataset.profileUuid = profile.uuid;

  const bar = document.createElement('div');
  bar.className = 'panel-box__bar';

  const title = document.createElement('div');
  title.className = 'panel-box__title';
  title.textContent = profile.name || '(unnamed profile)';
  bar.appendChild(title);

  const state = document.createElement('span');
  const isRunning = task !== null;
  state.className = `nvr-state-pill ${isRunning ? 'nvr-state-pill--running' : 'nvr-state-pill--stopped'}`;
  state.textContent = isRunning ? 'running' : 'stopped';
  bar.appendChild(state);

  const spacer = document.createElement('span');
  spacer.style.flex = '1';
  bar.appendChild(spacer);

  // Only offered while the watcher is running: there is no frame to show otherwise,
  // and a button that always opens an empty modal teaches the user to ignore it.
  if (isRunning) {
    const liveBtn = _panelButton(
      'Live view',
      () => openWatcherLiveView({ taskUuid: task.taskUuid, profileName: profile.name }),
    );
    liveBtn.dataset.testid = 'nvr-live-view';
    bar.appendChild(liveBtn);
  }

  // `selectUuid` is not optional decoration: without it the Watchers pane opens on
  // whatever was last selected, so "Edit rules" on card B lands you in card A's editor.
  bar.appendChild(_panelButton(
    'Edit rules',
    () => navigateTabs('tab-configuration', 'tab-watcher-management', { selectUuid: profile.uuid }),
  ));

  // Start/stop lives in Runs for now rather than being half-duplicated here:
  // this pane is read-only until the watcher control surface is built out.
  bar.appendChild(_panelButton(
    'Show in Runs',
    // Active, not just the Runs tab: "show in Runs" means the running watcher,
    // and the bare main-tab click lands on whichever sub-tab was last open.
    () => navigateTabs('tab-runs', 'tab-runs-active'),
  ));

  card.appendChild(bar);

  const body = document.createElement('div');
  body.className = 'panel-box__body nvr-profile-card__body';

  const rules = profile.rules ?? profile.interactionRules ?? [];
  const meta = document.createElement('div');
  meta.className = 'nvr-profile-card__meta';
  meta.textContent = [
    `${rules.length} rule${rules.length === 1 ? '' : 's'}`,
    profile.trackerTypename || profile.tracker_typename,
    _analysisFpsText(profile, task),
  ].filter(Boolean).join(' · ');
  body.appendChild(meta);

  for (const rule of rules) {
    const row = document.createElement('div');
    row.className = 'nvr-rule-row';
    row.textContent = rule.name || '(unnamed rule)';
    body.appendChild(row);
  }

  card.appendChild(body);
  return card;
}

/**
 * Events — the recorded-event browser. Not built; says so rather than showing
 * an empty grid that looks broken.
 */
async function renderEvents({ target = 'nvr-events-box' } = {}) {
  const container = document.getElementById(target);
  if (!container) return;
  container.innerHTML = '';
  container.appendChild(_emptyState(
    'Event browser not built yet',
    'Recorded events and their clips will live here. The watch_events and recordings tables already exist; the browser itself is Phase 8 of STREAM_WATCHER_PLAN.',
  ));
}

/**
 * @param {string} title
 * @param {string} detail
 * @param {?{label: string, onClick: () => void}} [action]
 * @returns {HTMLElement}
 */
function _emptyState(title, detail, action = null) {
  const box = document.createElement('div');
  box.className = 'nvr-empty';

  const h = document.createElement('strong');
  h.textContent = title;
  box.appendChild(h);

  const p = document.createElement('p');
  p.textContent = detail;
  box.appendChild(p);

  if (action) {
    box.appendChild(_panelButton(action.label, action.onClick));
  }
  return box;
}

export async function init() {
  const { container } = setupWorkspaceTabs('#tab-nvr', async (tabName) => {
    if (tabName === 'tab-nvr-live') {
      await renderLive({ target: 'nvr-live-box' });
    } else if (tabName === 'tab-nvr-events') {
      await renderEvents({ target: 'nvr-events-box' });
    }
  });

  // activate default NVR tab
  const defaultTabBtn = container?.querySelector('.workspace-tab-button');
  if (defaultTabBtn) defaultTabBtn.click();
}
