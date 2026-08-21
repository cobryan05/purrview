/**
 * NVR — the surveillance half of the app.
 *
 * Deliberately placeholders for now. The structure exists so the two surfaces
 * have a home and a stable id to grow into; neither invents data it does not
 * have.
 *
 *  - **Live** lists the watch profiles you have defined and whether each is
 *    running. It reads the same task instances Jobs shows, over the same shared
 *    SSE stream, so the two views cannot disagree — Jobs answers "what is
 *    holding my GPU?", this answers "is the house being watched?".
 *  - **Events** is the media browser from STREAM_WATCHER_PLAN §7.5 (Phase 8),
 *    which is not started. `watch_events` and `recordings` exist with the
 *    foreign keys it needs, so this shows a real (currently empty) count rather
 *    than pretending.
 */
import { setupWorkspaceTabs } from '/app-static/js/app/utils/tabs.js';
import { fetchWatchProfiles } from '/app-static/js/shared/api/watchProfiles.js';
import { cancelTasks, fetchTaskStatusData, fetchTasksStatus } from '/app-static/js/shared/api/tasks.js';
import { launchRun } from '/app-static/js/ui/utils/launchRun.js';
import { subscribeTaskStream } from '/app-static/js/shared/api/taskStream.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { NEW_ITEM_SELECTION, navigateTabs } from '/app-static/js/app/utils/tabs.js';
import { Logger, showPaneError, toast } from '/app-static/js/ui/utils/index.js';
import { openWatcherLiveView } from '/app-static/js/app/nvr/liveViewModal.js';
import {
  DEFAULT_REFRESH_MS,
  REFRESH_RATES,
  attachCardPreviews,
  levelFromHealthLabel,
  stopAllCardPreviews,
} from '/app-static/js/app/nvr/cardPreview.js';

/** localStorage key prefix for one card's chosen thumbnail refresh rate. */
const REFRESH_KEY_PREFIX = 'nvr.cardRefreshMs.';

/**
 * The refresh rate this card should start at.
 *
 * Persisted per profile, and it has to be: this pane re-renders whenever a watcher
 * starts or stops, which rebuilds every card from scratch. Without a store, turning one
 * camera up would silently snap back to the default the next time any *other* watcher
 * changed state.
 *
 * @param {string} profileUuid
 * @returns {number} milliseconds
 */
function _storedRefreshMs(profileUuid) {
  try {
    const raw = Number(localStorage.getItem(REFRESH_KEY_PREFIX + profileUuid));
    // Only a rate still on the menu: a value left over from an older option list would
    // pin a card to an interval its dropdown can no longer show or undo.
    if (REFRESH_RATES.some(r => r.ms === raw)) return raw;
  } catch {
    // Private-browsing / disabled storage. The default is a fine answer.
  }
  return DEFAULT_REFRESH_MS;
}

function _storeRefreshMs(profileUuid, ms) {
  try {
    localStorage.setItem(REFRESH_KEY_PREFIX + profileUuid, String(ms));
  } catch {
    // Not worth telling the user about — the rate still applies for this session.
  }
}

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
/**
 * Which watchers were live as of the last read — see `_watcherSignature`.
 *
 * Module scope so the render path and the stream subscription share one value: if the
 * stream kept its own, a tab entry would render once from `fetchRunningWatchers` and
 * again on the next tick, restarting every preview poller for nothing.
 */
let _watcherSig = null;

async function fetchRunningWatchers() {
  const running = new Map();
  try {
    const { tasks } = await fetchTasksStatus();
    _watcherSig = _watcherSignature(tasks);
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
 * Each running watcher's sources, for the card thumbnails.
 *
 * One batched call for the whole pane. The live-preview endpoint could answer this too,
 * but its job is to return a *frame* — it would encode and ship a JPEG per card purely
 * to learn some names. The status payload already carries the sources and encodes
 * nothing, which is the difference between a free pane and one that costs a JPEG per
 * watcher every time you open it.
 *
 * @param {string[]} taskUuids
 * @returns {Promise<Map<string, Array<{sourceUuid: string, sourceName: string}>>>}
 */
async function fetchWatcherSources(taskUuids) {
  const byTask = new Map();
  if (!taskUuids.length) return byTask;
  try {
    const statusData = await fetchTaskStatusData(taskUuids);
    for (const [taskUuid, data] of statusData) {
      const sources = (data?.sources ?? []).map(row => ({
        sourceUuid: row.sourceUuid,
        // `source` is the status table's column name for the human name.
        sourceName: row.source,
        // Carried through for the card's counter. This is the only per-source health
        // that covers the sources *beyond* the four that get thumbnails — without it a
        // card silently reports on a subset of its own cameras.
        health: row.health,
      })).filter(s => s.sourceUuid);
      if (sources.length) byTask.set(taskUuid, sources);
    }
  } catch (err) {
    // Thumbnails are an enhancement; the cards are still worth showing without them.
    Logger.warn('nvr: could not read watcher sources', err);
  }
  return byTask;
}

/** Guards renderLive against overlapping calls — see the note there. */
let _rendering = false;
/** A render that arrived while one was in flight, and must happen after it. */
let _renderQueued = false;

/**
 * Live — one card per watch profile, showing whether it is currently running.
 *
 * ⚠️ Two callers can fire this at once: the tab activation and the task stream. It is
 * async between reading and writing, so an unguarded version clears the pane, awaits,
 * and appends — and two overlapping runs both append, listing every watcher twice.
 * (That is exactly what happened when the stream subscription was added.)
 *
 * Hence both halves of the fix: the guard below, and a single atomic `replaceChildren`
 * at the end instead of an early wipe. The atomic swap also means a slow render no
 * longer blanks the pane while it waits on three API calls.
 */
async function renderLive({ target = 'nvr-live-box' } = {}) {
  const container = document.getElementById(target);
  if (!container) return;
  if (_rendering) {
    // Not simply dropped: the caller that lost the race may be the one reporting a
    // watcher that just started, and swallowing it would leave the pane stale until
    // the next state change.
    _renderQueued = true;
    return;
  }
  _rendering = true;
  try {
    let profiles;
    try {
      // ⚠️ A Map keyed by uuid, not an array (same convention as fetchLabels —
      // see CLAUDE.md "Key Gotchas"). Iterate with .values().
      profiles = await fetchWatchProfiles();
    } catch (err) {
      Logger.warn('nvr: could not load watch profiles', err);
      stopAllCardPreviews();
      container.replaceChildren();
      showPaneError(container, 'Could not load watch profiles');
      return;
    }

    const running = await fetchRunningWatchers();
    const sourcesByTask = await fetchWatcherSources([...running.values()].map(t => t.taskUuid));
    const rows = [...(profiles?.values?.() ?? [])];

    // Every card about to be discarded owns polling timers, and a timer left running
    // against detached DOM is a leak that compounds each time the pane re-renders.
    // Deliberately here rather than at the top: until this point nothing has been
    // replaced, so an earlier render's thumbnails keep updating while this one loads.
    stopAllCardPreviews();

    if (!rows.length) {
      container.replaceChildren(_emptyState(
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
      list.appendChild(_profileCard(profile, task, sourcesByTask.get(task?.taskUuid) ?? []));
    }

    container.replaceChildren(list);
  } finally {
    _rendering = false;
    if (_renderQueued) {
      _renderQueued = false;
      void renderLive({ target });
    }
  }
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
 * "Start watching" / "Stop", for one card.
 *
 * Starting is exactly what the Jobs launcher does — a temporary task config plus an
 * instance — rather than a second mechanism: two ways to start a watcher that differ in
 * what they write to the DB is how you end up with watchers that behave differently
 * depending on which button started them.
 *
 * The pane is not re-rendered here. The task stream reports the state change (see
 * `init`), so a card that flipped itself optimistically would either duplicate that
 * render or, if the start failed, show a lie until the next one.
 *
 * @param {object} profile
 * @param {?object} task
 * @returns {HTMLButtonElement}
 */
function _startStopButton(profile, task) {
  const isRunning = task !== null;
  const btn = _panelButton(isRunning ? 'Stop' : 'Start watching', async () => {
    btn.disabled = true;
    btn.textContent = isRunning ? 'Stopping…' : 'Starting…';
    try {
      if (isRunning) {
        const outcome = await cancelTasks({ taskUuids: [task.taskUuid] });
        // The server now says whether the watcher actually went down inside the
        // cancel window. Reporting "Stopping" either way is what let a watcher that
        // never stopped look exactly like one that did.
        if (outcome.stopped.includes(task.taskUuid)) {
          toast(`Stopped ${profile.name}`, 3000, 'success');
        } else if (outcome.notRunning.includes(task.taskUuid)) {
          toast(`${profile.name} was not running`, 5000, 'warning');
        } else {
          toast(`Stopping ${profile.name} — it is still shutting down`, 5000, 'warning');
        }
      } else {
        await launchRun({
          typename: WATCHER_TYPENAME,
          name: `Watch ${profile.name}`,
          params: { watch_profile_uuid: profile.uuid },
        });
        toast(`Starting ${profile.name}`, 3000);
      }
    } catch (err) {
      Logger.error(err);
      toast(`Could not ${isRunning ? 'stop' : 'start'} ${profile.name}`, 5000, 'error');
      // Only on failure: on success the stream's re-render replaces this button, and
      // restoring it here would fight that render.
      btn.disabled = false;
      btn.textContent = isRunning ? 'Stop' : 'Start watching';
    }
  });
  btn.dataset.testid = isRunning ? 'nvr-stop-watcher' : 'nvr-start-watcher';
  return btn;
}

/**
 * The card's thumbnail refresh dropdown.
 *
 * Deliberately labelled "thumbnails": this changes how often the *browser* asks for a
 * picture, and nothing about how often the watcher analyses frames. Those two numbers
 * are both on the card and both in fps, so without the label the dropdown reads as a
 * control over the watcher's analysis rate — which it is not, and which would be a
 * dangerous thing to fiddle with from an overview pane.
 *
 * @param {string} profileUuid
 * @param {(ms: number) => void} onChange
 * @returns {HTMLElement}
 */
function _refreshRateControl(profileUuid, onChange) {
  const wrap = document.createElement('label');
  wrap.className = 'nvr-refresh-control';

  const text = document.createElement('span');
  text.textContent = 'Thumbnails';
  wrap.appendChild(text);

  const select = document.createElement('select');
  select.className = 'nvr-refresh-select';
  select.dataset.testid = 'nvr-refresh-rate';
  const current = _storedRefreshMs(profileUuid);
  for (const rate of REFRESH_RATES) {
    const opt = document.createElement('option');
    opt.value = String(rate.ms);
    opt.textContent = rate.label;
    opt.selected = rate.ms === current;
    select.appendChild(opt);
  }
  select.addEventListener('change', () => {
    const ms = Number(select.value) || DEFAULT_REFRESH_MS;
    _storeRefreshMs(profileUuid, ms);
    onChange(ms);
  });
  wrap.appendChild(select);
  return wrap;
}

/**
 * @param {object} profile
 * @param {?object} task - live task info, or null when not running
 * @param {Array<{sourceUuid: string, sourceName: string}>} [sources] - for the thumbnails
 * @returns {HTMLElement}
 */
function _profileCard(profile, task, sources = []) {
  const card = document.createElement('section');
  card.className = 'panel-box nvr-profile-card';
  // The one case that justifies a wide card: several cameras' thumbnails side by side.
  // A single-source watcher gets a normal grid cell, so a screen of them packs tightly
  // instead of each being a full-width line of text.
  if ((profile.sources ?? []).filter(s => s.enabled !== false).length > 1) {
    card.classList.add('nvr-profile-card--multi-source');
  }
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

  // "running" is true of a watcher whose only camera has been dark for an hour, and
  // that reading is what let a dead source go unnoticed for a whole run. These two sit
  // next to it and can contradict it, so the card cannot say "fine" on its own.
  //
  // The count is separate from the alert because it answers a question the alert
  // cannot: only four sources get thumbnails, so on a bigger profile the strip shows a
  // subset and "no red tiles" stops meaning "no problems". `n/m live` is over *every*
  // source, tiled or not.
  const countPill = document.createElement('span');
  countPill.className = 'nvr-state-pill nvr-state-pill--count';
  countPill.dataset.testid = 'nvr-source-count';
  countPill.hidden = true;
  bar.appendChild(countPill);

  const alertPill = document.createElement('span');
  alertPill.className = 'nvr-state-pill nvr-state-pill--offline';
  alertPill.dataset.testid = 'nvr-offline-pill';
  alertPill.hidden = true;
  bar.appendChild(alertPill);

  /**
   * Per-source level for the whole profile.
   *
   * Seeded from the status payload, which covers every source, then overwritten for
   * the four that have tiles as their polls come in — those are fresher (10 s) than
   * this pane's re-render, and they are the ones the user is looking at.
   */
  const levels = new Map(
    (sources ?? []).map(s => [s.sourceUuid, {
      sourceName: s.sourceName || s.sourceUuid,
      level: levelFromHealthLabel(s.health),
    }]),
  );

  function renderHealthPills() {
    const all = [...levels.values()];
    if (!isRunning || !all.length) {
      countPill.hidden = true;
      alertPill.hidden = true;
      card.classList.remove('nvr-profile-card--degraded');
      return;
    }
    const dead = all.filter(s => s.level === 'dead');
    const connecting = all.filter(s => s.level === 'connecting');
    const live = all.length - dead.length - connecting.length;

    countPill.hidden = false;
    countPill.textContent = `${live}/${all.length} live`;
    countPill.classList.toggle('nvr-state-pill--count-degraded', dead.length > 0);

    // The card is named after the profile, so the alert has to name the *camera* —
    // "1 source offline" on a six-camera card sends you hunting for which.
    const names = list => list.map(s => s.sourceName).join(', ');
    if (dead.length) {
      alertPill.hidden = false;
      alertPill.textContent = `⚠ no signal: ${names(dead)}`;
      card.classList.add('nvr-profile-card--degraded');
    } else if (connecting.length) {
      alertPill.hidden = false;
      alertPill.textContent = `connecting: ${names(connecting)}`;
      card.classList.remove('nvr-profile-card--degraded');
    } else {
      alertPill.hidden = true;
      alertPill.textContent = '';
      card.classList.remove('nvr-profile-card--degraded');
    }
  }

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

  // Start/stop belongs here as well as in Jobs. This pane is where you *look* to see
  // whether the house is being watched, and a card that reports "stopped" and then
  // sends you to another tab to do anything about it is a dead end.
  bar.appendChild(_startStopButton(profile, task));

  // `selectUuid` is not optional decoration: without it the Watchers pane opens on
  // whatever was last selected, so "Edit rules" on card B lands you in card A's editor.
  bar.appendChild(_panelButton(
    'Edit rules',
    () => navigateTabs('tab-configuration', 'tab-watcher-management', { selectUuid: profile.uuid }),
  ));

  // Start/stop lives in Jobs for now rather than being half-duplicated here:
  // this pane is read-only until the watcher control surface is built out.
  bar.appendChild(_panelButton(
    'Show in Jobs',
    // Running, not just the Jobs tab: "show in Jobs" means the running watcher,
    // and the bare main-tab click lands on whichever sub-tab was last open.
    () => navigateTabs('tab-runs', 'tab-runs-active'),
  ));

  card.appendChild(bar);

  const body = document.createElement('div');
  body.className = 'panel-box__body nvr-profile-card__body';

  // Only for a running watcher: a stopped one has no frames, and a strip of permanent
  // "no frame yet" placeholders would make every stopped card look broken.
  if (isRunning) {
    const previews = attachCardPreviews(body, {
      taskUuid: task.taskUuid,
      sources,
      refreshMs: _storedRefreshMs(profile.uuid),
      onHealth: (states) => {
        for (const state of states) {
          levels.set(state.sourceUuid, { sourceName: state.sourceName, level: state.level });
        }
        renderHealthPills();
      },
      // Clicking the picture is the obvious way to say "show me that camera", and it
      // carries the source with it — the bar's Live view button opens on whatever the
      // watcher lists first, which on a multi-camera profile is rarely the one you
      // just looked at.
      onOpenSource: (sourceUuid) => openWatcherLiveView({
        taskUuid: task.taskUuid,
        profileName: profile.name,
        sourceUuid,
      }),
    });
    // Appended after the strip exists so it sits under the thumbnails it governs,
    // rather than in the bar with the controls that act on the watcher itself. This
    // one changes only what the browser asks for; it does not touch the watcher.
    body.appendChild(_refreshRateControl(profile.uuid, previews.setRefreshMs));
  }

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

  // Seeded state, before any tile has polled: a card that has just been rendered onto
  // an already-broken watcher must say so immediately, not ten seconds later.
  renderHealthPills();

  card.appendChild(body);
  return card;
}

/**
 * Events — the recorded-event review page.
 *
 * Lazily imported rather than pulled in at module load: the Live tab is what opens
 * first, and the search page drags in a filter bar, a card grid and a video modal that
 * a user who never leaves Live pays nothing for.
 */
async function renderEvents({ target = 'nvr-events-box' } = {}) {
  const { renderEvents: render } = await import('/app-static/js/app/nvr/events/main.js');
  await render({ target });
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

/**
 * A signature of which watchers are live, for spotting real changes in the stream.
 *
 * The stream pushes the whole task map on every tick, and the pane must not re-render
 * on each one: a render tears down the preview strips and restarts their polling, so
 * following the stream literally would turn 10-second thumbnails into a hidden
 * per-tick refresh. Only the *set* of running watchers matters here.
 *
 * @param {object} tasks - the stream's task map
 * @returns {string}
 */
function _watcherSignature(tasks) {
  return Object.entries(tasks ?? {})
    .filter(([, t]) => t?.config?.typename === WATCHER_TYPENAME && LIVE_STATUSES.has(t?.instance?.status))
    .map(([uuid, t]) => `${uuid}:${t?.instance?.status}`)
    .sort()
    .join('|');
}

export async function init() {
  const { container } = setupWorkspaceTabs('#tab-nvr', async (tabName) => {
    if (tabName === 'tab-nvr-live') {
      await renderLive({ target: 'nvr-live-box' });
    } else if (tabName === 'tab-nvr-events') {
      await renderEvents({ target: 'nvr-events-box' });
    }
  });

  // Without this the pane only learned about a watcher starting on a full page reload —
  // including a watcher this pane's own Start button had just launched.
  let unsubscribe = null;

  function openStream() {
    if (unsubscribe !== null) return;
    unsubscribe = subscribeTaskStream({
      onMessage: (data) => {
        if (!data?.tasks) return;
        const next = _watcherSignature(data.tasks);
        if (next === _watcherSig) return;
        _watcherSig = next;
        void renderLive({ target: 'nvr-live-box' });
      },
    });
  }

  // Set by closeStream, cleared on the way back in. A plain "re-render on entry"
  // would also fire on the *first* entry — the one that lazy-loaded this module and
  // has already rendered — costing every card a second frame fetch on open.
  let previewsStopped = false;

  function closeStream() {
    // Both halves, or the pane keeps working off-screen: the stream is shared with Jobs
    // and costs a connection while subscribed, and the thumbnails would otherwise keep
    // fetching frames for a pane nobody is looking at.
    unsubscribe?.();
    unsubscribe = null;
    stopAllCardPreviews();
    previewsStopped = true;
  }

  events.subscribe(EventTypes.MAIN_TAB_CHANGED, ({ tabName }) => {
    if (tabName !== 'tab-nvr') {
      closeStream();
      return;
    }
    openStream();
    // Re-render, don't just re-subscribe: closeStream() stopped every card preview,
    // and nothing else brings them back on a main-tab switch — the sub-tab callback
    // doesn't re-fire, and the stream only re-renders when the watcher signature
    // changes. Without this, leaving NVR and returning left every tile frozen on its
    // last frame until a watcher happened to start or stop.
    if (previewsStopped) {
      previewsStopped = false;
      void renderLive({ target: 'nvr-live-box' });
    }
  });

  // Defining a watcher lives in Configuration, but wanting one happens here — looking
  // at the cameras. Hands the pane straight to a blank editor rather than dropping the
  // user on a list they then have to find the "+" in.
  document.getElementById('nvr-add-watcher')?.addEventListener('click', () => {
    navigateTabs('tab-configuration', 'tab-watcher-management', { selectUuid: NEW_ITEM_SELECTION });
  });

  // activate default NVR tab
  const defaultTabBtn = container?.querySelector('.workspace-tab-button');
  if (defaultTabBtn) defaultTabBtn.click();

  // This module is lazy-loaded on the NVR tab's first focus, so by the time init runs
  // the pane is on screen — and no MAIN_TAB_CHANGED will fire to say so.
  openStream();
}
