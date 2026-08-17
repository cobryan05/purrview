/**
 * Runs → Start a run.
 *
 * The single place anything is launched from. Configuration holds recipes —
 * a training config, a filter chain, a watch profile, a saved task config —
 * and this turns one into a running instance.
 *
 * Why this pane exists at all: starting a training used to be a button inside
 * its Configuration editor, along with a live progress bar, a log tail and a
 * Cancel. That put job monitoring inside a configuration screen. Recipes are
 * defined in Configuration; runs start and are watched here.
 */
import { fetchTrainingConfigs } from '/app-static/js/shared/api/trainingConfigs.js';
import { fetchFilterChains } from '/app-static/js/shared/api/filterChains.js';
import { fetchWatchProfiles } from '/app-static/js/shared/api/watchProfiles.js';
import { fetchTaskConfigs, fetchTasksStatus } from '/app-static/js/shared/api/tasks.js';
// Straight from the module rather than the fields barrel: one Set is not worth
// pulling every field widget into the launcher's import graph.
import { TERMINAL_STATUSES } from '/app-static/js/ui/utils/fields/ActiveTaskField.js';
import {
  fetchTrainingConfigReadiness,
  fetchTrainingConfigReadinessForImages,
} from '/app-static/js/shared/api/trainingConfigs.js';
import { loadPreviousRun, formatTrainedAt } from '/app-static/js/ui/utils/previousTrainingRun.js';
import { launchRun, launchSavedConfig } from '/app-static/js/ui/utils/launchRun.js';
import { navigateTabs } from '/app-static/js/app/utils/tabs.js';
import { showConfirmModal } from '/app-static/js/ui/utils/confirmModal.js';
import { openWatcherLiveView } from '/app-static/js/app/nvr/liveViewModal.js';
import { RUN_KINDS, SAVED_TASK_BADGE } from '/app-static/js/ui/utils/runKinds.js';
import { Logger, toast } from '/app-static/js/ui/utils/logging.js';

let _rendering = false;

/**
 * Normalise the assorted return shapes of the fetchers above.
 *
 * ⚠️ Several of these return a Map keyed by uuid rather than an array (the
 * fetchLabels convention — see CLAUDE.md "Key Gotchas"). A bare for...of over a
 * Map yields [key, value] pairs, so this is not cosmetic.
 *
 * @param {Map|Array|null} result
 * @returns {Array<object>}
 */
function asRows(result) {
  if (!result) return [];
  if (result instanceof Map) return [...result.values()];
  if (Array.isArray(result)) return result;
  return [];
}

/**
 * Resolve every group independently: one failing endpoint must not blank the
 * whole launcher, because the other groups are still launchable.
 *
 * @param {() => Promise<any>} fn
 * @returns {Promise<Array<object>|null>} null marks a group that failed to load
 */
async function settle(fn) {
  try {
    return asRows(await fn());
  } catch (err) {
    Logger.warn('runsLauncher: group failed to load', err);
    return null;
  }
}

// ── "+N since last run" on training cards ────────────────────────────────────
//
// Images that would go into a run started now, minus the images that trained the
// model this config last produced — the delta the training config's coverage
// panel shows, on the screen where the decision to start a run is made.
//
// Split into positives and negatives, and never shown as a single total: these
// datasets are overwhelmingly background, so "+1,100 since last run" routinely
// means 1,100 confirmed-absence images and no new training signal at all. The
// same reason readiness itself carries ready_positive_count/ready_negative_count.
//
// It costs two readiness queries plus a sidecar read per config — measured
// 0.05–0.4s each against the live database — so it is loaded off the render
// path, a few at a time, and cached briefly.

/** @type {Map<string, {at: number, value: {text: string, tone: string, title: string}}>} */
const _deltaCache = new Map();
const DELTA_TTL_MS = 60_000;
const DELTA_CONCURRENCY = 3;

/**
 * Run `tasks` a few at a time. Dozens of training configs firing a readiness
 * query each at once would put the database under a burst nobody asked for,
 * while the user is most likely about to start a run on it.
 *
 * @param {Array<() => Promise<any>>} tasks
 * @param {number} limit
 */
async function runThrottled(tasks, limit = DELTA_CONCURRENCY) {
  const queue = [...tasks];
  const workers = Array.from({ length: Math.min(limit, queue.length) }, async () => {
    for (let task = queue.shift(); task; task = queue.shift()) await task();
  });
  await Promise.all(workers);
}

/**
 * Signed, because a run can also shrink: images get deleted and datasets get
 * re-scoped, and "+-40" would be nonsense.
 * @param {number} n
 */
function _signed(n) {
  return `${n > 0 ? '+' : ''}${n.toLocaleString()}`;
}

/**
 * How many more images are ready for this training config than went into the
 * model it last trained, positives and negatives counted separately.
 *
 * `withLabelStats: false` is what keeps this affordable: the per-label replay
 * over the previous manifest is the expensive half of loadPreviousRun and none
 * of it is visible on a card.
 *
 * @param {{uuid: string, name: string}} config
 * @returns {Promise<{text: string, tone: string, title: string}>}
 */
async function trainingDelta(config) {
  const cached = _deltaCache.get(config.uuid);
  if (cached && Date.now() - cached.at < DELTA_TTL_MS) return cached.value;

  const [readiness, previousRun] = await Promise.all([
    fetchTrainingConfigReadiness(config.uuid),
    loadPreviousRun(config, { withLabelStats: false }),
  ]);
  const pos = readiness?.readyPositiveCount ?? 0;
  const neg = readiness?.readyNegativeCount ?? 0;

  let value;
  if (!previousRun) {
    // Never trained, or trained somewhere else — there is no baseline to
    // subtract, and claiming "+everything since" would be a lie.
    value = {
      text: `${pos.toLocaleString()} pos`,
      rest: ` · ${neg.toLocaleString()} neg ready · not trained yet`,
      tone: 'muted',
      title: 'No previous run recorded for this recipe',
    };
  } else {
    // The baseline is the same split over the manifest the model was trained on,
    // read through the same endpoint — so a re-labelled image moves both sides
    // rather than showing up as phantom growth.
    const before = await fetchTrainingConfigReadinessForImages(config.uuid, previousRun.imageUuids);
    const posDelta = pos - before.readyPositiveCount;
    const negDelta = neg - before.readyNegativeCount;

    const when = formatTrainedAt(previousRun.trainedAt);
    const title =
      `Last run (${previousRun.modelName}${when ? `, ${when}` : ''}): ` +
      `${before.readyPositiveCount.toLocaleString()} positive / ${before.readyNegativeCount.toLocaleString()} negative ` +
      `of ${previousRun.manifestCount.toLocaleString()} images\n` +
      `Ready now: ${pos.toLocaleString()} positive / ${neg.toLocaleString()} negative`;

    // Tone follows the positives alone. A card that went up by a thousand
    // confirmed-absence images has not gained training signal, and colouring it
    // green would say that it had.
    const tone = posDelta > 0 ? 'up' : posDelta < 0 ? 'down' : 'muted';
    value =
      posDelta === 0 && negDelta === 0
        ? { text: 'no new images since last run', tone: 'muted', title }
        : {
            text: `${_signed(posDelta)} pos`,
            rest: ` · ${_signed(negDelta)} neg since last run`,
            tone,
            title,
          };
  }

  _deltaCache.set(config.uuid, { at: Date.now(), value });
  return value;
}

// ── "Training in progress" on training cards ─────────────────────────────────
//
// A recipe whose model is training right now is the one card you should not be
// pressing, and the launcher had no way to say so: the run is visible on Active,
// one sub-tab away, and nowhere else.
//
// This deliberately owns no SSE stream. Runs already holds exactly one (opened
// in app/runs/main.js, closed when the last job ends off-tab — the leak
// test_leaks_and_races.py guards), so the launcher is *pushed* the task list
// that stream already delivers rather than subscribing to it a second time.

/** Live training runs by training-config uuid, rebuilt on every render. */
let _runStateByConfig = new Map();

/**
 * Which task types a launch card can recognise a live instance of lives in
 * RUN_KINDS, shared with the Active pane so the two screens cannot disagree
 * about what a running instance was launched from.
 *
 * Only kinds carrying a `badge` are shown here: the wording is the whole point
 * ("Training in progress" and "Watching now" answer different questions, and a
 * generic "Running" on a wall of watcher cards says nothing useful), so a kind
 * with nothing worth saying gets no badge rather than a filler one.
 */

/**
 * How a live run should read on the card that launches it.
 * Anything not terminal counts: a queued run is as much a reason not to press
 * Start as a running one.
 *
 * @param {string} status
 * @param {{running: string, paused: string}} badge - the kind's per-status wording
 * @returns {?{text: string, tone: string}}
 */
function _runLabel(status, badge) {
  if (TERMINAL_STATUSES.has(status)) return null;
  if (status === 'cancelled') return null;
  if (status === 'paused') return { text: badge.paused, tone: 'paused' };
  if (status === 'running') return { text: badge.running, tone: 'running' };
  // 'new'/'pending', plus any status this UI has not been taught yet — treating
  // an unknown state as live is the safe direction.
  return { text: 'Queued', tone: 'queued' };
}

/**
 * Statuses whose instance can actually be looked at live.
 *
 * A queued watcher has no frames yet and a paused one has stopped producing
 * them, so opening a live view on either shows an empty modal — which teaches
 * people that the button does not work.
 *
 * Only the *live view* is gated on this. Every live run, queued and paused
 * included, can still be shown on Active: that pane's whole job is to list runs
 * in whatever state they are in.
 */
const _VIEWABLE_TONES = new Set(['running']);

/**
 * Show a running instance where runs are actually watched.
 *
 * The default answer to pressing a card whose run is already going. Refusing
 * with a toast told the user what would not happen and left them to find the
 * run themselves, one sub-tab away; this takes them to it and marks it.
 *
 * @param {?string} taskUuid
 */
function showRunOnActive(taskUuid) {
  navigateTabs('tab-runs', 'tab-runs-active', { selectUuid: taskUuid ?? null });
}

/**
 * Index the live runs in a task-status payload by the thing a launch card
 * launches: the definition (training config / watch profile) a throwaway config
 * was built around, or — for a saved task, which is started as itself — the
 * task config's own uuid.
 *
 * @param {Record<string, any>} tasks - `tasks` from fetchTasksStatus / the stream
 * @returns {Map<string, {text: string, tone: string}>}
 */
function indexLiveRuns(tasks) {
  const byConfig = new Map();

  const record = (configUuid, label, entry, taskId) => {
    if (!configUuid || !label) return;
    // The instance uuid rides along so a card can open the running thing, not
    // just describe it. Falls back to the payload's own key, which is what the
    // Active pane hangs its rows off (`data-task-uuid`), so the highlight lands
    // even if an entry ever arrives without an inner uuid.
    label.taskUuid = entry.instance?.uuid ?? taskId ?? null;
    // Whether that instance is worth *looking at* is a narrower question than
    // whether it can be pointed at on Active.
    label.liveViewable = _VIEWABLE_TONES.has(label.tone);
    // Running beats queued when a definition has both in flight — the stronger
    // statement is the true one, and two badges do not fit on a card.
    const current = byConfig.get(configUuid);
    if (current && current.tone === 'running') return;
    byConfig.set(configUuid, label);
  };

  for (const [taskId, entry] of Object.entries(tasks || {})) {
    const status = entry?.instance?.status;
    const kind = RUN_KINDS[entry?.config?.typename];
    if (kind?.badge) {
      record(entry.config?.params?.[kind.paramKey], _runLabel(status, kind.badge), entry, taskId);
    }
    // A saved task config is started directly rather than cloned, so a live
    // instance of it points back at the very config a Saved-tasks card offers.
    // Temporary configs are launcher scaffolding and appear on no card, so
    // indexing them would only key the map on uuids nothing can match — and a
    // run launched from a training/watcher card is already badged above, on the
    // card that actually launched it.
    if (!entry?.config?.isTemporary) {
      record(entry?.config?.uuid, _runLabel(status, SAVED_TASK_BADGE), entry, taskId);
    }
  }
  return byConfig;
}

/**
 * Paint the live-run badges from a task-status payload.
 *
 * Exported so the Runs tab can hand over the task list its own SSE subscription
 * already receives. No-ops once the launcher has been re-rendered or was never
 * rendered, so a stream message that arrives while the user is elsewhere costs
 * a Map lookup and nothing else.
 *
 * @param {Record<string, any>} tasks
 */
export function applyRunsLauncherTaskState(tasks) {
  if (_runStateByConfig.size === 0) return;
  const live = indexLiveRuns(tasks);
  for (const [configUuid, setBadge] of _runStateByConfig) {
    setBadge(live.get(configUuid) ?? null);
  }
}

/**
 * One launchable thing: a big Start target, plus a link to what defines it.
 *
 * The card is a `<div>` holding two buttons rather than one big `<button>`,
 * because a button inside a button is invalid HTML — browsers recover by
 * hoisting the inner one out, which loses the click handler.
 *
 * @param {Object} opts
 * @param {string} opts.name
 * @param {string} opts.hint - what it will do, in the user's terms
 * @param {() => Promise<string>} opts.start - resolves to the task instance uuid
 * @param {() => void} opts.onStarted
 * @param {string} opts.configTab - Configuration sub-tab that defines this thing
 * @param {?string} [opts.configUuid] - the definition this card launches, so the
 *   Configuration screen opens on it rather than on whatever was last edited
 * @param {?{title: string, message: string}} [opts.confirm] - gate the start
 *   behind a confirmation, for runs that are destructive or unscoped.
 * @param {?() => Promise<{text: string, tone?: string, title?: string}>} [opts.detail] -
 *   a slow, per-card number that must not hold up the card itself. When given, a
 *   third line is reserved on the card and filled in when this resolves.
 * @param {?(reload: () => Promise<void>) => void} [opts.registerDetail] - handed
 *   the card's reload function, so the group can re-run every card's detail.
 * @param {?(setBadge: (label: ?{text: string, tone: string}) => void) => void} [opts.registerRunState] -
 *   handed a setter for the card's live-run badge, so task-status updates can
 *   reach this card without re-rendering the launcher.
 * @param {boolean} [opts.preventDuplicate] - refuse to start a second run while
 *   this card's definition already has a live one. On by default: nothing the
 *   launcher starts is improved by a second concurrent copy, and a card can
 *   only ever be guarded when it has a badge to guard from — without
 *   registerRunState there is no live label and this is inert.
 * @param {?(label: {text: string, tone: string, taskUuid: ?string, liveViewable: boolean}) => void} [opts.onLiveClick] -
 *   what pressing an already-running card does instead of starting. Defaults to
 *   showing the run on Active; watchers override it with their live view.
 * @returns {HTMLElement}
 */
function launchCard({
  name,
  hint,
  start,
  onStarted,
  configTab,
  configUuid = null,
  confirm = null,
  detail = null,
  registerDetail = null,
  registerRunState = null,
  preventDuplicate = true,
  onLiveClick = null,
}) {
  // Every live card leads somewhere: to the run itself on Active, unless the
  // caller has something better to show (a watcher's live frames).
  const livePress = onLiveClick ?? ((label) => showRunOnActive(label.taskUuid));
  // The card's current live-run label, kept in the closure so the click handler
  // can read what the badge is showing. Only meaningful with registerRunState.
  let liveLabel = null;
  const card = document.createElement('div');
  card.className = 'runs-launch-card';
  card.dataset.tour = 'runs.launchCard';
  card.dataset.runName = name;

  const startBtn = document.createElement('button');
  startBtn.type = 'button';
  startBtn.className = 'runs-launch-card__start';
  startBtn.dataset.testid = 'runs-launch-start';

  const top = document.createElement('span');
  top.className = 'runs-launch-card__name';
  top.textContent = name;
  startBtn.appendChild(top);

  const sub = document.createElement('span');
  sub.className = 'runs-launch-card__hint';
  sub.textContent = hint;
  startBtn.appendChild(sub);

  if (registerRunState) {
    // Sits above the name so it reads first — the point of the badge is to be
    // seen before the card is pressed, not after.
    const badge = document.createElement('span');
    badge.className = 'runs-launch-card__run-badge';
    badge.dataset.testid = 'runs-launch-run-badge';
    badge.hidden = true;
    startBtn.insertBefore(badge, top);

    registerRunState((label) => {
      liveLabel = label;
      badge.hidden = label === null;
      card.classList.toggle('is-live', label !== null);
      // A live card is never *blocked*, only repurposed: it stops being a Start
      // button and becomes the way into the run it already has — the live view
      // for a watcher, the Active pane for everything else. So it keeps the
      // normal pointer, is never `disabled` or aria-disabled (both would be a
      // lie to assistive tech about a control that does act), and says in its
      // title what the press will now do.
      if (preventDuplicate) {
        if (label === null) startBtn.removeAttribute('title');
        else startBtn.title = onLiveClick
          ? `Open the live view for ${name}`
          : `${name} is already running — show it on Active`;
      }
      if (label === null) return;
      badge.className = `runs-launch-card__run-badge is-${label.tone}`;
      badge.textContent = label.text;
    });
  }

  if (detail) {
    // The slot is appended now and filled later: the launcher must paint at the
    // speed of the four list endpoints, not at the speed of one readiness query
    // per card.
    const slot = document.createElement('span');
    slot.className = 'runs-launch-card__detail is-pending';
    slot.dataset.testid = 'runs-launch-detail';
    slot.textContent = '…';
    startBtn.appendChild(slot);

    // Two spans, because only the first half carries a colour: a card whose
    // negatives grew by a thousand must not read as a thousand-image win.
    const lead = document.createElement('span');
    lead.dataset.testid = 'runs-launch-detail-lead';
    const rest = document.createElement('span');
    rest.className = 'runs-launch-card__detail-rest';

    registerDetail?.(async () => {
      slot.className = 'runs-launch-card__detail is-pending';
      slot.replaceChildren('…');
      slot.removeAttribute('title');
      try {
        const { text, rest: restText = '', tone = 'muted', title = '' } = await detail();
        slot.className = 'runs-launch-card__detail';
        lead.className = `runs-launch-card__detail-lead is-${tone}`;
        lead.textContent = text;
        rest.textContent = restText;
        slot.replaceChildren(lead, rest);
        if (title) slot.title = title;
      } catch (err) {
        // One card's number failing is not worth a toast — the card still starts
        // runs, which is what it is for.
        Logger.warn('runsLauncher: card detail failed', err);
        slot.className = 'runs-launch-card__detail is-muted';
        slot.replaceChildren('count unavailable');
      }
    });
  }

  card.appendChild(startBtn);

  // Starting something you have not looked at in a while is exactly when you
  // want to check its settings first — and this is the only screen listing all
  // of them, so it is the natural jumping-off point.
  const configBtn = document.createElement('button');
  configBtn.type = 'button';
  configBtn.className = 'runs-launch-card__config';
  configBtn.dataset.testid = 'runs-launch-config';
  configBtn.dataset.tour = 'runs.launchCardConfig';
  configBtn.textContent = '⚙';
  configBtn.title = `Configure ${name}`;
  configBtn.setAttribute('aria-label', `Configure ${name}`);
  configBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    navigateTabs('tab-configuration', configTab, { selectUuid: configUuid });
  });
  card.appendChild(configBtn);

  startBtn.addEventListener('click', async () => {
    // Held for the whole round trip — including the confirm — because a second
    // click would otherwise stack a second modal, and past it start the run
    // twice. A training is expensive enough that that matters.
    //
    // Disabling the *button*, not the card: the card is a div now, where
    // `disabled` is an inert unknown property that silently does nothing.
    // Nothing this launcher starts is improved by a second concurrent copy:
    // two watchers on one profile open the same cameras and double-fire every
    // action (duplicate clips, duplicate MQTT, and a "stop" that stops one of
    // them); two trainings from one recipe contend for the GPU and race on the
    // same output model; a saved task run twice does its work twice. So the
    // guard is on by default and a card opts *out* of it, not in — the only
    // cards it cannot cover are the ones with no way to detect a live run
    // (a filter chain), where an unguarded press means "we cannot tell".
    //
    // Read from the badge rather than re-queried, so what happens matches what
    // the user could actually see when they pressed.
    if (preventDuplicate && liveLabel) {
      // A card that is already running is the wrong place to offer "start
      // again" and the right place to offer "show me it".
      try {
        livePress(liveLabel);
      } catch (err) {
        Logger.warn('runsLauncher: could not show the running instance', err);
        toast(err.message || `Could not show the run for ${name}`, 5000, 'error');
      }
      return;
    }
    startBtn.disabled = true;
    const original = top.textContent;
    try {
      if (confirm && !await showConfirmModal({
        title: confirm.title,
        message: confirm.message,
        confirmText: 'Start run',
        danger: true,
      })) {
        return;
      }

      top.textContent = `Starting ${name}…`;
      await start();
      toast(`Started ${name}`, 3000);
      onStarted?.();
    } catch (err) {
      Logger.warn('runsLauncher: start failed', err);
      toast(err.message || `Could not start ${name}`, 5000, 'error');
    } finally {
      top.textContent = original;
      startBtn.disabled = false;
    }
  });

  return card;
}

/**
 * @param {string} title
 * @param {Array<HTMLElement>} cards
 * @param {?{label: string, tab: string}} emptyAction - where to go to create one
 * @param {string} emptyText
 * @param {?{label: string, title: string, onClick: (btn: HTMLButtonElement) => any}} [headerAction]
 * @returns {HTMLElement}
 */
function group(title, cards, emptyText, emptyAction, headerAction = null) {
  const section = document.createElement('section');
  section.className = 'runs-launch-group';

  const heading = document.createElement('h4');
  heading.className = 'runs-launch-group__heading';
  heading.textContent = title;
  if (headerAction && cards.length) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'runs-launch-group__action';
    btn.dataset.testid = 'runs-launch-group-action';
    btn.textContent = headerAction.label;
    btn.title = headerAction.title;
    btn.setAttribute('aria-label', headerAction.title);
    btn.addEventListener('click', async () => {
      btn.disabled = true;
      try {
        await headerAction.onClick(btn);
      } finally {
        btn.disabled = false;
      }
    });
    heading.appendChild(btn);
  }
  section.appendChild(heading);

  if (cards.length) {
    const grid = document.createElement('div');
    grid.className = 'runs-launch-grid';
    cards.forEach(c => grid.appendChild(c));
    section.appendChild(grid);
    return section;
  }

  const empty = document.createElement('p');
  empty.className = 'runs-launch-empty';
  empty.append(`${emptyText} `);
  if (emptyAction) {
    const link = document.createElement('a');
    link.href = '#';
    link.textContent = emptyAction.label;
    link.addEventListener('click', (e) => {
      e.preventDefault();
      navigateTabs('tab-configuration', emptyAction.tab);
    });
    empty.appendChild(link);
  }
  section.appendChild(empty);
  return section;
}

/**
 * Render the launcher.
 *
 * @param {Object} opts
 * @param {string} [opts.target]
 * @param {() => void} [opts.onStarted] - called after any successful start,
 *   so the caller can show the Active pane where the run now appears.
 */
export async function renderRunsLauncher({ target = 'runs-launcher-box', onStarted = null } = {}) {
  const container = document.getElementById(target);
  if (!container) return;

  // Re-entrancy: two rapid tab activations would otherwise interleave two
  // renders and append both.
  if (_rendering) return;
  _rendering = true;

  try {
    const [trainings, chains, watchers, taskConfigs] = await Promise.all([
      settle(fetchTrainingConfigs),
      settle(fetchFilterChains),
      settle(fetchWatchProfiles),
      // Temporary configs are launcher scaffolding, not things to launch —
      // filtered server-side rather than in JS so the payload stays small.
      settle(() => fetchTaskConfigs(null, { excludeTemporary: true })),
    ]);

    const frag = document.createDocumentFragment();

    /** @type {Array<() => Promise<void>>} */
    const deltaLoaders = [];
    /** Replaced wholesale, so a stream message can never paint a stale card. */
    const runStateSetters = new Map();

    frag.appendChild(group(
      'Training',
      (trainings ?? []).map(cfg => launchCard({
        name: cfg.name,
        hint: 'Train a model from this recipe',
        start: () => launchRun({
          typename: 'TrainModelTask',
          name: `Train ${cfg.name}`,
          params: { training_config_uuid: cfg.uuid },
          // The training itself lives on the training server; restarting
          // tasks_server only detaches the watcher. Persistent lets the
          // recovered task re-attach to the still-running job by job_id.
          persistent: true,
        }),
        configTab: 'tab-training-management',
        configUuid: cfg.uuid,
        onStarted,
        detail: () => trainingDelta(cfg),
        registerDetail: (reload) => deltaLoaders.push(reload),
        registerRunState: (setBadge) => runStateSetters.set(cfg.uuid, setBadge),
        // Guarded by the default: two trainings from one recipe contend for the
        // GPU and race on the same output model.
      })),
      trainings === null ? 'Could not load training configs.' : 'No training configs yet —',
      trainings === null ? null : { label: 'define one', tab: 'tab-training-management' },
      {
        label: '↻',
        title: 'Recount images since the last training run',
        onClick: async () => {
          _deltaCache.clear();
          await runThrottled(deltaLoaders);
        },
      },
    ));

    frag.appendChild(group(
      'Filter chains',
      (chains ?? []).map(chain => launchCard({
        name: chain.name,
        // Only filter_chain_uuid is required by ApplyFilterChainTask; every
        // scope param (directory, dataset, tag, label) is optional, and omitting
        // them means the whole database. Say so, and confirm before starting —
        // a chain can discard and relabel images, so an unscoped one-click run
        // is not something to do by accident.
        hint: 'Runs over the whole database',
        confirm: {
          title: `Run ${chain.name} over everything?`,
          message: 'Starting it from here applies the chain to every image in the database, with no directory, dataset, tag or label scope. To scope a run, create a task config for it in Configuration → Tasks instead.',
        },
        start: () => launchRun({
          typename: 'ApplyFilterChainTask',
          name: `Run ${chain.name}`,
          params: { filter_chain_uuid: chain.uuid },
        }),
        configTab: 'tab-filterchain-management',
        configUuid: chain.uuid,
        onStarted,
      })),
      chains === null ? 'Could not load filter chains.' : 'No filter chains yet —',
      chains === null ? null : { label: 'define one', tab: 'tab-filterchain-management' },
    ));

    frag.appendChild(group(
      'Watchers',
      (watchers ?? []).map(profile => launchCard({
        name: profile.name,
        hint: 'Watch its sources and fire its rules',
        start: () => launchRun({
          typename: 'StreamWatcherTask',
          name: `Watch ${profile.name}`,
          params: { watch_profile_uuid: profile.uuid },
          // persistent deliberately omitted: StreamWatcherTask.DEFAULT_PERSISTENT
          // decides, so a watcher survives a service restart wherever it was
          // started from.
        }),
        configTab: 'tab-watcher-management',
        configUuid: profile.uuid,
        onStarted,
        registerRunState: (setBadge) => runStateSetters.set(profile.uuid, setBadge),
        // Guarded by the default: a second watcher on one profile is never what
        // was meant. Pressing one that is already watching opens the same
        // drill-down the NVR pane uses — "Watching now" invites a press, and
        // the useful answer to that press is the live view, which is more than
        // the default (a row on Active) can show.
        onLiveClick: ({ taskUuid, liveViewable }) => {
          if (!liveViewable || !taskUuid) {
            // A queued watcher has no frames yet and a paused one has stopped
            // producing them, so the modal would open empty. Fall back to the
            // default: the run itself, on the pane that lists it in that state.
            toast(`${profile.name} has no live view right now`, 4000);
            showRunOnActive(taskUuid);
            return;
          }
          openWatcherLiveView({ taskUuid, profileName: profile.name });
        },
      })),
      watchers === null ? 'Could not load watch profiles.' : 'No watch profiles yet —',
      watchers === null ? null : { label: 'define one', tab: 'tab-watcher-management' },
    ));

    // Saved task configs start as themselves — they are definitions the user
    // authored, so cloning one per press (as the temporary path does) would
    // litter the Tasks list with duplicates.
    frag.appendChild(group(
      'Saved tasks',
      (taskConfigs ?? []).map(cfg => launchCard({
        name: cfg.name,
        hint: cfg.typename,
        start: () => launchSavedConfig(cfg.uuid),
        configTab: 'tab-task-management',
        configUuid: cfg.uuid,
        onStarted,
        // Started as itself, so a live instance names this very config — the
        // card can both say it is already running and, via the default guard,
        // decline to start a second copy of work already in flight.
        registerRunState: (setBadge) => runStateSetters.set(cfg.uuid, setBadge),
      })),
      taskConfigs === null ? 'Could not load task configs.' : 'No saved tasks yet —',
      taskConfigs === null ? null : { label: 'define one', tab: 'tab-task-management' },
    ));

    container.replaceChildren(frag);
    _runStateByConfig = runStateSetters;

    // Seeded from one fetch, then kept fresh by the Runs tab handing over its
    // stream messages. Without the seed a card arrives blank and stays blank
    // until something happens to a task — precisely the case where it should
    // already read "Training in progress".
    fetchTasksStatus()
      .then(({ tasks }) => applyRunsLauncherTaskState(tasks))
      .catch(err => Logger.warn('runsLauncher: could not seed run state', err));

    // Deliberately not awaited: the launcher is usable the moment the cards are
    // in the DOM, and each card fills its own number in as it arrives. The
    // release of _rendering must not wait on it either, or a tab switch during
    // the fill would be swallowed by the re-entrancy guard.
    runThrottled(deltaLoaders).catch(err => Logger.warn('runsLauncher: delta pass failed', err));
  } finally {
    _rendering = false;
  }
}
