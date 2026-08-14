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
import { fetchTaskConfigs } from '/app-static/js/shared/api/tasks.js';
import { launchRun, launchSavedConfig } from '/app-static/js/ui/utils/launchRun.js';
import { navigateTabs } from '/app-static/js/app/utils/tabs.js';
import { showConfirmModal } from '/app-static/js/ui/utils/confirmModal.js';
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

/**
 * One launchable thing.
 *
 * @param {Object} opts
 * @param {string} opts.name
 * @param {string} opts.hint - what it will do, in the user's terms
 * @param {() => Promise<string>} opts.start - resolves to the task instance uuid
 * @param {() => void} opts.onStarted
 * @param {?{title: string, message: string}} [opts.confirm] - gate the start
 *   behind a confirmation, for runs that are destructive or unscoped.
 * @returns {HTMLElement}
 */
function launchCard({ name, hint, start, onStarted, confirm = null }) {
  const card = document.createElement('button');
  card.type = 'button';
  card.className = 'runs-launch-card';
  card.dataset.tour = 'runs.launchCard';
  card.dataset.runName = name;

  const top = document.createElement('span');
  top.className = 'runs-launch-card__name';
  top.textContent = name;
  card.appendChild(top);

  const sub = document.createElement('span');
  sub.className = 'runs-launch-card__hint';
  sub.textContent = hint;
  card.appendChild(sub);

  card.addEventListener('click', async () => {
    // Held for the whole round trip — including the confirm — because a second
    // click would otherwise stack a second modal, and past it start the run
    // twice. A training is expensive enough that that matters.
    card.disabled = true;
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
      card.disabled = false;
    }
  });

  return card;
}

/**
 * @param {string} title
 * @param {Array<HTMLElement>} cards
 * @param {?{label: string, tab: string}} emptyAction - where to go to create one
 * @param {string} emptyText
 * @returns {HTMLElement}
 */
function group(title, cards, emptyText, emptyAction) {
  const section = document.createElement('section');
  section.className = 'runs-launch-group';

  const heading = document.createElement('h4');
  heading.className = 'runs-launch-group__heading';
  heading.textContent = title;
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

    frag.appendChild(group(
      'Training',
      (trainings ?? []).map(cfg => launchCard({
        name: cfg.name,
        hint: 'Train a model from this recipe',
        start: () => launchRun({
          typename: 'TrainModelTask',
          name: `Train ${cfg.name}`,
          params: { training_config_uuid: cfg.uuid },
        }),
        onStarted,
      })),
      trainings === null ? 'Could not load training configs.' : 'No training configs yet —',
      trainings === null ? null : { label: 'define one', tab: 'tab-training-management' },
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
        }),
        onStarted,
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
        onStarted,
      })),
      taskConfigs === null ? 'Could not load task configs.' : 'No saved tasks yet —',
      taskConfigs === null ? null : { label: 'define one', tab: 'tab-task-management' },
    ));

    container.replaceChildren(frag);
  } finally {
    _rendering = false;
  }
}
