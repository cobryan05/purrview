/**
 * What a running task was launched *from*.
 *
 * The launcher starts these through a throwaway task config carrying the
 * definition's uuid, so that param is the only link between a running instance
 * and the recipe that defines it. Two screens need that link and must agree
 * about it:
 *
 *  - Jobs → Start a job, to badge a card whose definition is already running
 *    (`badge`), and to open the live view for the one that is;
 *  - Jobs → Running, to send you from a running instance back to the recipe it
 *    is running (`configTab`).
 *
 * Kinds without a `badge` are still navigable — a filter chain has a definition
 * worth jumping to, it just has no wording worth putting on a launch card.
 */

/** Configuration sub-tab holding the saved task configs themselves. */
export const TASK_CONFIG_TAB = 'tab-task-management';

export const RUN_KINDS = {
  TrainModelTask: {
    paramKey: 'training_config_uuid',
    configTab: 'tab-training-management',
    badge: { running: 'Training in progress', paused: 'Training paused' },
  },
  StreamWatcherTask: {
    paramKey: 'watch_profile_uuid',
    configTab: 'tab-watcher-management',
    badge: { running: 'Watching now', paused: 'Watcher paused' },
  },
  ApplyFilterChainTask: {
    paramKey: 'filter_chain_uuid',
    configTab: 'tab-filterchain-management',
  },
};

/**
 * Wording for a saved task config that is running as itself.
 *
 * Saved tasks are not launched through a throwaway config carrying a
 * definition's uuid — the launcher starts them directly (launchSavedConfig), so
 * the running instance *is* an instance of the card's own config and the link
 * is `config.uuid`, not a param. Deliberately generic: a saved task can be any
 * task type, so there is no single true verb to put here the way "Training in
 * progress" is true of every TrainModelTask.
 */
export const SAVED_TASK_BADGE = { running: 'Running now', paused: 'Paused' };

/**
 * Where "show me what this run is running" should land.
 *
 * Prefers the *definition* (training config, watch profile, filter chain) over
 * the task config, because that is the thing a user edits and recognises — the
 * task config behind a launcher-started run is throwaway scaffolding with a
 * generated name. Falls back to the task config for everything else, and to
 * null when even that is temporary: sending someone to a Tasks list that hides
 * temporary configs by default would land them on a blank editor.
 *
 * @param {?{typename?: string, uuid?: string, params?: object, isTemporary?: boolean}} config
 *   the `config` block of a task-status entry
 * @returns {?{tab: string, uuid: string}}
 */
export function configTargetForTask(config) {
  if (!config) return null;
  const kind = RUN_KINDS[config.typename];
  const definitionUuid = kind ? config.params?.[kind.paramKey] : null;
  if (definitionUuid) return { tab: kind.configTab, uuid: definitionUuid };
  if (config.isTemporary || !config.uuid) return null;
  return { tab: TASK_CONFIG_TAB, uuid: config.uuid };
}
