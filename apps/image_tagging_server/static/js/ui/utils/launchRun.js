/**
 * Start a job, from anywhere.
 *
 * There is no "run" record in the API: starting something means creating a task
 * config and then starting an instance of it. For a one-off — a training, a
 * filter chain over a dataset — the config is pure scaffolding, so it is
 * written as `isTemporary` and hidden from the Tasks list behind that pane's
 * system/temporary toggle.
 *
 * This exists so Jobs and the config editors agree on how that is done. It used
 * to live inline in taskRunnerSection, which was the only caller back when the
 * only way to start a training was a button inside its config editor.
 */
import { createTaskConfig, startTask } from '/app-static/js/shared/api/tasks.js';

/**
 * @param {Object} opts
 * @param {string} opts.typename - Task class to run, e.g. 'TrainModelTask'.
 * @param {string} opts.name - Human-readable job name, shown in Jobs.
 * @param {Object} opts.params - Task params; the shape is the task's own.
 * @param {boolean} [opts.persistent] - Survive a server restart and auto-resume.
 *   Leave undefined to use the task type's own default: batch work stays
 *   non-persistent (a transient runner must never silently resume a training
 *   the user started by hand hours ago), while a watcher defaults to
 *   persistent, since a watcher that quietly stopped watching is the one
 *   failure an NVR must not have. Pass an explicit boolean to override.
 * @returns {Promise<string>} the started task instance's uuid
 */
export async function launchRun({ typename, name, params, persistent }) {
  const configUuid = await createTaskConfig({
    typename,
    name,
    params,
    persistent,
    // Scaffolding, not a definition the user authored — keep it out of the
    // Tasks list, which is for configs you meant to keep.
    isTemporary: true,
  });
  return startTask({ configUuid });
}

/**
 * Start an already-saved task config, without cloning it.
 *
 * The `isTemporary` path above would write a duplicate config every press; a
 * config the user actually authored is started directly.
 *
 * @param {string} configUuid
 * @returns {Promise<string>} the started task instance's uuid
 */
export async function launchSavedConfig(configUuid) {
  return startTask({ configUuid });
}
