/**
 * Launching a blip harvest, and meeting its result.
 *
 * Shared by the two surfaces that start one: a single event's blips panel, and the
 * events grid's selection bar. Both want the same three things after the click — a
 * toast naming the durable way back, a live wait on the task, and the preview-triage
 * modal opened on the run the moment frames land — and a harvest whose frames are
 * unreachable is the failure this exists to prevent.
 */
import { Logger, toast } from '/app-static/js/ui/utils/index.js';
import { launchRun } from '/app-static/js/ui/utils/launchRun.js';
import { TaskStatus, taskResultData } from '/app-static/js/shared/api/tasks.js';
import { subscribeTaskStream } from '/app-static/js/shared/api/taskStream.js';

export const HARVEST_TYPENAME = 'HarvestBlipFramesTask';

/**
 * Start a harvest over a set of blips and open its frames when it finishes.
 *
 * @param {Object} opts
 * @param {string[]} opts.blipUuids - must already be filtered to harvestable blips; the
 *   task refuses tracker faults rather than silently dropping them.
 * @param {string} opts.name - the Jobs card's name.
 * @param {string} [opts.notice] - the toast shown at launch.
 * @returns {Promise<string>} the task uuid.
 */
export async function startHarvest({ blipUuids, name, notice = '' }) {
  const taskUuid = await launchRun({
    typename: HARVEST_TYPENAME,
    name,
    params: { blip_uuids: blipUuids },
  });
  // Said even though the run also auto-opens: the watch below lives only as long as
  // this page, so someone who navigates away still needs a route back.
  toast(
    notice
      || `Harvesting frames from ${blipUuids.length} incident(s) — the review panel will open when it's done, `
        + 'or find them any time under the "Harvested" image tag',
    7000,
  );
  watchHarvest(taskUuid);
  return taskUuid;
}

/**
 * Open the preview-triage modal as soon as `taskUuid` completes.
 *
 * Best-effort: a live SSE subscription tied to the current page, not a durable wait. If
 * the tab closes the harvest still finishes and is reached through the toast's routes.
 */
export function watchHarvest(taskUuid) {
  const unsubscribe = subscribeTaskStream({
    onMessage: async (data) => {
      const taskInfo = data.tasks?.[taskUuid];
      if (!taskInfo) return;
      const status = taskInfo.instance?.status;
      if (status !== TaskStatus.COMPLETED && status !== TaskStatus.ERROR) return;
      unsubscribe();

      if (status === TaskStatus.ERROR) {
        toast(`Harvest failed: ${taskInfo.instance?.errorMessage || 'unknown error'}`, 6000, 'error');
        return;
      }
      // `.data`, not the result itself: the task's dict arrives wrapped in a
      // `{ status, data }` envelope, and reading past the envelope reports "no frames"
      // for a run that filed plenty.
      const runUuid = taskResultData(taskInfo.instance).previewRunUuid;
      if (!runUuid) {
        toast('Harvest finished — no frames to review', 5000);
        return;
      }
      try {
        const { openPreviewRunModal } = await import('/app-static/js/app/review/previewPanel.js');
        await openPreviewRunModal({ runUuid });
      } catch (err) {
        Logger.error(err);
      }
    },
    onError: (err) => Logger.error(err),
  });
  return unsubscribe;
}
