/**
 * The Runs tab's live-work counter.
 *
 * Lives here rather than inside `app/runs/main.js` because the badge is visible
 * from every tab but Runs' own init only happens the first time someone opens
 * the tab. Seeding it at page load is the whole point: on a fresh load the badge
 * used to read blank until you visited Runs, which is exactly backwards — the
 * number is there to tell you something is running *without* going to look.
 *
 * The count is "everything under Active that has not finished": jobs and
 * continuous runs alike. It used to exclude continuous work (a watcher stays
 * armed for weeks, so it pins the badge at >= 1) but that made the number
 * unexplainable from the screen it describes — Active lists a running watcher
 * and the badge silently disagreed. A steady 1 is honest; the tooltip breaks it
 * down into jobs vs continuous for anyone who cares which kind is which.
 */
import { fetchTasksStatus } from '/app-static/js/shared/api/tasks.js';
import { countLiveTasks, countLiveTasksByKind } from '/app-static/js/ui/tasks.js';

/**
 * Both places the same count appears: the main Runs tab button (visible from
 * anywhere in the app) and the Active sub-tab, which is the pane the count is
 * actually about — from inside Runs the main-tab badge is out of the eyeline.
 */
const BADGE_IDS = ['task-activity-badge', 'task-activity-badge-active'];

function _plural(n, word) {
  return `${n} ${word}${n === 1 ? '' : 's'}`;
}

/**
 * Human-readable breakdown for the badge tooltip, e.g. "1 job, 2 continuous
 * runs". Answers "why does it say 3?" without opening the tab.
 *
 * @param {{jobs: number, continuous: number}} counts
 * @returns {string}
 */
function _badgeTitle({ jobs, continuous }) {
  const parts = [];
  if (jobs) parts.push(_plural(jobs, 'job'));
  if (continuous) parts.push(`${_plural(continuous, 'continuous run')}`);
  if (!parts.length) return 'Nothing running';
  return `${parts.join(', ')} — click to see them`;
}

/**
 * Paint the badge from a task-status map.
 *
 * Hidden at zero rather than showing "0": a permanent badge stops being a
 * signal.
 *
 * @param {Record<string, any>} tasks
 * @returns {number} the count that was painted
 */
export function updateRunsBadge(tasks) {
  const n = countLiveTasks(tasks);
  const title = _badgeTitle(countLiveTasksByKind(tasks));
  for (const id of BADGE_IDS) {
    const badge = document.getElementById(id);
    if (!badge) continue;
    badge.textContent = String(n);
    badge.hidden = n === 0;
    badge.title = title;
  }
  return n;
}

/**
 * Seed the badge from a one-off fetch.
 *
 * Called at page load (before Runs has ever been opened) and again whenever
 * Runs regains focus. SSE keeps the badge fresh while the stream is open, but
 * the stream only pushes *on change* — so without this the badge stays blank on
 * arrival until something happens to a task, which is precisely the case where
 * it should already read "2".
 *
 * @param {?(tasks: Record<string, any>) => void} onTasks
 *   Optional: receives the fetched task map instead of the default paint. Runs
 *   passes its own `updateBadge`, which paints *and* re-derives whether its SSE
 *   stream is still worth holding — a seed that only painted left that count
 *   stale at zero and dropped the stream on the way out of the tab.
 * @returns {Promise<void>}
 */
export async function seedRunsBadge(onTasks = null) {
  try {
    const { tasks } = await fetchTasksStatus();
    if (onTasks) onTasks(tasks);
    else updateRunsBadge(tasks);
  } catch {
    // A failed poll is not worth a toast; the SSE stream corrects it later.
  }
}
