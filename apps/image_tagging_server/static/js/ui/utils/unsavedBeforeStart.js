import { showUnsavedBeforeStartModal } from '/app-static/js/ui/utils/confirmModal.js';

/**
 * Gate a "start / run" action on the editor's dirty state.
 *
 * The list-editor screens already guard *navigation* away from a dirty form
 * (listEditorScreen's `confirmIfDirty`), but the action buttons that launch
 * server-side work — Task Configs' "▶ Start", Training Runs' "▶ Start Training"
 * — read the config from the server, so pressing them with unsaved edits
 * quietly runs the previous settings. This is the same prompt shape for both.
 *
 * @param {Object} opts
 * @param {() => boolean} opts.getDirty - reads the panel bar's dirty flag.
 * @param {() => Promise<boolean>} opts.save - saves the form; resolves false if
 *   the save failed (validation error, server error), in which case the start
 *   is aborted rather than run against stale settings.
 * @param {string} [opts.actionText='Start'] - verb phrase used in the buttons.
 * @returns {Promise<boolean>} true = proceed with the start, false = abort.
 */
export async function confirmStartIfDirty({ getDirty, save, actionText = 'Start' }) {
  if (!getDirty()) return true;
  const choice = await showUnsavedBeforeStartModal({ actionText });
  if (choice === 'cancel') return false;
  if (choice === 'save') return await save();
  return true; // 'discard' — run the saved settings, deliberately
}
