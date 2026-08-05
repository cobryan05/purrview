import { showUnsavedConfigChangesModal } from '/app-static/js/ui/utils/confirmModal.js';

/**
 * Intercept tab navigation away from a list-editor screen while it holds
 * unsaved changes, offering Save / Discard / Cancel first.
 *
 * Extracted from four byte-identical copies (`ui/datasets.js`,
 * `ui/modelConfigs.js`, `ui/trainingConfigs.js`, `ui/filterChains.js`) that
 * differed only in which tab id counts as "staying put".
 *
 * Runs in the **capture** phase so it can stop propagation before the tab
 * system sees the click; on success it re-fires `btn.click()` now that the
 * guard is satisfied.
 *
 * @param {Object} opts
 * @param {string} opts.ownTabId - the screen's own sub-tab; clicking it is not navigating away.
 * @param {string} [opts.parentTabId] - the enclosing main tab, likewise not navigating away.
 * @param {AbortSignal} opts.signal - removes the listener, and marks an in-flight
 *   prompt stale: if a re-render supersedes this guard mid-prompt, `signal.aborted`
 *   is true and the answer is discarded rather than applied to the new editor.
 * @param {() => boolean} opts.isDirty - read at event time, so it sees the latest editor state.
 * @param {() => Promise<boolean>} opts.save - returns false to abort navigation (e.g. validation failed).
 * @param {() => void} opts.markClean - clear the dirty flag once the user has chosen.
 */
export function installUnsavedNavGuard({
  ownTabId,
  parentTabId = 'tab-configuration',
  signal,
  isDirty,
  save,
  markClean,
}) {
  document.addEventListener('click', async (e) => {
    const btn = e.target.closest('.workspace-tab-button, .main-tab-button');
    if (!btn) return;
    const targetTab = btn.dataset.tab;
    if (targetTab === ownTabId || targetTab === parentTabId) return;
    if (!isDirty()) return;

    e.stopPropagation();
    const result = await showUnsavedConfigChangesModal();
    if (result === 'cancel') return;
    if (result === 'save') {
      const ok = await save();
      if (!ok) return;
    }
    if (signal.aborted) return; // guard superseded by a re-render; answer is stale
    markClean();
    btn.click(); // re-fire now that the guard is satisfied
  }, { capture: true, signal });
}
