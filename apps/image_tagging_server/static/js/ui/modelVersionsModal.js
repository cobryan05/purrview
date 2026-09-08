import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';
import { showConfirmModal } from '/app-static/js/ui/utils/confirmModal.js';
import { toast } from '/app-static/js/ui/utils/index.js';
import { fetchModelVersions, restoreModelVersion, notifyModelsChanged } from '/app-static/js/shared/api/models.js';
import { parseApiDate } from '/app-static/js/shared/time.js';

function _fmtBytes(n) {
  if (!Number.isFinite(n) || n <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return `${v >= 10 || i === 0 ? Math.round(v) : v.toFixed(1)} ${units[i]}`;
}

function _fmtWhen(iso) {
  const d = parseApiDate(iso);
  return d ? d.toLocaleString() : 'unknown date';
}

/**
 * A model's previous generations, and the way back to one of them.
 *
 * Each training run archives the weights it replaces (see
 * `apps/helpers/modelVersions.py`), so this list is the model's history. How many
 * entries it can hold is the "Keep N previous versions" setting — this modal
 * shows what survived that bound, which is why an empty list is a normal state
 * and says so rather than looking broken.
 *
 * @param {string} modelName
 */
export async function showModelVersionsModal(modelName) {
  const { dialog, open, close } = createModalShell();
  dialog.dataset.testid = 'model-versions-modal';
  dialog.style.minWidth = 'min(36em, 90vw)';

  const title = document.createElement('strong');
  title.textContent = `History — ${modelName}`;
  dialog.appendChild(title);

  const body = document.createElement('div');
  body.style.cssText = 'margin: 0.6em 0; display: grid; gap: 0.4em;';
  dialog.appendChild(body);

  const btnRow = document.createElement('div');
  btnRow.style.cssText = 'display: flex; gap: 0.5em; justify-content: flex-end;';
  const closeBtn = document.createElement('button');
  closeBtn.type = 'button';
  closeBtn.className = 'secondary';
  closeBtn.textContent = 'Close';
  closeBtn.addEventListener('click', () => close());
  btnRow.appendChild(closeBtn);
  dialog.appendChild(btnRow);

  async function render() {
    body.innerHTML = '';
    body.textContent = 'Loading…';

    let versions;
    try {
      versions = await fetchModelVersions(modelName);
    } catch (err) {
      body.textContent = err.message || 'Failed to load history';
      return;
    }

    body.innerHTML = '';

    const current = document.createElement('div');
    current.dataset.testid = 'model-versions-current';
    current.style.cssText = 'font-size: 0.85em; padding-bottom: 0.3em;';
    current.textContent = versions.length === 0
      ? 'No previous versions yet — nothing has replaced this model since it was created.'
      : `${versions.length} previous version${versions.length === 1 ? '' : 's'} kept. `
        + 'Restoring one archives the model that is live now, so this is reversible.';
    body.appendChild(current);

    for (const version of versions) {
      const row = document.createElement('div');
      row.className = 'model-version-row';
      row.dataset.testid = 'model-version-row';
      row.dataset.versionId = version.versionId;
      row.style.cssText = 'display: flex; align-items: center; gap: 0.6em; flex-wrap: wrap; '
        + 'padding: 0.35em 0; border-top: 1px solid var(--pico-muted-border-color, #333);';

      const when = document.createElement('div');
      when.style.cssText = 'flex: 1 1 14em; min-width: 0;';
      const primary = document.createElement('div');
      // Trained-at is the mtime of the weights themselves; archived-at is when
      // they stopped being live. The first is what identifies a generation.
      primary.textContent = `Trained ${_fmtWhen(version.createdAt)}`;
      const secondary = document.createElement('small');
      secondary.style.cssText = 'opacity: 0.75;';
      secondary.textContent = `Replaced ${_fmtWhen(version.archivedAt)} · ${_fmtBytes(version.bytes)}`;
      when.append(primary, secondary);

      const restoreBtn = document.createElement('button');
      restoreBtn.type = 'button';
      restoreBtn.className = 'secondary';
      restoreBtn.dataset.testid = 'model-version-restore';
      restoreBtn.textContent = 'Restore';
      restoreBtn.addEventListener('click', async () => {
        const ok = await showConfirmModal({
          title: 'Restore this version?',
          message: `"${modelName}" goes back to the weights trained ${_fmtWhen(version.createdAt)}. `
            + 'The version live now is archived first, so you can undo this. Anything currently '
            + 'running inference picks up the restored weights on its next call.',
          confirmText: 'Restore',
        });
        if (!ok) return;
        restoreBtn.disabled = true;
        try {
          await restoreModelVersion(modelName, version.versionId);
          notifyModelsChanged();
          toast(`Restored "${modelName}" to its ${_fmtWhen(version.createdAt)} weights`, 5000);
          await render();
        } catch (err) {
          toast(err.message || 'Failed to restore version', 6000, 'error');
          restoreBtn.disabled = false;
        }
      });

      row.append(when, restoreBtn);
      body.appendChild(row);
    }
  }

  open();
  await render();
}
