import { showConfirmModal } from '/app-static/js/ui/utils/confirmModal.js';
import { toast } from '/app-static/js/ui/utils/logging.js';

/**
 * Populates a panel editor header bar with a dirty indicator, title, and
 * Save / Delete buttons. Manages the save-in-flight and dirty state
 * internally; returns getters/setters so the caller can read and update them.
 *
 * @param {{
 *   container: HTMLElement,     the panel-box__bar element to clear and populate
 *   item: object | null,        the entity being edited (must have .name); null = new
 *   newTitle: string,           title text shown when item is null, e.g. 'New Dataset'
 *   deleteTitle: string,        confirm-modal title, e.g. 'Delete dataset'
 *   onSave: () => void,         called when the Save/Create button is clicked
 *   onDelete?: () => Promise<void>,  called after the user confirms deletion
 * }} opts
 * @returns {{
 *   setDirty(v: boolean): void,
 *   setSaving(v: boolean): void,
 *   getDirty(): boolean,
 *   updateTitle(name: string): void,
 * }}
 */
export function buildPanelEditorBar({ container, item, newTitle, deleteTitle, onSave, onDelete }) {
  container.innerHTML = '';
  let _dirty = false;
  let _saving = false;

  const dirtyDot = document.createElement('span');
  dirtyDot.className = 'panel-dirty-dot';

  const titleSpan = document.createElement('span');
  titleSpan.className = 'panel-box__title';
  titleSpan.textContent = item ? item.name : newTitle;

  // A 7px dot next to an uppercase title was too quiet to notice before hitting
  // an action button (Start / Start Training), so unsaved state also says so in
  // words and lights the Save button up.
  const dirtyLabel = document.createElement('span');
  dirtyLabel.className = 'panel-dirty-label';
  dirtyLabel.textContent = 'Unsaved changes';
  dirtyLabel.dataset.testid = 'panel-dirty-label';
  dirtyLabel.hidden = true;

  const saveBtn = document.createElement('button');
  saveBtn.type = 'button';
  saveBtn.textContent = item ? 'Save' : 'Create';
  saveBtn.className = 'panel-btn';
  saveBtn.disabled = true;
  saveBtn.addEventListener('click', onSave);

  const setDirty = (v) => {
    _dirty = v;
    dirtyDot.classList.toggle('visible', v);
    dirtyLabel.hidden = !v;
    saveBtn.classList.toggle('panel-btn--attention', v);
    // Don't re-enable the button while a save is in flight — setDirty(true) fires
    // on every keystroke and would otherwise allow a second concurrent submission.
    if (!_saving) saveBtn.disabled = !v;
  };
  const setSaving = (v) => {
    _saving = v;
    // Always disable while saving; restore to dirty-based state when done.
    saveBtn.disabled = v || !_dirty;
  };
  const getDirty = () => _dirty;
  const updateTitle = (name) => { titleSpan.textContent = name; };

  container.append(dirtyDot, titleSpan, dirtyLabel);

  if (item && onDelete) {
    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.textContent = 'Delete';
    deleteBtn.className = 'panel-btn secondary';
    deleteBtn.addEventListener('click', async () => {
      // Disable save during the async confirm to prevent a concurrent save+delete.
      setSaving(true);
      const ok = await showConfirmModal({
        title: deleteTitle,
        message: `Delete "${item.name}"? This cannot be undone.`,
        confirmText: 'Delete',
        cancelText: 'Cancel',
        danger: true,
      });
      if (!ok) { setSaving(false); return; }
      deleteBtn.disabled = true;
      try {
        await onDelete();
      } catch (err) {
        toast(err.message, 5000, 'error');
        deleteBtn.disabled = false;
        setSaving(false);
      }
    });
    container.appendChild(deleteBtn);
  }

  container.appendChild(saveBtn);

  return { setDirty, setSaving, getDirty, updateTitle };
}
