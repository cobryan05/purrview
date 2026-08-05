import { showUnsavedConfigChangesModal } from '/app-static/js/ui/utils/confirmModal.js';
import { installUnsavedNavGuard } from '/app-static/js/ui/utils/unsavedNavGuard.js';

/**
 * Assemble the list-beside-editor screen shared by Datasets, Model Blueprints,
 * Training Runs and Filter Chains.
 *
 * All four hand-built the identical structure — a `panel-box <prefix>-list-panel`
 * with `__bar`/`__title`/`__body`/`__footer` beside a `panel-box <prefix>-editor`
 * with `__bar`/`__body` (+ optional `__footer`) — and then re-declared the same
 * three pieces of editor bookkeeping on top of it: the `_barHandle` /
 * `_currentDoSave` pair, the `_confirmIfDirty()` prompt that reads them, and the
 * `installUnsavedNavGuard` call that reads them again.
 *
 * The screens keep their own `<prefix>-*` classes; `panel-box` is the shared CSS
 * primitive and the prefixes carry each screen's own width/height rules, so this
 * extracts **assembly and bookkeeping only** — no CSS is unified here.
 *
 * @param {Object} opts
 * @param {HTMLElement} opts.container - emptied by the caller; the split is appended to it.
 * @param {string} opts.prefix - class prefix, e.g. `'ds'` → `.ds-list-panel` / `.ds-editor`.
 *   Training Runs deliberately passes `'mc'`: it shares Model Blueprints' CSS.
 * @param {string} opts.title - list panel title.
 * @param {string} opts.itemUuidAttr - the `dataset.*` key list rows carry their uuid in,
 *   e.g. `'datasetUuid'` — read by `updateListSelection`.
 * @param {boolean} [opts.withEditorFooter=false] - add a sticky editor footer for
 *   always-visible action buttons. Only the screens that have footer actions ask for one.
 * @param {() => void} opts.onNew - click handler for the footer's `+ New` button.
 * @param {string} opts.ownTabId - this screen's sub-tab id, for the nav guard.
 * @param {AbortSignal} opts.navGuardSignal - the caller's re-render controller signal.
 * @returns {Object} the assembled elements plus the editor-state bookkeeping:
 *   `setBarHandle` / `setDoSave` (assigned at different points in a form build, so
 *   two setters rather than one), `confirmIfDirty()` and `updateListSelection(uuid)`.
 */
export function createListEditorScreen({
  container,
  prefix,
  title,
  itemUuidAttr,
  withEditorFooter = false,
  onNew,
  ownTabId,
  navGuardSignal,
}) {
  const split = document.createElement('div');
  split.className = 'list-editor-split';
  container.appendChild(split);

  // ── Left: list panel ────────────────────────────────────────────────────
  const listPanel = document.createElement('div');
  listPanel.className = `panel-box ${prefix}-list-panel`;
  split.appendChild(listPanel);

  const listBar = document.createElement('div');
  listBar.className = 'panel-box__bar';
  const listTitle = document.createElement('span');
  listTitle.className = 'panel-box__title';
  listTitle.textContent = title;
  listBar.appendChild(listTitle);
  listPanel.appendChild(listBar);

  const listBody = document.createElement('div');
  listBody.className = 'panel-box__body';
  listPanel.appendChild(listBody);

  const listFooter = document.createElement('div');
  listFooter.className = 'panel-box__footer';
  listPanel.appendChild(listFooter);

  // Footer "+ New" button is static — created once here, not on every refreshList.
  const newBtn = document.createElement('button');
  newBtn.type = 'button';
  newBtn.textContent = '+ New';
  newBtn.className = 'panel-btn panel-btn--block';
  newBtn.addEventListener('click', () => onNew());
  listFooter.appendChild(newBtn);

  // ── Right: editor panel ─────────────────────────────────────────────────
  const editorPanel = document.createElement('div');
  editorPanel.className = `panel-box ${prefix}-editor`;
  split.appendChild(editorPanel);

  const editorBar = document.createElement('div');
  editorBar.className = 'panel-box__bar';
  editorPanel.appendChild(editorBar);

  const editorBody = document.createElement('div');
  editorBody.className = 'panel-box__body';
  editorPanel.appendChild(editorBody);

  // Sticky footer: always-visible action buttons regardless of scroll position
  let editorFooter = null;
  if (withEditorFooter) {
    editorFooter = document.createElement('div');
    editorFooter.className = 'panel-box__footer';
    editorPanel.appendChild(editorFooter);
  }

  // ── Editor state ────────────────────────────────────────────────────────
  // The active form's bar handle (dirty flag) and its save-only fn. Both are
  // nulled while a form's async fetches are in flight, hence the optional calls.
  let _barHandle = null;
  let _doSave = null;

  const isDirty = () => !!_barHandle?.getDirty();

  /** Prompt when switching away from a dirty form. False = caller should abort. */
  async function confirmIfDirty() {
    if (!isDirty()) return true;
    const result = await showUnsavedConfigChangesModal();
    if (result === 'cancel') return false;
    if (result === 'save') return await (_doSave?.() ?? false);
    return true; // 'discard'
  }

  function updateListSelection(uuid) {
    for (const item of listBody.querySelectorAll(`.${prefix}-list-item`)) {
      item.classList.toggle('active', item.dataset[itemUuidAttr] === uuid);
    }
  }

  installUnsavedNavGuard({
    ownTabId,
    signal: navGuardSignal,
    isDirty,
    save: () => _doSave?.(),
    markClean: () => _barHandle?.setDirty(false),
  });

  return {
    split,
    listPanel,
    listBar,
    listTitle,
    listBody,
    listFooter,
    newBtn,
    editorPanel,
    editorBar,
    editorBody,
    editorFooter,
    setBarHandle: (handle) => { _barHandle = handle; },
    setDoSave: (fn) => { _doSave = fn; },
    isDirty,
    confirmIfDirty,
    updateListSelection,
  };
}
