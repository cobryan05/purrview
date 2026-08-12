// ================= Sources Manager =================
//
// A list-beside-editor screen (ui/utils/listEditorScreen.js), the same shape as
// Datasets / Model Shapes / Training Runs / Filter Chains.
//
// It used to be a `.config-column` holding the source list *and* a permanently
// expanded "Create Source" form, with editing done inline inside a list row.
// That put the schema form in half a pane, under everything else, so filling one
// in meant scrolling past every existing source first. Here the list is a narrow
// column, the form owns the rest of the pane, and Save/Delete/Test live in bars
// that don't scroll away.

import { runPreviewSourceTest, showPaneError, toast } from './utils/index.js';
import { buildPanelEditorBar } from '/app-static/js/ui/utils/panelEditorBar.js';
import { createListEditorScreen } from '/app-static/js/ui/utils/listEditorScreen.js';
import { createSource, deleteSources, fetchSourceList, updateSource } from '/app-static/js/shared/api/sources.js';
import { FIELD_EDIT_EVENT, SourceConfigField } from '/app-static/js/ui/utils/fields/index.js';

let _rendering = false;
let _navGuardController = null;

/**
 * Mark the form dirty on any edit anywhere inside it.
 *
 * The field classes (SourceConfigField → SchemaField → the per-type fields) each
 * own their own `onChange`, and there is no single hook that fires for all of
 * them — SchemaField's `onChange` isn't even wired by SourceConfigField. Rather
 * than thread a callback through four layers of field plumbing, listen once on
 * the container: every control these fields build is a real `<input>`/`<select>`/
 * `<textarea>`, so delegation catches all of them, including controls added
 * later when the provider dropdown swaps the schema.
 *
 * The exception is button-driven edits (ArrayField's add/delete/reorder), which
 * fire no native event — those fields dispatch FIELD_EDIT_EVENT themselves.
 *
 * @param {HTMLElement} root
 * @param {() => void} onEdit
 */
function watchForEdits(root, onEdit) {
  root.addEventListener('input', onEdit);
  root.addEventListener('change', onEdit);
  root.addEventListener(FIELD_EDIT_EVENT, onEdit);
}

/**
 * Renders the UI for managing sources (list, create, edit, delete, test).
 * @param {Object} options
 * @param {string} options.target - ID of the container to render into.
 */
export async function renderSourceManager({ target = 'source-management-box' } = {}) {
  if (_rendering) return;
  _rendering = true;

  _navGuardController?.abort();
  _navGuardController = new AbortController();

  const container = document.getElementById(target);
  if (!container) { _rendering = false; return; }
  container.innerHTML = '';

  try {
    const screen = createListEditorScreen({
      container,
      prefix: 'src',
      title: 'Sources',
      itemUuidAttr: 'sourceUuid',
      withEditorFooter: true,
      searchable: true,
      searchPlaceholder: 'Filter sources…',
      newLabel: '+ New Source',
      onNew: () => _showForm(null),
      ownTabId: 'tab-source-management',
      navGuardSignal: _navGuardController.signal,
    });
    const { listBody, editorBar, editorBody, editorFooter } = screen;

    let _selectedUuid = null;
    let _switching = false;

    async function _showForm(source, { force = false } = {}) {
      if (_switching && !force) return;
      _switching = true;
      try {
        if (!force && !await screen.confirmIfDirty()) return;
        screen.setBarHandle(null);
        screen.setDoSave(null);
        _selectedUuid = source?.uuid ?? null;

        editorBody.innerHTML = '';
        editorFooter.innerHTML = '';

        const barHandle = buildPanelEditorBar({
          container: editorBar,
          item: source,
          newTitle: 'New Source',
          deleteTitle: 'Delete source',
          onSave: () => _doSave(true),
          onDelete: source ? async () => {
            await deleteSources({ uuids: [source.uuid] });
            toast('Source deleted', 3000);
            await refreshList();
            await _showForm(null, { force: true });
          } : undefined,
        });
        screen.setBarHandle(barHandle);
        const { setDirty, setSaving, updateTitle } = barHandle;

        // Building the field is async; a second click during the fetch would
        // otherwise append a second form into the same body.
        const field = await SourceConfigField.create({
          value: source
            ? {
              sourceName: source.name,
              provider: source.typename,
              uuid: source.uuid,
              providerParams: source.params,
            }
            : { sourceName: '', provider: '', providerParams: {} },
          editMode: true,
        });

        const formWrap = document.createElement('div');
        formWrap.className = 'src-form';
        formWrap.appendChild(await field.renderEdit());
        editorBody.appendChild(formWrap);

        // The provider dropdown replaces the schema form in place, which counts
        // as an edit — delegation on the wrapper keeps catching it afterwards.
        watchForEdits(formWrap, () => setDirty(true));

        // Test output belongs to the source being edited, so it renders under
        // the form rather than in a column of its own. runPreviewSourceTest
        // looks this element up by id.
        const preview = document.createElement('div');
        preview.id = 'source-preview-box';
        preview.className = 'src-preview';
        editorBody.appendChild(preview);

        setDirty(false);

        const testBtn = document.createElement('button');
        testBtn.type = 'button';
        testBtn.textContent = 'Test Source';
        testBtn.title = 'Test Source';
        testBtn.className = 'panel-btn secondary';
        testBtn.addEventListener('click', async () => {
          const { provider, providerParams } = field.getValue();
          await runPreviewSourceTest({ provider, providerParams });
        });
        editorFooter.appendChild(testBtn);

        async function _doSave(navigateAfterCreate = false) {
          const { sourceName, provider, providerParams } = field.getValue();
          if (!sourceName) { toast('Name is required', 3000, 'error'); return false; }
          if (!provider) { toast('Provider is required', 3000, 'error'); return false; }
          setSaving(true);
          try {
            if (source) {
              await updateSource({ uuid: source.uuid, name: sourceName, provider, providerParams });
              setDirty(false);
              setSaving(false);
              updateTitle(sourceName);
              toast('Source saved', 3000);
              await refreshList();
            } else {
              // createSource returns the full created record and writes it into
              // the sources cache, so refreshList() below already sees it.
              const created = await createSource({ name: sourceName, provider, providerParams });
              setDirty(false);
              toast(`Source "${sourceName}" created`, 3000);
              await refreshList();
              if (navigateAfterCreate) {
                await _showForm(created, { force: true });
              } else {
                setSaving(false);
              }
            }
            return true;
          } catch (err) {
            toast(err.message || 'Failed to save source', 5000, 'error');
            setSaving(false);
            return false;
          }
        }

        screen.setDoSave(() => _doSave(false));
        screen.updateListSelection(source?.uuid ?? null);
      } finally {
        _switching = false;
      }
    }

    async function refreshList() {
      listBody.innerHTML = '';
      // fetchSourceList() returns a Map, not an array — same trap as
      // fetchLabels() (see CLAUDE.md). Iterate .values(), size not length.
      const sources = await fetchSourceList();
      screen.setCount(sources.size);

      if (sources.size === 0) {
        listBody.appendChild(screen.createListEmpty('No sources yet.'));
      }

      for (const source of sources.values()) {
        listBody.appendChild(screen.createListItem({
          uuid: source.uuid,
          name: source.name,
          meta: source.typename ?? '',
          onClick: () => _showForm(source),
        }));
      }

      screen.applyFilter();
      screen.updateListSelection(_selectedUuid);
    }

    await refreshList();
    await _showForm(null, { force: true });
  } catch (err) {
    showPaneError(container, err.message);
  } finally {
    _rendering = false;
  }
}
