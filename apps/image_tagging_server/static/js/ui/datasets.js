import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import {
  fetchDatasets, fetchDatasetsAndViews, createDataset, updateDataset, deleteDataset,
} from '/app-static/js/shared/api/datasets.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { toast, makeFieldRow, buildPanelEditorBar } from '/app-static/js/ui/utils/index.js';
import { showConfirmModal } from '/app-static/js/ui/utils/confirmModal.js';
import { createListEditorScreen } from '/app-static/js/ui/utils/listEditorScreen.js';
import { createFilterPanel } from '/app-static/js/app/review/filters.js';
import { reviewStateToDatasetForm, datasetFormToReviewState } from '/app-static/js/shared/filterUtils.js';

let _rendering = false;
let _unsubDatasets = null;
let _navGuardController = null;
let _pendingPrefill = null;

/** Queue filter values to pre-fill the next new-dataset form (consumed once on open). */
export function setPendingDatasetPrefill({ filterJson, labelUuids, name = '', description = '' }) {
  _pendingPrefill = { filterJson, labelUuids, name, description };
}

// Returns a human-readable one-line summary of a dataset's active filters for the list card.
function _datasetMetaSummary(dataset) {
  const parts = [];

  const labelCount = dataset.labelUuids?.length ?? 0;
  if (labelCount > 0) parts.push(`${labelCount} label${labelCount !== 1 ? 's' : ''}`);

  const f = dataset.filterJson ?? {};
  const terms = f.terms ?? [];

  const tagCount = terms
    .filter(t => t.kind === 'bboxTag')
    .reduce((n, t) => n + (t.tagUuids?.length ?? 0), 0);
  if (tagCount > 0) parts.push(`${tagCount} tag filter${tagCount !== 1 ? 's' : ''}`);

  const condCount = terms.filter(t => t.kind === 'imageLabel').length;
  if (condCount > 0) parts.push(`${condCount} image condition${condCount !== 1 ? 's' : ''}`);

  const extras = [];
  if (terms.some(t => t.kind === 'filename')) extras.push('filename');
  if (terms.some(t => t.kind === 'date')) extras.push('date range');
  if (f.verificationCheck) extras.push('verification');
  if (extras.length > 0) parts.push(extras.join(', '));

  return parts.length > 0 ? parts.join(' · ') : '(no filters)';
}

// -------------------------------------------------------
// Dataset form body — fields only (no buttons)
// Uses createFilterPanel for filter editing (labels + tags + date + filename + verification).
// Returns { getName, getDescription, getLabelUuids, getFilterJson, destroy }
// -------------------------------------------------------

function buildDatasetForm(container, { name = '', description = '', labelUuids = [], filterJson = {} }, labelsMap, tagsMap, { onAnyChange } = {}) {
  // Name
  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.value = name;
  nameInput.placeholder = 'Required';
  nameInput.addEventListener('input', () => onAnyChange?.());
  container.appendChild(makeFieldRow('Name', nameInput));

  // Description
  const descInput = document.createElement('input');
  descInput.type = 'text';
  descInput.value = description;
  descInput.placeholder = 'Optional';
  descInput.addEventListener('input', () => onAnyChange?.());
  container.appendChild(makeFieldRow('Description', descInput));

  // Filter panel (handles Tags + Labels + Date + Filename + Verification + Image Label Conditions)
  // inlineControls: Labels and Tags render as always-expanded scrollable boxes.
  const filterPanel = createFilterPanel(labelsMap, tagsMap, () => onAnyChange?.(), {
    inlineControls: true,
  });

  // One mount: every control, Tags and Labels included, is inside the panel.
  // They used to be handed out separately as `toolbarControls` and re-hosted
  // here in a `ds-inline-controls` row of the form's own making; they now live
  // in the panel's Labels & Tags row, which carries its own section headings.
  container.appendChild(filterPanel.element);

  // Populate from saved state
  filterPanel.setState(datasetFormToReviewState(filterJson, labelUuids));

  return {
    getName: () => nameInput.value.trim(),
    getDescription: () => descInput.value.trim(),
    getFormData: () => reviewStateToDatasetForm(filterPanel.getState()),
    getLabelUuids() { return this.getFormData().labelUuids; },
    getFilterJson() { return this.getFormData().filterJson; },
    setFilterState: (newFilterJson, newLabelUuids) => {
      filterPanel.setState(datasetFormToReviewState(newFilterJson, newLabelUuids));
      onAnyChange?.();
    },
    destroy: () => filterPanel.destroy(),
  };
}

// -------------------------------------------------------
// Entry point
// -------------------------------------------------------

export async function renderDatasetManager({ target = 'dataset-management-box' } = {}) {
  if (_rendering) return;
  _rendering = true;

  // Replace any previously-installed nav guard (handles re-render on tab revisit)
  _navGuardController?.abort();
  _navGuardController = new AbortController();

  const container = document.getElementById(target);
  if (!container) { _rendering = false; return; }
  container.innerHTML = '';

  try {
    // Warm all three caches in parallel so first panel open is near-instant
    await Promise.all([fetchLabels(), fetchTags(), fetchDatasetsAndViews()]);

    const screen = createListEditorScreen({
      container,
      prefix: 'ds',
      title: 'Datasets & Views',
      itemUuidAttr: 'datasetUuid',
      withEditorFooter: true,
      onNew: () => _showForm(null),
      ownTabId: 'tab-dataset-management',
      navGuardSignal: _navGuardController.signal,
    });
    const { listBody, editorBar, editorBody, editorFooter } = screen;
    const _confirmIfDirty = screen.confirmIfDirty;
    const _updateListSelection = screen.updateListSelection;

    // ── Shared state ──────────────────────────────────────────────────────
    let _selectedUuid = null;
    let _switching = false;
    let _formDestroyFn = null; // cleanup for active-filter subscription in current form

    // Single entry point for both edit (dataset != null) and create (dataset == null).
    async function _showForm(dataset, { force = false } = {}) {
      if (_switching && !force) return;
      _switching = true;
      try {
        if (!force && !await _confirmIfDirty()) return;
        _formDestroyFn?.();
        _formDestroyFn = null;
        screen.setBarHandle(null); // mark clean while async fetches are in flight
        screen.setDoSave(null);
        _selectedUuid = dataset?.uuid ?? null;

        // Re-fetch so pickers always reflect current data
        const [labelsMap, tagsMap] = await Promise.all([fetchLabels(), fetchTags()]);
        editorBody.innerHTML = '';
        editorFooter.innerHTML = '';

        // A row is a Dataset or a View depending on `promoted`; the editor is
        // shared, only its vocabulary and the promote control differ.
        const isView = dataset ? !dataset.promoted : false;
        const kindNoun = isView ? 'View' : 'Dataset';

        const barHandle = buildPanelEditorBar({
          container: editorBar,
          item: dataset,
          newTitle: 'New Dataset',
          deleteTitle: `Delete ${kindNoun.toLowerCase()}`,
          onSave: () => _doSave(true),
          onDelete: dataset ? async () => {
            await deleteDataset(dataset.uuid);
            toast(`${kindNoun} deleted`, 3000);
            // deleteDataset() fires DATASETS_UPDATED → subscription calls refreshList(); no need to call it again.
            await _showForm(null, { force: true });
          } : undefined,
        });
        screen.setBarHandle(barHandle);
        const { setDirty, setSaving, updateTitle, getDirty } = barHandle;

        // Render form fields into editorBody (destroy cleans up active-filter subscription)
        const prefill = dataset ? null : _pendingPrefill;
        if (!dataset) _pendingPrefill = null;
        const formHandle = buildDatasetForm(
          editorBody,
          {
            name: dataset?.name ?? prefill?.name ?? '',
            description: dataset?.description ?? prefill?.description ?? '',
            labelUuids: dataset?.labelUuids ?? prefill?.labelUuids ?? [],
            filterJson: dataset?.filterJson ?? prefill?.filterJson ?? {},
          },
          labelsMap,
          tagsMap,
          { onAnyChange: () => setDirty(true) },
        );
        _formDestroyFn = formHandle.destroy ?? null;

        // The filter panel's initial setState fires onAnyChange → setDirty(true).
        // Reset to clean so the Save/Create button starts disabled.
        setDirty(false);

        // Preview Images — lives in the sticky footer so it's always visible
        // regardless of how far the user has scrolled the form.
        const previewBtn = document.createElement('button');
        previewBtn.type = 'button';
        previewBtn.textContent = 'Preview Images';
        previewBtn.className = 'panel-btn';
        previewBtn.style.marginBottom = '0';
        previewBtn.addEventListener('click', async () => {
          previewBtn.disabled = true;
          previewBtn.textContent = 'Loading…';
          try {
            const { showReviewModal } = await import('/app-static/js/app/review/main.js');

            const unsavedNote = getDirty() ? ' — unsaved' : '';
            const previewTitle = `Preview: ${dataset?.name ?? 'New Dataset'}${unsavedNote}`;

            // For existing datasets: open an editable preview with live filter panel
            // that can push changes back to the in-memory form (user must still Save).
            // For new (unsaved) datasets: same editable mode.
            showReviewModal({
              title: previewTitle,
              editableDataset: {
                filterJson: formHandle.getFilterJson(),
                labelUuids: formHandle.getLabelUuids(),
                onUpdateFilters: (filterJson, labelUuids) => {
                  formHandle.setFilterState(filterJson, labelUuids);
                  setDirty(true);
                },
              },
            }).catch(err => toast(err.message || 'Failed to open preview', 5000, 'error'));
          } catch (err) {
            toast(err.message || 'Failed to preview', 5000, 'error');
          } finally {
            previewBtn.disabled = false;
            previewBtn.textContent = 'Preview Images';
          }
        });
        editorFooter.appendChild(previewBtn);

        // Promote / demote — the flag flip that turns a saved search into
        // something training can consume, and back. Only meaningful on a saved
        // row, so it is absent on the new-item form.
        if (dataset) {
          const promoteBtn = document.createElement('button');
          promoteBtn.type = 'button';
          promoteBtn.className = 'panel-btn secondary';
          promoteBtn.style.marginBottom = '0';
          promoteBtn.dataset.testid = 'dataset-promote-toggle';
          promoteBtn.textContent = isView ? '▲ Promote to Dataset' : '▼ Demote to View';
          promoteBtn.title = isView
            ? 'Make this View available to training configs and model pickers'
            : 'Remove this Dataset from training configs and model pickers';

          promoteBtn.addEventListener('click', async () => {
            // Promotion is a consequential action (decision #2) — it changes
            // what training is allowed to consume — so it is always confirmed.
            const ok = await showConfirmModal({
              title: isView ? 'Promote to Dataset' : 'Demote to View',
              message: isView
                ? `Promote "${dataset.name}" to a Dataset? It becomes selectable in training `
                  + 'configs and model pickers.'
                : `Demote "${dataset.name}" to a View? It stays saved and editable, but `
                  + 'training configs and model pickers will no longer offer it.',
              confirmText: isView ? 'Promote' : 'Demote',
              danger: !isView,
            });
            if (!ok) return;

            // Unsaved form edits are unrelated to the flag and would be lost on
            // the re-render below, so settle them the same way tab navigation does.
            if (!await _confirmIfDirty()) return;

            promoteBtn.disabled = true;
            try {
              const updated = await updateDataset(dataset.uuid, { promoted: isView });
              toast(isView ? `"${updated.name}" promoted to Dataset` : `"${updated.name}" demoted to View`, 3000);
              // Re-open on the updated record so the editor's vocabulary, the
              // promote button and the list section all reflect the new kind.
              await _showForm(updated, { force: true });
            } catch (err) {
              // The server refuses demotion while a training config references
              // the row — surface that reason rather than a generic failure.
              toast(err.message || 'Failed to change promotion', 6000, 'error');
              promoteBtn.disabled = false;
            }
          });
          editorFooter.appendChild(promoteBtn);
        }

        // Shared save logic used by both the Save button (navigateAfterCreate=true)
        // and the nav-guard (screen.setDoSave, navigateAfterCreate=false).
        async function _doSave(navigateAfterCreate = false) {
          const name = formHandle.getName();
          if (!name) { toast('Name is required', 3000, 'error'); return false; }
          setSaving(true);
          try {
            if (dataset) {
              const updated = await updateDataset(dataset.uuid, {
                name,
                description: formHandle.getDescription(),
                filterJson: formHandle.getFilterJson(),
                labelUuids: formHandle.getLabelUuids(),
              });
              setDirty(false);
              setSaving(false);
              updateTitle(updated.name);
              toast(`${kindNoun} saved`, 3000);
              const nameEl = listBody.querySelector(`[data-dataset-uuid="${dataset.uuid}"] .ds-list-item__name`);
              if (nameEl) nameEl.textContent = updated.name;
            } else {
              const created = await createDataset({
                name,
                description: formHandle.getDescription(),
                filterJson: formHandle.getFilterJson(),
                labelUuids: formHandle.getLabelUuids(),
                // "+ New" on this screen means a Dataset. Stated explicitly
                // rather than leaning on the server default: with one table and
                // two nouns, every creating path must name the flag (PR 5.1).
                promoted: true,
              });
              setDirty(false);
              toast(`Dataset "${created.name}" created`, 3000);
              // refreshList() is triggered by the DATASETS_UPDATED event fired inside
              // createDataset(); calling it again here would race and duplicate list items.
              if (navigateAfterCreate) {
                await _showForm(created, { force: true });
              } else {
                setSaving(false);
              }
            }
            return true;
          } catch (err) {
            toast(err.message, 5000, 'error');
            setSaving(false);
            return false;
          }
        }

        screen.setDoSave(() => _doSave(false));
        _updateListSelection(dataset?.uuid ?? null);
      } finally {
        _switching = false;
      }
    }

    function _renderListSection(title, rows, emptyText) {
      const header = document.createElement('div');
      header.className = 'ds-list-section__header';
      header.textContent = `${title} (${rows.length})`;
      header.dataset.testid = `dataset-section-${title.toLowerCase()}`;
      listBody.appendChild(header);

      if (rows.length === 0) {
        const empty = document.createElement('p');
        empty.className = 'ds-list-empty';
        empty.textContent = emptyText;
        listBody.appendChild(empty);
        return;
      }

      for (const dataset of rows) {
        const item = document.createElement('div');
        item.className = 'ds-list-item';
        item.dataset.datasetUuid = dataset.uuid;
        item.dataset.promoted = String(!!dataset.promoted);

        const nameLine = document.createElement('div');
        nameLine.className = 'ds-list-item__name';
        nameLine.textContent = dataset.name;
        item.appendChild(nameLine);

        if (dataset.description) {
          const descLine = document.createElement('div');
          descLine.className = 'ds-list-item__desc';
          descLine.textContent = dataset.description;
          item.appendChild(descLine);
        }

        const metaLine = document.createElement('div');
        metaLine.className = 'ds-list-item__meta';
        metaLine.textContent = _datasetMetaSummary(dataset);
        item.appendChild(metaLine);

        item.addEventListener('click', () => _showForm(dataset));
        listBody.appendChild(item);
      }
    }

    async function refreshList() {
      listBody.innerHTML = '';

      // Both nouns live in one table separated by `promoted` (decision #10), and
      // this is the one screen that shows both — promotion is meaningless if you
      // can't see what you're promoting from.
      const allRows = await fetchDatasetsAndViews();
      const datasets = [...allRows.values()].filter(d => d.promoted);
      const views = [...allRows.values()].filter(d => !d.promoted);

      _renderListSection('Datasets', datasets, 'No datasets yet.');
      _renderListSection('Views', views, 'No saved Views yet — save one from the Review tab.');

      _updateListSelection(_selectedUuid);
    }

    // Replace any previous subscription so we don't double-refresh on tab re-activation
    _unsubDatasets?.();
    _unsubDatasets = events.subscribe(EventTypes.DATASETS_UPDATED, async () => {
      try { await refreshList(); } catch (err) { toast(err.message || 'Failed to refresh datasets', 5000, 'error'); }
    });

    await refreshList();
    await _showForm(null, { force: true });
  } catch (err) {
    const errMsg = document.createElement('p');
    errMsg.style.color = 'red';
    errMsg.textContent = `Error loading datasets: ${err.message}`;
    container.appendChild(errMsg);
  } finally {
    _rendering = false;
  }
}
