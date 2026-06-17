import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import {
  fetchDatasets, createDataset, updateDataset, deleteDataset,
} from '/app-static/js/shared/api/datasets.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { toast, makeFieldRow, buildPanelEditorBar } from '/app-static/js/ui/utils/index.js';
import { showUnsavedConfigChangesModal } from '/app-static/js/ui/utils/confirmModal.js';
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

  // Filter panel (handles Tags + Labels + Date + Filename + Verification + Image Tag Conditions)
  // inlineControls: Labels and Tags render as always-expanded scrollable boxes.
  // allowImageTagUnset: false — datasets are for curated training data; unset has no meaning there.
  const filterPanel = createFilterPanel(labelsMap, tagsMap, () => onAnyChange?.(), {
    inlineControls: true,
    allowImageTagUnset: false,
  });

  // Date + Verification + Image Tag Conditions rows are in filterPanel.element
  container.appendChild(filterPanel.element);

  // Tags and Labels as side-by-side expanded boxes below
  const inlineRow = document.createElement('div');
  inlineRow.className = 'ds-inline-controls';
  const [tagsEl, labelsEl] = filterPanel.toolbarControls;
  for (const [title, el] of [['Tags', tagsEl], ['Labels', labelsEl]]) {
    const box = document.createElement('div');
    box.className = 'ds-inline-control';
    const lbl = document.createElement('div');
    lbl.className = 'ds-inline-control__label';
    lbl.textContent = title;
    box.appendChild(lbl);
    box.appendChild(el);
    inlineRow.appendChild(box);
  }
  container.appendChild(inlineRow);

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
    await Promise.all([fetchLabels(), fetchTags(), fetchDatasets()]);

    const split = document.createElement('div');
    split.style.cssText = 'display: flex; gap: 1em; align-items: flex-start;';
    container.appendChild(split);

    // ── Left: list panel ──────────────────────────────────────────────────
    const listPanel = document.createElement('div');
    listPanel.className = 'panel-box ds-list-panel';
    split.appendChild(listPanel);

    const listBar = document.createElement('div');
    listBar.className = 'panel-box__bar';
    const listTitle = document.createElement('span');
    listTitle.className = 'panel-box__title';
    listTitle.textContent = 'Datasets';
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
    newBtn.className = 'panel-btn';
    newBtn.style.cssText = 'width: 100%; margin-bottom: 0;';
    newBtn.addEventListener('click', () => _showForm(null));
    listFooter.appendChild(newBtn);

    // ── Right: editor panel ───────────────────────────────────────────────
    const editorPanel = document.createElement('div');
    editorPanel.className = 'panel-box ds-editor';
    split.appendChild(editorPanel);

    const editorBar = document.createElement('div');
    editorBar.className = 'panel-box__bar';
    editorPanel.appendChild(editorBar);

    const editorBody = document.createElement('div');
    editorBody.className = 'panel-box__body';
    editorPanel.appendChild(editorBody);

    // Sticky footer: always-visible action buttons regardless of scroll position
    const editorFooter = document.createElement('div');
    editorFooter.className = 'panel-box__footer';
    editorPanel.appendChild(editorFooter);

    // ── Shared state ──────────────────────────────────────────────────────
    let _selectedUuid = null;
    let _barHandle = null; // set by _showForm; tracks dirty state via getDirty()/setDirty()
    let _currentDoSave = null;
    let _switching = false;
    let _formDestroyFn = null; // cleanup for active-filter subscription in current form

    async function _confirmIfDirty() {
      if (!_barHandle?.getDirty()) return true;
      const result = await showUnsavedConfigChangesModal();
      if (result === 'cancel') return false;
      if (result === 'save') return await (_currentDoSave?.() ?? false);
      return true; // 'discard'
    }

    function _updateListSelection(uuid) {
      for (const item of listBody.querySelectorAll('.ds-list-item')) {
        item.classList.toggle('active', item.dataset.datasetUuid === uuid);
      }
    }

    // Single entry point for both edit (dataset != null) and create (dataset == null).
    async function _showForm(dataset, { force = false } = {}) {
      if (_switching && !force) return;
      _switching = true;
      try {
        if (!force && !await _confirmIfDirty()) return;
        _formDestroyFn?.();
        _formDestroyFn = null;
        _barHandle = null; // mark clean while async fetches are in flight
        _currentDoSave = null;
        _selectedUuid = dataset?.uuid ?? null;

        // Re-fetch so pickers always reflect current data
        const [labelsMap, tagsMap] = await Promise.all([fetchLabels(), fetchTags()]);
        editorBody.innerHTML = '';
        editorFooter.innerHTML = '';

        _barHandle = buildPanelEditorBar({
          container: editorBar,
          item: dataset,
          newTitle: 'New Dataset',
          deleteTitle: 'Delete dataset',
          onSave: () => _doSave(true),
          onDelete: dataset ? async () => {
            await deleteDataset(dataset.uuid);
            toast('Dataset deleted', 3000);
            await refreshList();
            await _showForm(null, { force: true });
          } : undefined,
        });
        const { setDirty, setSaving, updateTitle, getDirty } = _barHandle;

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

        // Shared save logic used by both the Save button (navigateAfterCreate=true)
        // and the nav-guard (_currentDoSave, navigateAfterCreate=false).
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
              toast('Dataset saved', 3000);
              const nameEl = listBody.querySelector(`[data-dataset-uuid="${dataset.uuid}"] .ds-list-item__name`);
              if (nameEl) nameEl.textContent = updated.name;
            } else {
              const created = await createDataset({
                name,
                description: formHandle.getDescription(),
                filterJson: formHandle.getFilterJson(),
                labelUuids: formHandle.getLabelUuids(),
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

        _currentDoSave = () => _doSave(false);
        _updateListSelection(dataset?.uuid ?? null);
      } finally {
        _switching = false;
      }
    }

    async function refreshList() {
      listBody.innerHTML = '';

      const datasetsMap = await fetchDatasets();

      if (datasetsMap.size === 0) {
        const empty = document.createElement('p');
        empty.style.cssText = 'color: var(--pico-muted-color, #888); font-style: italic; font-size: 0.85em; padding: 0.5em 0.75em; margin: 0;';
        empty.textContent = 'No datasets yet.';
        listBody.appendChild(empty);
      }

      for (const dataset of datasetsMap.values()) {
        const item = document.createElement('div');
        item.className = 'ds-list-item';
        item.dataset.datasetUuid = dataset.uuid;

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
        metaLine.textContent = `${dataset.labelUuids?.length ?? 0} labels`;
        item.appendChild(metaLine);

        item.addEventListener('click', () => _showForm(dataset));
        listBody.appendChild(item);
      }

      _updateListSelection(_selectedUuid);
    }

    // ── Navigation guard ─────────────────────────────────────────────────
    // Intercept workspace-tab and main-tab clicks while there are unsaved changes.
    const guardSignal = _navGuardController.signal;
    document.addEventListener('click', async (e) => {
      const btn = e.target.closest('.workspace-tab-button, .main-tab-button');
      if (!btn) return;
      const targetTab = btn.dataset.tab;
      // Allow clicking the Datasets tab itself or the Configuration main tab (no navigation away)
      if (targetTab === 'tab-dataset-management' || targetTab === 'tab-configuration') return;
      if (!_barHandle?.getDirty()) return;

      e.stopPropagation();
      const result = await showUnsavedConfigChangesModal();
      if (result === 'cancel') return;
      if (result === 'save') {
        const ok = await _currentDoSave?.();
        if (!ok) return;
      }
      if (guardSignal.aborted) return; // guard superseded by a re-render; discard is stale
      _barHandle?.setDirty(false);
      btn.click(); // re-fire now that guard is satisfied
    }, { capture: true, signal: guardSignal });

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
