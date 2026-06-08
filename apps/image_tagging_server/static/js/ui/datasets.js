import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import {
  fetchDatasets, createDataset, updateDataset, deleteDataset,
} from '/app-static/js/shared/api/datasets.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { toast, makeFieldRow, buildPanelEditorBar } from '/app-static/js/ui/utils/index.js';
import { showUnsavedConfigChangesModal } from '/app-static/js/ui/utils/confirmModal.js';
import { activeModel } from '/app-static/js/shared/activeModel.js';

let _rendering = false;
let _unsubDatasets = null;
let _navGuardController = null;
let _pendingPrefill = null;

/** Queue filter values to pre-fill the next new-dataset form (consumed once on open). */
export function setPendingDatasetPrefill({ conditions, labelUuids, name = '', description = '' }) {
  _pendingPrefill = { conditions, labelUuids, name, description };
}

// -------------------------------------------------------
// Label checkbox picker
// -------------------------------------------------------

function renderLabelCheckboxList(container, labelsMap, selectedUuids, onChange) {
  container.innerHTML = '';
  const selected = new Set(selectedUuids);

  const controls = document.createElement('div');
  controls.style.marginBottom = '4px';
  const selectAllBtn = document.createElement('button');
  selectAllBtn.type = 'button';
  selectAllBtn.textContent = 'Select All';
  selectAllBtn.className = 'panel-btn';
  selectAllBtn.style.marginRight = '4px';
  const clearAllBtn = document.createElement('button');
  clearAllBtn.type = 'button';
  clearAllBtn.textContent = 'Clear All';
  clearAllBtn.className = 'panel-btn';
  controls.appendChild(selectAllBtn);
  controls.appendChild(clearAllBtn);
  container.appendChild(controls);

  const list = document.createElement('div');
  list.className = 'panel-tree-box';
  container.appendChild(list);

  // Build hierarchy from labelsMap (Map<uuid, label>)
  const byParent = new Map();
  for (const label of labelsMap.values()) {
    const key = label.parentUuid ?? null;
    if (!byParent.has(key)) byParent.set(key, []);
    byParent.get(key).push(label);
  }
  for (const children of byParent.values()) {
    children.sort((a, b) => a.name.localeCompare(b.name));
  }

  function renderGroup(parentUuid, indent) {
    for (const label of (byParent.get(parentUuid) ?? [])) {
      const row = document.createElement('label');
      row.style.cssText = `display:flex; align-items:center; gap:6px; padding:2px 0; padding-left:${indent * 16}px; cursor:pointer; user-select:none;`;

      const swatch = document.createElement('span');
      swatch.style.cssText = `width:12px; height:12px; border-radius:2px; flex-shrink:0; background:${label.color || '#ccc'};`;

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = selected.has(label.uuid);
      cb.addEventListener('change', () => {
        if (cb.checked) selected.add(label.uuid);
        else selected.delete(label.uuid);
        onChange([...selected]);
      });

      row.appendChild(swatch);
      row.appendChild(cb);
      row.appendChild(document.createTextNode(' ' + label.name));
      list.appendChild(row);
      renderGroup(label.uuid, indent + 1);
    }
  }

  renderGroup(null, 0);

  selectAllBtn.addEventListener('click', () => {
    for (const cb of list.querySelectorAll('input[type=checkbox]')) cb.checked = true;
    for (const label of labelsMap.values()) selected.add(label.uuid);
    onChange([...selected]);
  });
  clearAllBtn.addEventListener('click', () => {
    for (const cb of list.querySelectorAll('input[type=checkbox]')) cb.checked = false;
    selected.clear();
    onChange([]);
  });
}

// -------------------------------------------------------
// Condition description helper
// -------------------------------------------------------

function describeCondition(cond, tagsMap) {
  switch (cond.type) {
    case 'image_age': {
      const field = cond.field === 'updated_at' ? 'Updated' : 'Created';
      if (cond.withinDays) return `Image ${field} within ${cond.withinDays} days`;
      if (cond.after && cond.before) return `Image ${field} between ${cond.after.slice(0, 10)} and ${cond.before.slice(0, 10)}`;
      if (cond.after) return `Image ${field} after ${cond.after.slice(0, 10)}`;
      if (cond.before) return `Image ${field} before ${cond.before.slice(0, 10)}`;
      return `Image ${field} (unspecified)`;
    }
    case 'bbox_age': {
      if (cond.withinDays) return `BBox updated within ${cond.withinDays} days`;
      if (cond.after && cond.before) return `BBox updated between ${cond.after.slice(0, 10)} and ${cond.before.slice(0, 10)}`;
      if (cond.after) return `BBox updated after ${cond.after.slice(0, 10)}`;
      if (cond.before) return `BBox updated before ${cond.before.slice(0, 10)}`;
      return 'BBox updated (unspecified)';
    }
    case 'tag': {
      const tagName = tagsMap.get(cond.tagUuid)?.name ?? cond.tagUuid;
      return cond.exclude ? `Tag absent: ${tagName}` : `Tag present: ${tagName}`;
    }
    case 'dataset_label_status':
      return cond.check === 'all_verified' ? 'Label status: all verified' : 'Label status: exclude uncertain';
    case 'filename':
      return `Filename contains: "${cond.pattern}"`;
    default:
      return JSON.stringify(cond);
  }
}

// -------------------------------------------------------
// Conditions editor (inline, expandable)
// -------------------------------------------------------

function renderConditionsEditor(parentEl, initialConditions, tagsMap, { onChange } = {}) {
  let conditions = [...initialConditions];

  const heading = document.createElement('p');
  heading.className = 'panel-section-label';
  heading.style.marginTop = '0.6em';
  heading.textContent = 'Filter Conditions';
  parentEl.appendChild(heading);

  const listEl = document.createElement('div');
  parentEl.appendChild(listEl);

  function renderList() {
    listEl.innerHTML = '';
    if (conditions.length === 0) {
      const empty = document.createElement('p');
      empty.style.cssText = 'color:var(--pico-muted-color,#888); font-style:italic; font-size:0.85em; margin:0;';
      empty.textContent = 'No conditions — all images included.';
      listEl.appendChild(empty);
      return;
    }
    for (let i = 0; i < conditions.length; i++) {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex; align-items:center; gap:8px; margin-bottom:4px; padding:4px 8px; background:var(--pico-card-background-color); border:1px solid var(--pico-border-color,#ddd); border-radius:4px;';

      const label = document.createElement('span');
      label.style.flex = '1';
      label.textContent = describeCondition(conditions[i], tagsMap);
      row.appendChild(label);

      const rmBtn = document.createElement('button');
      rmBtn.type = 'button';
      rmBtn.textContent = '✕';
      rmBtn.className = 'panel-btn secondary';
      const idx = i;
      rmBtn.addEventListener('click', () => {
        conditions.splice(idx, 1);
        renderList();
        onChange?.();
      });
      row.appendChild(rmBtn);
      listEl.appendChild(row);
    }
  }

  renderList();

  // "Add condition" expandable section
  const addSection = document.createElement('details');
  addSection.style.marginTop = '6px';
  const summary = document.createElement('summary');
  summary.style.cssText = 'cursor:pointer; font-weight:bold; font-size:0.85em;';
  summary.textContent = '+ Add Filter Condition';
  addSection.appendChild(summary);
  parentEl.appendChild(addSection);

  const typeRow = document.createElement('div');
  typeRow.style.marginTop = '6px';
  const typeSelect = document.createElement('select');
  typeSelect.innerHTML = `
    <option value="">— Choose type —</option>
    <option value="image_age">Image age</option>
    <option value="bbox_age">BBox age</option>
    <option value="tag_present">Tag present</option>
    <option value="tag_absent">Tag absent</option>
    <option value="label_status_all_verified">Label status: all verified</option>
    <option value="label_status_exclude_uncertain">Label status: exclude uncertain</option>
    <option value="filename">Filename contains</option>
  `;
  typeRow.appendChild(typeSelect);
  addSection.appendChild(typeRow);

  const hintEl = document.createElement('p');
  hintEl.style.cssText = 'font-size:0.82em; color:var(--pico-muted-color,#888); margin:4px 0 0; display:none;';
  addSection.appendChild(hintEl);

  const inputsEl = document.createElement('div');
  inputsEl.style.marginTop = '6px';
  addSection.appendChild(inputsEl);

  const _conditionHints = {
    image_age: 'Filters images by when they were created or last updated.',
    bbox_age: 'Filters images that have a bounding box created/updated within the given window.',
    tag_present: 'Includes only images that have this tag.',
    tag_absent: 'Excludes images that have this tag.',
    label_status_all_verified: 'Includes only images where every dataset label is verified. Click Add to confirm.',
    label_status_exclude_uncertain: 'Excludes images with any unverified (uncertain) label. Click Add to confirm.',
    filename: 'Includes images whose filename contains the given text (case-insensitive).',
  };

  function resetAddForm() {
    typeSelect.value = '';
    inputsEl.innerHTML = '';
    hintEl.style.display = 'none';
    hintEl.textContent = '';
    addSection.open = false;
  }

  typeSelect.addEventListener('change', () => {
    inputsEl.innerHTML = '';
    const type = typeSelect.value;
    if (!type) {
      hintEl.style.display = 'none';
      hintEl.textContent = '';
      return;
    }
    hintEl.textContent = _conditionHints[type] || '';
    hintEl.style.display = hintEl.textContent ? '' : 'none';

    switch (type) {
      case 'image_age':
      case 'bbox_age': {
        let fieldSelect = null;
        if (type === 'image_age') {
          const fieldRow = document.createElement('div');
          fieldRow.style.marginBottom = '4px';
          fieldSelect = document.createElement('select');
          fieldSelect.innerHTML = '<option value="updated_at">Updated at</option><option value="created_at">Created at</option>';
          fieldRow.appendChild(document.createTextNode('Field: '));
          fieldRow.appendChild(fieldSelect);
          inputsEl.appendChild(fieldRow);
        }

        const modeRow = document.createElement('div');
        modeRow.style.marginBottom = '4px';
        const modeSelect = document.createElement('select');
        modeSelect.innerHTML = '<option value="within_days">Within N days</option><option value="range">Date range</option>';
        modeRow.appendChild(document.createTextNode('Mode: '));
        modeRow.appendChild(modeSelect);
        inputsEl.appendChild(modeRow);

        const valueEl = document.createElement('div');
        valueEl.style.marginBottom = '4px';
        inputsEl.appendChild(valueEl);

        function renderValueInputs(mode) {
          valueEl.innerHTML = '';
          if (mode === 'within_days') {
            const inp = document.createElement('input');
            inp.type = 'number';
            inp.min = '1';
            inp.placeholder = '60';
            inp.dataset.key = 'within_days';
            inp.style.width = '80px';
            valueEl.appendChild(document.createTextNode('Days: '));
            valueEl.appendChild(inp);
          } else {
            const afterInp = document.createElement('input');
            afterInp.type = 'date';
            afterInp.dataset.key = 'after';
            const beforeInp = document.createElement('input');
            beforeInp.type = 'date';
            beforeInp.dataset.key = 'before';
            valueEl.appendChild(document.createTextNode('After: '));
            valueEl.appendChild(afterInp);
            valueEl.appendChild(document.createTextNode('  Before: '));
            valueEl.appendChild(beforeInp);
          }
        }
        renderValueInputs(modeSelect.value);
        modeSelect.addEventListener('change', () => renderValueInputs(modeSelect.value));

        const addBtn = document.createElement('button');
        addBtn.type = 'button';
        addBtn.textContent = 'Add';
        addBtn.className = 'panel-btn';
        addBtn.addEventListener('click', () => {
          const condType = type === 'image_age' ? 'image_age' : 'bbox_age';
          const cond = { type: condType, field: fieldSelect?.value ?? 'updated_at' };
          if (modeSelect.value === 'within_days') {
            const v = parseInt(inputsEl.querySelector('[data-key="within_days"]')?.value);
            if (!v || v < 1) { toast('Enter a valid number of days', 5000, 'error'); return; }
            cond.withinDays = v;
          } else {
            const a = inputsEl.querySelector('[data-key="after"]')?.value;
            const b = inputsEl.querySelector('[data-key="before"]')?.value;
            if (!a && !b) { toast('Enter at least one date', 5000, 'error'); return; }
            if (a) cond.after = a;
            if (b) cond.before = b;
          }
          conditions.push(cond);
          renderList();
          resetAddForm();
          onChange?.();
        });
        inputsEl.appendChild(addBtn);
        break;
      }

      case 'tag_present':
      case 'tag_absent': {
        const exclude = type === 'tag_absent';
        const tagSel = document.createElement('select');
        tagSel.innerHTML = '<option value="">— Choose tag —</option>';
        for (const tag of tagsMap.values()) {
          const opt = document.createElement('option');
          opt.value = tag.uuid;
          opt.textContent = tag.name;
          tagSel.appendChild(opt);
        }
        const row = document.createElement('div');
        row.style.marginBottom = '4px';
        row.appendChild(document.createTextNode('Tag: '));
        row.appendChild(tagSel);
        inputsEl.appendChild(row);

        const addBtn = document.createElement('button');
        addBtn.type = 'button';
        addBtn.textContent = 'Add';
        addBtn.className = 'panel-btn';
        addBtn.addEventListener('click', () => {
          if (!tagSel.value) { toast('Choose a tag', 5000, 'error'); return; }
          conditions.push({ type: 'tag', tagUuid: tagSel.value, exclude });
          renderList();
          resetAddForm();
          onChange?.();
        });
        inputsEl.appendChild(addBtn);
        break;
      }

      case 'label_status_all_verified':
      case 'label_status_exclude_uncertain': {
        const check = type === 'label_status_all_verified' ? 'all_verified' : 'exclude_uncertain';
        const addBtn = document.createElement('button');
        addBtn.type = 'button';
        addBtn.textContent = 'Add';
        addBtn.className = 'panel-btn';
        addBtn.addEventListener('click', () => {
          conditions.push({ type: 'dataset_label_status', check });
          renderList();
          resetAddForm();
          onChange?.();
        });
        inputsEl.appendChild(addBtn);
        break;
      }

      case 'filename': {
        const inp = document.createElement('input');
        inp.type = 'text';
        inp.placeholder = 'e.g. .jpg';
        const row = document.createElement('div');
        row.style.marginBottom = '4px';
        row.appendChild(document.createTextNode('Pattern: '));
        row.appendChild(inp);
        inputsEl.appendChild(row);

        const addBtn = document.createElement('button');
        addBtn.type = 'button';
        addBtn.textContent = 'Add';
        addBtn.className = 'panel-btn';
        addBtn.addEventListener('click', () => {
          if (!inp.value.trim()) { toast('Enter a pattern', 5000, 'error'); return; }
          conditions.push({ type: 'filename', pattern: inp.value.trim() });
          renderList();
          resetAddForm();
          onChange?.();
        });
        inputsEl.appendChild(addBtn);
        break;
      }
    }
  });

  return {
    getConditions: () => [...conditions],
    setConditions: (newConditions) => {
      conditions = [...newConditions];
      renderList();
    },
  };
}

// -------------------------------------------------------
// Dataset form body — fields only (no buttons)
// Returns { getName, getDescription, getLabelUuids, getConditions }
// -------------------------------------------------------

function buildDatasetForm(container, { name = '', description = '', labelUuids = [], filterConditions = [] }, labelsMap, tagsMap, { onAnyChange } = {}) {
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

  // Label picker
  const labelHeading = document.createElement('p');
  labelHeading.className = 'panel-section-label';
  labelHeading.textContent = 'Labels';
  container.appendChild(labelHeading);

  let selectedLabelUuids = [...labelUuids];

  const labelFilterNotice = document.createElement('p');
  labelFilterNotice.className = 'label-filter-notice';
  labelFilterNotice.hidden = true;
  container.appendChild(labelFilterNotice);

  const labelPickerEl = document.createElement('div');
  container.appendChild(labelPickerEl);

  function rerenderLabelPicker() {
    const displayMap = activeModel.applyLabelFilter(labelsMap);
    const hidden = labelsMap.size - displayMap.size;
    labelFilterNotice.textContent = `⚠ ${hidden} label${hidden !== 1 ? 's' : ''} hidden by model filter`;
    labelFilterNotice.hidden = hidden === 0;
    renderLabelCheckboxList(labelPickerEl, displayMap, selectedLabelUuids, (uuids) => {
      selectedLabelUuids = uuids;
      onAnyChange?.();
    });
  }
  rerenderLabelPicker();

  // Re-render picker when the active model filter toggle changes
  const unsubActiveFilter = events.subscribe(EventTypes.ACTIVE_MODEL_CHANGED, rerenderLabelPicker);

  // Conditions
  const condEditor = renderConditionsEditor(container, filterConditions, tagsMap, { onChange: onAnyChange });

  return {
    getName: () => nameInput.value.trim(),
    getDescription: () => descInput.value.trim(),
    getLabelUuids: () => selectedLabelUuids,
    getConditions: () => condEditor.getConditions(),
    setLabelUuids: (uuids) => {
      selectedLabelUuids = [...uuids];
      rerenderLabelPicker();
      onAnyChange?.();
    },
    setConditions: (newConditions) => {
      condEditor.setConditions(newConditions);
      onAnyChange?.();
    },
    destroy: () => unsubActiveFilter(),
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
            filterConditions: dataset?.filterJson?.conditions ?? prefill?.conditions ?? [],
          },
          labelsMap,
          tagsMap,
          { onAnyChange: () => setDirty(true) },
        );
        _formDestroyFn = formHandle.destroy ?? null;

        // Preview Images — always available, reflects current (possibly unsaved) form state
        const hr = document.createElement('hr');
        hr.style.cssText = 'margin: 0.8em 0 0.5em; opacity: 0.4;';
        editorBody.appendChild(hr);

        const viewRow = document.createElement('div');
        viewRow.style.cssText = 'display: flex; gap: 8px; flex-wrap: wrap;';
        editorBody.appendChild(viewRow);

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
                filterJson: { conditions: formHandle.getConditions() },
                labelUuids: formHandle.getLabelUuids(),
                onUpdateFilters: (conditions, labelUuids) => {
                  formHandle.setConditions(conditions);
                  formHandle.setLabelUuids(labelUuids);
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
        viewRow.appendChild(previewBtn);

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
                filterJson: { conditions: formHandle.getConditions() },
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
                filterJson: { conditions: formHandle.getConditions() },
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
