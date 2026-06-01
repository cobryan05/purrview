import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import {
  fetchDatasets, fetchDatasetImages, previewDatasetImages,
  createDataset, updateDataset, deleteDataset,
} from '/app-static/js/shared/api/datasets.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { toast } from '/app-static/js/ui/utils/index.js';

let _rendering = false;
let _unsubDatasets = null;

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
  selectAllBtn.style.marginRight = '4px';
  const clearAllBtn = document.createElement('button');
  clearAllBtn.type = 'button';
  clearAllBtn.textContent = 'Clear All';
  controls.appendChild(selectAllBtn);
  controls.appendChild(clearAllBtn);
  container.appendChild(controls);

  const list = document.createElement('div');
  list.style.cssText = 'height:300px; min-height:60px; overflow-y:auto; border:1px solid var(--pico-border-color, #ddd); padding:4px; border-radius:4px; resize:vertical;';
  container.appendChild(list);

  // Build hierarchy from labelsMap (Map<uuid, label>)
  const byParent = new Map();
  for (const label of labelsMap.values()) {
    const key = label.parent_uuid ?? null;
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
      if (cond.within_days) return `Image ${field} within ${cond.within_days} days`;
      if (cond.after && cond.before) return `Image ${field} between ${cond.after.slice(0, 10)} and ${cond.before.slice(0, 10)}`;
      if (cond.after) return `Image ${field} after ${cond.after.slice(0, 10)}`;
      if (cond.before) return `Image ${field} before ${cond.before.slice(0, 10)}`;
      return `Image ${field} (unspecified)`;
    }
    case 'bbox_age': {
      if (cond.within_days) return `BBox updated within ${cond.within_days} days`;
      if (cond.after && cond.before) return `BBox updated between ${cond.after.slice(0, 10)} and ${cond.before.slice(0, 10)}`;
      if (cond.after) return `BBox updated after ${cond.after.slice(0, 10)}`;
      if (cond.before) return `BBox updated before ${cond.before.slice(0, 10)}`;
      return 'BBox updated (unspecified)';
    }
    case 'tag': {
      const tagName = tagsMap.get(cond.tag_uuid)?.name ?? cond.tag_uuid;
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

  const heading = document.createElement('h5');
  heading.style.margin = '8px 0 4px';
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
      rmBtn.style.padding = '2px 8px';
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
  summary.style.cssText = 'cursor:pointer; font-weight:bold;';
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
        addBtn.addEventListener('click', () => {
          const condType = type === 'image_age' ? 'image_age' : 'bbox_age';
          const cond = { type: condType, field: fieldSelect?.value ?? 'updated_at' };
          if (modeSelect.value === 'within_days') {
            const v = parseInt(inputsEl.querySelector('[data-key="within_days"]')?.value);
            if (!v || v < 1) { toast('Enter a valid number of days', 5000, 'error'); return; }
            cond.within_days = v;
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
        addBtn.addEventListener('click', () => {
          if (!tagSel.value) { toast('Choose a tag', 5000, 'error'); return; }
          conditions.push({ type: 'tag', tag_uuid: tagSel.value, exclude });
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
  };
}

// -------------------------------------------------------
// Shared form fields builder
// -------------------------------------------------------

function buildDatasetForm(container, { name = '', description = '', labelUuids = [], filterConditions = [] }, labelsMap, tagsMap, { onAnyChange } = {}) {
  let _dirty = false;

  function _markDirty() {
    if (!_dirty) {
      _dirty = true;
      onAnyChange?.();
    }
  }

  // Name
  const nameLabel = document.createElement('label');
  nameLabel.style.display = 'block';
  nameLabel.textContent = 'Name';
  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.value = name;
  nameInput.placeholder = 'Dataset name';
  nameInput.style.width = '100%';
  nameInput.addEventListener('input', _markDirty);
  nameLabel.appendChild(nameInput);
  container.appendChild(nameLabel);

  // Description
  const descLabel = document.createElement('label');
  descLabel.style.display = 'block';
  descLabel.style.marginTop = '8px';
  descLabel.textContent = 'Description';
  const descInput = document.createElement('textarea');
  descInput.value = description;
  descInput.placeholder = 'Optional description';
  descInput.style.cssText = 'width:100%; height:60px; resize:vertical;';
  descInput.addEventListener('input', _markDirty);
  descLabel.appendChild(descInput);
  container.appendChild(descLabel);

  // Label picker
  const labelHeading = document.createElement('h5');
  labelHeading.style.margin = '8px 0 4px';
  labelHeading.textContent = 'Labels';
  container.appendChild(labelHeading);

  let selectedLabelUuids = [...labelUuids];
  const labelPickerEl = document.createElement('div');
  container.appendChild(labelPickerEl);
  renderLabelCheckboxList(labelPickerEl, labelsMap, selectedLabelUuids, (uuids) => {
    selectedLabelUuids = uuids;
    _markDirty();
  });

  // Conditions
  const condEditor = renderConditionsEditor(container, filterConditions, tagsMap, { onChange: _markDirty });

  return {
    getName: () => nameInput.value.trim(),
    getDescription: () => descInput.value.trim(),
    getLabelUuids: () => selectedLabelUuids,
    getConditions: () => condEditor.getConditions(),
    isDirty: () => _dirty,
    markClean: () => { _dirty = false; },
  };
}

// -------------------------------------------------------
// Detail panel (edit existing dataset)
// -------------------------------------------------------

function _renderDetailPanel(container, dataset, { labelsMap, tagsMap, onDeleted, onDirtyChange }) {
  container.innerHTML = '';

  const h4 = document.createElement('h4');
  h4.style.marginBottom = '8px';
  h4.textContent = dataset.name;
  container.appendChild(h4);

  const form = buildDatasetForm(
    container,
    {
      name: dataset.name,
      description: dataset.description || '',
      labelUuids: dataset.label_uuids ?? [],
      filterConditions: dataset.filter_json?.conditions ?? [],
    },
    labelsMap,
    tagsMap,
    { onAnyChange: () => { updateSaveBtnStyle(); onDirtyChange?.(() => form.isDirty()); } },
  );

  // Track current name so the delete confirm stays accurate after a rename+save
  let currentName = dataset.name;

  // Register dirty getter immediately so navigation checks work from the start
  onDirtyChange?.(() => form.isDirty());

  // ── Action buttons ───────────────────────────────────────────────
  const actionsEl = document.createElement('div');
  actionsEl.style.cssText = 'margin-top:12px; display:flex; gap:8px; flex-wrap:wrap; align-items:center;';
  container.appendChild(actionsEl);

  // View group: Preview + Open in Review
  const viewGroup = document.createElement('div');
  viewGroup.style.cssText = 'display:flex; gap:8px; flex-wrap:wrap;';
  actionsEl.appendChild(viewGroup);

  // Separator
  const sep = document.createElement('div');
  sep.style.cssText = 'width:1px; min-height:28px; background:var(--pico-border-color,#ddd); align-self:stretch; margin:0 2px;';
  actionsEl.appendChild(sep);

  // Edit group: Save + Delete
  const editGroup = document.createElement('div');
  editGroup.style.cssText = 'display:flex; gap:8px; flex-wrap:wrap;';
  actionsEl.appendChild(editGroup);

  // Preview Count — uses current form state, not the last-saved dataset
  const previewBtn = document.createElement('button');
  previewBtn.type = 'button';
  previewBtn.textContent = 'Preview Count';
  previewBtn.addEventListener('click', async () => {
    previewBtn.disabled = true;
    previewBtn.textContent = 'Loading…';
    try {
      const { images, hasMore } = await previewDatasetImages({
        filterJson: { conditions: form.getConditions() },
        labelUuids: form.getLabelUuids(),
        limit: 500,
      });
      const suffix = hasMore ? '500+' : String(images.length);
      toast(`Dataset contains ${suffix} images${form.isDirty() ? ' (unsaved)' : ''}`, 4000);
    } catch (err) {
      toast(err.message || 'Failed to preview', 5000, 'error');
    } finally {
      previewBtn.disabled = false;
      previewBtn.textContent = 'Preview Count';
    }
  });
  viewGroup.appendChild(previewBtn);

  // Browse Images — opens an inline review modal for this dataset
  const browseBtn = document.createElement('button');
  browseBtn.type = 'button';
  browseBtn.textContent = 'Browse Images';
  browseBtn.addEventListener('click', async () => {
    browseBtn.disabled = true;
    try {
      const { showReviewModal } = await import('/app-static/js/app/review/main.js');
      await showReviewModal({
        title: `Browse: ${dataset.name}`,
        fetchImages: async ({ sortBy, sortOrder, afterUuid, afterSortValue, limit }) =>
          fetchDatasetImages(dataset.uuid, {
            sort_by: sortBy,
            sort_order: sortOrder,
            after_uuid: afterUuid,
            after_sort_value: afterSortValue,
            limit,
          }),
      });
    } finally {
      browseBtn.disabled = false;
    }
  });
  viewGroup.appendChild(browseBtn);

  // Save
  const saveBtn = document.createElement('button');
  saveBtn.type = 'button';
  saveBtn.textContent = 'Save';

  function updateSaveBtnStyle() {
    if (form.isDirty()) {
      saveBtn.style.outline = '2px solid var(--pico-primary-background, #36c)';
      saveBtn.style.outlineOffset = '2px';
      saveBtn.style.fontWeight = 'bold';
      saveBtn.textContent = 'Save ●';
    } else {
      saveBtn.style.outline = '';
      saveBtn.style.outlineOffset = '';
      saveBtn.style.fontWeight = '';
      saveBtn.textContent = 'Save';
    }
  }

  saveBtn.addEventListener('click', async () => {
    const name = form.getName();
    if (!name) { toast('Name is required', 5000, 'error'); return; }
    saveBtn.disabled = true;
    try {
      const updated = await updateDataset(dataset.uuid, {
        name,
        description: form.getDescription(),
        filterJson: { conditions: form.getConditions() },
        labelUuids: form.getLabelUuids(),
      });
      h4.textContent = updated.name;
      currentName = updated.name;
      form.markClean();
      updateSaveBtnStyle();
      onDirtyChange?.(() => form.isDirty());
      toast('Dataset saved', 3000);
    } catch (err) {
      toast(err.message || 'Failed to save', 5000, 'error');
    } finally {
      saveBtn.disabled = false;
    }
  });
  editGroup.appendChild(saveBtn);

  // Delete
  const deleteBtn = document.createElement('button');
  deleteBtn.type = 'button';
  deleteBtn.textContent = 'Delete';
  deleteBtn.style.color = '#c00';
  deleteBtn.addEventListener('click', async () => {
    if (!window.confirm(`Delete dataset "${currentName}"?`)) return;
    deleteBtn.disabled = true;
    try {
      await deleteDataset(dataset.uuid);
      onDirtyChange?.(() => false);
      toast('Dataset deleted', 3000);
      onDeleted?.();
    } catch (err) {
      toast(err.message || 'Failed to delete', 5000, 'error');
      deleteBtn.disabled = false;
    }
  });
  editGroup.appendChild(deleteBtn);
}

// -------------------------------------------------------
// Create panel (new dataset)
// -------------------------------------------------------

function _renderCreatePanel(container, { labelsMap, tagsMap, onCreated, onDirtyChange }) {
  container.innerHTML = '';

  const h4 = document.createElement('h4');
  h4.style.marginBottom = '8px';
  h4.textContent = 'New Dataset';
  container.appendChild(h4);

  const form = buildDatasetForm(container, {}, labelsMap, tagsMap, {
    onAnyChange: () => onDirtyChange?.(() => form.isDirty()),
  });

  onDirtyChange?.(() => form.isDirty());

  const btnRow = document.createElement('div');
  btnRow.style.cssText = 'margin-top:12px; display:flex; gap:8px; flex-wrap:wrap;';
  container.appendChild(btnRow);

  const previewBtn = document.createElement('button');
  previewBtn.type = 'button';
  previewBtn.textContent = 'Preview Count';
  previewBtn.addEventListener('click', async () => {
    previewBtn.disabled = true;
    previewBtn.textContent = 'Loading…';
    try {
      const { images, hasMore } = await previewDatasetImages({
        filterJson: { conditions: form.getConditions() },
        labelUuids: form.getLabelUuids(),
        limit: 500,
      });
      const suffix = hasMore ? '500+' : String(images.length);
      toast(`Filter matches ${suffix} images`, 4000);
    } catch (err) {
      toast(err.message || 'Failed to preview', 5000, 'error');
    } finally {
      previewBtn.disabled = false;
      previewBtn.textContent = 'Preview Count';
    }
  });
  btnRow.appendChild(previewBtn);

  const createBtn = document.createElement('button');
  createBtn.type = 'button';
  createBtn.textContent = 'Create Dataset';
  createBtn.addEventListener('click', async () => {
    const name = form.getName();
    if (!name) { toast('Name is required', 5000, 'error'); return; }
    createBtn.disabled = true;
    try {
      const dataset = await createDataset({
        name,
        description: form.getDescription(),
        filterJson: { conditions: form.getConditions() },
        labelUuids: form.getLabelUuids(),
      });
      form.markClean();
      onDirtyChange?.(() => false);
      toast(`Dataset "${dataset.name}" created`, 3000);
      await onCreated?.(dataset);
    } catch (err) {
      toast(err.message || 'Failed to create dataset', 5000, 'error');
      createBtn.disabled = false;
    }
  });
  btnRow.appendChild(createBtn);
}

// -------------------------------------------------------
// Entry point
// -------------------------------------------------------

export async function renderDatasetManager({ target = 'dataset-management-box' } = {}) {
  if (_rendering) return;
  _rendering = true;

  try {
    const container = document.getElementById(target);
    if (!container) return;
    container.innerHTML = '';

    const panel = document.createElement('div');
    panel.className = 'config-panel';
    container.appendChild(panel);

    const leftCol = document.createElement('div');
    leftCol.className = 'config-column';
    panel.appendChild(leftCol);

    const rightCol = document.createElement('div');
    rightCol.className = 'config-column';
    panel.appendChild(rightCol);

    // Warm caches so first panel open is near-instant
    await Promise.all([fetchLabels(), fetchTags()]);

    let _selectedUuid = null;

    // Dirty-state tracker for the currently displayed right panel.
    // Both _renderDetailPanel and _renderCreatePanel register a getter via onDirtyChange.
    let _getDirtyState = () => false;

    function _confirmIfDirty() {
      if (!_getDirtyState()) return true;
      return window.confirm('You have unsaved changes. Discard them?');
    }

    // Update row highlight in-place without re-fetching the list
    function _updateListSelection(uuid) {
      for (const row of leftCol.querySelectorAll('[data-dataset-uuid]')) {
        const sel = row.dataset.datasetUuid === uuid;
        row.style.background = sel ? 'var(--pico-primary-background, #eef)' : '';
        row.style.borderColor = sel ? 'var(--pico-primary-border, #aad)' : 'transparent';
      }
    }

    async function showDetail(dataset) {
      if (!_confirmIfDirty()) return;
      _selectedUuid = dataset.uuid;
      _getDirtyState = () => false;
      // Re-fetch so the label/tag pickers always reflect current data
      const [labelsMap, tagsMap] = await Promise.all([fetchLabels(), fetchTags()]);
      _renderDetailPanel(rightCol, dataset, {
        labelsMap,
        tagsMap,
        onDirtyChange: (getter) => { _getDirtyState = getter; },
        onDeleted: () => {
          _selectedUuid = null;
          _getDirtyState = () => false;
          rightCol.innerHTML = '';
        },
      });
      _updateListSelection(dataset.uuid);
    }

    async function showCreate() {
      if (!_confirmIfDirty()) return;
      _selectedUuid = null;
      _getDirtyState = () => false;
      const [labelsMap, tagsMap] = await Promise.all([fetchLabels(), fetchTags()]);
      _renderCreatePanel(rightCol, {
        labelsMap,
        tagsMap,
        onDirtyChange: (getter) => { _getDirtyState = getter; },
        onCreated: async (dataset) => {
          _getDirtyState = () => false;
          await showDetail(dataset);
        },
      });
      _updateListSelection(null);
    }

    async function refreshList() {
      leftCol.innerHTML = '';

      const h4 = document.createElement('h4');
      h4.style.marginBottom = '8px';
      h4.textContent = 'Datasets';
      leftCol.appendChild(h4);

      const datasetsMap = await fetchDatasets();

      const listEl = document.createElement('div');
      leftCol.appendChild(listEl);

      if (datasetsMap.size === 0) {
        const empty = document.createElement('p');
        empty.style.cssText = 'color:var(--pico-muted-color,#888); font-style:italic; font-size:0.85em;';
        empty.textContent = 'No datasets yet.';
        listEl.appendChild(empty);
      }

      for (const dataset of datasetsMap.values()) {
        const row = document.createElement('div');
        row.style.cssText = 'cursor:pointer; padding:6px 8px; margin-bottom:4px; border-radius:4px; border:1px solid transparent; background:var(--pico-card-background-color);';
        row.dataset.datasetUuid = dataset.uuid;

        const nameLine = document.createElement('div');
        nameLine.style.fontWeight = 'bold';
        nameLine.textContent = dataset.name;
        row.appendChild(nameLine);

        if (dataset.description) {
          const descLine = document.createElement('div');
          descLine.style.cssText = 'font-size:0.82em; color:var(--pico-muted-color,#666); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:220px;';
          descLine.textContent = dataset.description;
          row.appendChild(descLine);
        }

        const metaLine = document.createElement('div');
        metaLine.style.cssText = 'font-size:0.75em; color:var(--pico-muted-color,#999);';
        metaLine.textContent = `${dataset.label_uuids?.length ?? 0} labels`;
        row.appendChild(metaLine);

        row.addEventListener('mouseenter', () => { if (row.dataset.datasetUuid !== _selectedUuid) row.style.background = 'var(--pico-card-sectioning-background-color, #f5f5f5)'; });
        row.addEventListener('mouseleave', () => { if (row.dataset.datasetUuid !== _selectedUuid) row.style.background = 'var(--pico-card-background-color)'; });
        row.addEventListener('click', async () => { await showDetail(dataset); });
        listEl.appendChild(row);
      }

      const newBtn = document.createElement('button');
      newBtn.type = 'button';
      newBtn.textContent = '+ New Dataset';
      newBtn.style.cssText = 'margin-top:8px; width:100%;';
      newBtn.addEventListener('click', async () => {
        try { await showCreate(); } catch (err) { toast(err.message || 'Failed to open form', 5000, 'error'); }
      });
      leftCol.appendChild(newBtn);

      // Re-apply selection styling after DOM is rebuilt
      _updateListSelection(_selectedUuid);
    }

    // Replace any previous subscription so we don't double-refresh on tab re-activation
    _unsubDatasets?.();
    _unsubDatasets = events.subscribe(EventTypes.DATASETS_UPDATED, async () => {
      try { await refreshList(); } catch (err) { toast(err.message || 'Failed to refresh datasets', 5000, 'error'); }
    });

    await refreshList();
    await showCreate();
  } finally {
    _rendering = false;
  }
}
