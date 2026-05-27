/**
 * Filter panel for the Review page.
 *
 * createFilterPanel(labels, tags, onFilterChange)
 *   → { element, getState, isDefault }
 *
 * Calls onFilterChange(state) whenever any filter value changes.
 * The panel element starts hidden; the caller controls visibility.
 */
import {
  UNVERIFIED_TAG_UUID,
  VERIFIED_TAG_UUID,
  FALSE_POS_TAG_UUID,
} from '/app-static/js/shared/api/review.js';

// System tags appear first in the tag checkbox list, in this order.
const SYSTEM_TAG_ORDER = [UNVERIFIED_TAG_UUID, VERIFIED_TAG_UUID, FALSE_POS_TAG_UUID];

const DEFAULT_STATE = () => ({
  tagUuids: [UNVERIFIED_TAG_UUID],
  sortBy: 'created_at',
  sortOrder: 'desc',
  filenameSearch: '',
  dateFrom: null,
  dateTo: null,
  imageLabelFilters: [],
});

/**
 * @param {Map<string, object>} labels   - label uuid → label record
 * @param {Map<string, object>} tags     - tag uuid → tag record
 * @param {(state: object) => void} onFilterChange
 */
export function createFilterPanel(labels, tags, onFilterChange) {
  let _state = DEFAULT_STATE();

  // ── Panel root ──────────────────────────────────────────────────────────
  const panel = document.createElement('div');
  panel.className = 'review-filter-panel';

  // ── Helper: emit change ─────────────────────────────────────────────────
  function emit() {
    onFilterChange({ ..._state });
  }

  // ── Row builder ─────────────────────────────────────────────────────────
  function makeRow(labelText) {
    const row = document.createElement('div');
    row.className = 'filter-row';
    const lbl = document.createElement('span');
    lbl.className = 'filter-label';
    lbl.textContent = labelText;
    row.appendChild(lbl);
    return row;
  }

  // ── 1. Tags row ─────────────────────────────────────────────────────────
  const tagRow = makeRow('Tags');
  const tagCheckboxes = document.createElement('div');
  tagCheckboxes.className = 'filter-tag-checkboxes';

  // Sort: system tags first (in SYSTEM_TAG_ORDER), then alphabetical custom tags.
  const sortedTags = [
    ...SYSTEM_TAG_ORDER.map(u => tags.get(u)).filter(Boolean),
    ...[...tags.values()]
      .filter(t => !SYSTEM_TAG_ORDER.includes(t.uuid))
      .sort((a, b) => a.name.localeCompare(b.name)),
  ];

  const tagCheckboxMap = new Map(); // uuid → checkbox input

  for (const tag of sortedTags) {
    const label = document.createElement('label');
    label.style.cssText = 'display:flex;align-items:center;gap:0.3em;cursor:pointer;white-space:nowrap;';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = tag.uuid;
    cb.checked = tag.uuid === UNVERIFIED_TAG_UUID; // default: only Unverified
    tagCheckboxMap.set(tag.uuid, cb);

    // Color swatch
    if (tag.color) {
      const swatch = document.createElement('span');
      swatch.style.cssText = `display:inline-block;width:0.7em;height:0.7em;border-radius:2px;background:${tag.color};flex-shrink:0;`;
      label.appendChild(swatch);
    }
    label.appendChild(cb);
    label.appendChild(document.createTextNode(tag.name));
    tagCheckboxes.appendChild(label);

    cb.addEventListener('change', () => {
      _state.tagUuids = [...tagCheckboxMap.entries()]
        .filter(([, input]) => input.checked)
        .map(([uuid]) => uuid);
      emit();
    });
  }

  tagRow.appendChild(tagCheckboxes);
  panel.appendChild(tagRow);

  // ── 2. Sort row ─────────────────────────────────────────────────────────
  const sortRow = makeRow('Sort by');

  const sortBySelect = document.createElement('select');
  for (const [value, text] of [['created_at', 'Date created'], ['updated_at', 'Date modified']]) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = text;
    sortBySelect.appendChild(opt);
  }
  sortBySelect.value = _state.sortBy;
  sortBySelect.addEventListener('change', () => {
    _state.sortBy = sortBySelect.value;
    emit();
  });

  const sortOrderSelect = document.createElement('select');
  for (const [value, text] of [['desc', 'Newest first'], ['asc', 'Oldest first']]) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = text;
    sortOrderSelect.appendChild(opt);
  }
  sortOrderSelect.value = _state.sortOrder;
  sortOrderSelect.addEventListener('change', () => {
    _state.sortOrder = sortOrderSelect.value;
    emit();
  });

  sortRow.appendChild(sortBySelect);
  sortRow.appendChild(sortOrderSelect);
  panel.appendChild(sortRow);

  // ── 3. Date range row ────────────────────────────────────────────────────
  const dateRow = makeRow('Date range');

  function makeDateInput(placeholder) {
    const input = document.createElement('input');
    input.type = 'datetime-local';
    input.title = placeholder;
    input.style.cssText = 'font-size:0.85em;';
    return input;
  }

  const dateFromInput = makeDateInput('From');
  const dateSep = document.createTextNode(' — ');
  const dateToInput = makeDateInput('To');

  dateFromInput.addEventListener('change', () => {
    _state.dateFrom = dateFromInput.value ? new Date(dateFromInput.value).toISOString() : null;
    emit();
  });
  dateToInput.addEventListener('change', () => {
    _state.dateTo = dateToInput.value ? new Date(dateToInput.value).toISOString() : null;
    emit();
  });

  dateRow.appendChild(dateFromInput);
  dateRow.appendChild(dateSep);
  dateRow.appendChild(dateToInput);
  panel.appendChild(dateRow);

  // ── 4. Filename search row ───────────────────────────────────────────────
  const filenameRow = makeRow('Filename');

  const filenameInput = document.createElement('input');
  filenameInput.type = 'text';
  filenameInput.placeholder = 'substring…';
  filenameInput.style.cssText = 'flex:1;min-width:120px;max-width:280px;font-size:0.85em;';

  let _filenameDebounce = null;
  filenameInput.addEventListener('input', () => {
    clearTimeout(_filenameDebounce);
    _filenameDebounce = setTimeout(() => {
      _state.filenameSearch = filenameInput.value.trim();
      emit();
    }, 400);
  });

  filenameRow.appendChild(filenameInput);
  panel.appendChild(filenameRow);

  // ── 5. Image-label filter row ─────────────────────────────────────────────
  const imLabelRow = makeRow('Image labels');
  imLabelRow.style.flexWrap = 'wrap';
  imLabelRow.style.gap = '0.4em';

  // Builder controls
  const builderDiv = document.createElement('div');
  builderDiv.className = 'filter-imlabel-builder';

  const labelSelect = document.createElement('select');
  labelSelect.style.fontSize = '0.85em';
  const labelPlaceholder = document.createElement('option');
  labelPlaceholder.value = '';
  labelPlaceholder.textContent = '— label —';
  labelSelect.appendChild(labelPlaceholder);
  for (const lbl of [...labels.values()].sort((a, b) => a.name.localeCompare(b.name))) {
    const opt = document.createElement('option');
    opt.value = lbl.uuid;
    opt.textContent = lbl.name;
    labelSelect.appendChild(opt);
  }

  const statusSelect = document.createElement('select');
  statusSelect.style.fontSize = '0.85em';
  for (const [value, text] of [['uncertain', 'Uncertain'], ['present', 'Present'], ['absent', 'Absent']]) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = text;
    statusSelect.appendChild(opt);
  }
  statusSelect.value = 'absent';

  const excludeLabel = document.createElement('label');
  excludeLabel.style.cssText = 'display:flex;align-items:center;gap:0.2em;cursor:pointer;font-size:0.85em;white-space:nowrap;';
  const excludeCb = document.createElement('input');
  excludeCb.type = 'checkbox';
  excludeCb.checked = true; // default: exclude
  excludeLabel.appendChild(excludeCb);
  excludeLabel.appendChild(document.createTextNode('Exclude'));

  const addBtn = document.createElement('button');
  addBtn.textContent = '＋ Add';
  addBtn.style.cssText = 'font-size:0.8em;padding:0.2em 0.5em;cursor:pointer;';

  builderDiv.appendChild(labelSelect);
  builderDiv.appendChild(statusSelect);
  builderDiv.appendChild(excludeLabel);
  builderDiv.appendChild(addBtn);
  imLabelRow.appendChild(builderDiv);

  // Active filters list
  const filterList = document.createElement('ul');
  filterList.className = 'filter-imlabel-list';
  imLabelRow.appendChild(filterList);

  function renderFilterChips() {
    filterList.innerHTML = '';
    for (let i = 0; i < _state.imageLabelFilters.length; i++) {
      const f = _state.imageLabelFilters[i];
      const labelName = labels.get(f.labelUuid)?.name ?? f.labelUuid;
      const li = document.createElement('li');
      li.textContent = `${labelName} is${f.exclude ? ' NOT' : ''} ${f.status}`;
      const removeBtn = document.createElement('button');
      removeBtn.textContent = '×';
      removeBtn.title = 'Remove filter';
      removeBtn.addEventListener('click', () => {
        _state.imageLabelFilters.splice(i, 1);
        renderFilterChips();
        emit();
      });
      li.appendChild(removeBtn);
      filterList.appendChild(li);
    }
  }

  addBtn.addEventListener('click', () => {
    if (!labelSelect.value) return;
    _state.imageLabelFilters.push({
      labelUuid: labelSelect.value,
      status: statusSelect.value,
      exclude: excludeCb.checked,
    });
    labelSelect.value = '';
    renderFilterChips();
    emit();
  });

  panel.appendChild(imLabelRow);

  // ── Public API ────────────────────────────────────────────────────────────
  function isDefault() {
    const s = _state;
    return (
      s.tagUuids.length === 1 &&
      s.tagUuids[0] === UNVERIFIED_TAG_UUID &&
      s.sortBy === 'created_at' &&
      s.sortOrder === 'desc' &&
      !s.filenameSearch &&
      !s.dateFrom &&
      !s.dateTo &&
      s.imageLabelFilters.length === 0
    );
  }

  return {
    element: panel,
    getState: () => ({
      ..._state,
      // showAllBboxes: true skips the server-side tag filter entirely.
      // We map "no tags selected" → showAll so the user sees everything rather
      // than nothing; callers should use this field instead of inferring from tagUuids.
      showAllBboxes: _state.tagUuids.length === 0,
      imageLabelFilters: [..._state.imageLabelFilters],
    }),
    isDefault,
  };
}
