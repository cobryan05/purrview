/**
 * Filter panel for the Review page.
 *
 * createFilterPanel(labels, tags, onFilterChange)
 *   → { element, getState, isDefault }
 *
 * Calls onFilterChange(state) whenever any filter value changes.
 * The panel element starts hidden; the caller controls visibility.
 *
 * Layout: two compact horizontal rows.
 *
 *   Row 1 (primary):
 *     [□ Unverified …]  |  Date: [from]—[to]  |  File: [____]
 *
 *   Row 2 (labels):
 *     Labels: [— label —▼] [status▼] [□ Exclude] [＋ Add]   chip chip …
 *
 * Sort controls are on the review toolbar (always visible).
 */
import {
  UNVERIFIED_TAG_UUID,
} from '/app-static/js/shared/api/review.js';

// System tags appear first in the tag checkbox list, in this order.
const SYSTEM_TAG_ORDER = [UNVERIFIED_TAG_UUID];

const DEFAULT_STATE = () => ({
  tagUuids: [UNVERIFIED_TAG_UUID],
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

  // ── Helpers ─────────────────────────────────────────────────────────────
  function emit() { onFilterChange({ ..._state }); }

  const makeSep = () => {
    const s = document.createElement('span');
    s.className = 'filter-sep';
    return s;
  };
  const makeLabel = (text) => {
    const s = document.createElement('span');
    s.className = 'filter-label';
    s.textContent = text;
    return s;
  };

  // ── Primary row ──────────────────────────────────────────────────────────
  const primaryRow = document.createElement('div');
  primaryRow.className = 'filter-row-primary';

  // ── 1. Tag checkboxes ───────────────────────────────────────────────────
  const tagCheckboxes = document.createElement('div');
  tagCheckboxes.className = 'filter-tag-checkboxes';

  // System tags first (SYSTEM_TAG_ORDER), then alphabetical custom tags.
  const sortedTags = [
    ...SYSTEM_TAG_ORDER.map(u => tags.get(u)).filter(Boolean),
    ...[...tags.values()]
      .filter(t => !SYSTEM_TAG_ORDER.includes(t.uuid))
      .sort((a, b) => a.name.localeCompare(b.name)),
  ];

  const tagCheckboxMap = new Map(); // uuid → checkbox input

  for (const tag of sortedTags) {
    const lbl = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = tag.uuid;
    cb.checked = tag.uuid === UNVERIFIED_TAG_UUID;
    tagCheckboxMap.set(tag.uuid, cb);

    if (tag.color) {
      const swatch = document.createElement('span');
      swatch.style.cssText = `display:inline-block;width:0.65em;height:0.65em;border-radius:2px;background:${tag.color};flex-shrink:0;`;
      lbl.appendChild(swatch);
    }
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(tag.name));
    tagCheckboxes.appendChild(lbl);

    cb.addEventListener('change', () => {
      _state.tagUuids = [...tagCheckboxMap.entries()]
        .filter(([, input]) => input.checked)
        .map(([uuid]) => uuid);
      emit();
    });
  }

  primaryRow.appendChild(tagCheckboxes);

  // ── 2. Date range ───────────────────────────────────────────────────────
  primaryRow.appendChild(makeSep());
  primaryRow.appendChild(makeLabel('Date:'));

  const dateFromInput = document.createElement('input');
  dateFromInput.type = 'datetime-local';
  dateFromInput.title = 'From date';
  dateFromInput.addEventListener('change', () => {
    _state.dateFrom = dateFromInput.value ? new Date(dateFromInput.value).toISOString() : null;
    emit();
  });

  const dateToInput = document.createElement('input');
  dateToInput.type = 'datetime-local';
  dateToInput.title = 'To date';
  dateToInput.addEventListener('change', () => {
    _state.dateTo = dateToInput.value ? new Date(dateToInput.value).toISOString() : null;
    emit();
  });

  primaryRow.appendChild(dateFromInput);
  primaryRow.appendChild(document.createTextNode('–'));
  primaryRow.appendChild(dateToInput);

  // ── 3. Filename search ──────────────────────────────────────────────────
  primaryRow.appendChild(makeSep());
  primaryRow.appendChild(makeLabel('File:'));

  const filenameInput = document.createElement('input');
  filenameInput.type = 'text';
  filenameInput.placeholder = 'substring…';

  let _filenameDebounce = null;
  filenameInput.addEventListener('input', () => {
    clearTimeout(_filenameDebounce);
    _filenameDebounce = setTimeout(() => {
      _state.filenameSearch = filenameInput.value.trim();
      emit();
    }, 400);
  });

  primaryRow.appendChild(filenameInput);
  panel.appendChild(primaryRow);

  // ── Labels row ───────────────────────────────────────────────────────────
  const labelsRow = document.createElement('div');
  labelsRow.className = 'filter-row-labels';

  labelsRow.appendChild(makeLabel('Labels:'));

  // Builder controls
  const builderDiv = document.createElement('div');
  builderDiv.className = 'filter-imlabel-builder';

  const labelSelect = document.createElement('select');
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
  for (const [value, text] of [['uncertain', 'Uncertain'], ['present', 'Present'], ['absent', 'Absent']]) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = text;
    statusSelect.appendChild(opt);
  }
  statusSelect.value = 'absent';

  const excludeLabel = document.createElement('label');
  excludeLabel.className = 'filter-exclude-label';
  const excludeCb = document.createElement('input');
  excludeCb.type = 'checkbox';
  excludeCb.checked = true;
  excludeLabel.appendChild(excludeCb);
  excludeLabel.appendChild(document.createTextNode('Exclude'));

  const addBtn = document.createElement('button');
  addBtn.textContent = '＋ Add';

  builderDiv.appendChild(labelSelect);
  builderDiv.appendChild(statusSelect);
  builderDiv.appendChild(excludeLabel);
  builderDiv.appendChild(addBtn);
  labelsRow.appendChild(builderDiv);

  // Active filter chips
  const filterList = document.createElement('ul');
  filterList.className = 'filter-imlabel-list';
  labelsRow.appendChild(filterList);

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

  panel.appendChild(labelsRow);

  // ── Public API ────────────────────────────────────────────────────────────
  function isDefault() {
    const s = _state;
    return (
      s.tagUuids.length === 1 &&
      s.tagUuids[0] === UNVERIFIED_TAG_UUID &&
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
