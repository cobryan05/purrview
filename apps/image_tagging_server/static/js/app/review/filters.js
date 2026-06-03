/**
 * Filter panel for the Review page.
 *
 * createFilterPanel(labels, tags, onFilterChange)
 *   → { element, getState, isDefault, getActiveFilterCount }
 *
 * Calls onFilterChange(state) whenever any filter value changes.
 * The panel element starts visible; callers may toggle element.hidden as needed.
 *
 * Layout: two compact horizontal rows.
 *
 *   Row 1 (primary):
 *     [□ Unverified …]  |  Label: [— All —▼]  |  Date: [from]—[to]  |  File: [____]
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
  labelUuid: '',
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

  // ── 1b. Bbox label filter ───────────────────────────────────────────────
  primaryRow.appendChild(makeSep());
  primaryRow.appendChild(makeLabel('Label:'));

  const sortedLabels = [...labels.values()].sort((a, b) => a.name.localeCompare(b.name));

  // Builds a label <select> with a blank placeholder option followed by all labels.
  function buildLabelSelect(placeholderText, className) {
    const sel = document.createElement('select');
    if (className) sel.className = className;
    const ph = document.createElement('option');
    ph.value = '';
    ph.textContent = placeholderText;
    sel.appendChild(ph);
    for (const lbl of sortedLabels) {
      const opt = document.createElement('option');
      opt.value = lbl.uuid;
      opt.textContent = lbl.name;
      sel.appendChild(opt);
    }
    return sel;
  }

  const bboxLabelSelect = buildLabelSelect('— All —', 'filter-label-select');
  bboxLabelSelect.title = 'Filter by bbox label';
  bboxLabelSelect.addEventListener('change', () => {
    _state.labelUuid = bboxLabelSelect.value;
    emit();
  });
  primaryRow.appendChild(bboxLabelSelect);

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

  const labelSelect = buildLabelSelect('— label —');

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
  /** Count of active filter "slots" — shown on the toggle button even in default state. */
  function getActiveFilterCount() {
    const s = _state;
    return s.tagUuids.length
      + (s.labelUuid ? 1 : 0)
      + (s.filenameSearch ? 1 : 0)
      + (s.dateFrom ? 1 : 0)
      + (s.dateTo ? 1 : 0)
      + s.imageLabelFilters.length;
  }

  /** True only when state matches the out-of-the-box defaults (Unverified tag, nothing else). */
  function isDefault() {
    return getActiveFilterCount() === 1 && _state.tagUuids[0] === UNVERIFIED_TAG_UUID;
  }

  /**
   * Programmatically set filter state. Merges partial state, syncs all DOM
   * controls to match, and fires onFilterChange. Used by task action buttons
   * ("See results"). Every controlled input is kept in sync so the visible UI
   * matches what the active filter is actually doing.
   * @param {Partial<typeof _state>} partial
   */
  function setState(partial) {
    _state = { ..._state, ...partial };

    if (partial.tagUuids !== undefined) {
      tagCheckboxMap.forEach((input, uuid) => {
        input.checked = _state.tagUuids.includes(uuid);
      });
    }
    if (partial.labelUuid !== undefined) {
      bboxLabelSelect.value = _state.labelUuid ?? '';
    }
    if (partial.dateFrom !== undefined) {
      // datetime-local expects 'YYYY-MM-DDTHH:MM' (no seconds/timezone)
      dateFromInput.value = _state.dateFrom
        ? new Date(_state.dateFrom).toISOString().slice(0, 16)
        : '';
    }
    if (partial.dateTo !== undefined) {
      dateToInput.value = _state.dateTo
        ? new Date(_state.dateTo).toISOString().slice(0, 16)
        : '';
    }
    if (partial.filenameSearch !== undefined) {
      filenameInput.value = _state.filenameSearch ?? '';
    }

    emit();
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
    getActiveFilterCount,
    setState,
  };
}
