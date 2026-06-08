/**
 * Filter panel for the Review page.
 *
 * createFilterPanel(labels, tags, onFilterChange)
 *   → {
 *       element,            // panel DOM node (Date + Filename + Advanced filters)
 *       toolbarControls,    // [tagsDropdown.element, labelsDropdown.element] — place in toolbar
 *       getState, setState, isDefault, getActiveFilterCount,
 *       destroy,
 *     }
 *
 * Calls onFilterChange(state) whenever any filter value changes.
 * The panel element starts visible; callers may toggle element.hidden as needed.
 *
 * Layout:
 *   Toolbar (toolbarControls):  [Tags ▾]  [Labels ▾]
 *   Filter panel (element):     Date: [from]–[to]  |  File: [____]
 *                               ▸ Advanced filters (collapsible)
 */
import { UNVERIFIED_TAG_UUID } from '/app-static/js/shared/api/review.js';
import { createMultiSelectDropdown, makeLabelFilterNotice } from '/app-static/js/ui/utils/index.js';
import { flattenHierarchy } from '/app-static/js/ui/labelList.js';
import { activeModel } from '/app-static/js/shared/activeModel.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';

// System tags appear first in the tags dropdown, in this order.
const SYSTEM_TAG_ORDER = [UNVERIFIED_TAG_UUID];

const DEFAULT_STATE = () => ({
  tagUuids: [UNVERIFIED_TAG_UUID],
  tagExcludeUuids: [],
  tagMatchMode: 'any',
  labelUuids: [],
  labelExcludeUuids: [],
  labelMatchMode: 'any',
  labelIncludeChildren: false,
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

  function emit() { onFilterChange({ ..._state }); }

  // ── Toolbar: Tags multi-select dropdown ──────────────────────────────────
  const sortedTags = [
    ...SYSTEM_TAG_ORDER.map(u => tags.get(u)).filter(Boolean),
    ...[...tags.values()]
      .filter(t => !SYSTEM_TAG_ORDER.includes(t.uuid))
      .sort((a, b) => a.name.localeCompare(b.name)),
  ];

  const tagsDropdown = createMultiSelectDropdown({
    label: 'Tags',
    items: sortedTags.map(t => ({ uuid: t.uuid, name: t.name, color: t.color || null })),
    defaultSelected: [UNVERIFIED_TAG_UUID],
    defaultMode: 'any',
    filterKey: 'tags',
    onChange: (selected, excluded, mode) => {
      _state.tagUuids = selected;
      _state.tagExcludeUuids = excluded;
      _state.tagMatchMode = mode;
      emit();
    },
  });
  // ── Toolbar: Labels multi-select dropdown (tree-ordered, with depth indentation) ──
  // Wrapped in a container div so it can be rebuilt when the active-model filter changes.
  const labelsDropdownContainer = document.createElement('div');
  let _labelsDropdown = null;
  // Create advLabelSelect here (before buildLabelsControls) so the first call
  // can populate it immediately — avoiding a second full rebuild call later.
  const advLabelSelect = document.createElement('select');
  let _advLabelSelect = advLabelSelect;

  function buildLabelsControls() {
    const filteredLabels = activeModel.applyLabelFilter(labels);
    const flatLabels = flattenHierarchy(filteredLabels);

    // Destroy previous dropdown instance
    _labelsDropdown?.destroy();

    _labelsDropdown = createMultiSelectDropdown({
      label: 'Labels',
      items: flatLabels.map(l => ({ uuid: l.uuid, name: l.name, color: l.color || null, depth: l.depth, parentUuid: l.parentUuid ?? null })),
      defaultSelected: [],
      defaultMode: 'any',
      filterKey: 'labels',
      hierarchyToggle: {
        label: 'Include child labels',
        default: false,
        onChange: (val) => {
          _state.labelIncludeChildren = val;
          emit();
        },
      },
      onChange: (selected, excluded, mode) => {
        _state.labelUuids = selected;
        _state.labelExcludeUuids = excluded;
        _state.labelMatchMode = mode;
        emit();
      },
    });

    labelsDropdownContainer.innerHTML = '';
    labelsDropdownContainer.appendChild(_labelsDropdown.element);
    const notice = makeLabelFilterNotice(labels.size - filteredLabels.size);
    if (notice) labelsDropdownContainer.appendChild(notice);

    // Rebuild the advanced-filter label select if it already exists
    if (_advLabelSelect) {
      _advLabelSelect.innerHTML = '';
      const ph = document.createElement('option');
      ph.value = '';
      ph.textContent = '— label —';
      _advLabelSelect.appendChild(ph);
      for (const lbl of flatLabels) {
        const opt = document.createElement('option');
        opt.value = lbl.uuid;
        opt.textContent = lbl.name;
        _advLabelSelect.appendChild(opt);
      }
    }

    // Reset label state on rebuild (filtered items may differ)
    _state.labelUuids = [];
    _state.labelExcludeUuids = [];
    _state.labelMatchMode = 'any';
    _state.labelIncludeChildren = false;
  }

  buildLabelsControls();
  const _unsubActiveFilter = events.subscribe(EventTypes.ACTIVE_MODEL_CHANGED, buildLabelsControls);


  // ── Panel root ─────────────────────────────────────────────────────────────
  const panel = document.createElement('div');
  panel.className = 'review-filter-panel';

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

  // ── Primary row: Date + Filename ───────────────────────────────────────────
  const primaryRow = document.createElement('div');
  primaryRow.className = 'filter-row-primary';

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

  // ── Advanced filters (collapsible) ────────────────────────────────────────
  const advancedDetails = document.createElement('details');
  advancedDetails.className = 'filter-advanced';

  const advancedSummary = document.createElement('summary');
  advancedSummary.className = 'filter-label filter-advanced__summary';
  advancedSummary.textContent = 'Advanced filters';
  advancedSummary.title = 'Image-level classification filters (set during bulk confirm)';
  advancedDetails.appendChild(advancedSummary);

  const labelsRow = document.createElement('div');
  labelsRow.className = 'filter-row-labels';
  advancedDetails.appendChild(labelsRow);

  // Builder: label select + status + exclude checkbox + Add button
  const builderDiv = document.createElement('div');
  builderDiv.className = 'filter-imlabel-builder';

  // advLabelSelect was created above (before the first buildLabelsControls() call)
  // so options are already populated — no second call needed.

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

  builderDiv.append(advLabelSelect, statusSelect, excludeLabel, addBtn);
  labelsRow.appendChild(builderDiv);

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
    if (!advLabelSelect.value) return;
    _state.imageLabelFilters.push({
      labelUuid: advLabelSelect.value,
      status: statusSelect.value,
      exclude: excludeCb.checked,
    });
    advLabelSelect.value = '';
    renderFilterChips();
    emit();
  });

  panel.appendChild(advancedDetails);

  // ── Public API ────────────────────────────────────────────────────────────
  function getActiveFilterCount() {
    const s = _state;
    return s.tagUuids.length
      + s.tagExcludeUuids.length
      + s.labelUuids.length
      + s.labelExcludeUuids.length
      + (s.filenameSearch ? 1 : 0)
      + (s.dateFrom ? 1 : 0)
      + (s.dateTo ? 1 : 0)
      + s.imageLabelFilters.length;
  }

  function isDefault() {
    const s = _state;
    return (
      s.tagUuids.length === 1 && s.tagUuids[0] === UNVERIFIED_TAG_UUID
      && s.tagExcludeUuids.length === 0
      && s.tagMatchMode === 'any'
      && s.labelUuids.length === 0
      && s.labelExcludeUuids.length === 0
      && s.labelMatchMode === 'any'
      && !s.filenameSearch
      && !s.dateFrom
      && !s.dateTo
      && s.imageLabelFilters.length === 0
    );
  }

  /**
   * Programmatically set filter state. Syncs all DOM controls and fires
   * onFilterChange. Used by task action buttons ("See results").
   * @param {Partial<ReturnType<typeof DEFAULT_STATE>>} partial
   */
  function setState(partial) {
    // Normalize: if tagUuids is being replaced but tagExcludeUuids is absent
    // (e.g. a preset saved before exclusions existed), default the exclude list
    // to [] so stale exclusions aren't silently preserved.
    if (partial.tagUuids !== undefined && !('tagExcludeUuids' in partial)) {
      partial = { ...partial, tagExcludeUuids: [] };
    }
    if (partial.labelUuids !== undefined && !('labelExcludeUuids' in partial)) {
      partial = { ...partial, labelExcludeUuids: [] };
    }
    _state = { ..._state, ...partial };

    if (partial.tagUuids !== undefined || partial.tagMatchMode !== undefined || partial.tagExcludeUuids !== undefined) {
      tagsDropdown.setState(_state.tagUuids, _state.tagMatchMode, undefined, _state.tagExcludeUuids);
    }
    if (partial.labelUuids !== undefined || partial.labelMatchMode !== undefined || partial.labelIncludeChildren !== undefined || partial.labelExcludeUuids !== undefined) {
      _labelsDropdown?.setState(_state.labelUuids, _state.labelMatchMode, _state.labelIncludeChildren, _state.labelExcludeUuids);
    }
    if (partial.dateFrom !== undefined) {
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
    if (partial.imageLabelFilters !== undefined) {
      renderFilterChips();
    }

    emit();
  }

  function destroy() {
    tagsDropdown.destroy();
    _labelsDropdown?.destroy();
    _unsubActiveFilter();
  }

  return {
    element: panel,
    /** Dropdown elements to place in the sticky toolbar (Tags + Labels). */
    toolbarControls: [tagsDropdown.element, labelsDropdownContainer],
    getState: () => ({
      ..._state,
      // showAllBboxes: true skips the server-side tag filter entirely.
      // "no tags included" → show everything rather than nothing (exclusions
      // are passed separately via tagExcludeUuids and handled server-side).
      showAllBboxes: _state.tagUuids.length === 0,
    }),
    setState,
    isDefault,
    getActiveFilterCount,
    destroy,
  };
}
