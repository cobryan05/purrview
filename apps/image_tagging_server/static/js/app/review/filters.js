/**
 * Filter panel for the Review page.
 *
 * createFilterPanel(labels, tags, onFilterChange)
 *   → {
 *       element,            // panel DOM node (Date + Filename + Verification row)
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
 *                               Verification: [<select>]
 */
import { UNVERIFIED_TAG_UUID } from '/app-static/js/shared/api/review.js';
import { createMultiSelectDropdown, makeLabelFilterNotice } from '/app-static/js/ui/utils/index.js';
import { flattenHierarchy } from '/app-static/js/ui/labelList.js';
import { activeModel, isTagVisible, isDynamicLabelsVisible } from '/app-static/js/shared/activeModel.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';

// System tags appear first in the tags dropdown, in this order.
const SYSTEM_TAG_ORDER = [UNVERIFIED_TAG_UUID];

const DEFAULT_STATE = () => ({
  tagUuids: [],
  tagExcludeUuids: [],
  tagMatchMode: 'any',
  labelUuids: [],
  labelExcludeUuids: [],
  labelMatchMode: 'any',
  labelIncludeChildren: false,
  // Whole-image tags (image_tags table) — distinct from tagUuids above, which
  // filters by bbox-level tags. Selected = image must carry the tag; excluded =
  // image must not carry it. No any/all toggle: all selected tags are required.
  imageTagUuids: [],
  imageTagExcludeUuids: [],
  filenameSearch: '',
  dateFrom: null,
  dateTo: null,
  verificationCheck: null,
});

/**
 * @param {Map<string, object>} labels   - label uuid → label record
 * @param {Map<string, object>} tags     - tag uuid → tag record
 * @param {(state: object) => void} onFilterChange
 */
export function createFilterPanel(labels, tags, onFilterChange, { inlineControls = false } = {}) {
  let _state = DEFAULT_STATE();

  function emit() { onFilterChange({ ..._state }); }

  // ── Toolbar: Tags multi-select dropdown ──────────────────────────────────
  // Tags whose kind is globally hidden don't appear in the picker.
  // Icon-less system tags are always excluded (they're used only via presets).
  function _visibleTags() {
    return [...tags.values()].filter(isTagVisible);
  }

  function _buildTagsDropdown() {
    const visible = _visibleTags();
    const byKind = [
      { label: 'System',         tags: visible.filter(t => t.kind === 'system') },
      { label: 'Task-generated', tags: visible.filter(t => t.kind === 'task_generated') },
      { label: 'Generic',        tags: visible.filter(t => !t.kind || t.kind === 'generic') },
    ].filter(g => g.tags.length > 0);

    byKind.forEach(g => g.tags.sort((a, b) => a.name.localeCompare(b.name)));

    const useHeaders = byKind.length > 1;
    const items = [];
    for (const g of byKind) {
      if (useHeaders) items.push({ isSeparator: true, label: g.label });
      items.push(...g.tags.map(t => ({ uuid: t.uuid, name: t.name, color: t.color || null })));
    }

    return createMultiSelectDropdown({
      label: 'Tags',
      items,
      defaultSelected: [],
      defaultMode: 'any',
      filterKey: 'tags',
      inline: inlineControls,
      defaultWidth: '220px',
      defaultHeight: '300px',
      onChange: (selected, excluded, mode) => {
        _state.tagUuids = selected;
        _state.tagExcludeUuids = excluded;
        _state.tagMatchMode = mode;
        emit();
      },
    });
  }

  // Stable container so toolbarControls[0] reference stays valid across rebuilds.
  const tagsContainer = document.createElement('div');
  if (inlineControls) tagsContainer.style.width = '100%';
  let tagsDropdown = _buildTagsDropdown();
  tagsContainer.appendChild(tagsDropdown.element);

  // ── Toolbar: Image Tags multi-select dropdown ────────────────────────────
  // Whole-image tags (image_tags table) — a different axis than the bbox-level
  // Tags dropdown above. No any/all mode: every selected tag is required (AND).
  function _buildImageTagsDropdown() {
    const visible = _visibleTags();
    const items = [...visible]
      .sort((a, b) => a.name.localeCompare(b.name))
      .map(t => ({ uuid: t.uuid, name: t.name, color: t.color || null }));

    return createMultiSelectDropdown({
      label: 'Image Tags',
      items,
      defaultSelected: [],
      showModeToggle: false,
      filterKey: 'image-tags',
      inline: inlineControls,
      defaultWidth: '220px',
      defaultHeight: '300px',
      onChange: (selected, excluded) => {
        _state.imageTagUuids = selected;
        _state.imageTagExcludeUuids = excluded;
        emit();
      },
    });
  }

  const imageTagsContainer = document.createElement('div');
  if (inlineControls) imageTagsContainer.style.width = '100%';
  let imageTagsDropdown = _buildImageTagsDropdown();
  imageTagsContainer.appendChild(imageTagsDropdown.element);
  // ── Toolbar: Labels multi-select dropdown (tree-ordered, with depth indentation) ──
  // Wrapped in a container div so it can be rebuilt when the active-model filter changes.
  const labelsDropdownContainer = document.createElement('div');
  if (inlineControls) labelsDropdownContainer.style.width = '100%';
  let _labelsDropdown = null;

  function buildLabelsControls() {
    const filteredLabels = activeModel.applyLabelFilter(labels);
    const regularMap = new Map([...filteredLabels].filter(([, l]) => l.labelType !== 'dynamic'));
    const dynamicMap = isDynamicLabelsVisible()
      ? new Map([...filteredLabels].filter(([, l]) => l.labelType === 'dynamic'))
      : new Map();

    const flatRegular = flattenHierarchy(regularMap);
    const flatDynamic = flattenHierarchy(dynamicMap);

    const toItem = l => ({ uuid: l.uuid, name: l.name, color: l.color || null, depth: l.depth, parentUuid: l.parentUuid ?? null });
    const labelItems = [
      ...flatRegular.map(toItem),
      ...(flatDynamic.length > 0 ? [{ isSeparator: true, label: 'Dynamic' }, ...flatDynamic.map(toItem)] : []),
    ];

    // Destroy previous dropdown instance
    _labelsDropdown?.destroy();

    _labelsDropdown = createMultiSelectDropdown({
      label: 'Labels',
      items: labelItems,
      defaultWidth: '280px',
      defaultHeight: '400px',
      defaultSelected: [],
      defaultMode: 'any',
      filterKey: 'labels-v2',
      inline: inlineControls,
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

    // Reset label state on rebuild (filtered items may differ)
    _state.labelUuids = [];
    _state.labelExcludeUuids = [];
    _state.labelMatchMode = 'any';
    _state.labelIncludeChildren = false;
  }

  buildLabelsControls();
  const _unsubActiveFilter = events.subscribe(EventTypes.ACTIVE_MODEL_CHANGED, buildLabelsControls);
  const _unsubDynLabels = events.subscribe(EventTypes.DYNAMIC_LABELS_VISIBILITY_CHANGED, buildLabelsControls);

  // Rebuild the tags dropdown when global tag-kind visibility changes.
  const _unsubTagVis = events.subscribe(EventTypes.TAG_KIND_VISIBILITY_CHANGED, () => {
    tagsDropdown.destroy();
    tagsContainer.innerHTML = '';
    tagsDropdown = _buildTagsDropdown();
    tagsContainer.appendChild(tagsDropdown.element);
    // Reset tag state to empty (show all) when visibility changes.
    _state.tagUuids = [];
    _state.tagExcludeUuids = [];

    imageTagsDropdown.destroy();
    imageTagsContainer.innerHTML = '';
    imageTagsDropdown = _buildImageTagsDropdown();
    imageTagsContainer.appendChild(imageTagsDropdown.element);
    _state.imageTagUuids = [];
    _state.imageTagExcludeUuids = [];

    emit();
  });


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

  // ── Verification row ──────────────────────────────────────────────────────
  const verificationRow = document.createElement('div');
  verificationRow.className = 'filter-row-primary';

  verificationRow.appendChild(makeLabel('Verification:'));

  const verificationSelect = document.createElement('select');
  for (const [value, text] of [
    ['', 'Any'],
    ['all_verified', 'All labels verified'],
    ['has_unverified', 'Has unverified labels'],
  ]) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = text;
    verificationSelect.appendChild(opt);
  }
  verificationSelect.addEventListener('change', () => {
    _state.verificationCheck = verificationSelect.value || null;
    emit();
  });

  verificationRow.appendChild(verificationSelect);
  panel.appendChild(verificationRow);

  // ── Public API ────────────────────────────────────────────────────────────
  function getActiveFilterCount() {
    const s = _state;
    return s.tagUuids.length
      + s.tagExcludeUuids.length
      + s.labelUuids.length
      + s.labelExcludeUuids.length
      + s.imageTagUuids.length
      + s.imageTagExcludeUuids.length
      + (s.filenameSearch ? 1 : 0)
      + (s.dateFrom ? 1 : 0)
      + (s.dateTo ? 1 : 0)
      + (s.verificationCheck ? 1 : 0);
  }

  function isDefault() {
    const s = _state;
    return (
      s.tagUuids.length === 0
      && s.tagExcludeUuids.length === 0
      && s.tagMatchMode === 'any'
      && s.labelUuids.length === 0
      && s.labelExcludeUuids.length === 0
      && s.labelMatchMode === 'any'
      && s.imageTagUuids.length === 0
      && s.imageTagExcludeUuids.length === 0
      && !s.filenameSearch
      && !s.dateFrom
      && !s.dateTo
      && !s.verificationCheck
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
    if (partial.imageTagUuids !== undefined && !('imageTagExcludeUuids' in partial)) {
      partial = { ...partial, imageTagExcludeUuids: [] };
    }
    _state = { ..._state, ...partial };

    if (partial.tagUuids !== undefined || partial.tagMatchMode !== undefined || partial.tagExcludeUuids !== undefined) {
      tagsDropdown.setState(_state.tagUuids, _state.tagMatchMode, undefined, _state.tagExcludeUuids);
    }
    if (partial.labelUuids !== undefined || partial.labelMatchMode !== undefined || partial.labelIncludeChildren !== undefined || partial.labelExcludeUuids !== undefined) {
      _labelsDropdown?.setState(_state.labelUuids, _state.labelMatchMode, _state.labelIncludeChildren, _state.labelExcludeUuids);
    }
    if (partial.imageTagUuids !== undefined || partial.imageTagExcludeUuids !== undefined) {
      imageTagsDropdown.setState(_state.imageTagUuids, undefined, undefined, _state.imageTagExcludeUuids);
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
    if (partial.verificationCheck !== undefined) {
      verificationSelect.value = _state.verificationCheck ?? '';
    }

    emit();
  }

  function destroy() {
    tagsDropdown.destroy();
    imageTagsDropdown.destroy();
    _labelsDropdown?.destroy();
    _unsubActiveFilter();
    _unsubDynLabels();
    _unsubTagVis();
  }

  return {
    element: panel,
    /** Dropdown elements to place in the sticky toolbar (Tags + Image Tags + Labels). */
    toolbarControls: [tagsContainer, labelsDropdownContainer, imageTagsContainer],
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
