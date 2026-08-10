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
 *                               Verification: [<select>]  |  Image Tag Filters: [condition list]
 */
import { VERIFICATION_TAG_UUIDS } from '/app-static/js/shared/api/review.js';
import { createMultiSelectDropdown, makeLabelFilterNotice } from '/app-static/js/ui/utils/index.js';
import { createImageLabelConditions, createImageTagConditions } from '/app-static/js/ui/imageTagConditions.js';
import { createBboxVerificationControl } from '/app-static/js/app/review/bboxVerificationControl.js';
import { flattenHierarchy } from '/app-static/js/ui/labelList.js';
import { filterVisibleTags, groupTagsByKind } from '/app-static/js/shared/activeModel.js';
import { activeTrainingConfig } from '/app-static/js/shared/activeTrainingConfig.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';

const DEFAULT_STATE = () => ({
  tagUuids: [],
  tagExcludeUuids: [],
  tagMatchMode: 'any',
  labelUuids: [],
  labelExcludeUuids: [],
  labelMatchMode: 'any',
  labelIncludeChildren: false,
  // Per-image-label conditions: [{ labelUuid, state: 'present'|'absent'|'uncertain' }]
  // Each entry is an AND condition applied against the image_labels table.
  imageLabelConditions: [],
  // Per-image-tag conditions: [{ tagUuid, state: 'present'|'absent'|'unset' }]
  // Each entry is an AND condition applied against the image_tags table.
  // Distinct from `tagUuids` above, which filters on tags carried by an image's
  // BBOXES — an image with no bboxes can only ever be matched by these.
  imageTagConditions: [],
  filenameSearch: '',
  filenameNegate: false,
  dateFrom: null,
  dateTo: null,
  dateNegate: false,
  verificationCheck: null,
  // Bbox verification tri-state (V1) — filters bboxes by their VERIFIED /
  // UNVERIFIED tag. Distinct from `verificationCheck` above,
  // which is the whole-image label-completeness check. 'any' = no-op, same
  // as filterPredicate.js's defaultCuratedFilterState().
  bboxVerification: 'any',
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
    // Verification tags (VERIFIED/UNVERIFIED) are owned exclusively
    // by the bbox-verification tri-state control below — excluded here so
    // there's only one way to filter by verification status (V1 decision).
    return filterVisibleTags(tags.values()).filter(t => !VERIFICATION_TAG_UUIDS.has(t.uuid));
  }

  function _buildTagsDropdown() {
    const visible = _visibleTags();
    const byKind = groupTagsByKind(visible, ['system', 'task_generated', 'generic']);

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

  // ── Toolbar: Bbox Labels multi-select dropdown ────────────────────────────
  // Shows only object-scoped labels — filter images by bbox label presence.
  // Use the Image Label Filters section to filter by whole-image label state.
  const labelsDropdownContainer = document.createElement('div');
  if (inlineControls) labelsDropdownContainer.style.width = '100%';
  let _labelsDropdown = null;

  function buildLabelsControls() {
    const filteredLabels = activeTrainingConfig.applyLabelFilter(labels);
    // Only object-scoped labels belong in the bbox-label filter dropdown.
    const regularMap = new Map(
      [...filteredLabels].filter(([, l]) => l.labelType !== 'dynamic' && l.scope !== 'image')
    );
    // Dynamic labels are always filterable here, even though the "Show dynamic
    // labels" preference (activeModel.js) hides them from label-assignment
    // pickers by default — that preference is about decluttering label
    // *creation* UI, not about whether existing dynamic-labeled bboxes can be
    // searched for.
    const dynamicMap = new Map(
      [...filteredLabels].filter(([, l]) => l.labelType === 'dynamic' && l.scope !== 'image')
    );

    const flatRegular = flattenHierarchy(regularMap);
    const flatDynamic = flattenHierarchy(dynamicMap);

    const toItem = l => ({ uuid: l.uuid, name: l.name, color: l.color || null, depth: l.depth, parentUuid: l.parentUuid ?? null });
    const labelItems = [
      ...flatRegular.map(toItem),
      ...(flatDynamic.length > 0 ? [{ isSeparator: true, label: 'Dynamic' }, ...flatDynamic.map(toItem)] : []),
    ];

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

    _state.labelUuids = [];
    _state.labelExcludeUuids = [];
    _state.labelMatchMode = 'any';
    _state.labelIncludeChildren = false;
  }

  buildLabelsControls();

  // ── Image label conditions component ──────────────────────────────────────
  // Returns the array of image-scoped labels visible to the current active model.
  function _imageScopedLabels() {
    // Include all non-dynamic labels (both object- and image-scoped) since image_label
    // rows exist for any label that appears on verified bboxes, not just image-scoped ones.
    const filtered = activeTrainingConfig.applyLabelFilter(labels);
    return [...filtered.values()].filter(l => l.labelType !== 'dynamic');
  }

  const imageLabelConditionsComp = createImageLabelConditions(
    _imageScopedLabels(),
    (conditions) => {
      _state.imageLabelConditions = conditions;
      emit();
    },
  );

  // ── Image tag conditions component ────────────────────────────────────────
  // Whole-image tag state, against the image_tags table. Shares the Tags
  // dropdown's visibility rules so the same tags appear in both surfaces.
  const imageTagConditionsComp = createImageTagConditions(
    _visibleTags(),
    (conditions) => {
      _state.imageTagConditions = conditions;
      emit();
    },
  );

  // Destroy `dropdown`, rebuild it via `buildFn` into `container`, and return the new instance.
  function _rebuildDropdown(dropdown, container, buildFn) {
    dropdown.destroy();
    container.innerHTML = '';
    const next = buildFn();
    container.appendChild(next.element);
    return next;
  }

  // Rebuild the tags dropdown when global tag-kind visibility changes; reset tag state.
  function _onTagKindVisibilityChanged() {
    tagsDropdown = _rebuildDropdown(tagsDropdown, tagsContainer, _buildTagsDropdown);
    _state.tagUuids = [];
    _state.tagExcludeUuids = [];
    // The image-tag conditions draw from the same visibility rules, so a
    // now-hidden tag must drop out of both surfaces, not just the dropdown.
    imageTagConditionsComp.setAvailableTags(_visibleTags());
    _state.imageTagConditions = [];
    imageTagConditionsComp.setState([]);
    emit();
  }

  // Rebuild tag dropdowns when a tag is created, renamed, or deleted.
  // Preserve the current selection so an in-progress filter is not lost.
  let _tagsUpdateInFlight = false;
  let _tagsUpdatePending = false;
  async function _onTagsUpdated() {
    if (_tagsUpdateInFlight) {
      _tagsUpdatePending = true;
      return;
    }
    _tagsUpdateInFlight = true;
    try {
      do {
        _tagsUpdatePending = false;
        tags = await fetchTags().catch(() => tags);
        tagsDropdown = _rebuildDropdown(tagsDropdown, tagsContainer, _buildTagsDropdown);
        tagsDropdown.setState(_state.tagUuids, _state.tagMatchMode, undefined, _state.tagExcludeUuids);
        // Refresh names/colours in the image-tag picker too; the selection is
        // preserved here for the same reason it is above.
        imageTagConditionsComp.setAvailableTags(_visibleTags());
        imageTagConditionsComp.setState(_state.imageTagConditions);
      } while (_tagsUpdatePending);
    } finally {
      _tagsUpdateInFlight = false;
    }
  }

  // Rebuild the labels dropdown and image label conditions when a label is created, renamed, or deleted.
  let _labelsUpdateInFlight = false;
  let _labelsUpdatePending = false;
  async function _onLabelsUpdated() {
    if (_labelsUpdateInFlight) {
      _labelsUpdatePending = true;
      return;
    }
    _labelsUpdateInFlight = true;
    try {
      do {
        _labelsUpdatePending = false;
        labels = await fetchLabels().catch(() => labels);
        buildLabelsControls();
        imageLabelConditionsComp.setAvailableLabels(_imageScopedLabels());
      } while (_labelsUpdatePending);
    } finally {
      _labelsUpdateInFlight = false;
    }
  }

  const _unsubAll = events.subscribeMany([
    [EventTypes.ACTIVE_TRAINING_CONFIG_CHANGED,    ({ statsUpdated }) => {
      // A stats-only refresh must not reset user selections — but the Config
      // Ready enabled/disabled state is cheap and safe to re-evaluate either way.
      _updateConfigReadyEnabled();
      if (statsUpdated) return; // stats refreshes don't change the label set; don't reset selections
      buildLabelsControls();
      imageLabelConditionsComp.setAvailableLabels(_imageScopedLabels());
    }],
    [EventTypes.TAG_KIND_VISIBILITY_CHANGED,       _onTagKindVisibilityChanged],
    [EventTypes.TAGS_UPDATED,                      _onTagsUpdated],
    [EventTypes.LABELS_UPDATED,                    _onLabelsUpdated],
  ]);

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

  const dateNegateLabel = document.createElement('label');
  dateNegateLabel.className = 'itc-negate-label filter-date-negate';
  dateNegateLabel.title = 'Exclude this range (date is NOT between from/to)';
  const dateNegateCb = document.createElement('input');
  dateNegateCb.type = 'checkbox';
  dateNegateCb.addEventListener('change', () => {
    _state.dateNegate = dateNegateCb.checked;
    emit();
  });
  dateNegateLabel.append(dateNegateCb, document.createTextNode('¬'));
  primaryRow.appendChild(dateNegateLabel);

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

  const filenameNegateLabel = document.createElement('label');
  filenameNegateLabel.className = 'itc-negate-label filter-filename-negate';
  filenameNegateLabel.title = 'Exclude matches (filename does NOT contain this)';
  const filenameNegateCb = document.createElement('input');
  filenameNegateCb.type = 'checkbox';
  filenameNegateCb.addEventListener('change', () => {
    _state.filenameNegate = filenameNegateCb.checked;
    emit();
  });
  filenameNegateLabel.append(filenameNegateCb, document.createTextNode('¬'));
  primaryRow.appendChild(filenameNegateLabel);

  panel.appendChild(primaryRow);

  // ── Secondary row: Verification + Image Label Filters side by side ────────
  const secondaryRow = document.createElement('div');
  secondaryRow.className = 'filter-row-secondary';

  // "Config Ready" section (V2) — whole-image label-completeness check
  // (`verificationCheck`), repointed from the Labels-dropdown selection to the
  // ACTIVE TRAINING CONFIG's target labels (see setConfigReadyAvailable below
  // and main.js, which supplies `verificationLabelUuids` on every request that
  // sends this field). Image-view only — disabled by main.js in bboxes view.
  // Renamed from "Verification" in place; underlying state field name
  // (`verificationCheck`) and values ('' | 'all_verified' | 'has_unverified')
  // are unchanged so the server contract for the check itself is untouched.
  const verificationSection = document.createElement('div');
  verificationSection.className = 'filter-section';

  const verificationSectionLabel = document.createElement('div');
  verificationSectionLabel.className = 'filter-section-label';
  verificationSectionLabel.textContent = 'Config Ready';
  verificationSection.appendChild(verificationSectionLabel);

  const verificationSelect = document.createElement('select');
  verificationSelect.className = 'filter-verification-select';
  for (const [value, text] of [
    ['', 'Any'],
    ['all_verified', 'Ready'],
    ['has_unverified', 'Needs review'],
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
  verificationSection.appendChild(verificationSelect);

  // Disabled + hint when there's no active training config (the check has
  // nothing meaningful to run against) — kept in sync by
  // _updateConfigReadyEnabled below, called on init and on
  // ACTIVE_TRAINING_CONFIG_CHANGED. `setConfigReadyAvailable` (exposed on the
  // returned API) additionally disables it in bboxes view; both gates AND
  // together via `_configReadyViewAvailable`.
  const NO_ACTIVE_CONFIG_HINT = 'Requires an active training config';
  let _configReadyViewAvailable = true;

  function _updateConfigReadyEnabled() {
    const hasActiveConfig = !!activeTrainingConfig.getActive();
    const enabled = hasActiveConfig && _configReadyViewAvailable;
    verificationSelect.disabled = !enabled;
    verificationSelect.title = !hasActiveConfig
      ? NO_ACTIVE_CONFIG_HINT
      : (!_configReadyViewAvailable ? 'Only available in Images view' : '');
    verificationSection.classList.toggle('filter-section--disabled', !enabled);
  }
  _updateConfigReadyEnabled();

  // Bbox verification control — Any/Verified/Unverified, filtering bboxes by
  // their VERIFIED/UNVERIFIED tag. Placed next to
  // the whole-image Verification select since both live in the "verification"
  // area of the panel, but they are deliberately distinct filters (see
  // filterPredicate.js's defaultCuratedFilterState doc comment).
  const bboxVerificationControl = createBboxVerificationControl({
    value: _state.bboxVerification,
    onChange: (value) => {
      _state.bboxVerification = value;
      emit();
    },
  });

  // Image label conditions section — filter by whole-image label status
  // (present / absent / uncertain) against the image_labels table.
  const imageLabelSection = document.createElement('div');
  imageLabelSection.className = 'filter-section filter-section--grow';

  const imageLabelSectionLabel = document.createElement('div');
  imageLabelSectionLabel.className = 'filter-section-label';
  imageLabelSectionLabel.textContent = 'Image Label Filters';
  imageLabelSection.appendChild(imageLabelSectionLabel);
  imageLabelSection.appendChild(imageLabelConditionsComp.element);

  // Image tag conditions section — filter by whole-image tag status against the
  // image_tags table. The Tags dropdown in the toolbar above is bbox-scoped, so
  // this is the only surface that can match an image carrying no bboxes (e.g. a
  // collect-everything training run, which saves unannotated frames).
  const imageTagSection = document.createElement('div');
  imageTagSection.className = 'filter-section filter-section--grow';

  const imageTagSectionLabel = document.createElement('div');
  imageTagSectionLabel.className = 'filter-section-label';
  imageTagSectionLabel.textContent = 'Image Tag Filters';
  imageTagSection.appendChild(imageTagSectionLabel);
  imageTagSection.appendChild(imageTagConditionsComp.element);

  secondaryRow.appendChild(verificationSection);
  secondaryRow.appendChild(bboxVerificationControl.element);
  secondaryRow.appendChild(imageLabelSection);
  secondaryRow.appendChild(imageTagSection);
  panel.appendChild(secondaryRow);

  // ── Public API ────────────────────────────────────────────────────────────
  function getActiveFilterCount() {
    const s = _state;
    return s.tagUuids.length
      + s.tagExcludeUuids.length
      + s.labelUuids.length
      + s.labelExcludeUuids.length
      + s.imageLabelConditions.length
      + s.imageTagConditions.length
      + (s.filenameSearch ? 1 : 0)
      + (s.dateFrom ? 1 : 0)
      + (s.dateTo ? 1 : 0)
      + (s.verificationCheck ? 1 : 0)
      + (s.bboxVerification !== 'any' ? 1 : 0);
  }

  // ── Filter chip bar (slice 3, PR 3.2) ───────────────────────────────────
  // One descriptor per active filter dimension, for main.js to render as an
  // always-visible chip row. Reuses the existing category-based state
  // wholesale (no atomic term-list rewrite — see UI_REDESIGN §32.6): Tags and
  // Object Labels are each one aggregate chip (the multi-select dropdowns
  // they summarize stay put in the toolbar, unchanged); Image Label
  // Filters, being already a real per-condition list, gets one chip per row.
  function _labelName(uuid) { return labels.get(uuid)?.name ?? uuid; }
  function _tagName(uuid) { return tags.get(uuid)?.name ?? uuid; }

  function getChipDescriptors() {
    const s = _state;
    const chips = [];

    if (s.tagUuids.length > 0 || s.tagExcludeUuids.length > 0) {
      const names = [...s.tagUuids.map(_tagName), ...s.tagExcludeUuids.map(u => `¬${_tagName(u)}`)];
      chips.push({
        key: 'tags',
        label: `Tags: ${names.join(', ')}`,
        onRemove: () => setState({ tagUuids: [], tagExcludeUuids: [] }),
      });
    }

    if (s.labelUuids.length > 0 || s.labelExcludeUuids.length > 0) {
      const names = [...s.labelUuids.map(_labelName), ...s.labelExcludeUuids.map(u => `¬${_labelName(u)}`)];
      const rollUp = s.labelIncludeChildren ? ' +sub' : '';
      chips.push({
        key: 'labels',
        label: `Labels: ${names.join(', ')}${rollUp}`,
        onRemove: () => setState({ labelUuids: [], labelExcludeUuids: [], labelIncludeChildren: false }),
      });
    }

    s.imageLabelConditions.forEach((c, idx) => {
      const negate = c.negate ? '¬' : '';
      const rollUp = c.rollUp !== false && c.state !== 'absent' ? ' +sub' : '';
      chips.push({
        key: `imageLabel-${c.labelUuid}-${idx}`,
        label: `${negate}${_labelName(c.labelUuid)} · ${c.state}${rollUp}`,
        onRemove: () => setState({ imageLabelConditions: s.imageLabelConditions.filter((_, i) => i !== idx) }),
      });
    });

    s.imageTagConditions.forEach((c, idx) => {
      chips.push({
        key: `imageTag-${c.tagUuid}-${idx}`,
        label: `Image tag: ${_tagName(c.tagUuid)} · ${c.state}`,
        onRemove: () => setState({ imageTagConditions: s.imageTagConditions.filter((_, i) => i !== idx) }),
      });
    });

    if (s.filenameSearch) {
      const negate = s.filenameNegate ? '¬' : '';
      chips.push({
        key: 'filename',
        label: `${negate}File: ${s.filenameSearch}`,
        onRemove: () => setState({ filenameSearch: '', filenameNegate: false }),
      });
    }

    if (s.dateFrom || s.dateTo) {
      const negate = s.dateNegate ? '¬' : '';
      const from = s.dateFrom ? new Date(s.dateFrom).toLocaleDateString() : '…';
      const to = s.dateTo ? new Date(s.dateTo).toLocaleDateString() : '…';
      chips.push({
        key: 'date',
        label: `${negate}Date: ${from}–${to}`,
        onRemove: () => setState({ dateFrom: null, dateTo: null, dateNegate: false }),
      });
    }

    if (s.verificationCheck) {
      const text = s.verificationCheck === 'all_verified' ? 'Ready' : 'Needs review';
      chips.push({ key: 'configReady', label: `Config Ready: ${text}`, onRemove: () => setState({ verificationCheck: null }) });
    }

    if (s.bboxVerification !== 'any') {
      chips.push({ key: 'bboxVerification', label: `Bbox: ${s.bboxVerification}`, onRemove: () => setState({ bboxVerification: 'any' }) });
    }

    return chips;
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
      && s.imageLabelConditions.length === 0
      && s.imageTagConditions.length === 0
      && !s.filenameSearch
      && !s.filenameNegate
      && !s.dateFrom
      && !s.dateTo
      && !s.dateNegate
      && !s.verificationCheck
      && s.bboxVerification === 'any'
    );
  }

  /**
   * Programmatically set filter state. Syncs all DOM controls and fires
   * onFilterChange. Used by task action buttons ("See results").
   * @param {Partial<ReturnType<typeof DEFAULT_STATE>>} partial
   */
  function setState(partial) {
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
    if (partial.imageLabelConditions !== undefined) {
      imageLabelConditionsComp.setState(_state.imageLabelConditions);
    }
    if (partial.imageTagConditions !== undefined) {
      imageTagConditionsComp.setState(_state.imageTagConditions);
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
    if (partial.filenameNegate !== undefined) {
      filenameNegateCb.checked = !!_state.filenameNegate;
    }
    if (partial.dateNegate !== undefined) {
      dateNegateCb.checked = !!_state.dateNegate;
    }
    if (partial.verificationCheck !== undefined) {
      verificationSelect.value = _state.verificationCheck ?? '';
    }
    if (partial.bboxVerification !== undefined) {
      bboxVerificationControl.setValue(_state.bboxVerification);
    }

    emit();
  }

  function destroy() {
    tagsDropdown.destroy();
    _labelsDropdown?.destroy();
    imageLabelConditionsComp.destroy();
    imageTagConditionsComp.destroy();
    bboxVerificationControl.destroy();
    _unsubAll();
  }

  /**
   * Enable/disable the Config Ready control based on view context (V2: the
   * check is whole-image-only and meaningless in bboxes view). Combined (AND)
   * with the "has an active training config" gate above — see
   * _updateConfigReadyEnabled. Called by main.js on init and whenever
   * _viewFilter changes.
   * @param {boolean} available
   */
  function setConfigReadyAvailable(available) {
    _configReadyViewAvailable = available;
    _updateConfigReadyEnabled();
  }

  return {
    element: panel,
    /** Dropdown elements to place in the sticky toolbar (Tags + Object Labels). */
    toolbarControls: [tagsContainer, labelsDropdownContainer],
    getState: () => ({
      ..._state,
      showAllBboxes: _state.tagUuids.length === 0,
    }),
    setState,
    isDefault,
    getActiveFilterCount,
    getChipDescriptors,
    setConfigReadyAvailable,
    destroy,
  };
}
