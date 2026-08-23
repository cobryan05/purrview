/**
 * Filter panel for the Review page.
 *
 * createFilterPanel(labels, tags, onFilterChange)
 *   → {
 *       element,            // panel DOM node — EVERY filter control lives in here
 *       getState, setState, isDefault, getActiveFilterCount,
 *       destroy,
 *     }
 *
 * Calls onFilterChange(state) whenever any filter value changes.
 * The panel element starts visible; callers may toggle element.hidden as needed.
 *
 * Layout — one surface, no controls handed out to the caller's toolbar:
 *   Labels & Tags row:  Bbox Tags [▾]  |  Bbox Labels [▾]
 *                       Image Label Filters: [conditions]  |  Image Tag Filters: [conditions]
 *   Primary row:        Date: [from]–[to]  |  File: [____]
 *   Secondary row:      Source [▾]  |  Config Ready: [<select>]
 *                       Box status: [<select>]  |  Confidence: [min%]–[max%]
 */
import { VERIFICATION_TAG_UUIDS } from '/app-static/js/shared/api/review.js';
import { createMultiSelectDropdown, makeLabelFilterNotice } from '/app-static/js/ui/utils/index.js';
import { createImageLabelConditions, createImageTagConditions } from '/app-static/js/ui/imageTagConditions.js';
import { createBboxVerificationControl } from '/app-static/js/app/review/bboxVerificationControl.js';
import { createConfidenceRangeControl } from '/app-static/js/app/review/confidenceRangeControl.js';
import { createFilenameConditions } from '/app-static/js/app/review/filenameConditions.js';
import { flattenHierarchy } from '/app-static/js/ui/labelList.js';
import { filterVisibleTags, groupTagsByKind } from '/app-static/js/shared/activeModel.js';
import { activeTrainingConfig } from '/app-static/js/shared/activeTrainingConfig.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { fetchTags, refreshTagsIfUnknown } from '/app-static/js/shared/api/tags.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchSourceList } from '/app-static/js/shared/api/sources.js';
import { NO_SOURCE_KEY } from '/app-static/js/shared/api/review.js';

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
  // Which sources produced the image (images.source_uuid) — "the camera that
  // took it". Selecting several ORs them: an image has exactly one source, so
  // there is no "all of" mode here. NO_SOURCE_KEY stands for images with no
  // source recorded (manual uploads, and anything collected before the column
  // existed) — otherwise unreachable, since NULL never satisfies IN.
  sourceUuids: [],
  sourceExcludeUuids: [],
  // Filename filters, one entry per row of the File list:
  // [{ pattern: string, negate: boolean }]. Includes are combined by
  // filenameMatchMode; every exclude applies. Replaces the old single
  // filenameSearch/filenameNegate pair, which could only hold one filter —
  // setState still accepts that older shape and folds it in.
  filenameConditions: [],
  filenameMatchMode: 'any',
  dateFrom: null,
  dateTo: null,
  dateNegate: false,
  verificationCheck: null,
  // Bbox verification tri-state (V1) — filters bboxes by their VERIFIED /
  // UNVERIFIED tag. Distinct from `verificationCheck` above,
  // which is the whole-image label-completeness check. 'any' = no-op, same
  // as filterPredicate.js's defaultCuratedFilterState().
  bboxVerification: 'any',
  // Model-confidence band on bboxes, 0–1 (the control talks percent). Both
  // bounds independently optional; null is the no-op. Pairs naturally with
  // bboxVerification: "unverified boxes the model was unsure about".
  confidenceMin: null,
  confidenceMax: null,
  confidenceIncludeMissing: false,
});

/**
 * @param {Map<string, object>} labels   - label uuid → label record
 * @param {Map<string, object>} tags     - tag uuid → tag record
 * @param {(state: object) => void} onFilterChange
 */
/**
 * @param {Map} labels
 * @param {Map} tags
 * @param {(state: Object) => void} onFilterChange
 * @param {Object} [opts]
 * @param {boolean} [opts.inlineControls]
 * @param {() => void} [opts.onNamesChanged] - called after a tag refetch changed
 *   the names this panel renders, so the caller can redraw its chip bar. Kept
 *   separate from `onFilterChange`, which re-runs the whole image query.
 */
export function createFilterPanel(labels, tags, onFilterChange, { inlineControls = false, onNamesChanged = null } = {}) {
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

  // Stable container: the dropdown inside is destroyed and rebuilt (tag-kind
  // visibility changes), so the panel must hold the wrapper, not the dropdown.
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

  // ── Source multi-select ───────────────────────────────────────────────────
  // Sources are fetched rather than passed in: unlike labels and tags they are
  // not part of the panel's constructor contract, and every other consumer of
  // the source list goes through the same cached fetcher.
  /** @type {Map<string, object>} */
  let _sources = new Map();
  let _sourcesDropdown = null;
  // Set by destroy(); read by the in-flight source fetch below.
  let _destroyed = false;

  function _buildSourcesDropdown() {
    const items = [...(_sources?.values?.() ?? [])]
      .map(s => ({ uuid: s.uuid, name: s.name || s.uuid }))
      .sort((a, b) => a.name.localeCompare(b.name));
    // Always offered, even with no sources configured: "images with no source"
    // is a meaningful thing to ask for on its own.
    items.push({ isSeparator: true, label: 'Other' });
    items.push({ uuid: NO_SOURCE_KEY, name: 'No source recorded' });

    return createMultiSelectDropdown({
      label: 'Source',
      items,
      defaultSelected: [],
      // An image has exactly one source, so "all of" would match nothing —
      // offering the toggle would only invite that dead end.
      showModeToggle: false,
      filterKey: 'sources',
      inline: inlineControls,
      defaultWidth: '220px',
      defaultHeight: '300px',
      onChange: (selected, excluded) => {
        _state.sourceUuids = selected;
        _state.sourceExcludeUuids = excluded;
        emit();
      },
    });
  }

  const sourcesContainer = document.createElement('div');
  if (inlineControls) sourcesContainer.style.width = '100%';
  _sourcesDropdown = _buildSourcesDropdown();
  sourcesContainer.appendChild(_sourcesDropdown.element);

  // Populate for real once the source list arrives. The selection is re-applied
  // rather than reset: setState() can run before the fetch resolves (a saved
  // View, a "See results" button), and dropping it would silently lose a filter
  // the caller already asked for.
  fetchSourceList()
    .then((map) => {
      // The panel can be torn down while this is in flight (a tab switch during
      // the fetch). Rebuilding then would install a fresh dropdown — with its
      // document-level listeners and resize observer — that nothing will ever
      // destroy again, since destroy() has already run.
      if (_destroyed) return;
      _sources = map;
      _sourcesDropdown.destroy();
      sourcesContainer.innerHTML = '';
      _sourcesDropdown = _buildSourcesDropdown();
      sourcesContainer.appendChild(_sourcesDropdown.element);
      _sourcesDropdown.setState(_state.sourceUuids, undefined, undefined, _state.sourceExcludeUuids);
    })
    .catch(() => { /* leave the "No source recorded" entry as the only option */ });

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
      // Chips render tag names straight out of `tags`; they are stale until told.
      onNamesChanged?.();
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

  // Multiple filename filters, each independently negatable — see
  // filenameConditions.js. Starts with one blank row so the common
  // single-filter case needs no extra click.
  const filenameConditionsComp = createFilenameConditions(({ conditions, matchMode }) => {
    _state.filenameConditions = conditions;
    _state.filenameMatchMode = matchMode;
    emit();
  });
  filenameConditionsComp.setState({ conditions: [{ pattern: '', negate: false }], matchMode: 'any' });

  primaryRow.appendChild(filenameConditionsComp.element);

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

  // Model-confidence band — filters bboxes by the inference confidence stored
  // in their metadata. Sits next to the bbox verification control because the
  // two are almost always used together (triage the low-confidence unverified
  // boxes first).
  const confidenceControl = createConfidenceRangeControl({
    min: _state.confidenceMin,
    max: _state.confidenceMax,
    includeMissing: _state.confidenceIncludeMissing,
    onChange: ({ min, max, includeMissing }) => {
      _state.confidenceMin = min;
      _state.confidenceMax = max;
      _state.confidenceIncludeMissing = includeMissing;
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

  // Source section — which camera/stream/file produced the image.
  const sourceSection = document.createElement('div');
  sourceSection.className = 'filter-section';

  const sourceSectionLabel = document.createElement('div');
  sourceSectionLabel.className = 'filter-section-label';
  sourceSectionLabel.textContent = 'Source';
  sourceSection.appendChild(sourceSectionLabel);
  sourceSection.appendChild(sourcesContainer);

  secondaryRow.appendChild(sourceSection);
  secondaryRow.appendChild(verificationSection);
  secondaryRow.appendChild(bboxVerificationControl.element);
  secondaryRow.appendChild(confidenceControl.element);
  panel.appendChild(secondaryRow);

  // ── Labels & Tags row ─────────────────────────────────────────────────────
  // All four label/tag dimensions live here, together, at the top of the panel.
  // The bbox-scoped Tags and Labels dropdowns used to be handed to main.js as
  // `toolbarControls` and mounted in the sticky toolbar instead, which split
  // label/tag filtering across two surfaces: two of them a click away in the
  // toolbar, the other two inside this popover. Nothing distinguished the two
  // groups except history, so there was no rule for which you reached for.
  //
  // Bbox- vs image-scoped is the distinction that actually matters, so it is
  // the one the section labels carry.
  const labelTagRow = document.createElement('div');
  labelTagRow.className = 'filter-row-secondary';

  const makeFilterSection = (text, child, grow = false) => {
    const section = document.createElement('div');
    section.className = grow ? 'filter-section filter-section--grow' : 'filter-section';
    const sectionLabel = document.createElement('div');
    sectionLabel.className = 'filter-section-label';
    sectionLabel.textContent = text;
    section.appendChild(sectionLabel);
    section.appendChild(child);
    return section;
  };

  // Inline mode (the dataset form) renders these two as always-expanded boxes
  // that fill their width, so they have to grow and wrap rather than sit at
  // their content size the way the popover's collapsed triggers do.
  labelTagRow.appendChild(makeFilterSection('Bbox Tags', tagsContainer, inlineControls));
  labelTagRow.appendChild(makeFilterSection('Bbox Labels', labelsDropdownContainer, inlineControls));
  labelTagRow.appendChild(imageLabelSection);
  labelTagRow.appendChild(imageTagSection);
  panel.insertBefore(labelTagRow, panel.firstChild);

  // ── Public API ────────────────────────────────────────────────────────────
  // The File list always carries at least one row, and a row the user hasn't
  // typed into yet is not a filter — every consumer (count, chips, isDefault,
  // the request body) sees only the non-blank ones.
  function _activeFilenameConditions() {
    return (_state.filenameConditions ?? [])
      .map(c => ({ pattern: (c.pattern ?? '').trim(), negate: !!c.negate }))
      .filter(c => c.pattern);
  }

  function getActiveFilterCount() {
    const s = _state;
    return s.tagUuids.length
      + s.tagExcludeUuids.length
      + s.labelUuids.length
      + s.labelExcludeUuids.length
      + s.imageLabelConditions.length
      + s.imageTagConditions.length
      + s.sourceUuids.length
      + s.sourceExcludeUuids.length
      + _activeFilenameConditions().length
      + (s.dateFrom ? 1 : 0)
      + (s.dateTo ? 1 : 0)
      + (s.verificationCheck ? 1 : 0)
      + (s.bboxVerification !== 'any' ? 1 : 0)
      // One chip's worth regardless of whether one bound or both are set.
      + (s.confidenceMin !== null || s.confidenceMax !== null ? 1 : 0);
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
  function _sourceName(uuid) {
    if (uuid === NO_SOURCE_KEY) return 'No source';
    return _sources.get(uuid)?.name ?? uuid;
  }

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

    if (s.sourceUuids.length > 0 || s.sourceExcludeUuids.length > 0) {
      const names = [...s.sourceUuids.map(_sourceName), ...s.sourceExcludeUuids.map(u => `¬${_sourceName(u)}`)];
      chips.push({
        key: 'sources',
        label: `Source: ${names.join(', ')}`,
        onRemove: () => setState({ sourceUuids: [], sourceExcludeUuids: [] }),
      });
    }

    // One chip per filename filter — removing a chip must drop only that
    // pattern, so the chip is keyed by its position in the (blank-free) list
    // and removal is expressed against the raw state at the same pattern.
    _activeFilenameConditions().forEach((c, idx) => {
      const negate = c.negate ? '¬' : '';
      chips.push({
        key: `filename-${idx}-${c.pattern}`,
        label: `${negate}File: ${c.pattern}`,
        onRemove: () => {
          const remaining = _activeFilenameConditions().filter((_, i) => i !== idx);
          setState({ filenameConditions: remaining });
        },
      });
    });

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
      // 'no_boxes' is a state name, not a description — spelled out here so the chip
      // reads as a sentence rather than as an identifier that leaked into the UI.
      const verificationChipText = s.bboxVerification === 'no_boxes' ? 'no boxes' : s.bboxVerification;
      chips.push({ key: 'bboxVerification', label: `Bbox: ${verificationChipText}`, onRemove: () => setState({ bboxVerification: 'any' }) });
    }

    if (s.confidenceMin !== null || s.confidenceMax !== null) {
      const pct = v => (v === null ? '…' : `${Math.round(v * 100)}%`);
      const missing = s.confidenceIncludeMissing ? ' +none' : '';
      chips.push({
        key: 'confidence',
        label: `Conf: ${pct(s.confidenceMin)}–${pct(s.confidenceMax)}${missing}`,
        onRemove: () => setState({ confidenceMin: null, confidenceMax: null, confidenceIncludeMissing: false }),
      });
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
      && s.sourceUuids.length === 0
      && s.sourceExcludeUuids.length === 0
      && _activeFilenameConditions().length === 0
      && !s.dateFrom
      && !s.dateTo
      && !s.dateNegate
      && !s.verificationCheck
      && s.bboxVerification === 'any'
      && s.confidenceMin === null
      && s.confidenceMax === null
      // confidenceIncludeMissing is deliberately NOT checked: with no bounds it
      // matches every bbox, so on its own it is not a filter — same rule
      // getActiveFilterCount and the chip row use.
    );
  }

  /**
   * Programmatically set filter state. Syncs all DOM controls and fires
   * onFilterChange. Used by task action buttons ("See results").
   * @param {Partial<ReturnType<typeof DEFAULT_STATE>>} partial
   */
  function setState(partial) {
    // Accept the pre-multi-filter shape (a single filenameSearch/filenameNegate
    // pair) and fold it into the condition list. Saved Views, task "See
    // results" buttons and anything else written before this list existed still
    // set it, and silently ignoring it would drop their filename filter.
    if (partial.filenameSearch !== undefined || partial.filenameNegate !== undefined) {
      const legacy = (partial.filenameSearch ?? '').trim();
      const legacyConditions = legacy ? [{ pattern: legacy, negate: !!partial.filenameNegate }] : [];
      const { filenameSearch: _fs, filenameNegate: _fn, ...rest } = partial;
      // An explicit filenameConditions in the same call wins — the caller is
      // then already speaking the current shape.
      partial = { filenameConditions: legacyConditions, ...rest };
    }
    if (partial.tagUuids !== undefined && !('tagExcludeUuids' in partial)) {
      partial = { ...partial, tagExcludeUuids: [] };
    }
    if (partial.labelUuids !== undefined && !('labelExcludeUuids' in partial)) {
      partial = { ...partial, labelExcludeUuids: [] };
    }
    if (partial.sourceUuids !== undefined && !('sourceExcludeUuids' in partial)) {
      partial = { ...partial, sourceExcludeUuids: [] };
    }
    _state = { ..._state, ...partial };

    // A filter can name a tag this session has never fetched — most often one a
    // task minted server-side. Heal the cache so the chip shows a name instead
    // of the raw uuid; the TAGS_UPDATED it publishes drives the re-render.
    refreshTagsIfUnknown([
      ..._state.tagUuids,
      ..._state.tagExcludeUuids,
      ..._state.imageTagConditions.map(c => c.tagUuid),
    ]).catch(() => {});

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
    if (partial.sourceUuids !== undefined || partial.sourceExcludeUuids !== undefined) {
      _sourcesDropdown?.setState(_state.sourceUuids, undefined, undefined, _state.sourceExcludeUuids);
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
    if (partial.filenameConditions !== undefined || partial.filenameMatchMode !== undefined) {
      // The list must never render empty — with no rows there is no input to
      // type into, and clearing the last filter would leave the File filter
      // unreachable until the panel was rebuilt.
      const rows = _state.filenameConditions.length > 0
        ? _state.filenameConditions
        : [{ pattern: '', negate: false }];
      filenameConditionsComp.setState({ conditions: rows, matchMode: _state.filenameMatchMode });
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
    if (partial.confidenceMin !== undefined || partial.confidenceMax !== undefined
        || partial.confidenceIncludeMissing !== undefined) {
      confidenceControl.setValue({
        min: _state.confidenceMin,
        max: _state.confidenceMax,
        includeMissing: _state.confidenceIncludeMissing,
      });
    }

    emit();
  }

  function destroy() {
    _destroyed = true;
    tagsDropdown.destroy();
    _labelsDropdown?.destroy();
    _sourcesDropdown?.destroy();
    imageLabelConditionsComp.destroy();
    imageTagConditionsComp.destroy();
    bboxVerificationControl.destroy();
    confidenceControl.destroy();
    filenameConditionsComp.destroy();
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
    getState: () => ({
      ..._state,
      // Blank rows are UI, not filter state — see _activeFilenameConditions.
      filenameConditions: _activeFilenameConditions(),
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
