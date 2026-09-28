/**
 * The NVR event search's filter bar.
 *
 * Deliberately *not* `app/review/filters.js`. That panel filters images by their
 * annotations — labels, tags, verification state, confidence bands — and every one of
 * its controls asks a question an event cannot answer. What the two share is the shape
 * of the interaction (a bar of controls that reports a state object upward, and a
 * results grid that re-queries on change), not the controls themselves.
 *
 * Every option offered here comes from the *facets* endpoint, i.e. from what history
 * actually contains. A dropdown built from the watcher config instead would list rules
 * that have never fired and omit the ones deleted last month that account for half the
 * events on screen.
 */
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchWatchEventFacets } from '/app-static/js/shared/api/watchProfiles.js';
import { localWallClockIso } from '/app-static/js/shared/time.js';
import { expandWithDescendants, flattenHierarchy } from '/app-static/js/ui/labelList.js';
import { createMultiSelectDropdown, Logger } from '/app-static/js/ui/utils/index.js';

/** @typedef {{sourceUuids: string[], ruleUuids: string[], subjectRefs: string[],
 *             targetRefs: string[], presentRefs: string[], sourceExcludeUuids: string[],
 *             ruleExcludeUuids: string[],
 *             subjectExcludeRefs: string[], targetExcludeRefs: string[], presentExcludeRefs: string[],
 *             subjectIncludeChildren: boolean, targetIncludeChildren: boolean,
 *             presentIncludeChildren: boolean,
 *             startFrom: ?string, startTo: ?string,
 *             hasRecording: ?boolean, recordingKind: ?string,
 *             reviewState: ?string, pinned: ?boolean}} EventFilterState */

/**
 * The five facet filters, as `[state key for includes, state key for excludes, facet key,
 * label, "no filter" text, testid, state key for the roll-up toggle or null]`.
 *
 * One table rather than four near-identical blocks: every one of them is the same
 * control over a different column, and the three places that need them — building,
 * restoring after a facet refresh, and reporting emptiness — were drifting apart when
 * each was written out longhand.
 *
 * Only the label-valued facets carry a roll-up key. A rule and a camera are flat
 * entities; a subject or target ref is (usually) a label uuid and `present` always is,
 * and labels are a tree.
 */
export const FACET_FILTERS = [
  ['ruleUuids', 'ruleExcludeUuids', 'rules', 'Event type', 'Any event type', 'event-filter-rule', null],
  ['sourceUuids', 'sourceExcludeUuids', 'sources', 'Camera', 'Any camera', 'event-filter-source', null],
  ['subjectRefs', 'subjectExcludeRefs', 'subjects', 'Acting', 'Anything acting', 'event-filter-subject', 'subjectIncludeChildren'],
  ['targetRefs', 'targetExcludeRefs', 'targets', 'Acted on', 'Anything acted on', 'event-filter-target', 'targetIncludeChildren'],
  // Not an actor slot: what the clip *contains*, at any point, whether or not the rule
  // cared — the catalogue `WatchEventLabel` keeps. The one facet backed by a one-to-many
  // table, which is why it is the only one whose value can appear on an event more than
  // once and the only one an event can have none of.
  ['presentRefs', 'presentExcludeRefs', 'present', 'Present', 'Anything present', 'event-filter-present', 'presentIncludeChildren'],
];

/** The roll-up state keys, for the places that must not treat one as a filter. */
const HIERARCHY_KEYS = FACET_FILTERS.map(([, , , , , , key]) => key).filter(Boolean);

/**
 * The Video select's options, as `[value, label]`.
 *
 * One control rather than two, because the four questions are alternatives a user picks
 * between — "no video" and "has a timelapse" cannot both be on. The value encodes which
 * of the two filters it sets: a bare boolean for the has/hasn't pair, `kind:<name>` for
 * a specific medium, which keeps the mapping in one place instead of spread across the
 * read and the write.
 */
const VIDEO_FILTER_OPTIONS = [
  ['', 'With or without video'],
  ['true', 'Has video'],
  ['false', 'No video'],
  ['kind:clip', 'Has a clip'],
  ['kind:timelapse', 'Has a timelapse'],
];

/**
 * The Review select's options, as `[value, label]`.
 *
 * Values are the `WatchEventReviewState` strings verbatim — `watchEventFilterParams`
 * passes `reviewState` straight through and the backend validates it against the enum,
 * so a label invented here would 422 rather than quietly matching nothing.
 *
 * "Labels were wrong" is the queue the correction flow fills: flagging an event is only
 * worth doing if the flagged ones can be found again afterwards.
 */
const REVIEW_FILTER_OPTIONS = [
  ['', 'Any review state'],
  ['unreviewed', 'Not yet reviewed'],
  ['flagged', 'Labels were wrong'],
  ['confirmed', 'Reviewed'],
  ['dismissed', 'Dismissed'],
];

/** The Pinned select's options, as `[value, label]`. */
const PINNED_FILTER_OPTIONS = [
  ['', 'Pinned or not'],
  ['true', '📌 Pinned only'],
  ['false', 'Not pinned'],
];

/** @param {EventFilterState} state @returns {string} the select value for a state */
export function videoFilterValue(state) {
  if (state.recordingKind) return `kind:${state.recordingKind}`;
  return state.hasRecording == null ? '' : String(state.hasRecording);
}

/**
 * Apply a Video select value to the state, clearing whichever of the two filters this
 * choice does not set — otherwise picking "has a timelapse" after "no video" would ask
 * for both at once and always return nothing.
 *
 * @param {EventFilterState} state
 * @param {string} value
 */
export function applyVideoFilter(state, value) {
  if (value.startsWith('kind:')) {
    state.recordingKind = value.slice('kind:'.length);
    state.hasRecording = null;
    return;
  }
  state.recordingKind = null;
  state.hasRecording = value === '' ? null : value === 'true';
}

/**
 * The relative range presets, as `[days, label]`. `null` days is "all of history".
 *
 * Whole *local* days, not rolling hours: the bounds have to survive a round trip through
 * `<input type="date">`, which cannot represent an hour. "Last 24 hours" would display as
 * a plain date and quietly mean something else — the same mismatch that keeps the
 * timeline's clicked-hour window out of the filter state entirely.
 */
export const RANGE_PRESETS = [
  [7, 'Last 7 days'],
  [1, 'Today'],
  [30, 'Last 30 days'],
  [90, 'Last 3 months'],
  [365, 'Last year'],
  [null, 'All time'],
];

/**
 * How much history the page shows before anyone touches a control.
 *
 * Unfiltered used to mean "everything", which grows without bound — the log is
 * append-only and already covers months. A default range is a filter the user did not
 * ask for, so the control naming it is always on screen and the empty state says so.
 */
export const DEFAULT_RANGE_DAYS = 7;

const DAY_MS = 24 * 60 * 60 * 1000;

/**
 * `YYYY-MM-DD` in the **viewer's** timezone.
 *
 * `toISOString().slice(0, 10)` is the obvious spelling and is wrong: it formats in UTC,
 * so for anyone west of Greenwich "today" becomes yesterday for part of every day.
 *
 * @param {Date} date
 * @returns {string}
 */
export function toDateInputValue(date) {
  const pad = n => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

/**
 * The `days` whole local days ending today.
 *
 * @param {number} days
 * @returns {{from: string, to: string}} date-input values
 */
export function rangeEndingToday(days) {
  const end = new Date();
  const start = new Date(end.getTime() - (days - 1) * DAY_MS);
  return { from: toDateInputValue(start), to: toDateInputValue(end) };
}

/**
 * Slide a window by its own length — what the prev/next buttons do.
 *
 * Derived from the window rather than from the chosen preset, so the buttons keep
 * working after the dates have been edited by hand: stepping a custom 3-day range moves
 * by 3 days, which is the only step that makes the two windows abut without a gap or an
 * overlap.
 *
 * @param {string} from - date-input value
 * @param {string} to
 * @param {number} direction - -1 for earlier, +1 for later
 * @returns {{from: string, to: string}}
 */
export function shiftRange(from, to, direction) {
  // Parsed as local midnight, not through `new Date('YYYY-MM-DD')`, which the spec says
  // to read as UTC — that would drift the window by a day for most of the world.
  const parse = value => {
    const [y, m, d] = value.split('-').map(Number);
    return new Date(y, m - 1, d);
  };
  const start = parse(from);
  const end = parse(to);
  const span = Math.round((end - start) / DAY_MS) + 1;
  return {
    from: toDateInputValue(new Date(start.getTime() + direction * span * DAY_MS)),
    to: toDateInputValue(new Date(end.getTime() + direction * span * DAY_MS)),
  };
}

/** @returns {EventFilterState} */
export function defaultFilterState() {
  const state = {
    startFrom: null,
    startTo: null,
    // Which preset the dates came from, or null for a hand-picked range. A UI mode
    // rather than a filter — `watchEventFilterParams` never sends it, and
    // `hasActiveFilter` skips it, because the narrowing it implies is already fully
    // expressed by `startFrom`/`startTo`.
    rangeDays: null,
    hasRecording: null,
    recordingKind: null,
    reviewState: null,
    pinned: null,
  };
  for (const [includeKey, excludeKey, , , , , hierarchyKey] of FACET_FILTERS) {
    state[includeKey] = [];
    state[excludeKey] = [];
    if (hierarchyKey) state[hierarchyKey] = false;
  }
  return state;
}

/**
 * Whether any filter is actually narrowing the search.
 *
 * An empty array is not a filter, so a plain "is anything non-null" test would report
 * every fresh page as filtered and offer the user a Clear button that changes nothing.
 *
 * @param {EventFilterState} state
 */
export function hasActiveFilter(state) {
  // The range the page opens on is not a filter the *user* applied, and counting it made
  // this always true: an install with no events at all answered "No events match these
  // filters — try widening the date range" instead of the one screen that explains where
  // events come from. Only a range that has been changed from the default counts.
  const skipDates = state?.rangeDays === DEFAULT_RANGE_DAYS;
  const ignored = new Set([
    // `rangeDays` only records which button produced the dates; counting it too would
    // make a range look like two filters and would report "filtered" for a state whose
    // dates had been cleared.
    'rangeDays',
    // Same for the roll-up toggles, and here it is not merely tidiness: they are plain
    // booleans, so a `v != null` test counts `false` and every fresh page would report
    // itself filtered. A roll-up only widens a selection that already counts on its own.
    ...HIERARCHY_KEYS,
    ...(skipDates ? ['startFrom', 'startTo'] : []),
  ]);
  return Object.entries(state).some(
    ([key, v]) => !ignored.has(key) && (Array.isArray(v) ? v.length > 0 : v != null),
  );
}

/**
 * The dropdown rows for one actor facet: its refs arranged as the label tree.
 *
 * Two things are not one-to-one with the facet list. **Ancestors are added even when
 * nothing ever acted as one** — `Bowl` has no events of its own, only its four children
 * do, and without it there is no row to tick for "at any bowl". That is a deliberate
 * exception to this file's rule against offering options history cannot match: with the
 * roll-up on, an ancestor matches exactly the events its descendants hold. **Refs that
 * are not labels** — a motion zone is named in the watcher profile, not the label tree —
 * keep their facet name and sit below a divider, since they have no place in the tree.
 *
 * @param {Array<{ref: string, name: string}>} options - one facet's values
 * @param {Map<string, object>} labels
 * @returns {Array<Object>} createMultiSelectDropdown items
 */
export function labelFacetItems(options, labels) {
  /** @type {Map<string, object>} the labels the tree needs: the refs and their ancestors */
  const tree = new Map();
  for (const { ref } of options) {
    let label = labels.get(ref);
    while (label && !tree.has(label.uuid)) {
      const parent = label.parentUuid ? labels.get(label.parentUuid) : null;
      // A dangling parentUuid — the parent was deleted — is re-rooted rather than left
      // as is: flattenHierarchy only descends from null, so an orphan keeping a parent
      // nothing holds would be dropped from the list entirely.
      tree.set(label.uuid, parent || !label.parentUuid ? label : { ...label, parentUuid: null });
      label = parent;
    }
  }
  const items = flattenHierarchy(tree).map(l => ({
    uuid: l.uuid,
    name: l.name,
    color: l.color || null,
    depth: l.depth,
    parentUuid: l.parentUuid ?? null,
  }));
  const others = options.filter(o => !labels.has(o.ref));
  if (others.length > 0) {
    if (items.length > 0) items.push({ isSeparator: true, label: 'Zones & other' });
    for (const o of others) items.push({ uuid: o.ref, name: o.name });
  }
  return items;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * A labelled control. The label is a real <label> bound to the control, so the whole
 * thing is clickable and screen readers announce the pair.
 */
function field(labelText, control, testid) {
  const wrap = el('div', 'event-filters__field');
  const id = `event-filter-${testid}`;
  control.id = id;
  if (testid) control.dataset.testid = testid;
  const label = el('label', 'event-filters__label', labelText);
  label.htmlFor = id;
  wrap.appendChild(label);
  wrap.appendChild(control);
  return wrap;
}

/**
 * A labelled multi-select dropdown, wrapped so the whole control carries the testid.
 *
 * The dropdown's own element is a div, not a form control, so it cannot be the target of
 * a `<label for>` the way the plain `<select>`s this replaced could.
 *
 * @param {string} labelText
 * @param {{element: HTMLElement}} dropdown
 * @param {string} testid
 */
function dropdownField(labelText, dropdown, testid) {
  const wrap = el('div', 'event-filters__field');
  wrap.dataset.testid = testid;
  wrap.appendChild(el('span', 'event-filters__label', labelText));
  wrap.appendChild(dropdown.element);
  return wrap;
}

function dateInput() {
  const node = document.createElement('input');
  node.type = 'date';
  node.className = 'event-filters__date';
  return node;
}

/**
 * Build the filter bar.
 *
 * @param {(state: EventFilterState) => void} onChange - called on every committed change
 * @returns {Promise<{element: HTMLElement, getState: () => EventFilterState,
 *                    reset: () => void, refreshFacets: () => Promise<void>}>}
 */
export async function createEventFilters(onChange) {
  const container = el('div', 'event-filters');
  container.dataset.testid = 'event-filters';

  const state = defaultFilterState();
  let facets = { rules: [], subjects: [], targets: [], sources: [], present: [] };
  // The label tree the two actor facets are drawn against, and the same tree the
  // roll-up expands through. Empty until the first fetch lands, and empty again only if
  // one fails — `labelFacetItems` then leaves every ref flat under the divider, which
  // is the old behaviour rather than a blank dropdown.
  /** @type {Map<string, object>} */
  let labels = new Map();

  const controls = el('div', 'event-filters__row');
  container.appendChild(controls);

  // Rebuilt on refresh rather than mutated: the option sets change wholesale when new
  // event kinds appear, and reconciling them in place is more code than re-rendering
  // them all.
  /** @type {Array<{destroy: () => void}>} live dropdowns, destroyed before each rebuild */
  let facetDropdowns = [];
  let clipSelect;
  let reviewSelect;
  let pinnedSelect;
  let rangeSelect;
  let prevBtn;
  let nextBtn;
  const fromInput = dateInput();
  const toInput = dateInput();

  /**
   * A copy no caller can write back through — the list values are arrays — with each
   * rolled-up facet expanded to its descendants.
   *
   * The expansion lives here rather than in the state because the state is what the
   * *dropdown* shows: writing the descendants back would tick eight bowls when the user
   * ticked one, and the next facet refresh would prune whichever of them history has no
   * events for. So the control keeps the choice and the query gets its closure.
   *
   * Exclusions expand too. One toggle reading "include child labels" that widened only
   * the ticked rows would leave "not any cat" matching every named cat.
   */
  function snapshot() {
    const copy = { ...state };
    for (const [includeKey, excludeKey, , , , , hierarchyKey] of FACET_FILTERS) {
      const rollUp = Boolean(hierarchyKey && state[hierarchyKey]);
      const resolve = refs => (rollUp ? [...expandWithDescendants(refs, labels)] : [...refs]);
      copy[includeKey] = resolve(state[includeKey]);
      copy[excludeKey] = resolve(state[excludeKey]);
    }
    return copy;
  }

  function commit() {
    onChange(snapshot());
  }

  function renderControls() {
    for (const dd of facetDropdowns) dd.destroy();
    facetDropdowns = [];
    controls.innerHTML = '';

    for (const [includeKey, excludeKey, facetKey, labelText, anyLabel, testid, hierarchyKey] of FACET_FILTERS) {
      const options = facets[facetKey] ?? [];
      const items = hierarchyKey ? labelFacetItems(options, labels) : options.map(o => ({ uuid: o.ref, name: o.name }));
      // The *rows*, not the facet values: a tree adds ancestors history has no events
      // for, and pruning against the raw facet list would clear a rolled-up parent
      // selection on the next refresh — the one selection the roll-up exists to make.
      const known = new Set(items.filter(i => !i.isSeparator).map(i => i.uuid));
      // Drop values history no longer holds (the last event with one was deleted).
      // Keeping them would show an empty result set with no visible cause; the rest of
      // the selection survives the rebuild, because a filter that silently cleared
      // itself on a facet refresh would look like it never worked.
      state[includeKey] = state[includeKey].filter(ref => known.has(ref));
      state[excludeKey] = state[excludeKey].filter(ref => known.has(ref));

      const dropdown = createMultiSelectDropdown({
        label: '',
        items,
        defaultSelected: [...state[includeKey]],
        // An event has exactly one camera, one rule, one subject and one target, so
        // "all of" could only ever match nothing — offering it would be a dead end. It
        // *would* mean something for `present`, where an event holds a set, but the
        // backend answers only "any of" (`present_refs`), so offering it here would
        // promise a narrowing that does not happen.
        showModeToggle: false,
        emptySummary: anyLabel,
        emptyText: 'Nothing recorded yet',
        searchable: items.length > 8,
        filterKey: `event-${facetKey}`,
        // Wider where the rows are a tree: two levels of indent plus a name do not fit
        // the flat facets' width.
        defaultWidth: hierarchyKey ? '260px' : '220px',
        defaultHeight: '260px',
        hierarchyToggle: hierarchyKey
          ? {
              label: 'Include child labels',
              default: state[hierarchyKey],
              onChange: value => {
                state[hierarchyKey] = value;
                commit();
              },
            }
          : null,
        onChange: (selected, excluded) => {
          state[includeKey] = selected;
          state[excludeKey] = excluded;
          commit();
        },
      });
      dropdown.setState(null, null, null, [...state[excludeKey]]);
      facetDropdowns.push(dropdown);
      controls.appendChild(dropdownField(labelText, dropdown, testid));
    }

    clipSelect = document.createElement('select');
    clipSelect.className = 'event-filters__select';
    for (const [value, label] of VIDEO_FILTER_OPTIONS) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      clipSelect.appendChild(option);
    }
    clipSelect.value = videoFilterValue(state);
    clipSelect.addEventListener('change', () => {
      applyVideoFilter(state, clipSelect.value);
      commit();
    });

    controls.appendChild(buildRangeControls());
    controls.appendChild(field('From', fromInput, 'event-filter-from'));
    controls.appendChild(field('To', toInput, 'event-filter-to'));
    controls.appendChild(field('Video', clipSelect, 'event-filter-clip'));

    reviewSelect = document.createElement('select');
    reviewSelect.className = 'event-filters__select';
    for (const [value, label] of REVIEW_FILTER_OPTIONS) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      reviewSelect.appendChild(option);
    }
    reviewSelect.value = state.reviewState ?? '';
    reviewSelect.addEventListener('change', () => {
      // Empty string means "any", which is the absence of a filter — stored as null so
      // hasActiveFilter does not count it and the Clear button stays honest.
      state.reviewState = reviewSelect.value || null;
      commit();
    });
    controls.appendChild(field('Review', reviewSelect, 'event-filter-review'));

    pinnedSelect = document.createElement('select');
    pinnedSelect.className = 'event-filters__select';
    for (const [value, label] of PINNED_FILTER_OPTIONS) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      pinnedSelect.appendChild(option);
    }
    pinnedSelect.value = state.pinned == null ? '' : String(state.pinned);
    pinnedSelect.addEventListener('change', () => {
      state.pinned = pinnedSelect.value === '' ? null : pinnedSelect.value === 'true';
      commit();
    });
    controls.appendChild(field('Pinned', pinnedSelect, 'event-filter-pinned'));

    const clear = document.createElement('button');
    clear.type = 'button';
    clear.className = 'panel-btn event-filters__clear';
    clear.textContent = 'Clear';
    clear.dataset.testid = 'event-filter-clear';
    clear.addEventListener('click', () => reset());
    controls.appendChild(clear);

    // The range controls are rebuilt along with everything else, so the freshly created
    // select starts on its first option and both buttons start enabled. Without this a
    // facet refresh silently relabels the current range as "Last 7 days".
    syncRangeControls();
  }

  /**
   * Write a window into the state and the two date inputs at once.
   *
   * The inputs stay the single visible truth about what is filtered — a relative preset
   * is a *way of choosing* dates, not a second, competing notion of time the user cannot
   * see. That is also what lets prev/next work on a hand-typed range.
   *
   * @param {?{from: string, to: string}} window - date-input values, or null for all time
   */
  function applyRange(window) {
    fromInput.value = window ? window.from : '';
    toInput.value = window ? window.to : '';
    // `To` covers the whole day it names — otherwise picking one day for both bounds
    // returns only events at exactly 00:00:00.
    state.startFrom = window ? localWallClockIso(window.from, '00:00:00') : null;
    state.startTo = window ? localWallClockIso(window.to, '23:59:59') : null;
    syncRangeControls();
  }

  function syncRangeControls() {
    if (!rangeSelect) return;
    rangeSelect.value = state.rangeDays == null ? (state.startFrom ? 'custom' : '') : String(state.rangeDays);
    const hasWindow = Boolean(fromInput.value && toInput.value);
    prevBtn.disabled = !hasWindow;
    // Stepping forward past today would land on a window that cannot contain anything
    // yet — a control that only ever empties the page is worse than a disabled one.
    nextBtn.disabled = !hasWindow || toInput.value >= toDateInputValue(new Date());
  }

  function step(direction) {
    if (!fromInput.value || !toInput.value) return;
    // Stepping leaves the preset behind: "Last 7 days" names a window ending today, and
    // the week before last is not that. The label has to stop claiming otherwise.
    state.rangeDays = null;
    applyRange(shiftRange(fromInput.value, toInput.value, direction));
    commit();
  }

  function buildRangeControls() {
    const wrap = el('div', 'event-filters__field event-filters__range');
    wrap.dataset.testid = 'event-filter-range';
    wrap.appendChild(el('span', 'event-filters__label', 'Range'));
    const row = el('div', 'event-filters__range-row');

    prevBtn = el('button', 'panel-btn event-filters__step', '‹');
    prevBtn.type = 'button';
    prevBtn.title = 'Jump back one window';
    prevBtn.dataset.testid = 'event-filter-range-prev';
    prevBtn.addEventListener('click', () => step(-1));

    rangeSelect = document.createElement('select');
    rangeSelect.className = 'event-filters__select';
    rangeSelect.dataset.testid = 'event-filter-range-select';
    for (const [days, label] of RANGE_PRESETS) {
      const option = document.createElement('option');
      option.value = days == null ? '' : String(days);
      option.textContent = label;
      rangeSelect.appendChild(option);
    }
    // Not offered, only reachable: "Custom" is what the control *becomes* when the dates
    // stop matching a preset, so picking it from the list would mean nothing.
    const custom = document.createElement('option');
    custom.value = 'custom';
    custom.textContent = 'Custom range';
    custom.disabled = true;
    rangeSelect.appendChild(custom);
    rangeSelect.addEventListener('change', () => {
      const value = rangeSelect.value;
      state.rangeDays = value === '' ? null : Number(value);
      applyRange(state.rangeDays == null ? null : rangeEndingToday(state.rangeDays));
      commit();
    });

    nextBtn = el('button', 'panel-btn event-filters__step', '›');
    nextBtn.type = 'button';
    nextBtn.title = 'Jump forward one window';
    nextBtn.dataset.testid = 'event-filter-range-next';
    nextBtn.addEventListener('click', () => step(1));

    row.appendChild(prevBtn);
    row.appendChild(rangeSelect);
    row.appendChild(nextBtn);
    wrap.appendChild(row);
    return wrap;
  }

  // A <input type="date"> gives a bare YYYY-MM-DD, which the backend would read as
  // midnight UTC. `To` therefore has to cover the whole day it names, or picking the
  // same day for both returns only events at exactly 00:00:00.
  //
  // Editing either by hand drops the preset: the dates are the truth, and a select still
  // reading "Last 7 days" over a range that is not the last 7 days would be a lie.
  fromInput.addEventListener('change', () => {
    state.startFrom = fromInput.value ? localWallClockIso(fromInput.value, '00:00:00') : null;
    state.rangeDays = null;
    syncRangeControls();
    commit();
  });
  toInput.addEventListener('change', () => {
    state.startTo = toInput.value ? localWallClockIso(toInput.value, '23:59:59') : null;
    state.rangeDays = null;
    syncRangeControls();
    commit();
  });

  function reset() {
    Object.assign(state, defaultFilterState());
    renderControls();
    // Back to the range the page opens on, not to unbounded history: "Clear" should put
    // the page in the state you would get by reloading it, and quietly widening to every
    // event ever recorded is a bigger change than the button appears to offer.
    state.rangeDays = DEFAULT_RANGE_DAYS;
    applyRange(rangeEndingToday(DEFAULT_RANGE_DAYS));
    commit();
  }

  /**
   * Replace the facet selections wholesale — how a group card drills into its events.
   *
   * Starts from `defaultFilterState()` rather than merging into what is already there:
   * "show me these 383" means *only* those, and merging would silently intersect the
   * drill-down with a selection the user had made three clicks ago and forgotten. The
   * dates are deliberately not touched, because narrowing to an identity is a different
   * question from narrowing to a day and the two compose.
   *
   * ⚠️ `seed` is not optional decoration for a caller that knows its values are new.
   * `renderControls` drops any selection the facet list does not contain, so that a
   * value whose last event was deleted cannot linger as a filter matching nothing. A
   * drill-down is the one case where that guard is wrong: the identity demonstrably
   * exists — the group card came from the same history a moment ago — but the facets
   * are only refreshed on load and on delete, so an identity that fired since the last
   * refresh is *not* in the list. Pruned, the drill-down silently clears its own filter
   * and shows every event instead of the group's, which looks like the feature working.
   * Seeding the option in tells the control what the caller already knows.
   *
   * @param {Object} partial - a partial EventFilterState
   * @param {Object} [opts]
   * @param {boolean} [opts.silent] - skip the onChange callback, for a caller that is
   *   about to re-search anyway and would otherwise do it twice
   * @param {Object} [opts.seed] - `{rules|subjects|targets|sources: [{ref, name}]}`
   *   options to add to the facet lists if history has not reported them yet
   */
  function setState(partial, { silent = false, seed = null } = {}) {
    for (const [facetKey, options] of Object.entries(seed ?? {})) {
      const known = new Set((facets[facetKey] ?? []).map(o => o.ref));
      for (const option of options) {
        if (option?.ref && !known.has(option.ref)) {
          facets[facetKey] = [...(facets[facetKey] ?? []), { ref: option.ref, name: option.name || option.ref }];
        }
      }
    }
    // The roll-up toggles are *not* carried over, for the same reason the other facets
    // are dropped: a drill-down names one identity, and rolling it up would return more
    // events than the card that opened it counted.
    const next = defaultFilterState();
    next.startFrom = state.startFrom;
    next.startTo = state.startTo;
    next.rangeDays = state.rangeDays;
    Object.assign(next, partial);
    Object.assign(state, next);
    fromInput.value = state.startFrom ? state.startFrom.slice(0, 10) : '';
    toInput.value = state.startTo ? state.startTo.slice(0, 10) : '';
    renderControls();
    syncRangeControls();
    if (!silent) commit();
  }

  async function refreshFacets() {
    // Alongside the facets, not before them: the two are independent and the label
    // fetch is cached, so a refresh after a delete costs one request, not two.
    const [facetResult, labelResult] = await Promise.allSettled([fetchWatchEventFacets(), fetchLabels()]);
    // `fetchLabels` answers a Map; the fallback must be a Map too, not `[]`.
    if (labelResult.status === 'fulfilled') labels = labelResult.value ?? new Map();
    else Logger.error(labelResult.reason);
    if (facetResult.status === 'rejected') {
      // A facets failure must not take the page down: the grid can still list events,
      // it just cannot offer *fresh* dropdowns. Logged so the cause is not invisible.
      //
      // ⚠️ The last good facets are kept rather than blanked. Blanking is not the
      // conservative choice it looks like: `renderControls` prunes the selection against
      // the option list, so four empty lists silently clear all four filters. This runs
      // fire-and-forget after every delete, concurrently with a search, so one transient
      // 500 would turn a drill-down showing 40 events into an unfiltered grid of
      // everything — with nothing on screen to say why. Stale options are a far smaller
      // lie than a filter that undid itself.
      //
      // `renderControls` still runs: it mounts the date inputs and the range controls as
      // well as the dropdowns, so returning early here would leave the whole filter bar
      // empty when the very first load is the one that fails.
      Logger.error(facetResult.reason);
    } else {
      facets = facetResult.value;
    }
    renderControls();
  }

  await refreshFacets();
  // After the controls exist — `applyRange` writes through them — and without
  // committing: the caller runs the first search itself, and firing `onChange` here
  // would make the page open with two identical requests in flight.
  state.rangeDays = DEFAULT_RANGE_DAYS;
  applyRange(rangeEndingToday(DEFAULT_RANGE_DAYS));

  return {
    element: container,
    getState: snapshot,
    setState,
    reset,
    refreshFacets,
  };
}
