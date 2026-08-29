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
import { fetchWatchEventFacets } from '/app-static/js/shared/api/watchProfiles.js';
import { createMultiSelectDropdown, Logger } from '/app-static/js/ui/utils/index.js';

/** @typedef {{sourceUuids: string[], ruleUuids: string[], subjectRefs: string[],
 *             targetRefs: string[], sourceExcludeUuids: string[], ruleExcludeUuids: string[],
 *             subjectExcludeRefs: string[], targetExcludeRefs: string[],
 *             startFrom: ?string, startTo: ?string,
 *             hasRecording: ?boolean, recordingKind: ?string}} EventFilterState */

/**
 * The four facet filters, as `[state key for includes, state key for excludes, facet key,
 * label, "no filter" text, testid]`.
 *
 * One table rather than four near-identical blocks: every one of them is the same
 * control over a different column, and the three places that need them — building,
 * restoring after a facet refresh, and reporting emptiness — were drifting apart when
 * each was written out longhand.
 */
export const FACET_FILTERS = [
  ['ruleUuids', 'ruleExcludeUuids', 'rules', 'Event type', 'Any event type', 'event-filter-rule'],
  ['sourceUuids', 'sourceExcludeUuids', 'sources', 'Camera', 'Any camera', 'event-filter-source'],
  ['subjectRefs', 'subjectExcludeRefs', 'subjects', 'Acting', 'Anything acting', 'event-filter-subject'],
  ['targetRefs', 'targetExcludeRefs', 'targets', 'Acted on', 'Anything acted on', 'event-filter-target'],
];

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

/** @returns {EventFilterState} */
export function defaultFilterState() {
  const state = {
    startFrom: null,
    startTo: null,
    hasRecording: null,
    recordingKind: null,
  };
  for (const [includeKey, excludeKey] of FACET_FILTERS) {
    state[includeKey] = [];
    state[excludeKey] = [];
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
  return Object.values(state).some(v => (Array.isArray(v) ? v.length > 0 : v != null));
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
  let facets = { rules: [], subjects: [], targets: [], sources: [] };

  const controls = el('div', 'event-filters__row');
  container.appendChild(controls);

  // Rebuilt on refresh rather than mutated: the option sets change wholesale when new
  // event kinds appear, and reconciling them in place is more code than re-rendering
  // four dropdowns.
  /** @type {Array<{destroy: () => void}>} live dropdowns, destroyed before each rebuild */
  let facetDropdowns = [];
  let clipSelect;
  const fromInput = dateInput();
  const toInput = dateInput();

  /** A copy no caller can write back through — the list values are arrays. */
  function snapshot() {
    const copy = { ...state };
    for (const [includeKey, excludeKey] of FACET_FILTERS) {
      copy[includeKey] = [...state[includeKey]];
      copy[excludeKey] = [...state[excludeKey]];
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

    for (const [includeKey, excludeKey, facetKey, labelText, anyLabel, testid] of FACET_FILTERS) {
      const options = facets[facetKey] ?? [];
      const known = new Set(options.map(o => o.ref));
      // Drop values history no longer holds (the last event with one was deleted).
      // Keeping them would show an empty result set with no visible cause; the rest of
      // the selection survives the rebuild, because a filter that silently cleared
      // itself on a facet refresh would look like it never worked.
      state[includeKey] = state[includeKey].filter(ref => known.has(ref));
      state[excludeKey] = state[excludeKey].filter(ref => known.has(ref));

      const dropdown = createMultiSelectDropdown({
        label: '',
        items: options.map(o => ({ uuid: o.ref, name: o.name })),
        defaultSelected: [...state[includeKey]],
        // An event has exactly one camera, one rule, one subject and one target, so
        // "all of" could only ever match nothing — offering it would be a dead end.
        showModeToggle: false,
        emptySummary: anyLabel,
        emptyText: 'Nothing recorded yet',
        searchable: options.length > 8,
        filterKey: `event-${facetKey}`,
        defaultWidth: '220px',
        defaultHeight: '260px',
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

    controls.appendChild(field('From', fromInput, 'event-filter-from'));
    controls.appendChild(field('To', toInput, 'event-filter-to'));
    controls.appendChild(field('Video', clipSelect, 'event-filter-clip'));

    const clear = document.createElement('button');
    clear.type = 'button';
    clear.className = 'panel-btn event-filters__clear';
    clear.textContent = 'Clear';
    clear.dataset.testid = 'event-filter-clear';
    clear.addEventListener('click', () => reset());
    controls.appendChild(clear);
  }

  // A <input type="date"> gives a bare YYYY-MM-DD, which the backend would read as
  // midnight UTC. `To` therefore has to cover the whole day it names, or picking the
  // same day for both returns only events at exactly 00:00:00.
  fromInput.addEventListener('change', () => {
    state.startFrom = fromInput.value ? `${fromInput.value}T00:00:00` : null;
    commit();
  });
  toInput.addEventListener('change', () => {
    state.startTo = toInput.value ? `${toInput.value}T23:59:59` : null;
    commit();
  });

  function reset() {
    Object.assign(state, defaultFilterState());
    fromInput.value = '';
    toInput.value = '';
    renderControls();
    commit();
  }

  async function refreshFacets() {
    try {
      facets = await fetchWatchEventFacets();
    } catch (err) {
      // A facets failure must not take the page down: the grid can still list events,
      // it just cannot offer the dropdowns. Logged so the cause is not invisible.
      Logger.error(err);
      facets = { rules: [], subjects: [], targets: [], sources: [] };
    }
    renderControls();
  }

  await refreshFacets();

  return {
    element: container,
    getState: snapshot,
    reset,
    refreshFacets,
  };
}
