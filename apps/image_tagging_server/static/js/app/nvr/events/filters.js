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
import { Logger } from '/app-static/js/ui/utils/index.js';

/** @typedef {{sourceUuid: ?string, ruleUuid: ?string, subjectRef: ?string,
 *             targetRef: ?string, startFrom: ?string, startTo: ?string,
 *             hasRecording: ?boolean}} EventFilterState */

/** @returns {EventFilterState} */
export function defaultFilterState() {
  return {
    sourceUuid: null,
    ruleUuid: null,
    subjectRef: null,
    targetRef: null,
    startFrom: null,
    startTo: null,
    hasRecording: null,
  };
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
 * @param {Array<{ref: string, name: string}>} options
 * @param {string} anyLabel - what "no filter" is called, e.g. "Any camera"
 */
function select(options, anyLabel) {
  const node = document.createElement('select');
  node.className = 'event-filters__select';
  const any = document.createElement('option');
  any.value = '';
  any.textContent = anyLabel;
  node.appendChild(any);
  for (const opt of options) {
    const option = document.createElement('option');
    option.value = opt.ref;
    option.textContent = opt.name;
    node.appendChild(option);
  }
  // Disabled rather than hidden when history holds nothing to filter on: a control that
  // vanishes reads as a missing feature, one that is greyed out reads as "not yet".
  node.disabled = options.length === 0;
  return node;
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
  // four <select>s.
  let ruleSelect;
  let sourceSelect;
  let subjectSelect;
  let targetSelect;
  let clipSelect;
  const fromInput = dateInput();
  const toInput = dateInput();

  function commit() {
    onChange({ ...state });
  }

  function bindSelect(node, key) {
    node.addEventListener('change', () => {
      state[key] = node.value || null;
      commit();
    });
  }

  function renderControls() {
    controls.innerHTML = '';

    ruleSelect = select(facets.rules, 'Any event type');
    sourceSelect = select(facets.sources, 'Any camera');
    subjectSelect = select(facets.subjects, 'Anything acting');
    targetSelect = select(facets.targets, 'Anything acted on');

    // Restore the current selection after a rebuild. A filter silently clearing itself
    // because the facets refreshed would re-run the search over everything, which looks
    // like the filter never worked.
    for (const [node, key] of [
      [ruleSelect, 'ruleUuid'],
      [sourceSelect, 'sourceUuid'],
      [subjectSelect, 'subjectRef'],
      [targetSelect, 'targetRef'],
    ]) {
      if (state[key] && [...node.options].some(o => o.value === state[key])) {
        node.value = state[key];
      } else if (state[key]) {
        // The value no longer exists in history (the last event with it was deleted).
        // Dropping it silently is right — keeping it would show an empty result set
        // with no visible cause.
        state[key] = null;
      }
      bindSelect(node, key);
    }

    clipSelect = document.createElement('select');
    clipSelect.className = 'event-filters__select';
    for (const [value, label] of [['', 'With or without video'], ['true', 'Has video'], ['false', 'No video']]) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      clipSelect.appendChild(option);
    }
    clipSelect.value = state.hasRecording == null ? '' : String(state.hasRecording);
    clipSelect.addEventListener('change', () => {
      state.hasRecording = clipSelect.value === '' ? null : clipSelect.value === 'true';
      commit();
    });

    controls.appendChild(field('Event type', ruleSelect, 'event-filter-rule'));
    controls.appendChild(field('Camera', sourceSelect, 'event-filter-source'));
    controls.appendChild(field('Acting', subjectSelect, 'event-filter-subject'));
    controls.appendChild(field('Acted on', targetSelect, 'event-filter-target'));
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
    getState: () => ({ ...state }),
    reset,
    refreshFacets,
  };
}
