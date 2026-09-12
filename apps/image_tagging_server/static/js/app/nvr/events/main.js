/**
 * NVR → Events: the recorded-event review page.
 *
 * The same shape as the Review page — a filter bar over a results grid, with a
 * drill-down modal per card — over a different subject. Review searches *images* by
 * their annotations; this searches *events* by what the watcher decided happened. The
 * two share the interaction, not the controls (see `./filters.js`).
 *
 * Paging is server-side offset paging rather than the review queue's keyset cursor:
 * events are a bounded append-only log ordered by the single scalar the user also
 * filters on, and the page shows a total, which needs the count anyway.
 */
import {
  deleteWatchEvents,
  fetchWatchEvent,
  fetchWatchEventGroups,
  fetchWatchEventHistogram,
  fetchWatchEventSessions,
  fetchWatchEvents,
  setWatchEventPinned,
  setWatchEventReviewStates,
} from '/app-static/js/shared/api/watchProfiles.js';
import { fetchSourceList } from '/app-static/js/shared/api/sources.js';
import { Logger, showConfirmModal, toast } from '/app-static/js/ui/utils/index.js';
import { parseApiDate } from '/app-static/js/shared/time.js';
import { buildEventCard } from './eventCard.js';
import { buildGroupCard } from './groupCard.js';
import { createEventFilters, hasActiveFilter } from './filters.js';
import { openEventDetailModal } from './eventDetailModal.js';
import { buildSessionCard } from './sessionCard.js';
import { buildTimeline } from './timeline.js';
import {
  GROUP_BY_MODES,
  GROUP_SORTS,
  createFocusController,
  dimensionsFor,
  foldGroups,
  groupDrilldownFilters,
  groupFacetSeeds,
  groupHeadline,
  identityKey,
  sortGroups,
} from './identity.js';
import { openBucketBrowseModal } from './bucketBrowseModal.js';

const PAGE_SIZE = 60;

/**
 * Most events a timeline mark will hand the detail modal to walk.
 *
 * A bin is one hour of one identity, so this is generous rather than tight — but it is
 * a bound, because a watcher left pointing at a busy road can put four figures in an
 * hour and the modal holds its siblings in memory. Beyond this the honest answer is the
 * grid, which the hover preview's "show these in the grid" button reaches.
 */
const BUCKET_MODAL_LIMIT = 200;

/**
 * The three projections of one query.
 *
 * `groups` is the default because it is the one that answers the question you have when
 * you *arrive*: 1277 events are 22 things that happened, and the flat grid shows the
 * wrong unit rather than too much detail. `all` is where you end up once you know what
 * you are looking for — which is also what the filters have always been for.
 */
const VIEWS = [
  ['groups', 'Groups', 'What keeps happening — one card per kind of event'],
  ['timeline', 'Timeline', 'When events happened, one lane per kind'],
  ['all', 'All events', 'Every event, newest first'],
];

const VIEW_STORAGE_KEY = 'nvr.events.view';

/** How many sparkline buckets a group card gets, across whatever range is filtered. */
const SPARK_BUCKETS = 24;

const SESSION_STORAGE_KEY = 'nvr.events.sessionGapS';
const GROUP_BY_STORAGE_KEY = 'nvr.events.groupBy';
const SORT_STORAGE_KEY = 'nvr.events.sortBy';

/**
 * The gap options offered for session grouping, in seconds.
 *
 * `0` is "off" rather than a zero-second gap — with grouping disabled the grid pages
 * over events, which is a different endpoint, not a degenerate case of this one.
 */
const SESSION_GAP_OPTIONS = [
  [0, 'Every event'],
  [60, 'Group repeats within 1 min'],
  [120, 'Group repeats within 2 min'],
  [300, 'Group repeats within 5 min'],
  [600, 'Group repeats within 10 min'],
  [900, 'Group repeats within 15 min'],
];

/** The triage states offered in bulk, as `[value, button label]`. */
const REVIEW_ACTIONS = [
  ['confirmed', 'Mark reviewed'],
  ['dismissed', 'Dismiss'],
  ['unreviewed', 'Reopen'],
];

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function button(text, onClick, { testid = null, className = 'panel-btn' } = {}) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = className;
  btn.textContent = text;
  if (testid) btn.dataset.testid = testid;
  btn.addEventListener('click', onClick);
  return btn;
}

/**
 * Camera uuid → name.
 *
 * Events store only the uuid, and a card captioned with one is unreadable. Fetched once
 * per page build rather than per card, and tolerant of failure: a missing camera name
 * costs a line of metadata, not the grid.
 *
 * @returns {Promise<Map<string, string>>}
 */
/**
 * Open one event's player, seeked to a moment, without the events grid existing.
 *
 * The entry point for "show me the clip this frame came from", published from the
 * preview panel while triaging a tracking-blip run. It fetches the single event rather
 * than going through the grid, so it works from any tab and costs one request — the
 * grid, its filters and its facets are irrelevant to answering one question about one
 * frame.
 *
 * Deliberately does not switch tabs. See OPEN_EVENT_CLIP in shared/events/types.js.
 *
 * @param {string} eventUuid
 * @param {{ atS?: ?number }} [opts]
 */
export async function openEventClipAt(eventUuid, { atS = null } = {}) {
  const [event, sourceNames] = await Promise.all([fetchWatchEvent(eventUuid), loadSourceNames()]);
  openEventDetailModal(event, { sourceNames, seekToS: atS });
}

/** The mounted page's own "filter to one source" handler, so a call from outside (the
 *  Live View button) can reach a panel that already exists — set near the end of
 *  `renderEvents`, cleared by nothing: the page is never unmounted, only re-rendered
 *  on top of itself, so the latest render's closure is always the live one. */
let _activeApplySourceFilter = null;
/** Queued source, applied once `renderEvents` has actually mounted — same reason
 *  Review's `applyFilterWhenReady` queues: the Events page inits lazily on first tab
 *  focus, and a caller can arrive before that has happened. */
let _pendingSourceFilter = null;

/**
 * Switch the mounted Events page to show only events from one camera. Queues the
 * request if the page has not mounted yet.
 *
 * @param {string} sourceUuid
 * @param {string} [sourceName]
 */
export function openEventsForSource(sourceUuid, sourceName) {
  if (!sourceUuid) return;
  if (_activeApplySourceFilter) {
    _activeApplySourceFilter(sourceUuid, sourceName);
  } else {
    _pendingSourceFilter = { sourceUuid, sourceName };
  }
}

async function loadSourceNames() {
  try {
    // A Map of uuid → source record, not an array — see the `fetchLabels` gotcha; a
    // bare `for...of` over it would yield [key, value] pairs.
    const sources = await fetchSourceList();
    return new Map([...sources.values()].map(s => [s.uuid, s.name]));
  } catch (err) {
    Logger.error(err);
    return new Map();
  }
}

/**
 * Build the Events page into a container.
 *
 * @param {string} [target] - element id
 * @returns {Promise<void>}
 */
export async function renderEvents({ target = 'nvr-events-box' } = {}) {
  const container = document.getElementById(target);
  if (!container) return;
  container.innerHTML = '';

  const page = el('div', 'event-search');
  page.dataset.testid = 'event-search';
  container.appendChild(page);

  const sourceNames = await loadSourceNames();

  const toolbar = el('div', 'event-search__toolbar');
  const summary = el('div', 'event-search__summary');
  summary.dataset.testid = 'event-search-summary';
  const grid = el('div', 'event-search__grid');
  grid.dataset.testid = 'event-search-grid';
  const pager = el('div', 'event-search__pager');

  let offset = 0;
  let total = 0;
  // The current page's events, in the order they are drawn: the shift-range anchor and
  // the detail modal's next/prev both index into this.
  let pageEvents = [];
  const selected = new Set();
  // Index in `pageEvents` of the last card the user checked — the other end of a
  // shift-range. -1 means there is no anchor yet, so shift behaves as a plain click.
  let anchorIndex = -1;
  // Guards a re-entrant render: a filter change while a fetch is in flight would
  // otherwise let the slower response paint over the newer one.
  let requestSeq = 0;

  // Which projection is on screen. Remembered across visits because it is a working
  // preference, not a navigation state — someone who thinks in groups thinks in groups
  // tomorrow too. A bad stored value falls back rather than blanking the page.
  let view = VIEWS.some(([id]) => id === localStorage.getItem(VIEW_STORAGE_KEY))
    ? localStorage.getItem(VIEW_STORAGE_KEY)
    : 'groups';
  let splitByCamera = false;
  // How coarsely the aggregate views group. Remembered like the view: "how much am I
  // summing" is a working preference, and re-picking it on every visit is friction.
  let groupBy = GROUP_BY_MODES.some(([id]) => id === localStorage.getItem(GROUP_BY_STORAGE_KEY))
    ? localStorage.getItem(GROUP_BY_STORAGE_KEY)
    : 'detail';
  const dimensions = () => dimensionsFor(groupBy);
  // Biggest-first answers "what dominates my day"; alphabetical is the only order you
  // can actually look something up in. Remembered alongside the grain for the same
  // reason — it is how this user reads the page, not where they are in it.
  let sortBy = GROUP_SORTS.some(([id]) => id === localStorage.getItem(SORT_STORAGE_KEY))
    ? localStorage.getItem(SORT_STORAGE_KEY)
    : 'count';
  // Seconds of quiet that end a session; 0 disables the grouping entirely. Remembered
  // like the view, because how coarsely you want repeats folded is a working preference.
  // Defaults to 600 (10 min) on a first-ever visit; a bad or missing stored value falls
  // back to that default rather than silently landing on "every event". `getItem` returns
  // null when nothing was ever stored, and 0 is itself a valid gap ("every event") — so a
  // missing value must be checked for before parsing, or Number(null) === 0 makes an
  // unset preference indistinguishable from an explicitly chosen "every event".
  const storedGap = localStorage.getItem(SESSION_STORAGE_KEY);
  let sessionGapS = storedGap !== null && SESSION_GAP_OPTIONS.some(([s]) => s === Number(storedGap))
    ? Number(storedGap)
    : 600;
  // The groups behind the current render, folded to the grain on screen.
  let pageGroups = [];
  // The last aggregate response, at the server's finest grain — the *input* to the fold
  // that produced `pageGroups`. Held so the grain, sort and split controls can re-fold
  // locally; null whenever the flat view is what is on screen, so those controls fall
  // back to a real search rather than redrawing stale rows.
  let aggregate = null;
  const selectedGroups = new Set();
  // A time window clicked out of the timeline, held *outside* the filter state on
  // purpose: it has hour precision and `<input type="date">` cannot show one, so
  // writing it into the date filters would display "26 Aug" for a filter that actually
  // means 14:00–15:00. A labelled chip says what is really being applied.
  let windowOverride = null;

  const focus = createFocusController(page);

  const filters = await createEventFilters(() => {
    // Any filter change is a new search, so the page resets — staying on page 4 of a
    // result set that no longer has four pages shows an empty grid for no visible
    // reason.
    offset = 0;
    // An explicit date change supersedes a window picked off the timeline; leaving both
    // on would apply the narrower one and make the date control look broken.
    windowOverride = null;
    void search();
  });

  /** The filter state actually sent to the server, timeline window included. */
  function queryState() {
    const state = filters.getState();
    if (windowOverride) {
      state.startFrom = windowOverride.from.toISOString();
      // A null `to` is an open-ended window — "from here on", which is what a visit that
      // has not finished yet actually spans. Sending no bound is not the same as sending
      // one equal to `from`: the latter is a zero-width window that matches nothing.
      if (windowOverride.to) state.startTo = windowOverride.to.toISOString();
    }
    return state;
  }

  // ── View switcher ───────────────────────────────────────────────────────────
  const viewSwitch = el('div', 'event-search__views');
  viewSwitch.dataset.testid = 'event-view-switch';
  viewSwitch.setAttribute('role', 'tablist');
  const viewButtons = new Map();
  for (const [id, label, description] of VIEWS) {
    const btn = button(label, () => setView(id), { testid: `event-view-${id}`, className: 'panel-btn event-search__view' });
    btn.title = description;
    btn.setAttribute('role', 'tab');
    viewButtons.set(id, btn);
    viewSwitch.appendChild(btn);
  }

  function syncViewUi() {
    for (const [id, btn] of viewButtons) {
      const on = id === view;
      btn.classList.toggle('event-search__view--active', on);
      btn.setAttribute('aria-selected', String(on));
    }
    splitLabel.hidden = view === 'all';
    // Grain only means anything where rows are folded; the chronological view
    // shows events, which are already at the finest grain there is.
    groupByLabel.hidden = view === 'all';
    // Same scope as the grain: the chronological view is ordered by the server,
    // newest first, which is the only order a log of events has.
    sortLabel.hidden = view === 'all';
    // Session grouping folds *consecutive* events, which only the chronological view
    // has; the aggregate views have already collapsed by identity, which subsumes it.
    gapLabel.hidden = view !== 'all';
    // The pager counts events; neither aggregate view is paged, because both already
    // show the whole filtered set in tens of rows.
    pager.hidden = view !== 'all';
    selectAllLabel.hidden = view === 'timeline' || (view === 'all' ? pageEvents.length === 0 : pageGroups.length === 0);
  }

  function setView(next) {
    if (next === view) return;
    view = next;
    localStorage.setItem(VIEW_STORAGE_KEY, next);
    // Selection is scoped to what is on screen and the two views select different kinds
    // of thing; carrying it across would arm a delete against rows in another unit.
    selected.clear();
    selectedGroups.clear();
    anchorIndex = -1;
    focus.clear();
    offset = 0;
    syncViewUi();
    void search();
  }

  /**
   * Redraw an aggregate view after a control that only changes the *fold*.
   *
   * The grain, the sort and split-by-camera are all applied client-side to rows already
   * in hand, so none of them is a reason to re-query. Falls back to a real search when
   * there is nothing cached — the flat view, or a control changed before the first
   * response lands — so the toggles still work from any state.
   */
  function refoldAggregate() {
    if (view !== 'all' && aggregate) renderAggregate();
    else void search();
  }

  // ── Grouping grain ──────────────────────────────────────────────────────────
  //
  // The finest grain is the unit an event happened at and is right for triage; it is the
  // wrong unit for "how much is each cat drinking", where one rule splits across four
  // vessels and nothing sums them. The backend already returns the finest grain, so this
  // only changes how the client folds — no request, no round trip.
  const groupBySelect = document.createElement('select');
  groupBySelect.className = 'event-filters__select';
  groupBySelect.dataset.testid = 'event-group-by';
  for (const [id, , label, description] of GROUP_BY_MODES) {
    const option = document.createElement('option');
    option.value = id;
    option.textContent = label;
    option.title = description;
    groupBySelect.appendChild(option);
  }
  groupBySelect.value = groupBy;
  const groupByLabel = el('label', 'event-search__group-by');
  groupByLabel.appendChild(el('span', 'event-search__group-by-text', 'Group by'));
  groupByLabel.appendChild(groupBySelect);
  groupBySelect.addEventListener('change', () => {
    groupBy = groupBySelect.value;
    localStorage.setItem(GROUP_BY_STORAGE_KEY, groupBy);
    // The rows are a different shape now, so a selection keyed on the old ones would
    // arm an action against groups that no longer exist.
    selectedGroups.clear();
    focus.clear();
    refoldAggregate();
  });

  // ── Sort order ──────────────────────────────────────────────────────────────
  //
  // Pure presentation over rows already in hand, like the grain: no request, no re-fetch.
  const sortSelect = document.createElement('select');
  sortSelect.className = 'event-filters__select';
  sortSelect.dataset.testid = 'event-sort-by';
  for (const [id, label] of GROUP_SORTS) {
    const option = document.createElement('option');
    option.value = id;
    option.textContent = label;
    sortSelect.appendChild(option);
  }
  sortSelect.value = sortBy;
  const sortLabel = el('label', 'event-search__sort');
  sortLabel.appendChild(el('span', 'event-search__sort-text', 'Sort'));
  sortLabel.appendChild(sortSelect);
  sortSelect.addEventListener('change', () => {
    sortBy = sortSelect.value;
    localStorage.setItem(SORT_STORAGE_KEY, sortBy);
    refoldAggregate();
  });

  const splitBox = document.createElement('input');
  splitBox.type = 'checkbox';
  splitBox.dataset.testid = 'event-split-by-camera';
  const splitLabel = el('label', 'event-search__split');
  splitLabel.appendChild(splitBox);
  splitLabel.appendChild(el('span', null, 'Split by camera'));
  splitLabel.title = 'Most identities happen on one camera; a few span several. '
    + 'Split them apart when you care which camera saw it.';
  splitBox.addEventListener('change', () => {
    splitByCamera = splitBox.checked;
    selectedGroups.clear();
    focus.clear();
    refoldAggregate();
  });

  // ── Session grouping (flat view only) ───────────────────────────────────────
  const gapSelect = document.createElement('select');
  gapSelect.className = 'event-filters__select';
  gapSelect.dataset.testid = 'event-session-gap';
  gapSelect.title = 'One visit to the water bowl re-fires its rule several times. '
    + 'Grouping repeats folds them into the grouping they describe.';
  for (const [value, label] of SESSION_GAP_OPTIONS) {
    const option = document.createElement('option');
    option.value = String(value);
    option.textContent = label;
    gapSelect.appendChild(option);
  }
  gapSelect.value = String(sessionGapS);
  const gapLabel = el('label', 'event-search__gap');
  gapLabel.appendChild(gapSelect);
  gapSelect.addEventListener('change', () => {
    sessionGapS = Number(gapSelect.value) || 0;
    localStorage.setItem(SESSION_STORAGE_KEY, String(sessionGapS));
    // Sessions and events are different units, and the offset counts whichever is on
    // screen — holding it across the switch would land on an arbitrary page.
    offset = 0;
    selected.clear();
    anchorIndex = -1;
    void search();
  });

  // ── Selection ───────────────────────────────────────────────────────────────
  //
  // Scoped to the page on screen, not to the whole result set. A "select all 4000
  // matches" that then deletes them is a different and much more dangerous feature than
  // this one, and the pager is right there for the rest.

  const selectAllBox = document.createElement('input');
  selectAllBox.type = 'checkbox';
  selectAllBox.dataset.testid = 'event-search-select-all';
  const selectAllLabel = el('label', 'event-search__select-all');
  selectAllLabel.appendChild(selectAllBox);
  selectAllLabel.appendChild(el('span', null, 'Select all'));
  selectAllBox.addEventListener('change', () => {
    selected.clear();
    selectedGroups.clear();
    if (selectAllBox.checked) {
      if (view === 'all') for (const e of pageEvents) selected.add(e.uuid);
      else for (const g of pageGroups) selectedGroups.add(g.key);
    }
    anchorIndex = -1;
    syncSelectionUi();
  });

  // ── The timeline's time-window scope ────────────────────────────────────────
  const windowChip = el('button', 'event-search__window-chip');
  windowChip.type = 'button';
  windowChip.hidden = true;
  windowChip.dataset.testid = 'event-window-chip';
  windowChip.title = 'Clear this time window';
  windowChip.addEventListener('click', () => {
    windowOverride = null;
    offset = 0;
    void search();
  });

  function syncWindowChip() {
    windowChip.hidden = !windowOverride;
    if (!windowOverride) return;
    const { from, to } = windowOverride;
    const time = d => d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    if (!to) {
      windowChip.textContent = `Since ${from.toLocaleDateString([], { day: 'numeric', month: 'short' })} ${time(from)} ✕`;
      return;
    }
    const sameDay = from.toDateString() === to.toDateString();
    windowChip.textContent = sameDay
      ? `${from.toLocaleDateString([], { day: 'numeric', month: 'short' })} ${time(from)}–${time(to)} ✕`
      : `${from.toLocaleString([], { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })} → `
        + `${to.toLocaleString([], { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })} ✕`;
  }

  /**
   * Narrow to one identity and show its events — the group and session cards' action.
   *
   * `ungroupSessions` is what makes opening a visit meaningful: with the collapse still
   * on, drilling into a session would land on the very card just clicked, since it is
   * the only thing in that window.
   */
  function drilldown(group, { window: win = null, ungroupSessions = false } = {}) {
    windowOverride = win;
    if (ungroupSessions && sessionGapS > 0) {
      sessionGapS = 0;
      gapSelect.value = '0';
      localStorage.setItem(SESSION_STORAGE_KEY, '0');
    }
    filters.setState(groupDrilldownFilters(group, { splitByCamera }), {
      silent: true,
      // The identity came from history a moment ago, so it is a legitimate filter even
      // if the facet list predates it — see `groupFacetSeeds`.
      seed: groupFacetSeeds(group),
    });
    offset = 0;
    view = 'all';
    localStorage.setItem(VIEW_STORAGE_KEY, view);
    selected.clear();
    selectedGroups.clear();
    focus.clear();
    syncViewUi();
    void search();
  }

  /**
   * Narrow to one camera and show its events — the Live View toolbar button's action.
   *
   * Same shape as `drilldown`, minus the session/window handling a group card carries:
   * the Live View only ever knows a source, not a rule or a time window.
   */
  function applySourceFilter(sourceUuid, sourceName) {
    filters.setState({ sourceUuids: [sourceUuid] }, {
      silent: true,
      // The camera may be a watcher's source that has never yet produced a logged
      // event, so it will not be in the facet list `createEventFilters` loaded at
      // mount — seed it in, same reason `groupFacetSeeds` exists for drilldown.
      seed: { sources: [{ ref: sourceUuid, name: sourceName || sourceUuid }] },
    });
    offset = 0;
    selected.clear();
    selectedGroups.clear();
    focus.clear();
    void search();
  }

  /**
   * The events behind one lane's bin, under the filters currently in force.
   *
   * Scoped to the lane's identity *and* the bin's hour, which is what keeps this
   * affordable: the timeline is drawn from a binned histogram precisely so the client
   * never holds every event, and resolving one bin on demand is the other half of that
   * bargain. Spread over `queryState()` so the page's own filters (review state, camera,
   * excluded rules) still apply — a preview that ignored them would show an event the
   * grid is deliberately hiding.
   *
   * @param {Object} group
   * @param {{from: Date, to: Date}} win
   * @param {number} limit
   */
  async function bucketEvents(group, win, limit) {
    const { events, total } = await fetchWatchEvents({
      ...queryState(),
      ...groupDrilldownFilters(group, { dimensions: dimensions(), splitByCamera }),
      startFrom: win.from.toISOString(),
      startTo: win.to.toISOString(),
      limit,
      offset: 0,
    });
    return { events, total };
  }

  /** One event's worth, for the hover snapshot. */
  async function previewBucket(group, win) {
    const { events, total } = await bucketEvents(group, win, 1);
    return { event: events[0] ?? null, total };
  }

  /**
   * Clicking a mark opens its events, over the timeline.
   *
   * A bin holding exactly one event goes straight to the player — an intermediate modal
   * listing a single card is pure friction. A bin holding several opens the browser
   * first, because prev/next can only show you one picture at a time and picking which
   * of fifty to watch is exactly the question a grid of cards answers.
   *
   * Either way the player is handed the whole bin as `siblings`, so prev/next still
   * walks the bin rather than dead-ending on whichever event was opened.
   */
  async function openBucketEvents(group, win) {
    try {
      const { events } = await bucketEvents(group, win, BUCKET_MODAL_LIMIT);
      if (!events.length) {
        // The histogram said there was something here, so an empty answer means the
        // events were deleted between the draw and the click. Say so and refresh
        // rather than opening an empty modal.
        toast('Those events are no longer there', 4000);
        void search();
        return;
      }
      const openDetail = (event, siblings) => openEventDetailModal(event, {
        sourceNames,
        siblings,
        onDeleted: () => void search(),
      });
      if (events.length > 1) {
        const { primary, secondary } = groupHeadline(group, dimensions());
        const span = `${win.from.toLocaleString([], { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })}`
          + ` – ${win.to.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
        openBucketBrowseModal({
          events,
          sourceNames,
          title: primary,
          subtitle: secondary ? `${secondary} · ${span}` : span,
          onOpenEvent: openDetail,
        });
        return;
      }
      openDetail(events[0], events);
    } catch (err) {
      Logger.error(err);
      toast(err.message || 'Could not open those events', 5000, 'error');
    }
  }

  function toggleSelectGroup(group, { checked }) {
    if (checked) selectedGroups.add(group.key);
    else selectedGroups.delete(group.key);
    syncSelectionUi();
  }

  const selectionActions = el('div', 'event-search__selection');
  selectionActions.dataset.testid = 'event-search-selection';
  const selectionCount = el('span', 'event-search__selection-count');
  selectionCount.dataset.testid = 'event-search-selection-count';
  const deleteBtn = button('Delete selected', () => void deleteSelected(), {
    testid: 'event-search-delete-selected',
    className: 'panel-btn panel-btn--danger',
  });
  selectionActions.appendChild(selectionCount);
  // Triage before delete, and separated from it: marking a run of events reviewed is
  // the common, reversible thing, and putting it beside the irreversible one makes the
  // safe action the easy one to reach.
  for (const [state, label] of REVIEW_ACTIONS) {
    selectionActions.appendChild(button(label, () => void applyReviewState(state), {
      testid: `event-search-review-${state}`,
    }));
  }
  selectionActions.appendChild(button('📌 Pin', () => void applyPinned(true), {
    testid: 'event-search-pin',
  }));
  selectionActions.appendChild(button('Unpin', () => void applyPinned(false), {
    testid: 'event-search-unpin',
  }));
  selectionActions.appendChild(deleteBtn);
  selectionActions.appendChild(button('Clear selection', () => {
    selected.clear();
    anchorIndex = -1;
    syncSelectionUi();
  }, { testid: 'event-search-clear-selection' }));

  const refreshBtn = button('Refresh', () => void search(), { testid: 'event-search-refresh' });
  toolbar.appendChild(viewSwitch);
  toolbar.appendChild(selectAllLabel);
  toolbar.appendChild(groupByLabel);
  toolbar.appendChild(sortLabel);
  toolbar.appendChild(splitLabel);
  toolbar.appendChild(gapLabel);
  toolbar.appendChild(summary);
  toolbar.appendChild(windowChip);
  toolbar.appendChild(refreshBtn);
  toolbar.appendChild(selectionActions);

  /** How many events the current selection covers, in either unit. */
  function selectedEventCount() {
    if (view === 'all') return selected.size;
    return pageGroups.filter(g => selectedGroups.has(g.key)).reduce((sum, g) => sum + g.count, 0);
  }

  /** Push the selection into the DOM: card classes, checkboxes, and the action bar. */
  function syncSelectionUi() {
    for (const card of grid.querySelectorAll('[data-event-uuid]')) {
      const isSelected = selected.has(card.dataset.eventUuid);
      card.classList.toggle('event-card--selected', isSelected);
      const box = card.querySelector('.event-card__select');
      if (box) box.checked = isSelected;
    }
    for (const card of grid.querySelectorAll('.event-group-card[data-identity-key]')) {
      const isSelected = selectedGroups.has(card.dataset.identityKey);
      card.classList.toggle('event-group-card--selected', isSelected);
      const box = card.querySelector('.event-group-card__select');
      if (box) box.checked = isSelected;
    }
    const units = view === 'all' ? selected.size : selectedGroups.size;
    const events = selectedEventCount();
    selectionActions.hidden = units === 0;
    selectionCount.textContent = view === 'all'
      ? `${units} selected`
      // A group's checkbox arms a delete against every event behind it, which is the
      // whole point of it and also the one thing that could badly surprise someone. Both
      // numbers, always, so "2 selected" can never quietly mean 665 rows.
      : `${units} group${units === 1 ? '' : 's'} — ${events} event${events === 1 ? '' : 's'}`;
    deleteBtn.textContent = `Delete ${events} event${events === 1 ? '' : 's'}`;
    const items = view === 'all' ? pageEvents.length : pageGroups.length;
    selectAllLabel.hidden = view === 'timeline' || items === 0;
    selectAllBox.checked = items > 0 && units === items;
    selectAllBox.indeterminate = units > 0 && units < items;
  }

  function toggleSelect(event, { shiftKey, checked }) {
    const index = pageEvents.findIndex(e => e.uuid === event.uuid);
    if (shiftKey && anchorIndex !== -1 && index !== -1) {
      // The whole range takes the *clicked* card's new state, so a shift-click can
      // deselect a run as well as select one.
      const lo = Math.min(anchorIndex, index);
      const hi = Math.max(anchorIndex, index);
      for (let i = lo; i <= hi; i++) {
        if (checked) selected.add(pageEvents[i].uuid);
        else selected.delete(pageEvents[i].uuid);
      }
    } else if (checked) {
      selected.add(event.uuid);
    } else {
      selected.delete(event.uuid);
    }
    // A shift-click extends *from* the anchor and does not become one. Moving it would
    // make the range unadjustable: shift-clicking one card further, or back over the
    // same card to undo the range, would each start measuring from where the last shift
    // landed instead of from the card the user actually anchored on.
    if (!shiftKey || anchorIndex === -1) anchorIndex = index;
    syncSelectionUi();
  }

  /**
   * Every event uuid behind the selected groups.
   *
   * Deferred to the delete click rather than paid while browsing, the same bargain the
   * Review page's Apply-to-All strikes: the grouped view's whole value is that it does
   * not fetch 383 rows to show you one card, and it would be self-defeating to fetch
   * them anyway just in case you tick the box. The exact count is therefore known at
   * confirm time, which is the only moment it has to be right.
   */
  async function eventsForSelectedGroups() {
    const groups = pageGroups.filter(g => selectedGroups.has(g.key));
    // Each group's page is an independent request — fire them together rather
    // than one at a time, which just adds up their latencies for no reason.
    const pages = await Promise.all(groups.map(group => fetchWatchEvents({
      ...queryState(),
      ...groupDrilldownFilters(group, { splitByCamera }),
      limit: Math.max(1, group.count),
      offset: 0,
    })));
    const events = [];
    for (const { events: page } of pages) events.push(...page);
    return events;
  }

  async function uuidsForSelectedGroups() {
    return (await eventsForSelectedGroups()).map(e => e.uuid);
  }

  /**
   * Move the selection to a triage state.
   *
   * Reuses the same uuid resolution as the delete, so ticking a group and marking it
   * reviewed covers all 383 of its events rather than the one card representing them.
   */
  async function applyReviewState(state) {
    try {
      const uuids = view === 'all' ? [...selected] : await uuidsForSelectedGroups();
      if (!uuids.length) return;
      const updated = await setWatchEventReviewStates(uuids, state);
      toast(`Updated ${updated} event${updated === 1 ? '' : 's'}`, 3000);
      selected.clear();
      selectedGroups.clear();
      anchorIndex = -1;
      await search();
    } catch (err) {
      Logger.error(err);
      toast(err.message || 'Could not update those events', 5000, 'error');
    }
  }

  /**
   * Move the selection to pinned or unpinned.
   *
   * Unlike delete, pinning itself is never refused — a pin is a promise about
   * *deletion*, not about who may set it, so both directions apply to the whole
   * selection regardless of an event's current state.
   */
  async function applyPinned(pinned) {
    try {
      const uuids = view === 'all' ? [...selected] : await uuidsForSelectedGroups();
      if (!uuids.length) return;
      const updated = await setWatchEventPinned(uuids, pinned);
      toast(`${pinned ? 'Pinned' : 'Unpinned'} ${updated} event${updated === 1 ? '' : 's'}`, 3000);
      selected.clear();
      selectedGroups.clear();
      anchorIndex = -1;
      await search();
    } catch (err) {
      Logger.error(err);
      toast(err.message || 'Could not update those events', 5000, 'error');
    }
  }

  async function deleteSelected() {
    let candidates;
    if (view === 'all') {
      candidates = pageEvents.filter(e => selected.has(e.uuid));
    } else {
      deleteBtn.disabled = true;
      deleteBtn.textContent = 'Loading events…';
      try {
        candidates = await eventsForSelectedGroups();
      } catch (err) {
        Logger.error(err);
        toast(err.message || 'Could not load the events in that group', 5000, 'error');
        return;
      } finally {
        deleteBtn.disabled = false;
        syncSelectionUi();
      }
    }
    // Pinned events are a promise against delete, not just against the automatic
    // sweeps — see `WatchEvent.pinned`. Kept out of the request entirely rather than
    // sent and silently skipped server-side, so the confirm dialog can say up front
    // what will and will not happen instead of the toast being the first anyone hears
    // of it.
    const pinnedCount = candidates.filter(e => e.pinned).length;
    const uuids = candidates.filter(e => !e.pinned).map(e => e.uuid);
    if (!uuids.length) {
      if (pinnedCount) toast(`${pinnedCount} pinned event${pinnedCount === 1 ? ' is' : 's are'} not deletable — unpin first`, 4000);
      return;
    }
    const confirmed = await showConfirmModal({
      title: `Delete ${uuids.length} event${uuids.length === 1 ? '' : 's'}?`,
      // Same promise the single-event delete makes, and for the same reason: the one
      // thing a user needs to know is that curated training frames are not in scope.
      message: `The events, their snapshots and any clips or timelapses of them will be removed. `
        + 'Frames you saved for annotation are not affected.'
        + (pinnedCount ? ` ${pinnedCount} pinned event${pinnedCount === 1 ? '' : 's'} in your selection will be kept.` : ''),
      confirmText: 'Delete',
      cancelText: 'Keep',
    });
    if (!confirmed) return;
    deleteBtn.disabled = true;
    try {
      const deleted = await deleteWatchEvents(uuids);
      toast(`Deleted ${deleted} event${deleted === 1 ? '' : 's'}`, 3000);
      selected.clear();
      // Both units, because either could have armed this delete: a group whose events
      // are gone but whose key survives the re-render (a partial delete, or an identity
      // that fired again in between) would come back still ticked, with the action bar
      // still armed against rows the user believes they already dealt with.
      selectedGroups.clear();
      anchorIndex = -1;
      void filters.refreshFacets();
      await search();
    } catch (err) {
      Logger.error(err);
      toast(err.message || 'Could not delete the selected events', 5000, 'error');
    } finally {
      deleteBtn.disabled = false;
    }
  }

  page.appendChild(filters.element);
  page.appendChild(toolbar);
  page.appendChild(grid);
  page.appendChild(pager);

  function renderPager() {
    pager.innerHTML = '';
    if (total <= PAGE_SIZE) return;
    const pageNumber = Math.floor(offset / PAGE_SIZE) + 1;
    const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));

    const prev = button('← Newer', () => {
      offset = Math.max(0, offset - PAGE_SIZE);
      void search();
    }, { testid: 'event-search-prev' });
    prev.disabled = offset === 0;

    const next = button('Older →', () => {
      offset += PAGE_SIZE;
      void search();
    }, { testid: 'event-search-next' });
    next.disabled = offset + PAGE_SIZE >= total;

    pager.appendChild(prev);
    pager.appendChild(el('span', 'event-search__page-count', `Page ${pageNumber} of ${pageCount}`));
    pager.appendChild(next);
  }

  function renderEmpty() {
    grid.innerHTML = '';
    const box = el('div', 'nvr-empty');
    box.dataset.testid = 'event-search-empty';
    const anyFilter = hasActiveFilter(filters.getState());
    box.appendChild(el('strong', null, anyFilter ? 'No events match these filters' : 'No events recorded yet'));
    box.appendChild(el('p', null, anyFilter
      ? 'Try widening the date range, or clearing a filter.'
      // Jobs → Start a job is where a watcher is actually launched. The Live tab shows
      // what a running watcher sees; sending someone there to start one is a dead end,
      // and this is the one screen a user reaches *before* they have ever run a watcher.
      : 'Events appear here as your watchers detect them. Start a watcher from Jobs → Start a job.'));
    if (anyFilter) box.appendChild(button('Clear filters', () => filters.reset()));
    grid.appendChild(box);
  }

  /**
   * Bucket a group's histogram rows into a fixed number of slots for its sparkline.
   *
   * Fixed count rather than fixed width so every card's spark spans the same window and
   * they can be read against each other — a card whose bars covered a different period
   * from its neighbour's would invite exactly the wrong comparison.
   *
   * @param {Array<Object>} rows - histogram rows for one identity
   * @param {?{start: number, end: number}} span
   * @returns {number[]}
   */
  function sparkCounts(rows, span) {
    if (!span || span.end <= span.start) return [];
    const counts = new Array(SPARK_BUCKETS).fill(0);
    for (const row of rows) {
      const at = parseApiDate(row.bucketStart)?.getTime();
      if (at == null) continue;
      const slot = Math.min(
        SPARK_BUCKETS - 1,
        Math.max(0, Math.floor(((at - span.start) / (span.end - span.start)) * SPARK_BUCKETS)),
      );
      counts[slot] += row.count;
    }
    return counts;
  }

  /**
   * The aggregate views' data: groups and the histogram, in one round trip each.
   *
   * Both views need both — the timeline needs the group list for its lane order and
   * headlines, the group cards need the histogram for their sparklines — so they are
   * fetched together and the view switch costs nothing beyond a re-render.
   */
  async function searchAggregate(seq) {
    const state = queryState();
    const [{ groups: raw, total: groupTotal }, histogram] = await Promise.all([
      fetchWatchEventGroups(state),
      fetchWatchEventHistogram({ ...state, bucket: 'hour' }),
    ]);
    if (seq !== requestSeq) return;

    // Kept so the grain/sort/split toggles can re-fold without asking again — see
    // `refoldAggregate`. Held at the finest grain the server returns, which is the input
    // `foldGroups` wants; `pageGroups` below is the folded output and cannot be re-folded.
    aggregate = { raw, groupTotal, histogram, state };
    renderAggregate();
  }

  /**
   * Draw the aggregate views from data already in hand.
   *
   * Split out from the fetch because the grouping grain, the sort order and split-by-
   * camera are all pure client-side folds over the same finest-grain rows — the comments
   * on those three controls have always said "no request, no round trip", and until this
   * split they each triggered two (groups *and* histogram). Nothing they change is a
   * query parameter.
   */
  function renderAggregate() {
    if (!aggregate) return;
    const { raw, groupTotal, histogram, state } = aggregate;

    total = groupTotal;
    pageGroups = sortGroups(foldGroups(raw, { dimensions: dimensions(), splitByCamera }), sortBy, dimensions());
    // Registrations point at elements that are about to be discarded; a Map still
    // holding them would pin the whole previous render in memory.
    focus.reset();
    grid.innerHTML = '';
    grid.classList.toggle('event-search__grid--timeline', view === 'timeline');

    if (!pageGroups.length) {
      renderEmpty();
    } else if (view === 'timeline') {
      grid.appendChild(buildTimeline({
        groups: pageGroups,
        buckets: histogram.buckets,
        bucket: histogram.bucket,
        filters: state,
        focus,
        dimensions: dimensions(),
        splitByCamera,
        sourceNames,
        onSelectGroup: group => drilldown(group),
        onSelectBucket: (group, win) => drilldown(group, { window: win }),
        onOpenBucket: (group, win) => void openBucketEvents(group, win),
        onPreviewBucket: (group, win) => previewBucket(group, win),
      }));
    } else {
      const byIdentity = new Map();
      for (const row of histogram.buckets) {
        const key = identityKey(row, { dimensions: dimensions(), splitByCamera });
        let list = byIdentity.get(key);
        if (!list) byIdentity.set(key, (list = []));
        list.push(row);
      }
      const times = histogram.buckets
        .map(b => parseApiDate(b.bucketStart)?.getTime())
        .filter(t => t != null);
      const span = times.length ? { start: Math.min(...times), end: Math.max(...times) + 3600000 } : null;

      for (const group of pageGroups) {
        const card = buildGroupCard(group, {
          sourceNames,
          sparkCounts: sparkCounts(byIdentity.get(group.key) ?? [], span),
          onDrilldown: g => drilldown(g),
          onOpenLatest: async g => {
            try {
              const event = await fetchWatchEvent(g.representativeUuid);
              openEventDetailModal(event, { sourceNames, onDeleted: () => void search() });
            } catch (err) {
              Logger.error(err);
              toast(err.message || 'Could not open that event', 5000, 'error');
            }
          },
          onToggleSelect: toggleSelectGroup,
          selected: selectedGroups.has(group.key),
        });
        focus.register(group.key, card);
        grid.appendChild(card);
      }
    }

    syncSelectionUi();
    syncWindowChip();
    const shown = pageGroups.length;
    summary.textContent = total
      ? `${total} event${total === 1 ? '' : 's'} in ${shown} group${shown === 1 ? '' : 's'}`
      : '';
    renderPager();
  }

  /**
   * The upper bound of the window that resolves a session back to its own events.
   *
   * Two traps, both of which produced an empty grid from a card that plainly showed
   * events behind it:
   *
   * 1. **A visit still in progress has no end.** Its `endedAt` is the last event's
   *    *start* (the backend coalesces, so a session always has an extent to draw), so a
   *    one-event open session reports `from === to`. Left alone that asks the server for
   *    events starting at exactly one instant. `null` means "no upper bound", which is
   *    what an unfinished visit actually spans.
   * 2. **`Date` is millisecond-resolution; the column is microsecond.** `started_at` is
   *    written by `datetime.now()`, so an event at `…04.098861` round-trips through
   *    `parseApiDate().toISOString()` as `…04.098` — *below* its own value. As
   *    `start_to` that is a bound the event itself fails, so any session whose extent
   *    collapses onto a single event's start (an event closed by a restart is the other
   *    way to get one) resolved to nothing. Rounding the bound up a millisecond puts the
   *    truncated instant back inside the window.
   *
   * @param {Object} session
   * @returns {?Date}
   */
  function sessionWindowEnd(session) {
    if (session.isOpen) return null;
    const end = parseApiDate(session.endedAt);
    return end ? new Date(end.getTime() + 1) : null;
  }

  /**
   * Open the one event behind a session that folded nothing.
   *
   * The session row carries a `representativeUuid` but not the event itself, so this
   * costs a fetch — paid on the click, never while browsing, the same bargain the group
   * card's "Latest" makes.
   *
   * @param {Object} session
   */
  async function openSessionEvent(session) {
    try {
      const event = await fetchWatchEvent(session.representativeUuid);
      openEventDetailModal(event, { sourceNames, onDeleted: () => void search() });
    } catch (err) {
      Logger.error(err);
      toast(err.message || 'Could not open that event', 5000, 'error');
    }
  }

  /**
   * The flat view with repeats folded into visits.
   *
   * Paged over sessions rather than events, which is why it has its own branch: the
   * offset means a different thing here, and a session can never be split across a page
   * boundary the way a client-side grouping would split it.
   */
  async function searchSessions(seq) {
    const { sessions, total: sessionTotal } = await fetchWatchEventSessions({
      ...queryState(),
      gapS: sessionGapS,
      limit: PAGE_SIZE,
      offset,
    });
    if (seq !== requestSeq) return;

    total = sessionTotal;
    if (!sessions.length && total > 0 && offset >= total) {
      offset = (Math.ceil(total / PAGE_SIZE) - 1) * PAGE_SIZE;
      await search();
      return;
    }

    grid.innerHTML = '';
    grid.classList.remove('event-search__grid--timeline');
    focus.reset();
    // A session is not an event, so nothing that selects events may survive into it —
    // the selection and delete paths below both key on event uuids.
    pageEvents = [];
    pageGroups = [];
    // The flat views draw from their own responses, so nothing aggregate is in hand any
    // more; the fold toggles must fetch rather than redraw rows for a different query.
    aggregate = null;
    selected.clear();
    anchorIndex = -1;

    if (!sessions.length) {
      renderEmpty();
    } else {
      for (const session of sessions) {
        const card = buildSessionCard(session, {
          sourceNames,
          // A visit of several events has no single clip to play, so it drills into its
          // own events and its own minutes rather than guessing which of the seven was
          // meant. A visit of *one* has exactly one, and an intermediate grid holding a
          // single card is pure friction — the same bargain the timeline strikes for a
          // bin holding one event.
          onOpen: s => {
            if (s.count === 1 && s.representativeUuid) {
              void openSessionEvent(s);
              return;
            }
            drilldown(s, {
              window: { from: parseApiDate(s.startedAt), to: sessionWindowEnd(s) },
              ungroupSessions: true,
            });
          },
        });
        focus.register(identityKey(session, { dimensions: dimensions(), splitByCamera }), card);
        grid.appendChild(card);
      }
    }
    syncSelectionUi();
    syncWindowChip();

    const folded = sessions.reduce((sum, s) => sum + s.count, 0);
    summary.textContent = total
      ? `${total} grouping${total === 1 ? '' : 's'}${folded > sessions.length ? ` — ${folded} events on this page` : ''}`
      : '';
    renderPager();
  }

  async function search() {
    const seq = ++requestSeq;
    grid.setAttribute('aria-busy', 'true');
    try {
      if (view !== 'all') {
        await searchAggregate(seq);
        return;
      }
      if (sessionGapS > 0) {
        await searchSessions(seq);
        return;
      }
      const result = await fetchWatchEvents({ ...queryState(), limit: PAGE_SIZE, offset });
      if (seq !== requestSeq) return;

      total = result.total;

      // The result set can shrink under a held offset — deleting the only event on the
      // last page is the ordinary way. Left alone, the empty response renders as "no
      // events match these filters" over a set that is merely one page shorter, and
      // renderPager() hides itself once total <= PAGE_SIZE, so there is no control left
      // to get back. Re-asking for the last page that does exist is the honest answer.
      // Cannot loop: offset is strictly below total on the retry, so the next response
      // is non-empty unless total reached zero, which this guard excludes.
      if (!result.events.length && total > 0 && offset >= total) {
        offset = (Math.ceil(total / PAGE_SIZE) - 1) * PAGE_SIZE;
        await search();
        return;
      }

      grid.innerHTML = '';
      grid.classList.remove('event-search__grid--timeline');
      focus.reset();
      // A selection is a set of cards on screen, so it cannot survive the cards. Kept
      // across a re-render it would silently arm "Delete 12 events" against a page the
      // user is no longer looking at.
      pageEvents = result.events;
      pageGroups = [];
      aggregate = null;
      selected.clear();
      anchorIndex = -1;
      if (!result.events.length) {
        renderEmpty();
      } else {
        for (const event of result.events) {
          const card = buildEventCard(event, {
            sourceNames,
            onToggleSelect: toggleSelect,
            selected: selected.has(event.uuid),
            onOpen: e => openEventDetailModal(e, {
              sourceNames,
              // The page, so the modal can walk it. The modal copies it and splices its
              // own deletes out of the copy, so an open modal keeps a coherent walk
              // even as the grid behind it re-searches onto a different set of rows.
              siblings: pageEvents,
              // A deleted event changes both the grid and the facet lists — the value
              // it was the last holder of is no longer filterable.
              onDeleted: () => {
                void filters.refreshFacets();
                void search();
              },
            }),
          });
          // Every tile joins the same highlight as its lane and its group card, so
          // hovering one "Maple → Tupperware" lights every other one on the page.
          focus.register(identityKey(event, { dimensions: dimensions(), splitByCamera }), card);
          grid.appendChild(card);
        }
      }
      syncSelectionUi();
      syncWindowChip();

      const shown = result.events.length;
      summary.textContent = total
        ? `${total} event${total === 1 ? '' : 's'}${total > shown ? ` — showing ${shown}` : ''}`
        : '';
      renderPager();
    } catch (err) {
      if (seq !== requestSeq) return;
      Logger.error(err);
      grid.innerHTML = '';
      const box = el('div', 'nvr-empty');
      box.appendChild(el('strong', null, 'Could not load events'));
      box.appendChild(el('p', null, err.message || 'The request failed.'));
      box.appendChild(button('Retry', () => void search()));
      grid.appendChild(box);
    } finally {
      if (seq === requestSeq) grid.removeAttribute('aria-busy');
    }
  }

  splitBox.checked = splitByCamera;
  syncViewUi();
  _activeApplySourceFilter = applySourceFilter;
  if (_pendingSourceFilter) {
    const { sourceUuid, sourceName } = _pendingSourceFilter;
    _pendingSourceFilter = null;
    applySourceFilter(sourceUuid, sourceName);
    return;
  }
  await search();
}
