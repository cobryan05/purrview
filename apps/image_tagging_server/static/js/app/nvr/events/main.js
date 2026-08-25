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
import { deleteWatchEvents, fetchWatchEvent, fetchWatchEvents } from '/app-static/js/shared/api/watchProfiles.js';
import { fetchSourceList } from '/app-static/js/shared/api/sources.js';
import { Logger, showConfirmModal, toast } from '/app-static/js/ui/utils/index.js';
import { buildEventCard } from './eventCard.js';
import { createEventFilters } from './filters.js';
import { openEventDetailModal } from './eventDetailModal.js';

const PAGE_SIZE = 60;

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

  const filters = await createEventFilters(() => {
    // Any filter change is a new search, so the page resets — staying on page 4 of a
    // result set that no longer has four pages shows an empty grid for no visible
    // reason.
    offset = 0;
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
    if (selectAllBox.checked) for (const e of pageEvents) selected.add(e.uuid);
    anchorIndex = -1;
    syncSelectionUi();
  });

  const selectionActions = el('div', 'event-search__selection');
  selectionActions.dataset.testid = 'event-search-selection';
  const selectionCount = el('span', 'event-search__selection-count');
  selectionCount.dataset.testid = 'event-search-selection-count';
  const deleteBtn = button('Delete selected', () => void deleteSelected(), {
    testid: 'event-search-delete-selected',
    className: 'panel-btn panel-btn--danger',
  });
  selectionActions.appendChild(selectionCount);
  selectionActions.appendChild(deleteBtn);
  selectionActions.appendChild(button('Clear selection', () => {
    selected.clear();
    anchorIndex = -1;
    syncSelectionUi();
  }, { testid: 'event-search-clear-selection' }));

  const refreshBtn = button('Refresh', () => void search(), { testid: 'event-search-refresh' });
  toolbar.appendChild(selectAllLabel);
  toolbar.appendChild(summary);
  toolbar.appendChild(refreshBtn);
  toolbar.appendChild(selectionActions);

  /** Push the selection into the DOM: card classes, checkboxes, and the action bar. */
  function syncSelectionUi() {
    for (const card of grid.querySelectorAll('[data-event-uuid]')) {
      const isSelected = selected.has(card.dataset.eventUuid);
      card.classList.toggle('event-card--selected', isSelected);
      const box = card.querySelector('.event-card__select');
      if (box) box.checked = isSelected;
    }
    selectionActions.hidden = selected.size === 0;
    selectionCount.textContent = `${selected.size} selected`;
    deleteBtn.textContent = `Delete ${selected.size} event${selected.size === 1 ? '' : 's'}`;
    selectAllLabel.hidden = pageEvents.length === 0;
    selectAllBox.checked = pageEvents.length > 0 && selected.size === pageEvents.length;
    selectAllBox.indeterminate = selected.size > 0 && selected.size < pageEvents.length;
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

  async function deleteSelected() {
    const uuids = [...selected];
    if (!uuids.length) return;
    const confirmed = await showConfirmModal({
      title: `Delete ${uuids.length} event${uuids.length === 1 ? '' : 's'}?`,
      // Same promise the single-event delete makes, and for the same reason: the one
      // thing a user needs to know is that curated training frames are not in scope.
      message: `The events, their snapshots and any clips or timelapses of them will be removed. `
        + 'Frames you saved for annotation are not affected.',
      confirmText: 'Delete',
      cancelText: 'Keep',
    });
    if (!confirmed) return;
    deleteBtn.disabled = true;
    try {
      const deleted = await deleteWatchEvents(uuids);
      toast(`Deleted ${deleted} event${deleted === 1 ? '' : 's'}`, 3000);
      selected.clear();
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
    const anyFilter = Object.values(filters.getState()).some(v => v != null);
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

  async function search() {
    const seq = ++requestSeq;
    grid.setAttribute('aria-busy', 'true');
    try {
      const result = await fetchWatchEvents({ ...filters.getState(), limit: PAGE_SIZE, offset });
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
      // A selection is a set of cards on screen, so it cannot survive the cards. Kept
      // across a re-render it would silently arm "Delete 12 events" against a page the
      // user is no longer looking at.
      pageEvents = result.events;
      selected.clear();
      anchorIndex = -1;
      if (!result.events.length) {
        renderEmpty();
      } else {
        for (const event of result.events) {
          grid.appendChild(buildEventCard(event, {
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
          }));
        }
      }
      syncSelectionUi();

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

  await search();
}
