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
import { fetchWatchEvents } from '/app-static/js/shared/api/watchProfiles.js';
import { fetchSourceList } from '/app-static/js/shared/api/sources.js';
import { Logger } from '/app-static/js/ui/utils/index.js';
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

  const summary = el('div', 'event-search__summary');
  summary.dataset.testid = 'event-search-summary';
  const grid = el('div', 'event-search__grid');
  grid.dataset.testid = 'event-search-grid';
  const pager = el('div', 'event-search__pager');

  let offset = 0;
  let total = 0;
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

  page.appendChild(filters.element);
  page.appendChild(summary);
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
      : 'Events appear here as your watchers detect them. Start a watcher on the Live tab.'));
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
      if (!result.events.length) {
        renderEmpty();
      } else {
        for (const event of result.events) {
          grid.appendChild(buildEventCard(event, {
            sourceNames,
            onOpen: e => openEventDetailModal(e, {
              sourceNames,
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
