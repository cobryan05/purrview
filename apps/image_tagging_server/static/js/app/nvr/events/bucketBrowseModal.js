/**
 * A set of events — one timeline mark, one group card, one visit — as browsable cards.
 *
 * The unit a card stands for is a *set*, not an event: a busy hour, or an identity that
 * fired 383 times. Opening the first one straight into the player made the rest
 * reachable only by clicking prev/next blind, with no way to see which was worth
 * watching — the picture is the answer, and prev/next hides every picture but one.
 *
 * So a set holding more than one event opens here. The player is still one click away,
 * and it still gets the loaded events as its siblings, so prev/next inside it works
 * once you have chosen where to start. This modal stays open underneath, which is what
 * makes "watch that one, no, the next one" cheap.
 *
 * Opening here instead of narrowing the page is also what keeps the page's filters: a
 * drilldown rewrites them, and the reader who spent a minute building a filter set to
 * find the group loses it by clicking the thing they were looking for. `actions` is
 * where that drilldown lives now — an explicit button, not the side effect of a click.
 */
import { createModalShell } from '/app-static/js/ui/utils/index.js';
import { buildEventCard } from './eventCard.js';

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * @param {Object} opts
 * @param {Array<Object>} opts.events - the first page, already fetched
 * @param {string} opts.title - the set's identity, e.g. "Luna → Feeder"
 * @param {string} [opts.subtitle] - the time span, or the rule behind it
 * @param {Map<string, string>} [opts.sourceNames]
 * @param {(event: Object, siblings: Array<Object>) => void} opts.onOpenEvent
 * @param {string} [opts.testid]
 * @param {?number} [opts.total] - events matching, when more exist than were fetched
 * @param {?(offset: number) => Promise<Array<Object>>} [opts.loadMore]
 * @param {Array<{label: string, title?: string, testid?: string, onClick: (close: () => void) => void}>} [opts.actions]
 * @returns {{ close: () => void }}
 */
export function openBucketBrowseModal({
  events,
  title,
  subtitle = '',
  sourceNames = new Map(),
  onOpenEvent,
  testid = 'bucket-browse-modal',
  total = null,
  loadMore = null,
  actions = [],
}) {
  const shell = createModalShell();
  shell.dialog.classList.add('app-modal-dialog--lg');
  shell.dialog.dataset.testid = testid;

  // Mutated as pages arrive, and handed to the player as its siblings, so prev/next
  // walks everything loaded so far rather than only the page the reader opened from.
  const loaded = [...events];

  const header = el('div', 'bucket-browse__header');
  const heading = el('h3', 'bucket-browse__title', title || 'Events');
  header.appendChild(heading);
  const count = el('span', 'bucket-browse__count');
  count.dataset.testid = 'bucket-browse-count';
  header.appendChild(count);
  shell.dialog.appendChild(header);

  if (subtitle) shell.dialog.appendChild(el('p', 'bucket-browse__subtitle', subtitle));

  // Its own scroller rather than letting the dialog grow: --lg caps the dialog at 85vh,
  // and a bin of fifty cards would otherwise push the heading off the top of it.
  const scroller = el('div', 'bucket-browse__scroll');
  const grid = el('div', 'event-search__grid');
  grid.dataset.testid = 'bucket-browse-grid';
  scroller.appendChild(grid);
  shell.dialog.appendChild(scroller);

  const footer = el('div', 'bucket-browse__footer');
  const more = el('button', 'panel-btn bucket-browse__more', 'Load more');
  more.type = 'button';
  more.dataset.testid = 'bucket-browse-more';
  footer.appendChild(more);
  for (const action of actions) {
    const btn = el('button', 'panel-btn', action.label);
    btn.type = 'button';
    if (action.testid) btn.dataset.testid = action.testid;
    if (action.title) btn.title = action.title;
    btn.addEventListener('click', () => action.onClick(shell.close));
    footer.appendChild(btn);
  }
  shell.dialog.appendChild(footer);

  function appendCards(batch) {
    for (const event of batch) {
      grid.appendChild(buildEventCard(event, {
        sourceNames,
        // No checkbox: selection acts on the page's grid, and a set opened over it has
        // no bulk bar to act through.
        onOpen: e => onOpenEvent(e, loaded),
      }));
    }
  }

  function syncCount() {
    const known = total ?? loaded.length;
    count.textContent = known > loaded.length
      ? `${loaded.length} of ${known} events`
      : `${loaded.length} event${loaded.length === 1 ? '' : 's'}`;
    more.hidden = !loadMore || known <= loaded.length;
    footer.hidden = more.hidden && !actions.length;
  }

  more.addEventListener('click', async () => {
    more.disabled = true;
    more.textContent = 'Loading…';
    try {
      const batch = await loadMore(loaded.length);
      // An empty page with a total still claiming more means the set shrank under us;
      // stop asking rather than leaving a button that can only disappoint.
      if (!batch.length) total = loaded.length;
      else {
        loaded.push(...batch);
        appendCards(batch);
      }
    } finally {
      more.disabled = false;
      more.textContent = 'Load more';
      syncCount();
    }
  });

  appendCards(events);
  syncCount();

  shell.open();
  return { close: shell.close };
}
