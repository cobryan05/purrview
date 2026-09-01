/**
 * The events behind one timeline mark, as browsable cards.
 *
 * A mark is a *bin*, not an event: a busy hour can hold fifty of them. Opening the
 * first one straight into the player made the other forty-nine reachable only by
 * clicking prev/next blind, with no way to see which was worth watching — the picture
 * is the answer, and prev/next hides every picture but one.
 *
 * So a mark holding more than one event opens here first. The player is still one
 * click away, and it still gets the whole bin as its siblings, so prev/next inside it
 * works exactly as before once you have chosen where to start. This modal stays open
 * underneath, which is what makes "watch that one, no, the next one" cheap.
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
 * @param {Array<Object>} opts.events - every event in the bin, already fetched
 * @param {string} opts.title - the lane's identity, e.g. "Luna → Feeder"
 * @param {string} [opts.subtitle] - the bin's time span
 * @param {Map<string, string>} [opts.sourceNames]
 * @param {(event: Object, siblings: Array<Object>) => void} opts.onOpenEvent
 * @returns {{ close: () => void }}
 */
export function openBucketBrowseModal({ events, title, subtitle = '', sourceNames = new Map(), onOpenEvent }) {
  const shell = createModalShell();
  shell.dialog.classList.add('app-modal-dialog--lg');
  shell.dialog.dataset.testid = 'bucket-browse-modal';

  const header = el('div', 'bucket-browse__header');
  const heading = el('h3', 'bucket-browse__title', title || 'Events');
  header.appendChild(heading);
  const count = el('span', 'bucket-browse__count',
    `${events.length} event${events.length === 1 ? '' : 's'}`);
  count.dataset.testid = 'bucket-browse-count';
  header.appendChild(count);
  shell.dialog.appendChild(header);

  if (subtitle) shell.dialog.appendChild(el('p', 'bucket-browse__subtitle', subtitle));

  // Its own scroller rather than letting the dialog grow: --lg caps the dialog at 85vh,
  // and a bin of fifty cards would otherwise push the heading off the top of it.
  const scroller = el('div', 'bucket-browse__scroll');
  const grid = el('div', 'event-search__grid');
  grid.dataset.testid = 'bucket-browse-grid';
  for (const event of events) {
    grid.appendChild(buildEventCard(event, {
      sourceNames,
      // No checkbox: selection acts on the page's grid, and a bin opened over the
      // timeline has no bulk bar to act through.
      onOpen: e => onOpenEvent(e, events),
    }));
  }
  scroller.appendChild(grid);
  shell.dialog.appendChild(scroller);

  shell.open();
  return { close: shell.close };
}
