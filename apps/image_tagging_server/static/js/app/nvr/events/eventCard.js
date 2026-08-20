/**
 * One result card in the NVR event search.
 *
 * A card has to answer three questions from across the room, in this order: **when**,
 * **what happened**, and **who did it to what**. Everything else — duration, camera,
 * confidence — is secondary and is sized accordingly. The snapshot is the card's
 * background rather than a thumbnail beside text, because at a glance the picture *is*
 * the answer and the text only qualifies it.
 */
import { watchEventSnapshotUrl } from '/app-static/js/shared/api/watchProfiles.js';
import { parseApiDate } from '/app-static/js/shared/time.js';

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * "Today 14:32:06" / "Yesterday 09:01" / "12 Aug 22:14".
 *
 * Relative day naming for the recent past because that is how someone reviewing an NVR
 * thinks ("was that this morning?"), with the absolute date taking over once "yesterday"
 * stops being a useful anchor.
 *
 * @param {?Date} date
 * @returns {string}
 */
export function formatEventTime(date) {
  if (!date) return 'Unknown time';
  const time = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const today = new Date();
  const isSameDay = (a, b) => a.toDateString() === b.toDateString();
  if (isSameDay(date, today)) return `Today ${time}`;
  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);
  if (isSameDay(date, yesterday)) return `Yesterday ${time}`;
  return `${date.toLocaleDateString([], { day: 'numeric', month: 'short' })} ${time}`;
}

/**
 * @param {?string} startedAt
 * @param {?string} endedAt
 * @returns {string} '' when the event is still open — an open event has no duration yet,
 *   and showing "0s" for one would be a wrong answer rather than a missing one
 */
export function formatDuration(startedAt, endedAt) {
  const start = parseApiDate(startedAt);
  const end = parseApiDate(endedAt);
  if (!start || !end) return '';
  const seconds = Math.max(0, (end - start) / 1000);
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds - minutes * 60)}s`;
}

/**
 * The headline: what happened, phrased as actor → acted-on when both are known.
 *
 * Falls back through progressively less specific descriptions rather than showing a
 * uuid: a card captioned with a raw id is worse than one captioned "Event".
 *
 * @param {Object} event
 * @returns {string}
 */
export function eventHeadline(event) {
  const rule = event.eventLabel || event.ruleName;
  if (event.subjectName && event.targetName) {
    return `${event.subjectName} → ${event.targetName}`;
  }
  return event.subjectName || rule || 'Event';
}

/**
 * @param {Object} event - a WatchEventRead
 * @param {Object} ctx
 * @param {(event: Object) => void} ctx.onOpen
 * @param {Map<string, string>} [ctx.sourceNames] - source uuid → camera name
 * @param {(event: Object, opts: {shiftKey: boolean, checked: boolean}) => void} [ctx.onToggleSelect]
 * @param {boolean} [ctx.selected]
 * @returns {HTMLElement}
 */
/**
 * The event's recording of one kind (`'clip'` / `'timelapse'`), or null.
 *
 * Shared with the detail modal so the card's badges and the modal's players can never
 * disagree about what an event has. Defensive about a missing list: the events endpoint
 * always sends one, but this is also called against the fixtures Playwright mocks with,
 * and a `.find` on undefined takes the whole grid down rather than one badge.
 */
export function findRecording(event, kind) {
  return (event?.recordings || []).find((r) => r.kind === kind) || null;
}

export function buildEventCard(event, { onOpen, onToggleSelect, selected = false, sourceNames = new Map() } = {}) {
  const card = el('article', 'event-card');
  card.dataset.testid = 'event-card';
  card.dataset.eventUuid = event.uuid;
  card.tabIndex = 0;
  card.setAttribute('role', 'button');
  card.classList.toggle('event-card--selected', selected);

  const started = parseApiDate(event.startedAt);
  const headline = eventHeadline(event);
  card.setAttribute('aria-label', `${headline}, ${formatEventTime(started)}`);

  const media = el('div', 'event-card__media');
  if (event.hasSnapshot) {
    const img = document.createElement('img');
    img.className = 'event-card__image';
    img.loading = 'lazy';
    img.alt = '';
    img.src = watchEventSnapshotUrl(event.uuid);
    // Card art can be reaped out from under a row; a broken-image icon in a grid reads
    // as a bug, whereas the placeholder reads as "no picture for this one".
    img.addEventListener('error', () => {
      img.remove();
      media.classList.add('event-card__media--empty');
    });
    media.appendChild(img);
  } else {
    media.classList.add('event-card__media--empty');
  }

  // One badge per kind of media, not one for "has video": an event can own both a clip
  // and a timelapse, and they answer different questions — the clip is what happened,
  // the timelapse is what a two-hour visit looked like in ten seconds. A card showing
  // only "Video" made the second one undiscoverable without opening every event.
  //
  // Stacked in a flow container rather than each pinned to the corner, which is what
  // they did when there could only ever be one: two absolutely positioned badges land
  // on precisely the same pixels.
  const badges = el('div', 'event-card__badges');
  if (findRecording(event, 'clip')) {
    const badge = el('span', 'event-card__badge event-card__badge--video', '▶ Video');
    badge.dataset.testid = 'event-card-video-badge';
    badges.appendChild(badge);
  }
  if (findRecording(event, 'timelapse')) {
    const badge = el('span', 'event-card__badge event-card__badge--timelapse', '⏩ Timelapse');
    badge.dataset.testid = 'event-card-timelapse-badge';
    badges.appendChild(badge);
  }
  if (badges.childElementCount) media.appendChild(badges);

  // The checkbox is the *only* thing on the card that selects on a plain click; the
  // card itself still opens the event. Review's grid trades the other way (click
  // selects, double-click opens) because there the common act is bulk-tagging a page of
  // boxes, whereas an event is nearly always opened to be watched — making a click
  // select would put a modal behind a double-click for the thing everybody came to do.
  // Shift/ctrl-click on the card body reaches selection without hunting for the box.
  if (onToggleSelect) {
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.className = 'event-card__select';
    box.checked = selected;
    box.dataset.testid = 'event-card-select';
    box.setAttribute('aria-label', `Select ${headline}`);
    box.addEventListener('click', e => {
      // Without this the card's own handler fires next and opens the modal over the
      // selection the user just made.
      e.stopPropagation();
      onToggleSelect(event, { shiftKey: e.shiftKey, checked: box.checked });
    });
    media.appendChild(box);
  }
  if (!event.endedAt) {
    // An open event is genuinely different from a finished one — its clip does not
    // exist yet and its duration is still growing.
    media.appendChild(el('span', 'event-card__badge event-card__badge--live', 'In progress'));
  }
  card.appendChild(media);

  const body = el('div', 'event-card__body');

  const time = el('time', 'event-card__time', formatEventTime(started));
  if (started) time.dateTime = started.toISOString();
  time.dataset.testid = 'event-card-time';
  body.appendChild(time);

  const title = el('h4', 'event-card__headline', headline);
  title.dataset.testid = 'event-card-headline';
  body.appendChild(title);

  // The rule is the *kind* of event; it sits under the actors because "Tabby → litter
  // box" is what identifies this event and "cat using litter box" is what identifies
  // its category.
  if (event.ruleName && event.ruleName !== headline) {
    body.appendChild(el('span', 'event-card__rule', event.ruleName));
  }

  const meta = el('div', 'event-card__meta');
  const camera = sourceNames.get(event.sourceUuid);
  if (camera) meta.appendChild(el('span', 'event-card__meta-item', camera));
  // An event closed by a restart has `ended_at` set to its own start, because the real
  // end was never recorded — see `close_stranded_watch_events`. Rendering that as "0.0s"
  // would present a guess as a measurement, which is the same wrong answer
  // `formatDuration` refuses to give for an open event.
  if (event.metadataJson?.closedBy === 'restart') {
    const note = el('span', 'event-card__meta-item event-card__meta-item--warn', 'ended unknown');
    note.title = 'The watcher stopped before this event closed, so how long it lasted was never recorded.';
    note.dataset.testid = 'event-card-stranded';
    meta.appendChild(note);
  } else {
    const duration = formatDuration(event.startedAt, event.endedAt);
    if (duration) meta.appendChild(el('span', 'event-card__meta-item', duration));
  }
  if (event.truncated) {
    const capped = el('span', 'event-card__meta-item event-card__meta-item--warn', 'capped');
    capped.title = 'Closed by the runaway-event cap, not by the subject leaving.';
    meta.appendChild(capped);
  }
  if (meta.childElementCount) body.appendChild(meta);

  card.appendChild(body);

  card.addEventListener('click', e => {
    // Shift extends a range and ctrl/meta toggles one card — the two gestures every
    // file manager already taught the user — and neither should also open the modal.
    if (onToggleSelect && (e.shiftKey || e.ctrlKey || e.metaKey)) {
      // Shift-clicking twice in a grid otherwise leaves the whole range text-selected.
      e.preventDefault();
      window.getSelection()?.removeAllRanges();
      // Read off the card, not the `selected` argument: the grid keeps selection state
      // in the class rather than rebuilding cards, so the argument is only ever true of
      // the moment this card was built.
      const isSelected = card.classList.contains('event-card--selected');
      onToggleSelect(event, { shiftKey: e.shiftKey, checked: !isSelected });
      return;
    }
    onOpen?.(event);
  });
  card.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onOpen?.(event);
    }
  });

  return card;
}
