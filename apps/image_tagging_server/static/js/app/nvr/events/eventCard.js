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

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * Parse an API timestamp.
 *
 * The API's timestamps carry no UTC offset, so `new Date(s)` reads them as local time
 * in some browsers and as UTC in others — the same bug `parseRecordDate` exists for on
 * the Review side. Appending `Z` makes the reading explicit and consistent.
 *
 * @param {?string} value
 * @returns {?Date}
 */
export function parseEventDate(value) {
  if (!value) return null;
  const normalised = /[Z+]|-\d{2}:\d{2}$/.test(value) ? value : `${value}Z`;
  const date = new Date(normalised);
  return Number.isNaN(date.getTime()) ? null : date;
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
  const start = parseEventDate(startedAt);
  const end = parseEventDate(endedAt);
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

export function buildEventCard(event, { onOpen, sourceNames = new Map() } = {}) {
  const card = el('article', 'event-card');
  card.dataset.testid = 'event-card';
  card.dataset.eventUuid = event.uuid;
  card.tabIndex = 0;
  card.setAttribute('role', 'button');

  const started = parseEventDate(event.startedAt);
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
  const duration = formatDuration(event.startedAt, event.endedAt);
  if (duration) meta.appendChild(el('span', 'event-card__meta-item', duration));
  if (event.truncated) {
    const capped = el('span', 'event-card__meta-item event-card__meta-item--warn', 'capped');
    capped.title = 'Closed by the runaway-event cap, not by the subject leaving.';
    meta.appendChild(capped);
  }
  if (meta.childElementCount) body.appendChild(meta);

  card.appendChild(body);

  card.addEventListener('click', () => onOpen?.(event));
  card.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onOpen?.(event);
    }
  });

  return card;
}
