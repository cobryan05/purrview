/**
 * One card per *visit* rather than per rule firing.
 *
 * A single trip to the water bowl re-fires its rule several times as `min_duration_s`
 * and `rearm_delay_s` cycle, so the flat grid shows seven tiles for one drink. This is
 * the same card as `eventCard.js` — the grid is still a grid of things that happened —
 * with the span replacing the instant and a repeat count where a single event has none.
 *
 * Sized as a *refinement*, not the headline: session grouping roughly halves the grid
 * (1277 → 653 on live history), where identity grouping cuts it by sixty. It makes the
 * flat view honest about what one visit is; it does not make the page comprehensible.
 */
import { watchEventSnapshotUrl } from '/app-static/js/shared/api/watchProfiles.js';
import { parseApiDate } from '/app-static/js/shared/time.js';
import { eventHeadline, formatDuration, formatEventTime } from './eventCard.js';

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * "14:32 → 14:47" for a visit, or just the start when it was one short event.
 *
 * The end is only worth the space when it says something the duration does not — a
 * session of one 20-second event is an instant, and rendering it as a range implies a
 * spread that is not there.
 *
 * @param {?Date} start
 * @param {?Date} end
 * @returns {string}
 */
export function formatSessionSpan(start, end) {
  if (!start) return 'Unknown time';
  const opening = formatEventTime(start);
  if (!end || end.getTime() - start.getTime() < 60000) return opening;
  return `${opening} → ${end.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
}

/**
 * @param {Object} session - a WatchEventSession
 * @param {Object} ctx
 * @param {(session: Object) => void} ctx.onOpen - show the events behind it
 * @param {Map<string, string>} [ctx.sourceNames]
 * @returns {HTMLElement}
 */
export function buildSessionCard(session, { onOpen, sourceNames = new Map() } = {}) {
  const card = el('article', 'event-card event-card--session');
  card.dataset.testid = 'event-session-card';
  card.tabIndex = 0;
  card.setAttribute('role', 'button');

  const start = parseApiDate(session.startedAt);
  const end = parseApiDate(session.endedAt);
  const headline = eventHeadline(session);
  card.setAttribute('aria-label', `${headline}, ${session.count} events from ${formatEventTime(start)}`);

  const media = el('div', 'event-card__media');
  if (session.representativeUuid) {
    const img = document.createElement('img');
    img.className = 'event-card__image';
    img.loading = 'lazy';
    img.alt = '';
    img.src = watchEventSnapshotUrl(session.representativeUuid);
    img.addEventListener('error', () => {
      img.remove();
      media.classList.add('event-card__media--empty');
    });
    media.appendChild(img);
  } else {
    media.classList.add('event-card__media--empty');
  }

  // Only when it actually collapsed something. A "×1" on most cards would be noise and
  // would make the badge stop meaning "several of these were folded together".
  if (session.count > 1) {
    const badge = el('span', 'event-card__badge event-card__badge--repeat', `×${session.count}`);
    badge.title = `${session.count} events, grouped as one visit because no long gap separates them.`;
    badge.dataset.testid = 'event-session-repeat-badge';
    media.appendChild(badge);
  }
  card.appendChild(media);

  const body = el('div', 'event-card__body');
  const time = el('time', 'event-card__time', formatSessionSpan(start, end));
  if (start) time.dateTime = start.toISOString();
  time.dataset.testid = 'event-session-time';
  body.appendChild(time);

  const title = el('h4', 'event-card__headline', headline);
  title.dataset.testid = 'event-session-headline';
  body.appendChild(title);
  if (session.ruleName && session.ruleName !== headline) {
    body.appendChild(el('span', 'event-card__rule', session.ruleName));
  }

  const meta = el('div', 'event-card__meta');
  const camera = sourceNames.get(session.sourceUuid);
  if (camera) meta.appendChild(el('span', 'event-card__meta-item', camera));
  // The span of the whole visit, which is the number a session exists to report — the
  // sum of its events' durations would leave out the gaps and understate the visit.
  const duration = formatDuration(session.startedAt, session.endedAt);
  if (duration) meta.appendChild(el('span', 'event-card__meta-item', duration));
  if (meta.childElementCount) body.appendChild(meta);
  card.appendChild(body);

  // A session has no single clip to play, so opening it shows its events rather than
  // guessing which of the seven the user meant.
  card.addEventListener('click', () => onOpen?.(session));
  card.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onOpen?.(session);
    }
  });
  return card;
}
