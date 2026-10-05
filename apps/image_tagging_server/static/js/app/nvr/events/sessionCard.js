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
import { parseApiDate } from '/app-static/js/shared/time.js';
import { eventHeadline, formatDuration, formatEventTime, setEventCardArt } from './eventCard.js';

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
 * @param {(session: Object) => void} ctx.onOpen - the events behind it, or its one event
 * @param {(session: Object, opts: {shiftKey: boolean, checked: boolean}) => void}
 *   [ctx.onToggleSelect] - omitted where the grid cannot act on a selection
 * @param {boolean} [ctx.selected]
 * @param {Map<string, string>} [ctx.sourceNames]
 * @returns {HTMLElement}
 */
export function buildSessionCard(
  session,
  { onOpen, onToggleSelect, selected = false, sourceNames = new Map() } = {},
) {
  const card = el('article', 'event-card event-card--session');
  card.dataset.testid = 'event-session-card';
  // Keyed on the representative rather than an identity: the same cat at the same bowl
  // is one identity and a dozen separate visits, and the grid has to tell them apart.
  if (session.representativeUuid) card.dataset.sessionKey = session.representativeUuid;
  card.classList.toggle('event-card--selected', selected);
  card.tabIndex = 0;
  card.setAttribute('role', 'button');

  const start = parseApiDate(session.startedAt);
  // An open visit's `endedAt` is its last event's *start* — see the field's note on the
  // API model. Treating it as an end would caption a session that is still running as a
  // finished one, and a one-event open session as an instantaneous one.
  const end = session.isOpen ? null : parseApiDate(session.endedAt);
  const headline = eventHeadline(session);
  card.setAttribute('aria-label', `${headline}, ${session.count} events from ${formatEventTime(start)}`);

  const media = el('div', 'event-card__media');
  if (session.representativeUuid) {
    const img = document.createElement('img');
    img.className = 'event-card__image';
    img.loading = 'lazy';
    img.alt = '';
    // Crop first, wide shot as the fallback — a session card is a grid card and asks
    // the grid's question. It carries no `hasCrop`, so it takes the two-step path.
    setEventCardArt(img, session.representativeUuid, {
      onGone: () => {
        img.remove();
        media.classList.add('event-card__media--empty');
      },
    });
    media.appendChild(img);
  } else {
    media.classList.add('event-card__media--empty');
  }

  // Only when it actually collapsed something. A "×1" on most cards would be noise and
  // would make the badge stop meaning "several of these were folded together".
  if (session.count > 1) {
    const badge = el('span', 'event-card__badge event-card__badge--repeat', `×${session.count}`);
    badge.title = `${session.count} events, folded into one grouping because no long gap separates them.`;
    badge.dataset.testid = 'event-session-repeat-badge';
    media.appendChild(badge);
  }
  // Which media the visit holds, in the same glyphs the flat card uses — ▶ for a clip,
  // ⏩ for a timelapse — abbreviated to the glyph alone because a folded card already
  // carries a count badge and a live badge in the same corner. A visit can show both:
  // a watcher recording both kinds produces one of each.
  if (session.hasClip || session.hasTimelapse) {
    const mediaKinds = el('div', 'event-card__badges event-card__badges--media');
    if (session.hasClip) {
      const badge = el('span', 'event-card__badge event-card__badge--video', '▶');
      badge.title = 'At least one event in this visit has a clip — real time, at the camera’s frame rate.';
      badge.dataset.testid = 'event-session-video-badge';
      mediaKinds.appendChild(badge);
    }
    if (session.hasTimelapse) {
      const badge = el('span', 'event-card__badge event-card__badge--timelapse', '⏩');
      badge.title = 'At least one event in this visit has a timelapse — sampled while it was open, '
        + 'so minutes or hours are compressed into seconds.';
      badge.dataset.testid = 'event-session-timelapse-badge';
      mediaKinds.appendChild(badge);
    }
    media.appendChild(mediaKinds);
  }
  // The same badge the flat grid's card carries, for the same reason: an open event's
  // clip does not exist yet and its duration is still growing. Folding repeats into
  // visits must not be the thing that hides that.
  if (session.isOpen) {
    const live = el('span', 'event-card__badge event-card__badge--live', 'In progress');
    live.dataset.testid = 'event-session-live-badge';
    media.appendChild(live);
  }
  // Same bargain as the event card's box: the checkbox is the only thing that selects
  // on a plain click, because a visit is opened far more often than it is bulk-acted on.
  // Without it this view had no multi-select at all — and it is the view the flat list
  // actually renders, since the gap preference defaults to grouping.
  if (onToggleSelect) {
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.className = 'event-card__select';
    box.checked = selected;
    box.dataset.testid = 'event-session-select';
    box.setAttribute('aria-label', `Select all ${session.count} ${headline} events in this visit`);
    box.addEventListener('click', e => {
      // Or the card's own handler opens the visit over the selection just made.
      e.stopPropagation();
      onToggleSelect(session, { shiftKey: e.shiftKey, checked: box.checked });
    });
    media.appendChild(box);
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
  const duration = formatDuration(session.startedAt, end?.toISOString() ?? null);
  if (duration) meta.appendChild(el('span', 'event-card__meta-item', duration));
  if (meta.childElementCount) body.appendChild(meta);
  card.appendChild(body);

  // What "open" means depends on how much this card folded — a visit of seven has no
  // single clip to play — so the decision belongs to the caller, which is the only side
  // that can act on either answer.
  card.addEventListener('click', (e) => {
    // Shift/ctrl-click reaches the selection without hunting for the 1.4em box, exactly
    // as it does on an event or group card.
    if (onToggleSelect && (e.shiftKey || e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      // Read off the card: the grid owns the selection and this closure's `selected` is
      // only ever the state at build time.
      const isSelected = card.classList.contains('event-card--selected');
      onToggleSelect(session, { shiftKey: e.shiftKey, checked: !isSelected });
      return;
    }
    onOpen?.(session);
  });
  card.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onOpen?.(session);
    }
  });
  return card;
}
