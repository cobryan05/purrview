/**
 * One card in the grouped event view — an *identity*, not an event.
 *
 * The flat grid answers "what happened at 14:32". This answers "what keeps happening",
 * which is the question you actually have when you open the page to see how the day
 * went. It is the same card language as `eventCard.js` — snapshot as background, when
 * first, then who did what to what — with the count taking the place the timestamp
 * holds there, because for a group the count *is* the headline fact.
 */
import { watchEventSnapshotUrl } from '/app-static/js/shared/api/watchProfiles.js';
import { parseApiDate } from '/app-static/js/shared/time.js';
import { formatEventTime } from './eventCard.js';
import { groupHeadline } from './identity.js';

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * "4h ago" / "2d ago" — how long since this identity last fired.
 *
 * Relative rather than absolute because the question a group answers is "is this still
 * going on?", where the gap matters and the wall-clock time does not. The individual
 * event card takes the opposite view and shows the clock, for the same reason.
 *
 * @param {?string} iso
 * @returns {string}
 */
export function formatSince(iso) {
  const date = parseApiDate(iso);
  if (!date) return '';
  const seconds = Math.max(0, (Date.now() - date.getTime()) / 1000);
  if (seconds < 90) return 'just now';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

/**
 * Total time spent, in the coarsest unit that is still honest.
 *
 * A group's duration runs to hours — "350447s" is not a number anyone reads — but
 * rounding a 40-second total up to "1m" would overstate it, so seconds survive below
 * the minute.
 *
 * @param {number} seconds
 * @returns {string}
 */
export function formatTotalDuration(seconds) {
  if (!seconds || seconds < 1) return '';
  if (seconds < 60) return `${Math.round(seconds)}s total`;
  const minutes = seconds / 60;
  if (minutes < 90) return `${Math.round(minutes)}m total`;
  return `${(minutes / 60).toFixed(1)}h total`;
}

/**
 * A bar per bucket, scaled to the busiest one.
 *
 * Inline SVG rather than a canvas or a chart library: it is a dozen rectangles, it
 * scales with the card instead of needing a resize observer, and it inherits
 * `currentColor` so the focus highlight recolours it for free.
 *
 * @param {number[]} counts - already bucketed, oldest first
 * @returns {?SVGElement} null when there is nothing to draw — one bar is not a trend
 */
export function buildSparkline(counts) {
  if (!counts || counts.length < 2) return null;
  const peak = Math.max(...counts);
  if (!peak) return null;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'event-group-card__spark');
  // A viewBox with no width/height: the card's CSS sizes it, so the same markup works
  // at any card width and on a phone without a second code path.
  svg.setAttribute('viewBox', `0 0 ${counts.length * 2 - 1} 10`);
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('aria-hidden', 'true');
  counts.forEach((count, i) => {
    // Floor of 1 unit so a bucket with events is never invisible: "one event here" and
    // "nothing here" must not render identically.
    const height = count ? Math.max(1, (count / peak) * 10) : 0;
    if (!height) return;
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', String(i * 2));
    rect.setAttribute('y', String(10 - height));
    rect.setAttribute('width', '1.4');
    rect.setAttribute('height', String(height));
    svg.appendChild(rect);
  });
  return svg;
}

/**
 * @param {Object} group - a folded WatchEventGroup (see identity.js `foldGroups`)
 * @param {Object} ctx
 * @param {(group: Object) => void} ctx.onDrilldown - "show me these 383 events"
 * @param {(group: Object) => void} [ctx.onOpenLatest] - open the newest one's player
 * @param {number[]} [ctx.sparkCounts]
 * @param {Map<string, string>} [ctx.sourceNames]
 * @param {(group: Object, opts: {shiftKey: boolean, checked: boolean}) => void} [ctx.onToggleSelect]
 * @param {boolean} [ctx.selected]
 * @returns {HTMLElement}
 */
export function buildGroupCard(group, {
  onDrilldown,
  onOpenLatest,
  sparkCounts = null,
  sourceNames = new Map(),
  onToggleSelect,
  selected = false,
} = {}) {
  const card = el('article', 'event-group-card');
  card.dataset.testid = 'event-group-card';
  card.dataset.identityKey = group.key;
  card.tabIndex = 0;
  card.setAttribute('role', 'button');
  card.classList.toggle('event-group-card--selected', selected);

  // Captioned for the grain the row is actually grouped at — see `groupHeadline`. A
  // per-actor card must not claim a target it only partly covers.
  const { primary: headline, secondary: ruleLine } = groupHeadline(group, group.dimensions ?? ['rule', 'subject', 'target']);
  card.setAttribute('aria-label', `${headline}, ${group.count} events`);

  const media = el('div', 'event-group-card__media');
  if (group.representativeUuid) {
    const img = document.createElement('img');
    img.className = 'event-group-card__image';
    img.loading = 'lazy';
    img.alt = '';
    img.src = watchEventSnapshotUrl(group.representativeUuid);
    // Same reasoning as the event card: card art can be reaped out from under a row,
    // and a broken-image icon reads as a bug where the placeholder reads as "no picture".
    img.addEventListener('error', () => {
      img.remove();
      media.classList.add('event-group-card__media--empty');
    });
    media.appendChild(img);
  } else {
    media.classList.add('event-group-card__media--empty');
  }

  const count = el('span', 'event-group-card__count', `${group.count}`);
  count.dataset.testid = 'event-group-card-count';
  media.appendChild(count);

  if (group.withBlips) {
    const badge = el('span', 'event-group-card__badge event-group-card__badge--blips',
      `⚠ ${group.withBlips}`);
    badge.title = `${group.withBlips} of these events have tracking issues the analyser flagged.`;
    badge.dataset.testid = 'event-group-card-blip-badge';
    media.appendChild(badge);
  }

  if (onToggleSelect) {
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.className = 'event-group-card__select';
    box.checked = selected;
    box.dataset.testid = 'event-group-card-select';
    box.setAttribute('aria-label', `Select all ${group.count} ${headline} events`);
    box.addEventListener('click', e => {
      e.stopPropagation();
      onToggleSelect(group, { shiftKey: e.shiftKey, checked: box.checked });
    });
    media.appendChild(box);
  }
  card.appendChild(media);

  const body = el('div', 'event-group-card__body');
  const title = el('h4', 'event-group-card__headline', headline);
  title.dataset.testid = 'event-group-card-headline';
  body.appendChild(title);
  if (ruleLine) body.appendChild(el('span', 'event-group-card__rule', ruleLine));

  const spark = buildSparkline(sparkCounts);
  if (spark) body.appendChild(spark);

  const meta = el('div', 'event-group-card__meta');
  const since = formatSince(group.lastSeen);
  if (since) meta.appendChild(el('span', 'event-group-card__meta-item', since));
  const total = formatTotalDuration(group.totalDurationS);
  if (total) meta.appendChild(el('span', 'event-group-card__meta-item', total));
  // Only when it is actually plural: "1 camera" on every card is noise, and naming the
  // single camera is what the split-by-camera toggle is for.
  const cameras = group.sourceUuids ?? [];
  if (cameras.length > 1) {
    meta.appendChild(el('span', 'event-group-card__meta-item', `${cameras.length} cameras`));
  } else if (group.sourceUuid && sourceNames.get(group.sourceUuid)) {
    meta.appendChild(el('span', 'event-group-card__meta-item', sourceNames.get(group.sourceUuid)));
  }
  if (meta.childElementCount) body.appendChild(meta);

  const actions = el('div', 'event-group-card__actions');
  const drill = el('button', 'panel-btn event-group-card__drill',
    `Show ${group.count} event${group.count === 1 ? '' : 's'}`);
  drill.type = 'button';
  drill.dataset.testid = 'event-group-card-drill';
  drill.addEventListener('click', e => {
    e.stopPropagation();
    onDrilldown?.(group);
  });
  actions.appendChild(drill);
  if (onOpenLatest && group.representativeUuid) {
    const latest = el('button', 'panel-btn event-group-card__latest', 'Latest');
    latest.type = 'button';
    latest.dataset.testid = 'event-group-card-latest';
    latest.title = `Open the most recent of these — ${formatEventTime(parseApiDate(group.lastSeen))}`;
    latest.addEventListener('click', e => {
      e.stopPropagation();
      onOpenLatest(group);
    });
    actions.appendChild(latest);
  }
  body.appendChild(actions);
  card.appendChild(body);

  // The card body drills in rather than opening a player. A group is a set, and the
  // obvious thing to want from a set is to see it — "Latest" is the deliberate escape
  // hatch to a single event, which is the opposite balance from the flat grid where a
  // click opens the video because an event is nearly always opened to be watched.
  card.addEventListener('click', e => {
    if (onToggleSelect && (e.shiftKey || e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      window.getSelection()?.removeAllRanges();
      const isSelected = card.classList.contains('event-group-card--selected');
      onToggleSelect(group, { shiftKey: e.shiftKey, checked: !isSelected });
      return;
    }
    onDrilldown?.(group);
  });
  card.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onDrilldown?.(group);
    }
  });

  return card;
}
