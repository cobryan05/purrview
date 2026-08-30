/**
 * NVR → Events: the timeline view.
 *
 * One lane per identity, time across the page. It answers the question the card grid
 * structurally cannot: *when* did these happen, and how do they sit against each other.
 * A grid page covers a variable, invisible slice of time — sixty cards might be four
 * hours or four days — so "was she at the bowl all afternoon or twice?" has no answer
 * there at any page size.
 *
 * ⚠️ **Marks are drawn from the binned histogram, never one node per event.** Live
 * history already holds 1277 events across 8 days and the log is append-only; a month at
 * the current rate is ~4500 marks and it only ever grows. The server bins by hour and
 * this draws one node per non-empty bin per lane — bounded by what is *visible* rather
 * than by how much history exists. Individual events are resolved on click, by a query
 * scoped to one lane and one bin.
 *
 * Positioning is by absolute timestamp, so it is timezone-agnostic; only the axis labels
 * are rendered in local time. That is the whole of "the client composes local days" —
 * the server's UTC hour buckets never need converting, just labelling.
 */
import { parseApiDate } from '/app-static/js/shared/time.js';
import { groupHeadline, identityKey } from './identity.js';

const HOUR_MS = 3600 * 1000;
const DAY_MS = 24 * HOUR_MS;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * The window the lanes are drawn across.
 *
 * Prefers the user's explicit date filter, because a timeline that silently retracted to
 * the extent of the data would make an empty Tuesday disappear rather than show up as
 * the gap it is — and the gaps are half of what a timeline is for. Falls back to the
 * data's own extent, padded to whole buckets so the first and last marks are not flush
 * against the edge.
 *
 * @param {Array<Object>} buckets
 * @param {Object} filters - an EventFilterState
 * @param {number} bucketMs
 * @returns {?{start: number, end: number}} epoch ms, or null when there is nothing to draw
 */
export function timelineRange(buckets, filters, bucketMs) {
  const from = parseApiDate(filters?.startFrom)?.getTime() ?? null;
  const to = parseApiDate(filters?.startTo)?.getTime() ?? null;
  if (from != null && to != null && to > from) return { start: from, end: to };

  const times = buckets.map(b => parseApiDate(b.bucketStart)?.getTime()).filter(t => t != null);
  if (!times.length) return null;
  let start = from ?? Math.min(...times);
  // The bucket's *end*, not its start: an event in the 14:00 bucket happened somewhere
  // in 14:00–15:00, and clipping the axis at 14:00 would put it on the boundary.
  let end = to ?? Math.max(...times) + bucketMs;
  if (end <= start) end = start + bucketMs;
  return { start, end };
}

/**
 * Axis ticks at a human interval, labelled in the viewer's local time.
 *
 * The step is chosen so a wide range does not grow a label per hour; the labels
 * themselves switch from clock to date once "which day" stops being obvious.
 *
 * @param {{start: number, end: number}} range
 * @returns {Array<{at: number, label: string, major: boolean}>}
 */
export function axisTicks({ start, end }) {
  const span = end - start;
  const step = span <= 2 * DAY_MS ? 6 * HOUR_MS : span <= 14 * DAY_MS ? DAY_MS : 7 * DAY_MS;
  const ticks = [];
  // Walk from the first local step boundary at or after `start`. Built with a Date so
  // the step lands on local midnight rather than UTC midnight — the whole reason this
  // is done here rather than in SQL.
  const cursor = new Date(start);
  cursor.setMinutes(0, 0, 0);
  while (cursor.getTime() < start || (cursor.getTime() - start) % step !== 0) {
    cursor.setHours(cursor.getHours() + 1);
    if (cursor.getTime() > end) break;
  }
  for (let t = cursor.getTime(); t <= end; t += step) {
    const date = new Date(t);
    const midnight = date.getHours() === 0;
    ticks.push({
      at: t,
      label: step >= DAY_MS || midnight
        ? date.toLocaleDateString([], { day: 'numeric', month: 'short' })
        : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      major: midnight,
    });
  }
  return ticks;
}

/**
 * Build the timeline.
 *
 * @param {Object} opts
 * @param {Array<Object>} opts.groups - folded groups, in the order lanes should appear
 * @param {Array<Object>} opts.buckets - histogram rows
 * @param {string} opts.bucket - the resolution the server actually used
 * @param {Object} opts.filters - the current EventFilterState
 * @param {Object} opts.focus - a focus controller (identity.js)
 * @param {boolean} [opts.splitByCamera]
 * @param {(group: Object) => void} [opts.onSelectGroup] - lane label clicked
 * @param {(group: Object, window: {from: Date, to: Date}) => void} [opts.onSelectBucket]
 * @returns {HTMLElement}
 */
export function buildTimeline({
  groups,
  buckets,
  bucket,
  filters,
  focus,
  dimensions = ['rule', 'subject', 'target'],
  splitByCamera = false,
  onSelectGroup,
  onSelectBucket,
}) {
  const bucketMs = bucket === 'day' ? DAY_MS : HOUR_MS;
  const root = el('div', 'event-timeline');
  root.dataset.testid = 'event-timeline';

  const range = timelineRange(buckets, filters, bucketMs);
  if (!range || !groups.length) {
    const empty = el('div', 'nvr-empty');
    empty.dataset.testid = 'event-timeline-empty';
    empty.appendChild(el('strong', null, 'Nothing to plot'));
    empty.appendChild(el('p', null, 'No events fall in this range.'));
    root.appendChild(empty);
    return root;
  }
  const span = range.end - range.start;
  /** Fraction of the way across the axis, clamped so a boundary event stays on screen. */
  const pct = t => Math.min(100, Math.max(0, ((t - range.start) / span) * 100));

  if (bucket === 'day') {
    // Said out loud rather than quietly drawn: the server coarsened the resolution
    // because an hour-by-hour answer would have been too large, and a user comparing
    // this against the clock deserves to know the marks are whole days.
    const note = el('p', 'event-timeline__note',
      'Showing whole days — the range is too wide for hour-by-hour detail. Narrow the dates to zoom in.');
    note.dataset.testid = 'event-timeline-coarse-note';
    root.appendChild(note);
  }

  // ── Axis ────────────────────────────────────────────────────────────────────
  const axis = el('div', 'event-timeline__axis');
  axis.dataset.testid = 'event-timeline-axis';
  axis.appendChild(el('div', 'event-timeline__axis-gutter'));
  const axisTrack = el('div', 'event-timeline__axis-track');
  for (const tick of axisTicks(range)) {
    const mark = el('span', `event-timeline__tick${tick.major ? ' event-timeline__tick--major' : ''}`, tick.label);
    mark.style.left = `${pct(tick.at)}%`;
    axisTrack.appendChild(mark);
  }
  axis.appendChild(axisTrack);
  root.appendChild(axis);

  // ── Lanes ───────────────────────────────────────────────────────────────────
  //
  // Bucketed by identity first so each lane walks only its own marks; doing it the
  // other way round is a scan of the whole histogram per lane, which at 22 lanes and a
  // month of history is the one quadratic in this file waiting to be written.
  const byIdentity = new Map();
  let peak = 1;
  for (const row of buckets) {
    const key = identityKey(row, { dimensions, splitByCamera });
    let list = byIdentity.get(key);
    if (!list) byIdentity.set(key, (list = []));
    list.push(row);
    if (row.count > peak) peak = row.count;
  }

  const lanes = el('div', 'event-timeline__lanes');
  for (const group of groups) {
    const rows = byIdentity.get(group.key) ?? [];
    const lane = el('div', 'event-timeline__lane');
    lane.dataset.testid = 'event-timeline-lane';
    lane.dataset.identityKey = group.key;

    const { primary, secondary } = groupHeadline(group, dimensions);
    const label = el('button', 'event-timeline__label');
    label.type = 'button';
    label.dataset.testid = 'event-timeline-label';
    const text = el('span', 'event-timeline__label-text');
    text.appendChild(el('span', 'event-timeline__label-primary', primary));
    // The rule on a second line when it is not already the headline: grouping per actor
    // otherwise gives two lanes both reading "Luna", one per rule she appears in.
    if (secondary) {
      text.appendChild(el('span', 'event-timeline__label-rule', secondary));
    }
    label.appendChild(text);
    label.appendChild(el('span', 'event-timeline__label-count', String(group.count)));
    label.title = `${primary}${secondary ? ` — ${secondary}` : ''} — ${group.count} events. `
      + 'Click to see them; the highlight follows this row across every view.';
    label.addEventListener('click', () => onSelectGroup?.(group));
    lane.appendChild(label);

    const track = el('div', 'event-timeline__track');
    // The lane's whole strip is the hover target, not just its marks: an identity that
    // fired twice in a week would otherwise be a pair of 3px targets, and the highlight
    // is the point of the view.
    for (const row of rows) {
      const at = parseApiDate(row.bucketStart)?.getTime();
      if (at == null || at > range.end || at + bucketMs < range.start) continue;
      const mark = el('span', 'event-timeline__mark');
      mark.dataset.testid = 'event-timeline-mark';
      mark.style.left = `${pct(at)}%`;
      // Width in percent so it scales with the container rather than needing a resize
      // observer; the CSS floors it at a tappable minimum.
      mark.style.width = `${Math.max(0.15, (bucketMs / span) * 100)}%`;
      // Density by opacity against the busiest bin on screen. A count of 1 and a count
      // of 52 must not look the same, and height would fight the lane's own baseline.
      mark.style.opacity = String(0.35 + 0.65 * (row.count / peak));
      const when = new Date(at);
      mark.title = `${row.count} event${row.count === 1 ? '' : 's'} — ${
        when.toLocaleString([], { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })}`;
      mark.addEventListener('click', e => {
        e.stopPropagation();
        onSelectBucket?.(group, { from: when, to: new Date(at + bucketMs) });
      });
      track.appendChild(mark);
    }
    lane.appendChild(track);
    lanes.appendChild(lane);

    // One registration for the lane, so hovering the label or any mark lights the whole
    // identity — here, on its group card, and on every matching tile in the flat grid.
    focus.register(group.key, lane);
    // Clicking the label pins, so the highlight survives the pointer leaving. Without
    // this the view is unusable by touch, which has no hover at all.
    label.addEventListener('click', () => focus.togglePin(group.key));
  }
  root.appendChild(lanes);
  return root;
}
