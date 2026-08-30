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
import { watchEventSnapshotUrl } from '/app-static/js/shared/api/watchProfiles.js';
import { parseApiDate } from '/app-static/js/shared/time.js';
import { groupHeadline, identityKey } from './identity.js';

const HOUR_MS = 3600 * 1000;
const DAY_MS = 24 * HOUR_MS;

/**
 * Hover dwell before a mark's preview is fetched.
 *
 * Marks are a few pixels wide and sit shoulder to shoulder, so sweeping the pointer
 * across a busy lane crosses a dozen of them. Without a dwell that sweep is a dozen
 * queries for previews nobody looked at.
 */
const PREVIEW_DELAY_MS = 180;

/**
 * The preview holds no controls, and must not.
 *
 * The first cut put a "show these in the grid" button in it, which requires a hover
 * bridge: the pointer has to cross the gap between the mark and the popup without the
 * popup deciding it has been left. That is fragile for a real pointer (any scroll or
 * re-render mid-transit closes it under the cursor) and simply absent on touch, which
 * has no hover at all. So the popup is `pointer-events: none` in CSS, it closes the
 * moment the mark is left, and the action it used to hold is a shift-click on the mark
 * itself — advertised in the mark's own tooltip, which is where someone looks.
 */

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
 * The hover preview shared by every mark in the timeline.
 *
 * One controller rather than one per mark: only one preview is ever on screen, and a
 * lane holds hundreds of marks. Per-mark state would mean hundreds of timers and
 * detached nodes to reason about, and two overlapping popups the first time the
 * pointer moved faster than the grace period.
 *
 * Deliberately *not* `openAnchoredPopup`: that helper acquires a modal layer and
 * registers an ESC handler, which is the right contract for a click-opened flyout the
 * user must dismiss and the wrong one for something that follows the pointer. A hover
 * preview that ate the ESC key would break the detail modal opened from underneath it.
 *
 * @param {Object} opts
 * @param {(group: Object, window: {from: Date, to: Date}) => Promise<{event: ?Object, total: number}>} [opts.onPreviewBucket]
 * @returns {{attach: (mark: HTMLElement, ctx: Object) => void, destroy: () => void}}
 */
/**
 * The preview controller belonging to the timeline currently on screen.
 *
 * Module-scoped because the popup is parented to document.body and so outlives the
 * subtree it was opened from: a re-render with the pointer still over a mark discards
 * the lane and leaves the popup floating over a timeline that no longer contains it.
 * The next `buildTimeline` closes it.
 */
let _activePreview = null;

function createMarkPreview({ onPreviewBucket }) {
  /** Keyed by lane+bin, so re-hovering a mark is free and a sweep back is not a refetch. */
  const cache = new Map();
  let popup = null;
  let openTimer = null;
  let closeTimer = null;
  let token = 0;

  function close() {
    clearTimeout(openTimer);
    clearTimeout(closeTimer);
    openTimer = closeTimer = null;
    // Invalidate any fetch still in flight: without this, a slow response for a mark
    // the pointer already left re-opens a popup over whatever is now under the cursor.
    token++;
    popup?.remove();
    popup = null;
  }

  /** Anchored to the mark, flipped below when there is no room above. */
  function place(node, mark) {
    const rect = mark.getBoundingClientRect();
    const box = node.getBoundingClientRect();
    const left = Math.min(
      Math.max(4, rect.left + rect.width / 2 - box.width / 2),
      window.innerWidth - box.width - 4,
    );
    let top = rect.top - box.height - 8;
    if (top < 4) top = rect.bottom + 8;
    node.style.left = `${left}px`;
    node.style.top = `${top}px`;
  }

  function render(mark, ctx, data) {
    close();
    const node = el('div', 'event-timeline__preview');
    node.dataset.testid = 'event-timeline-preview';

    const { count, when, primary } = ctx;
    if (data.event) {
      const img = el('img', 'event-timeline__preview-image');
      img.loading = 'lazy';
      img.alt = '';
      img.src = watchEventSnapshotUrl(data.event.uuid);
      // A snapshot can be missing (an event whose frame was never written, or one
      // whose file has since been pruned): drop the image and keep the text rather
      // than showing a broken-image glyph, then re-place, since the popup just shrank.
      img.addEventListener('error', () => {
        img.remove();
        if (popup === node) place(node, mark);
      });
      img.addEventListener('load', () => {
        if (popup === node) place(node, mark);
      });
      node.appendChild(img);
    } else {
      node.appendChild(el('p', 'event-timeline__preview-empty', 'No snapshot for this event.'));
    }

    const meta = el('div', 'event-timeline__preview-meta');
    meta.appendChild(el('strong', null, primary));
    meta.appendChild(el('span', null, `${count} event${count === 1 ? '' : 's'}`));
    meta.appendChild(el('span', null, when.toLocaleString([], {
      day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit',
    })));
    node.appendChild(meta);

    node.style.visibility = 'hidden';
    document.body.appendChild(node);
    popup = node;
    place(node, mark);
    node.style.visibility = '';
  }

  function attach(mark, ctx) {
    mark.addEventListener('mouseenter', () => {
      clearTimeout(closeTimer);
      clearTimeout(openTimer);
      if (!onPreviewBucket) return;
      openTimer = setTimeout(async () => {
        const mine = ++token;
        const key = `${ctx.group.key}|${ctx.when.getTime()}`;
        let pending = cache.get(key);
        if (!pending) {
          pending = onPreviewBucket(ctx.group, { from: ctx.when, to: ctx.until });
          cache.set(key, pending);
        }
        let data;
        try {
          data = await pending;
        } catch {
          // A preview is decoration; a failed fetch closes the popup and says nothing.
          // Dropped from the cache so the next hover is a fresh attempt rather than a
          // permanently poisoned bin.
          cache.delete(key);
          return;
        }
        if (mine !== token) return;
        render(mark, ctx, data);
      }, PREVIEW_DELAY_MS);
    });
    mark.addEventListener('mouseleave', close);
  }

  return { attach, destroy: close };
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
 * @param {Map<string, string>} [opts.sourceNames] - camera uuid → name
 * @param {(group: Object) => void} [opts.onSelectGroup] - lane label clicked
 * @param {(group: Object, window: {from: Date, to: Date}) => void} [opts.onSelectBucket] -
 *   narrow the whole page to this bin; shift-click on a mark.
 * @param {(group: Object, window: {from: Date, to: Date}) => void} [opts.onOpenBucket] - mark clicked
 * @param {(group: Object, window: {from: Date, to: Date}) => Promise<{event: ?Object, total: number}>} [opts.onPreviewBucket]
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
  sourceNames = null,
  onSelectGroup,
  onSelectBucket,
  onOpenBucket,
  onPreviewBucket,
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
  const ticks = axisTicks(range);
  const axis = el('div', 'event-timeline__axis');
  axis.dataset.testid = 'event-timeline-axis';
  axis.appendChild(el('div', 'event-timeline__axis-gutter'));
  const axisTrack = el('div', 'event-timeline__axis-track');
  for (const tick of ticks) {
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

  // ── Gridlines ───────────────────────────────────────────────────────────────
  //
  // One rule per axis tick, carried down through every lane, so a mark can be read back
  // up to the label that dates it. Without them the axis is a row of numbers at the top
  // of an undifferentiated field and "where does Tuesday end?" is answered by holding a
  // finger against the screen.
  //
  // A single overlay rather than lines per lane: at 20 lanes and a fortnight of ticks
  // the per-lane version is hundreds of nodes redrawn on every filter change, all to
  // paint what is geometrically one set of vertical lines. It shares the lane's grid
  // template (see the CSS) so it cannot drift out of alignment with the marks.
  const gridOverlay = el('div', 'event-timeline__grid');
  gridOverlay.setAttribute('aria-hidden', 'true');
  gridOverlay.appendChild(el('div', 'event-timeline__axis-gutter'));
  const gridTrack = el('div', 'event-timeline__grid-track');
  for (const tick of ticks) {
    const line = el('span', `event-timeline__gridline${tick.major ? ' event-timeline__gridline--major' : ''}`);
    line.style.left = `${pct(tick.at)}%`;
    gridTrack.appendChild(line);
  }
  gridOverlay.appendChild(gridTrack);
  lanes.appendChild(gridOverlay);

  // Close the previous timeline's preview before taking over as the active one.
  _activePreview?.destroy();
  const preview = createMarkPreview({ onPreviewBucket });
  _activePreview = preview;

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
    // Splitting by camera subdivides every identity, so without naming the camera the
    // toggle turns one lane into several that caption themselves identically — the
    // duplicate rows read as a bug in the grouping rather than as the split that was
    // asked for. Falls back to the uuid: an unnamed camera is still a distinct one, and
    // "Camera 4f2c…" at least tells you which lanes are the same one.
    const cameraName = splitByCamera && group.sourceUuid
      ? (sourceNames?.get(group.sourceUuid) || `Camera ${group.sourceUuid.slice(0, 8)}…`)
      : '';
    if (cameraName) {
      text.appendChild(el('span', 'event-timeline__label-camera', cameraName));
    }
    label.appendChild(text);
    label.appendChild(el('span', 'event-timeline__label-count', String(group.count)));
    label.title = `${primary}${secondary ? ` — ${secondary}` : ''}${cameraName ? ` — ${cameraName}` : ''}`
      + ` — ${group.count} events. `
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
      const until = new Date(at + bucketMs);
      mark.title = `${row.count} event${row.count === 1 ? '' : 's'} — ${
        when.toLocaleString([], { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })}`
        + '\nClick to open them · Shift-click to narrow the page to this hour';
      // Open the events themselves rather than rewriting the page's filters. Drilling
      // down replaced the timeline with a filtered grid, which is a one-way door: the
      // way back is to notice the window chip, clear it, re-pick the view and re-find
      // your lane. A modal leaves the timeline underneath, exactly as it was.
      //
      // Shift keeps the drill-down, which is still the right tool for acting on a whole
      // hour at once (select, review, delete) — it is just no longer what an unmodified
      // click does. The tooltip above is the only place that says so, which is why it
      // says it on the mark rather than in a popup you have to already be hovering.
      mark.addEventListener('click', e => {
        e.stopPropagation();
        preview.destroy();
        const win = { from: when, to: until };
        if (e.shiftKey) onSelectBucket?.(group, win);
        else onOpenBucket?.(group, win);
      });
      preview.attach(mark, { group, when, until, count: row.count, primary });
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
