// statusWidgets/Factories.js
//
// Registry of status-widget renderers, keyed by the `type` field of a
// Task.status_schema() entry (see LIVE_COLLECTION_PLAN.md, Phase 3).
// Structurally mirrors ui/utils/fields/Factories.js: a flat map from type
// name -> async factory function.
//
// Factory signature:
//   async (config, data, ctx) => HTMLElement | { el: HTMLElement, destroy?: () => void } | null
//
//   config — the widget's schema entry: { id, type, source, ...widget-specific
//            keys like `label`, `lines`, `fields`, `columns` }.
//   data   — the resolved value for this widget. For `source: "summary"`
//            widgets this is a field read off the already-polled task
//            payload; for `source: "status"` widgets it's
//            statusData[widget.id] from the per-instance status-data
//            endpoint (undefined before the first fetch lands / while the
//            card is collapsed).
//   ctx    — shared context: { taskUuid, taskStatus, sendCommand }.
//
// A factory may return `{ el, destroy }` when it owns a resource that needs
// explicit teardown (annotated_image's canvas instance); a bare HTMLElement
// is also fine when there's nothing to tear down. Returning null/undefined
// means "nothing to show" and the caller renders no slot content.
//
// Unknown widget types and factory exceptions are handled by the caller
// (ActiveTaskField.renderView), never here — this module only defines the
// known types.

import { createCanvasInstance } from '/app-static/js/app/image-tagging/canvas/instance.js';
import { seedRuntimeBboxes } from '/app-static/js/app/image-tagging/canvas/image.js';

function labelWrap(label) {
  const div = document.createElement('div');
  if (label) {
    const labelEl = document.createElement('div');
    labelEl.textContent = label;
    labelEl.style.cssText = 'color:var(--text-muted); margin-bottom:0.15em; font-size:0.9em;';
    div.appendChild(labelEl);
  }
  return div;
}

// Categorical series colours, assigned in this fixed order and never cycled —
// a chart that wants a 5th series wants to be two charts.
//
// Defined per theme in tokens.css (--chart-series-*), not as the --color-*
// semantic tokens: see the comment there for why a data mark can't wear a UI
// accent colour.
const SERIES_COLORS = [
  'var(--chart-series-1)',
  'var(--chart-series-2)',
  'var(--chart-series-3)',
  'var(--chart-series-4)',
];

const SVG_NS = 'http://www.w3.org/2000/svg';
const svgEl = (name, attrs = {}) => {
  const el = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
};

/** Format a metric value for display. `percent` renders a 0–1 fraction as %. */
function formatValue(value, valueFormat) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (valueFormat === 'percent') return `${(value * 100).toFixed(1)}%`;
  if (Math.abs(value) >= 1000 || (value !== 0 && Math.abs(value) < 0.001)) return value.toExponential(2);
  return String(Math.round(value * 10000) / 10000);
}

/** Axis-tick formatting: shorter than the tooltip's, to fit a 4-char gutter. */
function formatTick(value, valueFormat) {
  if (valueFormat === 'percent') return `${Math.round(value * 100)}%`;
  if (value === 0) return '0';
  const abs = Math.abs(value);
  if (abs >= 100 || abs < 0.01) return value.toExponential(0);
  return String(Math.round(value * 100) / 100);
}

/** "Nice" axis bounds: pad the data range out to a round step. */
function niceBounds(min, max) {
  if (!(max > min)) {
    // A flat series still needs a band to draw in.
    const pad = Math.abs(max) > 0 ? Math.abs(max) * 0.1 : 1;
    return [min - pad, max + pad];
  }
  const step = 10 ** Math.floor(Math.log10(max - min)) / 2;
  return [Math.floor(min / step) * step, Math.ceil(max / step) * step];
}

export const statusWidgets = {
  // data: number (0-100), null (indeterminate — no known denominator), or
  // undefined (nothing to show, e.g. the task isn't running).
  async progress_bar(config, data, ctx) {
    if (ctx?.taskStatus !== 'running') return null;

    const wrap = labelWrap(config.label);
    const row = document.createElement('div');
    row.style.cssText = 'display:flex; align-items:center; gap:0.5em; margin-bottom:0.3em;';
    const bar = document.createElement('progress');
    bar.style.cssText = 'flex:1; height:8px;';
    bar.setAttribute('role', 'progressbar');
    const pct = document.createElement('span');
    pct.style.cssText = 'min-width:3em; text-align:right;';

    const indeterminate = data === null || data === undefined;
    if (indeterminate) {
      // No `value` attribute at all → native indeterminate <progress>; CSS
      // (.progress-indeterminate) paints the striped/animated fill.
      bar.classList.add('progress-indeterminate');
      bar.removeAttribute('value');
      bar.setAttribute('aria-valuetext', 'Working…');
      pct.textContent = '';
    } else {
      const value = Number(data) || 0;
      bar.value = value;
      bar.max = 100;
      bar.setAttribute('aria-valuemin', '0');
      bar.setAttribute('aria-valuemax', '100');
      bar.setAttribute('aria-valuenow', String(Math.round(value)));
      pct.textContent = `${Math.round(value)}%`;
    }
    row.appendChild(bar);
    row.appendChild(pct);
    wrap.appendChild(row);
    return wrap;
  },

  async text(config, data) {
    if (!data) return null;
    const wrap = labelWrap(config.label);
    const msgDiv = document.createElement('div');
    msgDiv.textContent = String(data);
    msgDiv.style.cssText = 'color:var(--text); margin-bottom:0.3em;';
    wrap.appendChild(msgDiv);
    return wrap;
  },

  async log_tail(config, data, ctx) {
    const allLines = Array.isArray(data) ? data : (data ? [String(data)] : []);
    const lines = config.lines ? allLines.slice(-config.lines) : allLines;
    const wrap = document.createElement('div');

    if (lines.length > 0) {
      const logLabel = document.createElement('div');
      // Preserve the pre-schema-driven distinction: an errored task's log
      // gets an "Error log" heading rather than a plain "Logs" one.
      const defaultLabel = ctx?.taskStatus === 'error'
        ? `Error log (${lines.length} lines):`
        : `Logs (${lines.length} lines):`;
      logLabel.textContent = config.label || defaultLabel;
      logLabel.style.cssText = 'color:var(--text-muted); margin-bottom:0.15em;';
      wrap.appendChild(logLabel);

      const pre = document.createElement('pre');
      pre.style.cssText = 'margin:0; padding:0.4em; background:var(--surface-sunken); color:var(--text); border-radius:3px; max-height:10em; overflow-y:auto; white-space:pre-wrap; word-break:break-all; font-size:0.9em;';
      pre.textContent = lines.join('\n');
      wrap.appendChild(pre);
    } else {
      const noLog = document.createElement('div');
      noLog.textContent = 'No log output.';
      noLog.style.cssText = 'color:var(--text-faint);';
      wrap.appendChild(noLog);
    }
    return wrap;
  },

  // config.fields: either { key: label, ... } or [{ key, label }, ...] / [key, ...]
  // data: { key: value, ... }
  async stat_grid(config, data) {
    const wrap = labelWrap(config.label);
    const grid = document.createElement('div');
    grid.style.cssText = 'display:grid; grid-template-columns:repeat(auto-fit, minmax(7em, 1fr)); gap:0.4em; margin-bottom:0.3em;';

    const fields = config.fields ?? {};
    const entries = Array.isArray(fields)
      ? fields.map(f => (typeof f === 'string' ? [f, f] : [f.key ?? f.id, f.label ?? f.key ?? f.id]))
      : Object.entries(fields);

    const values = data || {};
    for (const [key, fieldLabel] of entries) {
      const cell = document.createElement('div');
      cell.style.cssText = 'background:var(--surface-raised); color:var(--text); border-radius:3px; padding:0.3em 0.5em;';
      const valEl = document.createElement('div');
      valEl.textContent = values[key] ?? '—';
      valEl.style.cssText = 'font-weight:bold; font-size:1.05em;';
      const labelEl = document.createElement('div');
      labelEl.textContent = fieldLabel;
      labelEl.style.cssText = 'color:var(--text-muted); font-size:0.75em;';
      cell.append(valEl, labelEl);
      grid.appendChild(cell);
    }
    wrap.appendChild(grid);
    return wrap;
  },

  // config: { label, x, xLabel, series: [{ key, label, colorIndex? }],
  //           yMin?, yMax?, valueFormat?: 'percent', height? }
  // data:   { points: [ { <x>: number, <seriesKey>: number, ... }, ... ] }
  //
  // One y-scale, always. A metric on a second scale gets its own line_chart
  // widget — a shared plot with two axes lets the reader see crossings and
  // convergences that are artefacts of where each axis was zeroed.
  //
  // A point missing a series key is a gap in that series' line, not a zero:
  // trainers legitimately don't report every metric (the detection trainers
  // send no val_loss), and drawing the absence as 0 invents a measurement.
  async line_chart(config, data) {
    const wrap = labelWrap(config.label);
    const points = Array.isArray(data?.points) ? data.points : [];
    const xKey = config.x || 'x';
    const valueFormat = config.valueFormat;

    const seriesDefs = (config.series ?? []).map((s, i) => ({
      key: s.key ?? s.id,
      label: s.label ?? s.key ?? s.id,
      color: SERIES_COLORS[(s.colorIndex ?? i) % SERIES_COLORS.length],
    }));

    // A series with no numeric samples anywhere is dropped outright rather than
    // drawn as an empty line with a legend entry promising data that never comes.
    const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
    const series = seriesDefs
      .map(def => ({
        ...def,
        values: points.map(p => num(p?.[def.key])),
      }))
      .filter(s => s.values.some(v => v !== null));

    if (points.length === 0 || series.length === 0) {
      const empty = document.createElement('div');
      empty.textContent = 'No epoch data yet.';
      empty.style.cssText = 'color:var(--text-faint);';
      wrap.appendChild(empty);
      return wrap;
    }

    const xs = points.map(p => num(p?.[xKey]) ?? 0);
    const allY = series.flatMap(s => s.values.filter(v => v !== null));
    let yLo = config.yMin ?? Math.min(...allY);
    let yHi = config.yMax ?? Math.max(...allY);
    if (config.yMin === undefined || config.yMax === undefined) {
      const [nLo, nHi] = niceBounds(Math.min(...allY), Math.max(...allY));
      if (config.yMin === undefined) yLo = nLo;
      if (config.yMax === undefined) yHi = nHi;
    }
    if (yHi === yLo) yHi = yLo + 1;
    const xLo = Math.min(...xs);
    const xHi = Math.max(...xs);

    // Fixed viewBox scaled to the card's width. Strokes are pinned with
    // vector-effect so line weight stays 2px however wide the card is.
    const W = 400, H = 170;
    const PAD = { top: 10, right: 12, bottom: 26, left: 42 };
    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;
    const sx = x => PAD.left + (xHi === xLo ? plotW / 2 : ((x - xLo) / (xHi - xLo)) * plotW);
    const sy = y => PAD.top + plotH - ((y - yLo) / (yHi - yLo)) * plotH;

    const frame = document.createElement('div');
    frame.style.cssText = 'position:relative; background:var(--surface-sunken); border-radius:3px; padding:0.2em;';
    const svg = svgEl('svg', {
      viewBox: `0 0 ${W} ${H}`,
      width: '100%',
      role: 'img',
      'aria-label': `${config.label || 'Chart'}: ${series.map(s => s.label).join(', ')} by ${config.xLabel || xKey}`,
    });
    // The viewBox gives the SVG its own aspect ratio, so width:100% is enough to
    // size it; config.height only overrides that when a card wants a taller plot.
    svg.style.cssText = `display:block; overflow:visible; touch-action:none;${config.height ? ` height:${config.height};` : ''}`;

    // ── Grid + y ticks (recessive: hairlines, muted ink) ──────────────────
    for (let i = 0; i <= 4; i++) {
      const value = yLo + ((yHi - yLo) * i) / 4;
      const y = sy(value);
      svg.appendChild(svgEl('line', {
        x1: PAD.left, x2: W - PAD.right, y1: y, y2: y,
        stroke: 'var(--border)', 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke',
      }));
      const tick = svgEl('text', {
        x: PAD.left - 5, y: y + 3, 'text-anchor': 'end', fill: 'var(--text-muted)', 'font-size': 9,
      });
      tick.textContent = formatTick(value, valueFormat);
      svg.appendChild(tick);
    }

    // ── X ticks: ends plus up to 3 interior, snapped to real samples ──────
    const tickIdxs = [...new Set(
      Array.from({ length: Math.min(5, points.length) }, (_, i) =>
        Math.round((i * (points.length - 1)) / Math.max(1, Math.min(5, points.length) - 1)))
    )];
    for (const idx of tickIdxs) {
      const label = svgEl('text', {
        x: sx(xs[idx]), y: H - PAD.bottom + 12, 'text-anchor': 'middle',
        fill: 'var(--text-muted)', 'font-size': 9,
      });
      label.textContent = String(xs[idx]);
      svg.appendChild(label);
    }
    if (config.xLabel) {
      const axisLabel = svgEl('text', {
        x: PAD.left + plotW / 2, y: H - 2, 'text-anchor': 'middle',
        fill: 'var(--text-faint)', 'font-size': 9,
      });
      axisLabel.textContent = config.xLabel;
      svg.appendChild(axisLabel);
    }

    // ── Series lines. Nulls break the path into segments (gaps, not zeros) ─
    const showDots = points.length <= 25;
    for (const s of series) {
      let run = [];
      const flush = () => {
        if (run.length === 0) return;
        if (run.length === 1) {
          // A lone sample between two gaps has no line to be seen on — give it
          // a dot regardless of the dot threshold, or the series vanishes.
          svg.appendChild(svgEl('circle', {
            cx: run[0][0], cy: run[0][1], r: 3, fill: s.color,
            stroke: 'var(--surface-sunken)', 'stroke-width': 1.5,
          }));
        } else {
          svg.appendChild(svgEl('polyline', {
            points: run.map(([x, y]) => `${x},${y}`).join(' '),
            fill: 'none', stroke: s.color, 'stroke-width': 2,
            'stroke-linejoin': 'round', 'stroke-linecap': 'round',
            'vector-effect': 'non-scaling-stroke',
          }));
        }
        run = [];
      };
      s.values.forEach((v, i) => (v === null ? flush() : run.push([sx(xs[i]), sy(v)])));
      flush();

      if (showDots) {
        s.values.forEach((v, i) => {
          if (v === null) return;
          svg.appendChild(svgEl('circle', {
            cx: sx(xs[i]), cy: sy(v), r: 3, fill: s.color,
            // A surface-coloured ring keeps overlapping marks legible where
            // two series cross.
            stroke: 'var(--surface-sunken)', 'stroke-width': 1.5,
          }));
        });
      }
    }

    // ── Hover layer: crosshair + nearest-sample tooltip ───────────────────
    const crosshair = svgEl('line', {
      y1: PAD.top, y2: PAD.top + plotH, stroke: 'var(--border-strong)',
      'stroke-width': 1, 'vector-effect': 'non-scaling-stroke', visibility: 'hidden',
    });
    svg.appendChild(crosshair);
    const markers = series.map(s => {
      const dot = svgEl('circle', {
        r: 4, fill: s.color, stroke: 'var(--surface-sunken)', 'stroke-width': 2, visibility: 'hidden',
      });
      svg.appendChild(dot);
      return dot;
    });

    const tooltip = document.createElement('div');
    tooltip.style.cssText = 'position:absolute; pointer-events:none; display:none; z-index:2; background:var(--surface-elevated); color:var(--text); border:1px solid var(--border-strong); border-radius:3px; padding:0.25em 0.45em; font-size:0.75em; white-space:nowrap; box-shadow:var(--shadow);';

    const hideHover = () => {
      crosshair.setAttribute('visibility', 'hidden');
      markers.forEach(m => m.setAttribute('visibility', 'hidden'));
      tooltip.style.display = 'none';
    };

    const showHover = (clientX) => {
      const rect = svg.getBoundingClientRect();
      if (!rect.width) return;
      // Pointer px -> viewBox units, then -> nearest sample index.
      const vbX = ((clientX - rect.left) / rect.width) * W;
      let best = 0;
      for (let i = 1; i < xs.length; i++) {
        if (Math.abs(sx(xs[i]) - vbX) < Math.abs(sx(xs[best]) - vbX)) best = i;
      }
      const px = sx(xs[best]);
      crosshair.setAttribute('x1', px);
      crosshair.setAttribute('x2', px);
      crosshair.setAttribute('visibility', 'visible');

      const rows = [`${config.xLabel || xKey} ${xs[best]}`];
      series.forEach((s, i) => {
        const v = s.values[best];
        if (v === null) {
          markers[i].setAttribute('visibility', 'hidden');
          return;
        }
        markers[i].setAttribute('cx', px);
        markers[i].setAttribute('cy', sy(v));
        markers[i].setAttribute('visibility', 'visible');
        rows.push(`${s.label}: ${formatValue(v, valueFormat)}`);
      });
      tooltip.textContent = rows.join('  ·  ');
      tooltip.style.display = 'block';
      // Flip the tooltip to the left of the crosshair past the halfway mark so
      // it never runs off the right edge of a narrow card.
      const leftPct = (px / W) * 100;
      tooltip.style.left = leftPct > 55 ? 'auto' : `calc(${leftPct}% + 8px)`;
      tooltip.style.right = leftPct > 55 ? `calc(${100 - leftPct}% + 8px)` : 'auto';
      tooltip.style.top = '2px';
    };

    svg.addEventListener('pointermove', e => showHover(e.clientX));
    svg.addEventListener('pointerleave', hideHover);
    frame.append(svg, tooltip);
    wrap.appendChild(frame);

    // ── Legend (>=2 series) + direct end-labels ───────────────────────────
    if (series.length >= 2) {
      const legend = document.createElement('div');
      legend.style.cssText = 'display:flex; flex-wrap:wrap; gap:0.75em; margin:0.25em 0 0.3em; font-size:0.75em; color:var(--text-muted);';
      for (const s of series) {
        const item = document.createElement('span');
        item.style.cssText = 'display:inline-flex; align-items:center; gap:0.3em;';
        const swatch = document.createElement('span');
        swatch.style.cssText = `width:0.7em; height:0.7em; border-radius:2px; background:${s.color}; flex:none;`;
        const text = document.createElement('span');
        // Identity is never colour alone: the swatch is always paired with the
        // series name, and the value below it repeats in the data table.
        text.textContent = `${s.label} — ${formatValue(s.values.filter(v => v !== null).at(-1), valueFormat)}`;
        item.append(swatch, text);
        legend.appendChild(item);
      }
      wrap.appendChild(legend);
    } else {
      const caption = document.createElement('div');
      caption.style.cssText = 'margin:0.25em 0 0.3em; font-size:0.75em; color:var(--text-muted);';
      caption.textContent = `${series[0].label} — latest ${formatValue(series[0].values.filter(v => v !== null).at(-1), valueFormat)}`;
      wrap.appendChild(caption);
    }

    // ── Table view: the same numbers, reachable without reading the chart ──
    const details = document.createElement('details');
    details.style.cssText = 'font-size:0.8em;';
    const summary = document.createElement('summary');
    summary.textContent = `Data table (${points.length} points)`;
    summary.style.cssText = 'color:var(--text-muted); cursor:pointer;';
    details.appendChild(summary);
    const table = await statusWidgets.table(
      {
        columns: [
          { key: xKey, label: config.xLabel || xKey },
          ...series.map(s => ({ key: s.key, label: s.label })),
        ],
      },
      points.map(p => {
        const row = { [xKey]: p?.[xKey] };
        for (const s of series) row[s.key] = formatValue(num(p?.[s.key]), valueFormat);
        return row;
      }),
    );
    details.appendChild(table);
    wrap.appendChild(details);

    return wrap;
  },

  // data: { bins: [{ label, count, marked }], total }
  //
  // Horizontal bars rather than vertical: bin labels are numeric distances that
  // stay readable on one line, and the widget lives in a narrow task card where
  // vertical bars would be a few pixels wide each. The `marked` bin is where the
  // configured threshold currently cuts — everything at or left of it counts as
  // a duplicate — so it's called out in both colour and text, never colour alone.
  async histogram(config, data) {
    const wrap = labelWrap(config.label);
    const bins = Array.isArray(data?.bins) ? data.bins : [];
    if (bins.length === 0) {
      const empty = document.createElement('div');
      empty.textContent = 'No measurements yet.';
      empty.style.cssText = 'color:var(--text-faint);';
      wrap.appendChild(empty);
      return wrap;
    }

    const max = Math.max(...bins.map(b => Number(b.count) || 0), 1);
    const chart = document.createElement('div');
    chart.style.cssText = 'display:flex; flex-direction:column; gap:1px; margin-bottom:0.3em;';

    for (const bin of bins) {
      const count = Number(bin.count) || 0;
      const row = document.createElement('div');
      row.style.cssText = 'display:flex; align-items:center; gap:0.4em; font-size:0.85em;';

      const tick = document.createElement('span');
      tick.textContent = bin.marked ? `${bin.label} ◀` : String(bin.label);
      tick.style.cssText = `min-width:4.5em; text-align:right; font-variant-numeric:tabular-nums; ${
        bin.marked ? 'font-weight:bold; color:var(--text);' : 'color:var(--text-muted);'
      }`;

      const track = document.createElement('div');
      track.className = 'status-histogram__track';
      track.style.cssText = 'flex:1; min-width:0; height:0.9em; background:var(--surface-sunken); border-radius:2px;';
      const fill = document.createElement('div');
      fill.className = 'status-histogram__fill';
      fill.style.cssText = `width:${(100 * count) / max}%; height:100%; border-radius:2px; background:${
        bin.marked ? 'var(--color-warning)' : 'var(--color-info)'
      };`;
      track.appendChild(fill);

      const value = document.createElement('span');
      value.textContent = String(count);
      value.style.cssText = 'min-width:2.5em; color:var(--text-muted); font-variant-numeric:tabular-nums;';

      row.append(tick, track, value);
      chart.appendChild(row);
    }
    wrap.appendChild(chart);

    const caption = document.createElement('div');
    const marked = bins.find(b => b.marked);
    caption.textContent = marked
      ? `${data.total} sample(s) · ◀ marks the current threshold (${marked.label})`
      : `${data.total} sample(s)`;
    caption.style.cssText = 'color:var(--text-muted); font-size:0.75em;';
    wrap.appendChild(caption);
    return wrap;
  },

  // config.columns: [{ key, label }, ...] / ["key", ...]
  // data: [ { key: value, ... }, ... ]
  async table(config, data) {
    const wrap = labelWrap(config.label);
    const rows = Array.isArray(data) ? data : [];
    const cols = (config.columns ?? []).map(c =>
      (typeof c === 'string' ? { key: c, label: c } : { key: c.key ?? c.id, label: c.label ?? c.key ?? c.id })
    );

    const table = document.createElement('table');
    table.style.cssText = 'width:100%; border-collapse:collapse; font-size:0.9em; margin-bottom:0.3em;';

    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    for (const col of cols) {
      const th = document.createElement('th');
      th.textContent = col.label;
      th.style.cssText = 'text-align:left; color:var(--text-muted); font-weight:normal; border-bottom:1px solid var(--border-strong); padding:2px 6px;';
      headRow.appendChild(th);
    }
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    for (const row of rows) {
      const tr = document.createElement('tr');
      for (const col of cols) {
        const td = document.createElement('td');
        td.textContent = row?.[col.key] ?? '';
        td.style.cssText = 'padding:2px 6px; border-bottom:1px solid var(--border);';
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);

    if (rows.length === 0) {
      const empty = document.createElement('div');
      empty.textContent = 'No data.';
      empty.style.cssText = 'color:var(--text-faint);';
      wrap.appendChild(empty);
    }
    return wrap;
  },

  // Images are passed BY REFERENCE ONLY — `data` is `{ path, bboxes }` (or
  // absent before the first status-data fetch lands). Pixels are pulled
  // through the existing image-loading pipeline (which itself goes through
  // the thumbnail-bounded fetch path) via createCanvasInstance + loadImage.
  // Never inline image bytes into the schema/status-data payload or here.
  async annotated_image(config, data) {
    const wrap = labelWrap(config.label);
    if (!data?.path) {
      const placeholder = document.createElement('div');
      placeholder.textContent = 'No image yet.';
      placeholder.style.cssText = 'color:var(--text-faint);';
      wrap.appendChild(placeholder);
      return wrap;
    }

    const height = config.height || '16em';
    const canvasContainer = document.createElement('div');
    canvasContainer.style.cssText = `width:100%; height:${height}; position:relative; background:var(--surface-sunken); border-radius:3px; overflow:hidden;`;
    wrap.appendChild(canvasContainer);

    let instance = null;
    // Deferred to mount(): createCanvasInstance measures the container's
    // clientWidth/clientHeight to fit-to-view the image, which read 0 (and
    // silently collapsed every bbox to (0,0)) if done here — `wrap` isn't
    // attached to the document yet at factory-call time, only once the
    // caller appends the returned element. mount() runs after that append.
    const mount = async () => {
      try {
        // 'none': these previews carry no enhance control, and several can be
        // on screen at once — don't spend CPU on an enhancement nobody asked
        // this widget for.
        //
        // hotkeys: false is load-bearing, not a tidiness flag. This canvas is
        // rebuilt on every STATUS_DATA_POLL_MS tick whose data changed, with no
        // user action involved. With hotkeys attached, each rebuild pushed this
        // read-only thumbnail onto the top of the hotkey stack and took every
        // shortcut away from whatever canvas the user had open — the keys still
        // fired, they just drove an invisible 16em preview with no
        // FileNavigation, so '[' / ']' looked completely dead.
        instance = await createCanvasInstance(canvasContainer, {
          tool: 'select', enhance: 'none', lockEnhance: true, hotkeys: false,
        });
        await instance.loadImage(data.path);
        if (Array.isArray(data.bboxes) && data.bboxes.length) {
          seedRuntimeBboxes(instance.runtime, data.bboxes);
        }
      } catch (err) {
        instance?.destroy?.();
        instance = null;
        const errEl = document.createElement('div');
        errEl.textContent = 'Failed to load image.';
        errEl.style.cssText = 'color:var(--color-danger); padding:0.5em;';
        canvasContainer.replaceWith(errEl);
      }
    };

    return { el: wrap, mount, destroy: () => instance?.destroy() };
  },
};
