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
    labelEl.style.cssText = 'color:#888; margin-bottom:0.15em; font-size:0.9em;';
    div.appendChild(labelEl);
  }
  return div;
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
    msgDiv.style.cssText = 'color:#555; margin-bottom:0.3em;';
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
      logLabel.style.cssText = 'color:#888; margin-bottom:0.15em;';
      wrap.appendChild(logLabel);

      const pre = document.createElement('pre');
      pre.style.cssText = 'margin:0; padding:0.4em; background:#f5f5f5; border-radius:3px; max-height:10em; overflow-y:auto; white-space:pre-wrap; word-break:break-all; font-size:0.9em;';
      pre.textContent = lines.join('\n');
      wrap.appendChild(pre);
    } else {
      const noLog = document.createElement('div');
      noLog.textContent = 'No log output.';
      noLog.style.cssText = 'color:#bbb;';
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
      cell.style.cssText = 'background:#f5f5f5; border-radius:3px; padding:0.3em 0.5em;';
      const valEl = document.createElement('div');
      valEl.textContent = values[key] ?? '—';
      valEl.style.cssText = 'font-weight:bold; font-size:1.05em;';
      const labelEl = document.createElement('div');
      labelEl.textContent = fieldLabel;
      labelEl.style.cssText = 'color:#888; font-size:0.75em;';
      cell.append(valEl, labelEl);
      grid.appendChild(cell);
    }
    wrap.appendChild(grid);
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
      empty.style.cssText = 'color:#bbb;';
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
        bin.marked ? 'font-weight:bold;' : 'color:#888;'
      }`;

      const track = document.createElement('div');
      track.className = 'status-histogram__track';
      track.style.cssText = 'flex:1; min-width:0; height:0.9em; background:#f0f0f0; border-radius:2px;';
      const fill = document.createElement('div');
      fill.className = 'status-histogram__fill';
      fill.style.cssText = `width:${(100 * count) / max}%; height:100%; border-radius:2px; background:${
        bin.marked ? '#e8833a' : '#7aa6d6'
      };`;
      track.appendChild(fill);

      const value = document.createElement('span');
      value.textContent = String(count);
      value.style.cssText = 'min-width:2.5em; color:#888; font-variant-numeric:tabular-nums;';

      row.append(tick, track, value);
      chart.appendChild(row);
    }
    wrap.appendChild(chart);

    const caption = document.createElement('div');
    const marked = bins.find(b => b.marked);
    caption.textContent = marked
      ? `${data.total} sample(s) · ◀ marks the current threshold (${marked.label})`
      : `${data.total} sample(s)`;
    caption.style.cssText = 'color:#888; font-size:0.75em;';
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
      th.style.cssText = 'text-align:left; color:#888; font-weight:normal; border-bottom:1px solid #e0e0e0; padding:2px 6px;';
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
        td.style.cssText = 'padding:2px 6px; border-bottom:1px solid #f0f0f0;';
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);

    if (rows.length === 0) {
      const empty = document.createElement('div');
      empty.textContent = 'No data.';
      empty.style.cssText = 'color:#bbb;';
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
      placeholder.style.cssText = 'color:#bbb;';
      wrap.appendChild(placeholder);
      return wrap;
    }

    const height = config.height || '16em';
    const canvasContainer = document.createElement('div');
    canvasContainer.style.cssText = `width:100%; height:${height}; position:relative; background:#111; border-radius:3px; overflow:hidden;`;
    wrap.appendChild(canvasContainer);

    let instance = null;
    // Deferred to mount(): createCanvasInstance measures the container's
    // clientWidth/clientHeight to fit-to-view the image, which read 0 (and
    // silently collapsed every bbox to (0,0)) if done here — `wrap` isn't
    // attached to the document yet at factory-call time, only once the
    // caller appends the returned element. mount() runs after that append.
    const mount = async () => {
      try {
        instance = await createCanvasInstance(canvasContainer, { tool: 'select' });
        await instance.loadImage(data.path);
        if (Array.isArray(data.bboxes) && data.bboxes.length) {
          seedRuntimeBboxes(instance.runtime, data.bboxes);
        }
      } catch (err) {
        instance?.destroy?.();
        instance = null;
        const errEl = document.createElement('div');
        errEl.textContent = 'Failed to load image.';
        errEl.style.cssText = 'color:#f44336; padding:0.5em;';
        canvasContainer.replaceWith(errEl);
      }
    };

    return { el: wrap, mount, destroy: () => instance?.destroy() };
  },
};
