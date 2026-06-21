import { createModalShell } from './utils/modalShell.js';
import { apiFetch } from '/app-static/js/shared/api/apiUtils.js';

const STATUS_COLOR = {
  running:       'var(--pico-ins-color,#4b8)',
  queued:        '#ca0',
  awaiting_data: '#888',
  done:          'var(--pico-muted-color,#888)',
  error:         'var(--pico-del-color,#e55)',
  cancelled:     'var(--pico-muted-color,#888)',
  interrupted:   '#e80',
};

const STATUS_ICON = {
  running:       '▶',
  queued:        '⏳',
  awaiting_data: '⌛',
  done:          '✓',
  error:         '✕',
  cancelled:     '–',
  interrupted:   '⚡',
};

function _fmtDate(iso) {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
  } catch (_) {
    return iso;
  }
}

const CANCELLABLE_STATUSES = new Set(['running', 'queued', 'awaiting_data', 'interrupted']);

function _renderJob(job, onCancel) {
  const row = document.createElement('div');
  row.style.cssText = `
    display: grid;
    grid-template-columns: 1.4em 1fr auto;
    gap: 6px;
    align-items: start;
    padding: 8px 0;
    border-bottom: 1px solid var(--pico-muted-border-color, #333);
  `;

  const color = STATUS_COLOR[job.status] ?? '#888';
  const icon  = STATUS_ICON[job.status]  ?? '?';

  const iconEl = document.createElement('span');
  iconEl.style.cssText = `color: ${color}; font-size: 1em; flex-shrink: 0; padding-top: 1px;`;
  iconEl.textContent = icon;
  iconEl.title = job.status;

  const body = document.createElement('div');
  body.style.cssText = 'min-width: 0;';

  const nameRow = document.createElement('div');
  nameRow.style.cssText = 'font-size: 0.85em; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;';
  nameRow.textContent = job.model_name || '(unnamed)';
  if (job.trainer_type) nameRow.title = `trainer: ${job.trainer_type}`;

  const metaRow = document.createElement('div');
  metaRow.style.cssText = 'font-size: 0.75em; color: var(--pico-muted-color, #888); margin-top: 2px;';
  const metaParts = [`${job.trainer_type || 'unknown trainer'}`];
  if (job.created_at) metaParts.push(_fmtDate(job.created_at));
  metaRow.textContent = metaParts.join('  ·  ');

  body.append(nameRow, metaRow);

  const rightCol = document.createElement('div');
  rightCol.style.cssText = 'display: flex; align-items: center; gap: 6px; padding-top: 2px;';

  const statusBadge = document.createElement('span');
  statusBadge.style.cssText = `font-size: 0.72em; color: ${color}; font-weight: 600; white-space: nowrap;`;
  const statusText = job.status === 'queued'
    ? `#${job.queue_position} queued`
    : job.status.replace(/_/g, ' ');
  statusBadge.textContent = statusText;
  rightCol.appendChild(statusBadge);

  if (CANCELLABLE_STATUSES.has(job.status) && onCancel) {
    const cancelBtn = document.createElement('button');
    cancelBtn.textContent = '✕';
    cancelBtn.className = 'secondary';
    cancelBtn.style.cssText = 'font-size: 0.68em; padding: 0.1em 0.35em; line-height: 1.4; min-width: 0;';
    cancelBtn.title = 'Cancel job';
    cancelBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      cancelBtn.disabled = true;
      cancelBtn.textContent = '…';
      await onCancel(job.job_id);
    });
    rightCol.appendChild(cancelBtn);
  }

  row.append(iconEl, body, rightCol);
  return row;
}

export function showTrainingServerModal() {
  const { dialog, open, close } = createModalShell();
  dialog.style.cssText = 'min-width: min(480px, 95vw); max-width: 560px;';

  // ── Header ─────────────────────────────────────────────────────────────────
  const header = document.createElement('div');
  header.style.cssText = 'display: flex; align-items: center; gap: 8px; margin-bottom: 12px;';

  const title = document.createElement('strong');
  title.textContent = '🏋  Training Server';
  title.style.fontSize = '1.05em';

  const refreshBtn = document.createElement('button');
  refreshBtn.textContent = '↻ Refresh';
  refreshBtn.className = 'secondary';
  refreshBtn.style.cssText = 'font-size: 0.78em; padding: 0.2em 0.6em; margin-left: auto;';

  const closeBtn = document.createElement('button');
  closeBtn.textContent = '✕';
  closeBtn.className = 'secondary';
  closeBtn.style.cssText = 'font-size: 0.78em; padding: 0.2em 0.5em;';
  closeBtn.addEventListener('click', close);

  header.append(title, refreshBtn, closeBtn);

  // ── Body ───────────────────────────────────────────────────────────────────
  const body = document.createElement('div');
  body.style.cssText = 'max-height: 60vh; overflow-y: auto;';

  const statusLine = document.createElement('div');
  statusLine.style.cssText = 'font-size: 0.78em; color: var(--pico-muted-color,#888); margin-top: 10px; text-align: right;';

  dialog.append(header, body, statusLine);

  async function refresh() {
    refreshBtn.disabled = true;
    refreshBtn.textContent = '↻ …';
    body.innerHTML = '';

    try {
      const res = await apiFetch('/api/training/jobs');
      const data = await res.json();
      const jobs = data.jobs ?? [];

      if (!jobs.length) {
        const empty = document.createElement('div');
        empty.style.cssText = 'font-size: 0.85em; color: var(--pico-muted-color,#888); padding: 12px 0;';
        empty.textContent = 'No training jobs.';
        body.appendChild(empty);
      } else {
        // Show active jobs first (running, queued, awaiting_data), then terminal ones
        const order = { running: 0, queued: 1, awaiting_data: 2, interrupted: 3, done: 4, error: 5, cancelled: 6 };
        const sorted = [...jobs].sort((a, b) => {
          const od = (order[a.status] ?? 9) - (order[b.status] ?? 9);
          if (od !== 0) return od;
          // Within same status, sort by queue_position then created_at
          if (a.queue_position !== b.queue_position) return a.queue_position - b.queue_position;
          return a.created_at < b.created_at ? -1 : 1;
        });
        for (const job of sorted) {
          body.appendChild(_renderJob(job, async (jobId) => {
            try {
              await apiFetch(`/api/training/jobs/${jobId}`, { method: 'DELETE' });
            } catch (_) { /* best-effort */ }
            await refresh();
          }));
        }
      }

      const now = new Date().toLocaleTimeString();
      statusLine.textContent = `Updated ${now}`;
    } catch (e) {
      const errEl = document.createElement('div');
      errEl.style.cssText = 'color: var(--pico-del-color,#e55); font-size: 0.85em; padding: 8px 0;';
      errEl.textContent = `Error: ${e.message}`;
      body.appendChild(errEl);
      statusLine.textContent = '';
    } finally {
      refreshBtn.disabled = false;
      refreshBtn.textContent = '↻ Refresh';
    }
  }

  refreshBtn.addEventListener('click', refresh);
  open();
  refresh();
}
