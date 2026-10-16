import { createModalShell } from './utils/modalShell.js';
import { apiFetch } from '/app-static/js/shared/api/apiUtils.js';
import { showPaneError } from '/app-static/js/ui/utils/index.js';
import { parseApiDate } from '/app-static/js/shared/time.js';
import { getSetting, TRAINING_ARTIFACTS_KEEP_SETTING_KEY } from '/app-static/js/shared/api/settings.js';

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
  const date = parseApiDate(iso);
  if (!date) return iso;
  return date.toLocaleString(undefined, {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

const CANCELLABLE_STATUSES = new Set(['running', 'queued', 'awaiting_data', 'interrupted']);

/**
 * A small action on a job row, styled like the cancel button beside it.
 *
 * @param {string} text
 * @param {string} title
 * @param {(e: Event) => Promise<void>} onClick
 * @param {string} [testid]
 */
function _rowButton(text, title, onClick, testid) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.textContent = text;
  btn.className = 'secondary';
  btn.style.cssText = 'font-size: 0.68em; padding: 0.1em 0.45em; line-height: 1.4; min-width: 0;';
  btn.title = title;
  if (testid) btn.dataset.testid = testid;
  btn.addEventListener('click', async (e) => {
    e.stopPropagation();
    btn.disabled = true;
    const original = btn.textContent;
    btn.textContent = '…';
    try {
      await onClick(e);
    } finally {
      btn.disabled = false;
      btn.textContent = original;
    }
  });
  return btn;
}

function _renderJob(job, onCancel, onRestart) {
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

  // An interrupted job is the one status with work still recoverable: the server died
  // mid-run and libreyolo left a per-epoch checkpoint behind. Resume leads, because
  // continuing is nearly always what was wanted — starting over throws away however
  // many epochs had already run, which at a couple of minutes each is the whole
  // afternoon. "Start over" stays available for the cases a resume cannot serve: a
  // changed dataset, or a changed architecture, which libreyolo refuses outright.
  if (job.status === 'interrupted' && onRestart) {
    const epoch = job.resumable_epoch;
    rightCol.appendChild(_rowButton(
      epoch ? `▸ Resume at ${epoch}` : '▸ Resume',
      epoch
        ? `Continue this run from its last checkpoint, at epoch ${epoch}. The optimizer and `
          + 'learning-rate schedule carry over, so it picks up where it stopped.'
        : 'Continue this run from its last checkpoint.',
      () => onRestart(job.job_id, true),
      'training-job-resume',
    ));
    rightCol.appendChild(_rowButton(
      'Start over',
      'Discard the checkpoint and train from pretrained weights again. Use this if the '
        + 'dataset or the architecture has changed since the run began.',
      () => onRestart(job.job_id, false),
      'training-job-start-over',
    ));
  }

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

/** Human bytes, matching the wording the server's own `fmt_bytes` produces. */
function _fmtBytes(n) {
  if (!Number.isFinite(n) || n <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return `${v >= 100 || i === 0 ? Math.round(v) : v.toFixed(1)} ${units[i]}`;
}

/**
 * Disk the training server is holding, and the two ways to give it back.
 *
 * The numbers were already computed — `/api/training/info` walks both directories
 * and the landing page renders them — but they were not reachable from the app,
 * so a cache that had grown to its 10 GB bound was invisible until you went
 * looking on the server itself. The two actions map to the endpoints that already
 * existed and had no caller:
 *
 *  - **Evict now** applies the configured age and size bounds immediately instead
 *    of waiting for the sweep timer. Safe by construction — it removes only what
 *    the server was going to remove anyway.
 *  - **Clear cache** drops everything unpinned. Images pinned by a live job are
 *    kept, because dropping those leaves a queued job unable to run without a
 *    fresh upload of every image it needs; the response says how many it spared.
 *
 *  - **Evict artifacts** applies the "Keep training job folders" bound from
 *    Settings → Model History. A job folder holds one finished run's weights and
 *    checkpoints and is what a restart or a re-download reads, so it is output
 *    rather than scratch — which is why it is bounded by its own number and not
 *    swept with the cache. Only finished runs are candidates: queued, running and
 *    interrupted jobs keep their folders, an interrupted one because it can still
 *    be restarted from the checkpoint inside it. The same sweep runs on its own
 *    after every completed training run; this button is for reclaiming the space
 *    without waiting for one.
 *
 * @param {() => Promise<void>} onChanged - re-read the panel after an action.
 * @returns {HTMLElement}
 */
function _renderStorage(info, onChanged) {
  const wrap = document.createElement('div');
  wrap.dataset.testid = 'training-storage';
  wrap.style.cssText = 'margin-top: 14px; padding-top: 10px; border-top: 1px solid var(--pico-muted-border-color, #333);';

  const title = document.createElement('div');
  title.style.cssText = 'font-size: 0.8em; font-weight: 600; margin-bottom: 6px;';
  title.textContent = 'Storage';
  wrap.appendChild(title);

  const cache = info?.cache ?? {};
  const artifacts = info?.artifacts ?? {};
  const pct = cache.max_bytes ? Math.round((cache.bytes / cache.max_bytes) * 100) : null;

  const lines = document.createElement('div');
  lines.style.cssText = 'font-size: 0.78em; color: var(--pico-muted-color,#888); display: grid; gap: 2px;';

  const cacheLine = document.createElement('div');
  cacheLine.dataset.testid = 'training-cache-usage';
  cacheLine.textContent =
    `Image cache: ${_fmtBytes(cache.bytes)}`
    + (cache.max_bytes ? ` of ${_fmtBytes(cache.max_bytes)}${pct === null ? '' : ` (${pct}%)`}` : '')
    + ` · ${cache.files ?? 0} file(s)`
    + (cache.pinned ? ` · ${cache.pinned} pinned by live jobs` : '');

  const artifactLine = document.createElement('div');
  artifactLine.dataset.testid = 'training-artifacts-usage';
  artifactLine.textContent = `Artifacts: ${_fmtBytes(artifacts.bytes)} · ${artifacts.jobs ?? 0} job folder(s)`;

  lines.append(cacheLine, artifactLine);
  wrap.appendChild(lines);

  const actions = document.createElement('div');
  actions.style.cssText = 'display: flex; gap: 6px; margin-top: 8px;';

  const status = document.createElement('div');
  status.style.cssText = 'font-size: 0.75em; color: var(--pico-muted-color,#888); margin-top: 6px; min-height: 1.1em;';
  status.dataset.testid = 'training-cache-status';

  /** Run one storage action, reporting what it freed and re-reading the figures. */
  const action = (label, title, run, unit = 'file') => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'secondary';
    btn.textContent = label;
    btn.title = title;
    btn.style.cssText = 'font-size: 0.72em; padding: 0.15em 0.5em; min-width: 0; margin: 0;';
    btn.addEventListener('click', async () => {
      const original = btn.textContent;
      btn.disabled = true;
      btn.textContent = '…';
      try {
        const result = await run();
        if (result === null) return; // declined at the confirm
        // Cache eviction counts files; artifact eviction names the job folders it
        // removed. Both arrive as `removed`.
        const rawRemoved = result?.removed ?? 0;
        const removed = Array.isArray(rawRemoved) ? rawRemoved.length : rawRemoved;
        status.textContent = removed
          ? `Removed ${removed} ${unit}(s), freeing ${_fmtBytes(result?.freedBytes ?? result?.freed_bytes ?? 0)}`
            + (result?.keptPinned ?? result?.kept_pinned ? ` · kept ${result.keptPinned ?? result.kept_pinned} pinned` : '')
          : 'Nothing to remove.';
        await onChanged();
      } catch (e) {
        status.textContent = `✗ ${e.message}`;
      } finally {
        btn.disabled = false;
        btn.textContent = original;
      }
    });
    return btn;
  };

  actions.append(
    action('Evict now', 'Apply the age and size limits immediately instead of waiting for the sweep', async () => {
      const res = await apiFetch('/api/training/cache/evict', { method: 'POST' });
      if (!res.ok) throw new Error(`Eviction failed (${res.status})`);
      return await res.json();
    }),
    action(
      'Evict artifacts',
      'Keep only the most recent finished runs, per Settings → Model History',
      async () => {
        // The limit lives in db_server settings; the training server may be on
        // another host and cannot read it, so it is sent with the request.
        let keep = null;
        try {
          const stored = await getSetting(TRAINING_ARTIFACTS_KEEP_SETTING_KEY);
          const parsed = stored != null && String(stored).trim() !== '' ? Number(stored) : NaN;
          if (Number.isFinite(parsed)) keep = Math.max(0, Math.trunc(parsed));
        } catch (_) {}
        const query = keep === null ? '' : `?keep=${keep}`;
        const res = await apiFetch(`/api/training/artifacts/evict${query}`, { method: 'POST' });
        if (!res.ok) throw new Error(`Artifact eviction failed (${res.status})`);
        return await res.json();
      },
      'folder',
    ),
    action('Clear cache', 'Delete every cached image except those a live job is using', async () => {
      if (!window.confirm(
        'Delete the training server\'s cached images?\n\n'
        + 'Images a running or queued job is using are kept. Everything else is re-uploaded '
        + 'automatically the next time a job needs it, so this costs upload time, not data.'
      )) return null;
      const res = await apiFetch('/api/training/cache', { method: 'DELETE' });
      if (!res.ok) throw new Error(`Clear failed (${res.status})`);
      return await res.json();
    }),
  );

  wrap.append(actions, status);
  return wrap;
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

  // Its own element rather than part of `body`, which refresh() empties: the
  // storage figures come from a separate, heavier endpoint and should not be torn
  // down and re-fetched every time the job list repaints.
  const storageBox = document.createElement('div');

  const statusLine = document.createElement('div');
  statusLine.style.cssText = 'font-size: 0.78em; color: var(--pico-muted-color,#888); margin-top: 10px; text-align: right;';

  dialog.append(header, body, storageBox, statusLine);

  async function refreshStorage() {
    try {
      const res = await apiFetch('/api/training/info');
      if (!res.ok) throw new Error(String(res.status));
      const info = await res.json();
      storageBox.innerHTML = '';
      storageBox.appendChild(_renderStorage(info, refreshStorage));
    } catch (_) {
      // The jobs list above already reports an unreachable server; a second
      // failure line for the same outage is noise.
      storageBox.innerHTML = '';
    }
  }

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
          body.appendChild(_renderJob(
            job,
            async (jobId) => {
              try {
                await apiFetch(`/api/training/jobs/${jobId}`, { method: 'DELETE' });
              } catch (_) { /* best-effort */ }
              await refresh();
            },
            async (jobId, resume) => {
              // Reported rather than swallowed, unlike the cancel beside it: a resume
              // that libreyolo refuses — a changed architecture, a config mismatch —
              // comes back as a 409 or a 500, and silently re-rendering an unchanged
              // row is how "the button does nothing" gets reported as a bug.
              try {
                await apiFetch(
                  `/api/training/jobs/${jobId}/restart?resume=${resume ? 'true' : 'false'}`,
                  { method: 'POST' },
                );
              } catch (e) {
                showPaneError(body, `Could not ${resume ? 'resume' : 'restart'} that job: ${e.message}`);
                return;
              }
              await refresh();
            },
          ));
        }
      }

      const now = new Date().toLocaleTimeString();
      statusLine.textContent = `Updated ${now}`;
    } catch (e) {
      showPaneError(body, `Training server unavailable: ${e.message}`);
      statusLine.textContent = '';
    } finally {
      refreshBtn.disabled = false;
      refreshBtn.textContent = '↻ Refresh';
    }
  }

  refreshBtn.addEventListener('click', () => { refresh(); refreshStorage(); });
  open();
  refresh();
  refreshStorage();
}
