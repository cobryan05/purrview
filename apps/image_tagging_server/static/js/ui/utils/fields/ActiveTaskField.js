import { Field } from './Field.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';

const STATUS_COLORS = {
  pending: '#888',
  running: '#2196F3',
  paused: '#FF9800',
  completed: '#4CAF50',
  error: '#F44336',
};

export class ActiveTaskField extends Field {
  constructor({ ...rest }) {
    super(rest);
  }

  static async create({ ...rest } = {}) {
    const instance = new ActiveTaskField({ ...rest });
    const { uuid, status, error_message, created_at, started_at, ended_at } = instance._value.instance || {};
    const { name, typename } = instance._value.config || {};
    instance._name = name || typename || uuid;
    instance._uuid = uuid;
    instance._status = status;
    instance._message = error_message;
    instance._progress = instance._value.progress ?? 0;
    instance._statusMessage = instance._value.status_message ?? '';
    instance._isPausedInPlace = instance._value.is_paused_in_place === true;
    // For active tasks use the live ring-buffer (may be empty on first poll).
    // For finished/suspended tasks fall back to the persisted log_json column.
    // Checking status rather than log length avoids showing a previous run's
    // persisted logs on the first poll of a fresh run.
    const liveLogs = instance._value.log_lines ?? [];
    const isActive = status === 'running' || status === 'pending';
    // TaskInstanceRead serializes with field aliases: log_lines→log_json, result→result_json.
    // Read the aliased keys from the JSON response.
    instance._logLines = (liveLogs.length > 0 || isActive)
      ? liveLogs
      : (instance._value.instance?.log_json ?? []);
    instance._createdAt = created_at ? new Date(created_at) : null;
    instance._startedAt = started_at ? new Date(started_at) : null;
    instance._endedAt = ended_at ? new Date(ended_at) : null;
    instance._durationSec = (instance._startedAt && instance._endedAt)
      ? Math.round((instance._endedAt - instance._startedAt) / 1000)
      : null;
    instance._logJoined = instance._logLines.join('\n');
    instance._typename = typename;
    instance._actions = instance._value.instance?.result_json?.actions ?? [];
    return instance;
  }

  async renderEdit() {
    const container = document.createElement('div');
    container.innerHTML = 'Not Implemented';
    return container;
  }

  async renderView() {
    const container = document.createElement('div');
    container.style.width = '100%';

    // Every task is a collapsible card — collapsed by default.
    const details = document.createElement('details');

    // ── Summary (always visible) ─────────────────────────────────────────
    const summary = document.createElement('summary');
    summary.style.cssText = 'display:flex; align-items:center; gap:0.5em; flex-wrap:wrap; cursor:pointer; user-select:none; padding:2px 0;';

    const nameSpan = document.createElement('strong');
    nameSpan.textContent = this._name;
    summary.appendChild(nameSpan);

    if (this._typename) {
      const typeBadge = document.createElement('span');
      typeBadge.textContent = this._typename;
      typeBadge.style.cssText = 'font-size:0.75em; background:#e0e0e0; border-radius:3px; padding:1px 5px; color:#444;';
      summary.appendChild(typeBadge);
    }

    const isPausedInPlace = this._status === 'running' && this._isPausedInPlace;
    const displayStatus = isPausedInPlace ? 'paused (in-place)' : this._status;
    const displayColor = isPausedInPlace ? STATUS_COLORS.paused : (STATUS_COLORS[this._status] ?? '#888');
    const statusBadge = document.createElement('span');
    statusBadge.textContent = `● ${displayStatus}`;
    statusBadge.style.cssText = `font-size:0.8em; font-weight:bold; color:${displayColor};`;
    summary.appendChild(statusBadge);

    details.appendChild(summary);

    // ── Body (visible when expanded) ─────────────────────────────────────
    const body = document.createElement('div');
    body.style.cssText = 'padding:0.4em 0 0.2em 1em; border-left:2px solid #e8e8e8; margin-top:0.3em; font-size:0.85em;';

    // Timestamps
    {
      const parts = [];
      if (this._createdAt) parts.push(`Created: ${this._createdAt.toLocaleString()}`);
      if (this._startedAt) parts.push(`Started: ${this._startedAt.toLocaleTimeString()}`);
      if (this._endedAt) parts.push(`Ended: ${this._endedAt.toLocaleTimeString()}`);
      if (this._durationSec !== null) parts.push(`Duration: ${this._durationSec}s`);
      if (parts.length) {
        const timeDiv = document.createElement('div');
        timeDiv.textContent = parts.join('  ·  ');
        timeDiv.style.cssText = 'color:#888; margin-bottom:0.35em;';
        body.appendChild(timeDiv);
      }
    }

    // Progress bar (running + progress > 0)
    if (this._status === 'running' && this._progress > 0) {
      const progressRow = document.createElement('div');
      progressRow.style.cssText = 'display:flex; align-items:center; gap:0.5em; margin-bottom:0.3em;';
      const bar = document.createElement('progress');
      bar.value = this._progress;
      bar.max = 100;
      bar.style.cssText = 'flex:1; height:8px;';
      progressRow.appendChild(bar);
      const pct = document.createElement('span');
      pct.textContent = `${Math.round(this._progress)}%`;
      pct.style.cssText = 'min-width:3em; text-align:right;';
      progressRow.appendChild(pct);
      body.appendChild(progressRow);
    }

    // Status / error message
    const displayMsg = this._statusMessage || this._message;
    if (displayMsg) {
      const msgDiv = document.createElement('div');
      msgDiv.textContent = displayMsg;
      msgDiv.style.cssText = 'color:#555; margin-bottom:0.3em;';
      body.appendChild(msgDiv);
    }

    // Log output (always rendered; shows placeholder when empty)
    if (this._logLines.length > 0) {
      const logLabel = document.createElement('div');
      logLabel.textContent = this._status === 'error'
        ? `Error log (${this._logLines.length} lines):`
        : `Logs (${this._logLines.length} lines):`;
      logLabel.style.cssText = 'color:#888; margin-bottom:0.15em;';
      body.appendChild(logLabel);

      const pre = document.createElement('pre');
      pre.style.cssText = 'margin:0; padding:0.4em; background:#f5f5f5; border-radius:3px; max-height:10em; overflow-y:auto; white-space:pre-wrap; word-break:break-all; font-size:0.9em;';
      pre.textContent = this._logJoined;
      body.appendChild(pre);
    } else {
      const noLog = document.createElement('div');
      noLog.textContent = 'No log output.';
      noLog.style.cssText = 'color:#bbb;';
      body.appendChild(noLog);
    }

    // Task-supplied action buttons (e.g. "See results")
    if (this._actions.length > 0) {
      const actionsRow = document.createElement('div');
      actionsRow.style.cssText = 'display:flex; gap:0.5em; margin-top:0.4em; flex-wrap:wrap;';
      for (const action of this._actions) {
        const btn = document.createElement('button');
        btn.textContent = action.label;
        btn.style.cssText = 'font-size:0.9em; padding:0.2em 0.6em; cursor:pointer;';
        btn.addEventListener('click', () => this._handleAction(action));
        actionsRow.appendChild(btn);
      }
      body.appendChild(actionsRow);
    }

    details.appendChild(body);
    container.appendChild(details);
    return container;
  }

  _handleAction(action) {
    if (action.type === 'open_review') {
      events.publish(EventTypes.OPEN_REVIEW_WITH_FILTERS, {
        tagUuids: action.payload?.tag_uuids ?? [],
      });
    }
  }

  getValue() {
    return {};
  }
}
