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
    instance._logLines = (liveLogs.length > 0 || isActive)
      ? liveLogs
      : (instance._value.instance?.log_lines ?? []);
    instance._createdAt = created_at ? new Date(created_at) : null;
    instance._startedAt = started_at ? new Date(started_at) : null;
    instance._endedAt = ended_at ? new Date(ended_at) : null;
    instance._typename = typename;
    instance._actions = instance._value.instance?.result?.actions ?? [];
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

    // Header row: name + type badge + status badge + timestamp
    const header = document.createElement('div');
    header.style.cssText = 'display:flex; align-items:center; gap:0.5em; flex-wrap:wrap;';

    const nameSpan = document.createElement('strong');
    nameSpan.textContent = this._name;
    header.appendChild(nameSpan);

    if (this._typename) {
      const typeBadge = document.createElement('span');
      typeBadge.textContent = this._typename;
      typeBadge.style.cssText = 'font-size:0.75em; background:#e0e0e0; border-radius:3px; padding:1px 5px; color:#444;';
      header.appendChild(typeBadge);
    }

    const statusBadge = document.createElement('span');
    const displayStatus = (this._status === 'running' && this._isPausedInPlace) ? 'paused (in-place)' : this._status;
    const displayColor = (this._status === 'running' && this._isPausedInPlace)
      ? STATUS_COLORS.paused
      : (STATUS_COLORS[this._status] ?? '#888');
    statusBadge.textContent = `● ${displayStatus}`;
    statusBadge.style.cssText = `font-size:0.8em; font-weight:bold; color:${displayColor};`;
    header.appendChild(statusBadge);

    const displayTime = this._startedAt ?? this._createdAt;
    if (displayTime) {
      const timeSpan = document.createElement('span');
      const label = this._startedAt ? 'Started' : 'Created';
      let timeText = `${label}: ${displayTime.toLocaleTimeString()}`;
      if (this._startedAt && this._endedAt) {
        const durationSec = Math.round((this._endedAt - this._startedAt) / 1000);
        timeText += ` · ${durationSec}s`;
      }
      timeSpan.textContent = timeText;
      timeSpan.style.cssText = 'font-size:0.75em; color:#888; margin-left:auto;';
      header.appendChild(timeSpan);
    }

    container.appendChild(header);

    // Progress bar + status message (only when running and progress > 0)
    if (this._status === 'running' && this._progress > 0) {
      const progressRow = document.createElement('div');
      progressRow.style.cssText = 'display:flex; align-items:center; gap:0.5em; margin-top:0.3em;';

      const bar = document.createElement('progress');
      bar.value = this._progress;
      bar.max = 100;
      bar.style.cssText = 'flex:1; height:8px;';
      progressRow.appendChild(bar);

      const pct = document.createElement('span');
      pct.textContent = `${Math.round(this._progress)}%`;
      pct.style.cssText = 'font-size:0.8em; min-width:3em; text-align:right;';
      progressRow.appendChild(pct);

      container.appendChild(progressRow);
    }

    // Status message
    const displayMsg = this._statusMessage || this._message;
    if (displayMsg) {
      const msgSpan = document.createElement('div');
      msgSpan.textContent = displayMsg;
      msgSpan.style.cssText = 'font-size:0.8em; color:#555; margin-top:0.2em;';
      container.appendChild(msgSpan);
    }

    // Collapsible log lines
    if (this._logLines.length > 0) {
      const details = document.createElement('details');
      details.style.cssText = 'margin-top:0.4em; font-size:0.8em;';

      const summary = document.createElement('summary');
      summary.textContent = `Logs (${this._logLines.length})`;
      summary.style.cssText = 'cursor:pointer; color:#666; user-select:none;';
      details.appendChild(summary);

      const pre = document.createElement('pre');
      pre.style.cssText = 'margin:0.3em 0 0 0; padding:0.4em; background:#f5f5f5; border-radius:3px; max-height:10em; overflow-y:auto; white-space:pre-wrap; word-break:break-all; font-size:0.9em;';
      pre.textContent = this._logLines.join('\n');
      details.appendChild(pre);

      container.appendChild(details);
    }

    // Task-supplied action buttons (e.g. "See results")
    if (this._actions.length > 0) {
      const actionsRow = document.createElement('div');
      actionsRow.style.cssText = 'display:flex; gap:0.5em; margin-top:0.5em; flex-wrap:wrap;';
      for (const action of this._actions) {
        const btn = document.createElement('button');
        btn.textContent = action.label;
        btn.style.cssText = 'font-size:0.8em; padding:0.2em 0.6em; cursor:pointer;';
        btn.addEventListener('click', () => this._handleAction(action));
        actionsRow.appendChild(btn);
      }
      container.appendChild(actionsRow);
    }

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
