import { Field } from './Field.js';

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
    const { uuid, status, error_message, created_at } = instance._value.instance || {};
    const { name, typename } = instance._value.config || {};
    instance._name = name || typename || uuid;
    instance._uuid = uuid;
    instance._status = status;
    instance._message = error_message;
    instance._progress = instance._value.progress ?? 0;
    instance._statusMessage = instance._value.status_message ?? '';
    instance._logLines = instance._value.log_lines ?? [];
    instance._createdAt = created_at ? new Date(created_at) : null;
    instance._typename = typename;
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
    statusBadge.textContent = `● ${this._status}`;
    statusBadge.style.cssText = `font-size:0.8em; font-weight:bold; color:${STATUS_COLORS[this._status] ?? '#888'};`;
    header.appendChild(statusBadge);

    if (this._createdAt) {
      const timeSpan = document.createElement('span');
      timeSpan.textContent = this._createdAt.toLocaleTimeString();
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

    return container;
  }

  getValue() {
    return {};
  }
}
