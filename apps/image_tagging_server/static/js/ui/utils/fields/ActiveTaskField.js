import { Field } from './Field.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { fetchTaskStatusSchemas, fetchTaskStatusData, sendTaskCommand } from '/app-static/js/shared/api/tasks.js';
import { statusWidgets } from '/app-static/js/ui/utils/statusWidgets/index.js';

/** Statuses that mean "this task is no longer doing anything". */
export const TERMINAL_STATUSES = new Set(['completed', 'error']);

const STATUS_COLORS = {
  pending: '#888',
  running: '#2196F3',
  paused: '#FF9800',
  completed: '#4CAF50',
  error: '#F44336',
};

// The base Task ships a status_schema() that reproduces this exact layout
// (progress bar + status/error message + log tail) — this is the fallback
// used when the backend hasn't landed yet, the fetch fails, or a typename
// has no schema registered. It is what makes every pre-Phase-3 task card
// render identically to before.
const DEFAULT_STATUS_SCHEMA = [
  { id: 'progress', type: 'progress_bar', source: 'summary' },
  { id: 'statusMessage', type: 'text', source: 'summary' },
  { id: 'logLines', type: 'log_tail', source: 'summary' },
];

// Poll cadence for status-data while a card is expanded. Independent of the
// SSE-driven all-tasks poll (which intentionally never carries status-data).
const STATUS_DATA_POLL_MS = 5000;

/**
 * Last status-data payload seen per task uuid.
 *
 * renderActiveTasks rebuilds every card whenever the poll fingerprint changes,
 * and a running task changes it constantly (each new log line). A freshly built
 * card starts with empty widget slots and only fills them once its own
 * status-data fetch returns — so an expanded card briefly collapses to a
 * fraction of its height, the browser clamps the page scroll to the shorter
 * document, and the user is thrown to the top mid-read. Seeding the rebuilt card
 * from this cache keeps its height stable across the swap (and stops the widgets
 * flashing empty every few seconds).
 */
const _lastStatusDataByTask = new Map();
// Rebuilt cards are keyed by uuid, so the map would otherwise grow for the life
// of the session on a long-lived tab. Only expanded cards ever write to it.
const _STATUS_DATA_CACHE_MAX = 50;

function _rememberStatusData(uuid, data) {
  if (!uuid) return;
  _lastStatusDataByTask.delete(uuid); // re-insert so iteration order is LRU
  _lastStatusDataByTask.set(uuid, data);
  while (_lastStatusDataByTask.size > _STATUS_DATA_CACHE_MAX) {
    _lastStatusDataByTask.delete(_lastStatusDataByTask.keys().next().value);
  }
}

/**
 * Which status-widget groups the user has opened, keyed `${taskUuid}::${groupId}`.
 *
 * A group's open state cannot live in the DOM alone: renderActiveTasks rebuilds
 * the entire card whenever the poll fingerprint changes — which, for a running
 * task, is every few seconds — so a group the user opened would snap shut under
 * them mid-read. tasks.js already restores the card-level <details>, but it
 * cannot know about widget groups, let alone nested ones.
 */
const _groupOpenByKey = new Map();
const _GROUP_OPEN_CACHE_MAX = 200;

function _rememberGroupOpen(key, open) {
  _groupOpenByKey.delete(key); // re-insert so iteration order is LRU
  _groupOpenByKey.set(key, open);
  while (_groupOpenByKey.size > _GROUP_OPEN_CACHE_MAX) {
    _groupOpenByKey.delete(_groupOpenByKey.keys().next().value);
  }
}

/**
 * Does this schema contain any `source: "status"` widget, at any nesting depth?
 *
 * Groups build their children lazily (see _buildGroup), so the flat slot list is
 * empty at the point the poller is wired up when every status widget happens to
 * sit inside a collapsed group. Asking the schema instead of the slot list is
 * what keeps such a card polling once the user opens the group.
 */
function _schemaHasStatusWidgets(widgets) {
  for (const widget of widgets ?? []) {
    if (widget.type === 'group') {
      if (_schemaHasStatusWidgets(widget.widgets)) return true;
      continue;
    }
    if ((widget.source ?? 'status') !== 'summary') return true;
  }
  return false;
}

/** Module-level, per-typename schema cache (parallels fetchTaskStatusSchemas'
 * own createCachedFetcher, but also memoizes the "which fallback did we use"
 * decision so repeated failures don't reissue requests every render). */
async function resolveStatusSchema(typename) {
  if (!typename) return DEFAULT_STATUS_SCHEMA;
  try {
    const schemas = await fetchTaskStatusSchemas([typename]);
    const schema = schemas.get(typename);
    if (Array.isArray(schema) && schema.length > 0) return schema;
  } catch (err) {
    // Backend may not have landed yet, or the request failed outright —
    // fall back so the card still renders exactly as before.
  }
  return DEFAULT_STATUS_SCHEMA;
}

/**
 * Renders a single schema widget, degrading gracefully rather than ever
 * throwing: an unknown `type` or a factory exception must not break sibling
 * widgets on the same card.
 *
 * @returns {Promise<{el: HTMLElement, destroy: (() => void)|null}|null>}
 */
async function renderStatusWidget(widget, data, ctx) {
  const factory = statusWidgets[widget.type];
  if (!factory) {
    const placeholder = document.createElement('div');
    placeholder.textContent = `[Unsupported status widget: ${widget.type}]`;
    placeholder.style.cssText = 'color:#bbb; font-size:0.8em; font-style:italic;';
    return { el: placeholder, destroy: null };
  }
  try {
    const result = await factory(widget, data, ctx);
    if (!result) return null;
    return (result instanceof HTMLElement) ? { el: result, destroy: null } : result;
  } catch (err) {
    console.error(`Status widget "${widget.type}" (id=${widget.id}) failed to render:`, err);
    const placeholder = document.createElement('div');
    placeholder.textContent = `[Widget error: ${widget.id}]`;
    placeholder.style.cssText = 'color:#f44336; font-size:0.8em; font-style:italic;';
    return { el: placeholder, destroy: null };
  }
}

export class ActiveTaskField extends Field {
  constructor({ ...rest }) {
    super(rest);
  }

  static async create({ ...rest } = {}) {
    const instance = new ActiveTaskField({ ...rest });
    const { uuid, status, errorMessage, createdAt, startedAt, endedAt } = instance._value.instance || {};
    const { name, typename } = instance._value.config || {};
    instance._name = name || typename || uuid;
    instance._uuid = uuid;
    instance._status = status;
    instance._message = errorMessage;
    // Wire signal for "indeterminate": a null/absent `progress` (as opposed to a
    // genuine numeric 0, which is a normal 0%-complete batch task). The `progress`
    // field is already optional-shaped on the wire (existing code coalesced it with
    // `?? 0`); we just stop collapsing that distinction before it reaches rendering.
    instance._progress = instance._value.progress ?? 0;
    instance._statusMessage = instance._value.statusMessage ?? '';
    instance._isPausedInPlace = instance._value.isPausedInPlace === true;
    // Set by renderActiveTasks when it is rebuilding a card the user had open,
    // so the new card can be seeded to its predecessor's height (see
    // _lastStatusDataByTask). Defaults false: a first-time card has nothing to
    // restore and must not pre-render widgets it may never show.
    instance._expanded = rest.expanded === true;
    // For active tasks use the live ring-buffer (may be empty on first poll).
    // For finished/suspended tasks fall back to the persisted logLines field.
    // Checking status rather than log length avoids showing a previous run's
    // persisted logs on the first poll of a fresh run.
    const liveLogs = instance._value.logLines ?? [];
    const isActive = status === 'running' || status === 'pending';
    instance._logLines = (liveLogs.length > 0 || isActive)
      ? liveLogs
      : (instance._value.instance?.logLines ?? []);
    instance._createdAt = createdAt ? new Date(createdAt) : null;
    instance._startedAt = startedAt ? new Date(startedAt) : null;
    instance._endedAt = endedAt ? new Date(endedAt) : null;
    instance._durationSec = (instance._startedAt && instance._endedAt)
      ? Math.round((instance._endedAt - instance._startedAt) / 1000)
      : null;
    instance._logJoined = instance._logLines.join('\n');
    instance._typename = typename;
    // Merge live status actions (available while running) with completed-result actions.
    const resultActions = instance._value.instance?.result?.actions ?? [];
    const statusActions = instance._value.statusActions ?? [];
    // Deduplicate by label: result actions win (they come from the completed final state).
    const seenLabels = new Set(resultActions.map(a => a.label));
    const mergedActions = [...resultActions, ...statusActions.filter(a => !seenLabels.has(a.label))];
    instance._actions = mergedActions;

    // Values readable by `source: "summary"` schema widgets — the field
    // named by a widget's `id` is looked up here first (falling back to the
    // raw already-polled payload for typenames with custom summary fields).
    // `progress` is only meaningful while running (see progress_bar's own
    // gate on ctx.taskStatus); statusMessage folds in the same error-message
    // fallback the old hardcoded rendering used.
    instance._summaryData = {
      progress: instance._value.progress,
      statusMessage: instance._statusMessage || instance._message || '',
      logLines: instance._logLines,
      log: instance._logLines, // alias — see status_schema wire-contract example using id "log"
    };

    instance._schema = await resolveStatusSchema(typename);

    return instance;
  }

  async renderEdit() {
    const container = document.createElement('div');
    container.innerHTML = 'Not Implemented';
    return container;
  }

  _getSummaryValue(widget) {
    if (widget.id in this._summaryData) return this._summaryData[widget.id];
    return this._value?.[widget.id];
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

    // ── Why it failed ──────────────────────────────────────────────────
    // Rendered here rather than as a status widget on purpose: a task type with
    // its own status_schema() (CollectTrainingImages, TrainModel, …) doesn't
    // include the default statusMessage widget, so a schema-driven error message
    // would be invisible on exactly the cards most likely to have one.
    if (this._status === 'error' && this._message) {
      const errDiv = document.createElement('div');
      errDiv.className = 'task-error-message';
      errDiv.dataset.testid = 'task-error-message';
      errDiv.textContent = this._message;
      body.appendChild(errDiv);
    }

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

    // ── Schema-driven status widgets ───────────────────────────────────
    const widgetsHost = document.createElement('div');
    body.appendChild(widgetsHost);

    const ctx = {
      taskUuid: this._uuid,
      taskStatus: this._status,
      sendCommand: (command, params = null) => sendTaskCommand({ taskUuid: this._uuid, command, params }),
    };

    // status-source widgets are fetched only while expanded; summary-source
    // widgets render immediately from the already-polled payload we have.
    const statusSlots = []; // { widget, slot } — groups may append to this later
    // Assigned once refresh() exists below; a group opened after the initial
    // build needs to pull data for the widgets it has just created rather than
    // sitting empty until the next 5s tick.
    let requestRefresh = () => {};

    /**
     * A `group` widget: a nestable <details> holding other widgets. It has no
     * data of its own and never appears in get_status_data() — it is pure
     * layout, which is why it is handled here rather than as a factory in
     * statusWidgets (its children need this method's slot/poll/teardown
     * bookkeeping, which a factory has no access to).
     */
    const buildGroup = async (widget) => {
      const key = `${this._uuid}::${widget.id}`;
      const details = document.createElement('details');
      details.className = 'status-group';
      details.dataset.statusGroupId = widget.id;

      const groupSummary = document.createElement('summary');
      groupSummary.className = 'status-group__summary';
      groupSummary.textContent = widget.label || widget.id;
      details.appendChild(groupSummary);

      const groupBody = document.createElement('div');
      groupBody.className = 'status-group__body';
      details.appendChild(groupBody);

      // Children are built on first open, not up front. A collapsed group then
      // costs nothing — and, more importantly, an annotated_image inside one
      // would otherwise mount its canvas into a zero-height box and fit every
      // bbox to (0,0), the same failure the top-level card avoids by deferring
      // its own mounts until it is expanded.
      let built = false;
      const build = async () => {
        if (built) return;
        built = true;
        await buildWidgets(widget.widgets, groupBody);
      };

      const remembered = _groupOpenByKey.get(key);
      if (remembered ?? widget.collapsed !== true) {
        details.open = true;
        await build();
      }

      details.addEventListener('toggle', async () => {
        _rememberGroupOpen(key, details.open);
        if (!details.open) return;
        await build();
        requestRefresh();
      });

      return details;
    };

    const buildWidgets = async (widgets, host) => {
      for (const widget of widgets ?? []) {
        if (widget.type === 'group') {
          host.appendChild(await buildGroup(widget));
          continue;
        }

        const slot = document.createElement('div');
        host.appendChild(slot);

        const source = widget.source ?? 'status';
        if (source === 'summary') {
          const value = this._getSummaryValue(widget);
          const rendered = await renderStatusWidget(widget, value, ctx);
          if (rendered) {
            slot.appendChild(rendered.el);
            // mount() runs post-attach layout (e.g. sizing a canvas) — some widget
            // factories (annotated_image) defer their real setup to it, so it must
            // run regardless of which source the widget is configured with.
            await rendered.mount?.();
          }
          // Summary widgets never own a resource that needs teardown.
        } else {
          statusSlots.push({ widget, slot });
        }
      }
    };

    await buildWidgets(this._schema || [], widgetsHost);

    if (_schemaHasStatusWidgets(this._schema) && this._uuid) {
      let pollTimer = null;
      let refreshing = false; // re-entrancy guard
      // Per-widget (not global) teardown + last-seen-data tracking: a poll tick
      // only tears down and re-renders the widgets whose data actually changed,
      // instead of unconditionally rebuilding every widget's DOM (and, for
      // annotated_image, its Konva canvas) every STATUS_DATA_POLL_MS regardless
      // of whether anything is new. Keyed by widget.id.
      const cleanupFnsByWidget = new Map();
      const lastDataJsonByWidget = new Map();

      const teardown = () => {
        for (const fn of cleanupFnsByWidget.values()) {
          try { fn(); } catch (err) { console.error('Status widget teardown failed:', err); }
        }
        cleanupFnsByWidget.clear();
        lastDataJsonByWidget.clear();
      };

      const stopPolling = () => {
        if (pollTimer !== null) {
          clearInterval(pollTimer);
          pollTimer = null;
        }
      };

      const refresh = async () => {
        // Self-healing cleanup: if this card's DOM was replaced by a later
        // poll/render cycle (renderActiveTasks does a full swap), stop.
        if (!details.isConnected) {
          stopPolling();
          return;
        }
        if (!details.open || refreshing) return;
        refreshing = true;
        try {
          let dataForTask = {};
          try {
            const dataMap = await fetchTaskStatusData([this._uuid]);
            dataForTask = dataMap.get(this._uuid) || {};
          } catch (err) {
            console.error('Failed to fetch task status data:', err);
            return;
          }
          // Card may have collapsed / been removed while the fetch was in flight.
          if (!details.open || !details.isConnected) return;
          _rememberStatusData(this._uuid, dataForTask);

          for (const { widget, slot } of statusSlots) {
            const data = dataForTask[widget.id];
            const dataJson = JSON.stringify(data ?? null);
            if (lastDataJsonByWidget.has(widget.id) && lastDataJsonByWidget.get(widget.id) === dataJson) {
              continue; // unchanged since the last poll — leave this widget's DOM/resources alone
            }
            lastDataJsonByWidget.set(widget.id, dataJson);

            const prevDestroy = cleanupFnsByWidget.get(widget.id);
            if (prevDestroy) {
              try { prevDestroy(); } catch (err) { console.error('Status widget teardown failed:', err); }
              cleanupFnsByWidget.delete(widget.id);
            }

            slot.innerHTML = '';
            const rendered = await renderStatusWidget(widget, data, ctx);
            if (rendered) {
              slot.appendChild(rendered.el);
              // mount() is an optional hook for widgets that need real layout
              // (e.g. a canvas) — called only once the element is attached to
              // the live document, unlike measuring during the factory call.
              await rendered.mount?.();
              if (rendered.destroy) cleanupFnsByWidget.set(widget.id, rendered.destroy);
            }
          }
        } finally {
          refreshing = false;
        }
      };
      requestRefresh = refresh;

      // Seed this card from the last payload its predecessor saw, so a rebuild
      // reuses the previous height instead of momentarily rendering empty (see
      // _lastStatusDataByTask). Only for a card that is being rebuilt expanded:
      // a collapsed one is not visible, so pre-rendering it — including spinning
      // up an annotated_image canvas — would be pure waste.
      const pendingMounts = [];
      if (this._expanded) {
        const seed = _lastStatusDataByTask.get(this._uuid);
        if (seed) {
          for (const { widget, slot } of statusSlots) {
            const data = seed[widget.id];
            // Record what was rendered so the first refresh() skips widgets whose
            // data has not moved on, rather than tearing them straight back down.
            lastDataJsonByWidget.set(widget.id, JSON.stringify(data ?? null));
            const rendered = await renderStatusWidget(widget, data, ctx);
            if (!rendered) continue;
            slot.appendChild(rendered.el);
            // mount() measures real layout, so it must wait until this card is
            // attached to the document — which happens after renderView returns.
            if (rendered.mount) pendingMounts.push(rendered.mount);
            if (rendered.destroy) cleanupFnsByWidget.set(widget.id, rendered.destroy);
          }
        }
      }

      // A finished task's status data is a snapshot persisted when it ended — it
      // cannot change, so fetch it once and don't hold a 5s timer open on a card
      // the user may leave expanded indefinitely.
      const isTerminal = TERMINAL_STATUSES.has(this._status);

      details.addEventListener('toggle', () => {
        if (details.open) {
          while (pendingMounts.length) {
            Promise.resolve(pendingMounts.shift()()).catch(err =>
              console.error('Status widget mount failed:', err));
          }
          refresh();
          if (!isTerminal && pollTimer === null) pollTimer = setInterval(refresh, STATUS_DATA_POLL_MS);
        } else {
          stopPolling();
          teardown();
        }
      });
    }

    // Task-supplied action buttons (e.g. "See results", or a `task_command`
    // back-channel action like "capture now").
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

  async _handleAction(action) {
    if (action.type === 'open_review') {
      const payload = {
        tagUuids: action.payload?.tag_uuids ?? [],
        imageTagUuids: action.payload?.image_tag_uuids ?? [],
      };
      // Only forwarded when the action asks for it, because the subscriber
      // treats `undefined` as "leave whatever the panel already had" — which is
      // what a plain "browse results" wants, and what an explicit
      // "browse unverified" must override.
      if (action.payload?.bbox_verification !== undefined) {
        payload.bboxVerification = action.payload.bbox_verification;
      }
      events.publish(EventTypes.OPEN_REVIEW_WITH_FILTERS, payload);
      return;
    }
    if (action.type === 'task_command') {
      const command = action.command ?? action.payload?.command;
      const params = action.params ?? action.payload?.params ?? null;
      if (!command || !this._uuid) return;
      try {
        await sendTaskCommand({ taskUuid: this._uuid, command, params });
      } catch (err) {
        console.error('Task command failed:', err);
      }
    }
  }

  getValue() {
    return {};
  }
}
