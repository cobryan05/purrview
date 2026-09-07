/**
 * Collector diagnostics modal — the tuning surface for one training collector.
 *
 * Opened from wherever a collector is doing live work (today: the Live pane's
 * watcher card). Three things live here together because tuning a collector is
 * one task, not three:
 *
 *   - the live stats and calibration widgets `TrainingCollector.status_stats()`
 *     already computes (dedup histograms/sweep tables, confusable-detection
 *     closeness, detector-miss/novelty counts) — the same widget definitions
 *     `CollectTrainingImagesTask`'s own Jobs card renders, from the one shared
 *     `collector_status_schema()` (collectorSpec.py), so the two surfaces can't
 *     drift apart;
 *   - "Browse unverified" and friends, so seeing a knob's effect and looking at
 *     the frames it kept are one click apart, not two tabs apart;
 *   - a link to the collector's own configuration, so acting on what you just
 *     saw doesn't mean hunting for it.
 *
 * The live widgets need a running task to poll (`fetchCollectorDiagnostics`
 * has nothing to answer for a stopped one) — the review buttons and the config
 * link don't, and stay available regardless.
 */
import { createModalShell, Logger } from '/app-static/js/ui/utils/index.js';
import { statusWidgets } from '/app-static/js/ui/utils/statusWidgets/index.js';
import { navigateTabs } from '/app-static/js/app/utils/tabs.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { fetchCollectorDiagnostics } from '/app-static/js/shared/api/tasks.js';
import {
  fetchCollectorStatusSchema,
  fetchTrainingCollectorReviewActions,
} from '/app-static/js/shared/api/trainingCollectors.js';

const POLL_MS = 4000;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * One leaf status widget. Mirrors `ActiveTaskField`'s own `renderStatusWidget`
 * (same factory registry, same "never let one widget's failure blank the
 * rest" contract) — kept as a short, separate copy rather than an import
 * because that function is private to a class with a much larger polling and
 * teardown contract this modal doesn't need.
 */
async function renderLeafWidget(widget, data) {
  const factory = statusWidgets[widget.type];
  if (!factory) return null;
  try {
    const result = await factory(widget, data, {});
    if (!result) return null;
    return result instanceof HTMLElement ? result : (result.el ?? null);
  } catch (err) {
    Logger.error(err);
    const placeholder = el('div', null, `[Widget error: ${widget.id}]`);
    placeholder.style.cssText = 'color:var(--text-danger, #f44336); font-size:0.8em; font-style:italic;';
    return placeholder;
  }
}

/**
 * A schema list into a host element. `groupOverrides` (widget.id -> bool) is
 * the user's explicit open/closed choice for a `group` widget, across a
 * poll's full rebuild — there is no cheap incremental diff here (unlike
 * ActiveTaskField, this modal has no per-widget teardown to preserve), so the
 * whole tree is rebuilt each tick.
 *
 * A plain "which ids are open" set is not enough: a group that defaults OPEN
 * (`collapsed` unset — Image/BBox distance) has to be able to record "the
 * user closed this", and a set can only say "not open", which reads back as
 * "no override yet" and re-opens it on every rebuild. The map records the
 * choice itself, not just one direction of it.
 */
async function renderWidgets(widgets, host, dataSource, groupOverrides) {
  for (const widget of widgets ?? []) {
    if (widget.type === 'group') {
      const details = document.createElement('details');
      details.className = 'status-group';
      details.dataset.statusGroupId = widget.id;
      details.open = groupOverrides.has(widget.id) ? groupOverrides.get(widget.id) : widget.collapsed !== true;
      details.addEventListener('toggle', () => {
        groupOverrides.set(widget.id, details.open);
      });
      const summary = el('summary', 'status-group__summary', widget.label || widget.id);
      details.appendChild(summary);
      const body = el('div', 'status-group__body');
      details.appendChild(body);
      // eslint-disable-next-line no-await-in-loop -- widgets render in schema order
      await renderWidgets(widget.widgets, body, dataSource, groupOverrides);
      host.appendChild(details);
      continue;
    }
    // eslint-disable-next-line no-await-in-loop -- widgets render in schema order
    const rendered = await renderLeafWidget(widget, dataSource[widget.id]);
    if (rendered) host.appendChild(rendered);
  }
}

/**
 * @param {Object} opts
 * @param {string} opts.collectorUuid
 * @param {string} opts.collectorName
 * @param {?string} [opts.taskUuid] - the running task instance to poll for live
 *   diagnostics, or null/omitted when nothing is currently running it.
 * @returns {{close: () => void}}
 */
export function openCollectorDiagnosticsModal({ collectorUuid, collectorName, taskUuid = null }) {
  const shell = createModalShell({ onClose: () => stop() });
  shell.dialog.classList.add('collector-diag-dialog');
  shell.dialog.dataset.testid = 'collector-diag-dialog';

  shell.dialog.appendChild(el('h3', null, `Collector — ${collectorName || 'training collector'}`));

  const linkRow = el('div', 'collector-diag-links');
  const configBtn = document.createElement('button');
  configBtn.type = 'button';
  configBtn.className = 'panel-btn';
  configBtn.textContent = 'Edit collector configuration';
  configBtn.dataset.testid = 'collector-diag-edit-config';
  configBtn.addEventListener('click', () => {
    shell.close();
    navigateTabs('tab-configuration', 'tab-collector-management', { selectUuid: collectorUuid });
  });
  linkRow.appendChild(configBtn);
  shell.dialog.appendChild(linkRow);

  const reviewBox = el('div', 'collector-diag-review');
  reviewBox.dataset.testid = 'collector-diag-review';
  reviewBox.hidden = true;
  shell.dialog.appendChild(reviewBox);

  const statusLine = el('div', 'collector-diag-status');
  statusLine.dataset.testid = 'collector-diag-status';
  shell.dialog.appendChild(statusLine);

  const noticesBox = el('div', 'collector-diag-notices');
  noticesBox.dataset.testid = 'collector-diag-notices';
  shell.dialog.appendChild(noticesBox);

  const widgetsHost = el('div', 'collector-diag-widgets');
  widgetsHost.dataset.testid = 'collector-diag-widgets';
  shell.dialog.appendChild(widgetsHost);

  const footer = el('div', 'watch-live-footer');
  const closeBtn = document.createElement('button');
  closeBtn.type = 'button';
  closeBtn.className = 'panel-btn';
  closeBtn.textContent = 'Close';
  closeBtn.dataset.testid = 'collector-diag-close';
  closeBtn.addEventListener('click', () => shell.close());
  footer.appendChild(closeBtn);
  shell.dialog.appendChild(footer);

  shell.open();

  let stopped = false;
  let pollTimer = null;
  const groupOverrides = new Map();

  function goToReview(action) {
    shell.close();
    const payload = {
      tagUuids: action.payload?.tag_uuids ?? [],
      imageTagUuids: action.payload?.image_tag_uuids ?? [],
    };
    if (action.payload?.bbox_verification !== undefined) {
      payload.bboxVerification = action.payload.bbox_verification;
    }
    events.publish(EventTypes.OPEN_REVIEW_WITH_FILTERS, payload);
  }

  async function loadReviewActions() {
    try {
      const actions = await fetchTrainingCollectorReviewActions(collectorUuid);
      if (!actions.length || stopped) return;
      reviewBox.innerHTML = '';
      reviewBox.appendChild(el('div', 'panel-section-label', 'Review this collector’s output'));
      const row = el('div', 'collector-review-box__row');
      for (const action of actions) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'panel-btn';
        btn.textContent = action.label;
        btn.dataset.testid = 'collector-diag-review-action';
        btn.addEventListener('click', () => goToReview(action));
        row.appendChild(btn);
      }
      reviewBox.appendChild(row);
      reviewBox.hidden = false;
    } catch (err) {
      Logger.error(err);
    }
  }

  async function pollDiagnostics() {
    if (stopped) return;
    if (!taskUuid) {
      statusLine.textContent = 'Not currently running — start the watcher to see live tuning stats.';
      return;
    }
    let diagnostics = null;
    try {
      diagnostics = await fetchCollectorDiagnostics(taskUuid, collectorUuid);
    } catch (err) {
      Logger.error(err);
      statusLine.textContent = 'Could not load live stats.';
      return;
    }
    if (stopped) return;
    if (!diagnostics) {
      statusLine.textContent = 'Not currently running — start the watcher to see live tuning stats.';
      widgetsHost.innerHTML = '';
      noticesBox.innerHTML = '';
      return;
    }
    statusLine.textContent = '';
    noticesBox.innerHTML = '';
    for (const notice of diagnostics.notices ?? []) {
      const row = el('div', `collector-diag-notice collector-diag-notice--${notice.level || 'info'}`, notice.message);
      noticesBox.appendChild(row);
    }
    const schema = await fetchCollectorStatusSchema();
    if (stopped) return;
    const dataSource = { ...diagnostics.stats, ...diagnostics.calibration };
    widgetsHost.innerHTML = '';
    await renderWidgets(schema, widgetsHost, dataSource, groupOverrides);
  }

  function stop() {
    stopped = true;
    if (pollTimer !== null) clearInterval(pollTimer);
  }

  void loadReviewActions();
  void pollDiagnostics();
  if (taskUuid) {
    pollTimer = setInterval(() => void pollDiagnostics(), POLL_MS);
  }

  return { close: () => shell.close() };
}

/**
 * "Which collector?" — only ever shown for the uncommon case of a watcher with
 * more than one attached (`openCollectorDiagnosticsModal` is opened directly
 * when there's exactly one, which is the common case). A picker, not a
 * dropdown baked into the diagnostics modal itself, because which collector
 * you're looking at has to be settled before that modal's title and links
 * mean anything.
 *
 * @param {Object} opts
 * @param {Array<{uuid: string, name: string}>} opts.rows
 * @param {(row: {uuid: string, name: string}) => void} opts.onPick
 */
export function openCollectorPickerModal({ rows, onPick }) {
  const shell = createModalShell({});
  shell.dialog.appendChild(el('h3', null, 'Which collector?'));
  const list = el('div', 'collector-diag-picker-list');
  for (const row of rows) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'panel-btn';
    btn.textContent = row.name;
    btn.dataset.testid = 'collector-diag-picker-option';
    btn.addEventListener('click', () => {
      shell.close();
      onPick(row);
    });
    list.appendChild(btn);
  }
  shell.dialog.appendChild(list);
  shell.open();
  return { close: () => shell.close() };
}
