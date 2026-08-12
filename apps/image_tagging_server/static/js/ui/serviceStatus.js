import { apiFetch } from '/app-static/js/shared/api/apiUtils.js';
import { prefs } from '/app-static/js/shared/prefs.js';
import { showTrainingServerModal } from '/app-static/js/ui/trainingServerModal.js';

/**
 * Live service-health panel for the Operations tab.
 *
 * This is pure monitoring — it persists nothing about the *services* — which is
 * why it lives here rather than in Settings (UI_REDESIGN §12: its presence in
 * Settings was the original miscategorisation).
 *
 * It is collapsed by default. Expanded it is five rows plus a heading — about
 * 38% of a 950px viewport — sitting above every Operations sub-tab, and the
 * answer it gives is almost always "everything is green". The collapsed summary
 * carries the part worth a permanent slot (are all services up, and the few
 * live counters), and the rows are one click away. Whether it is open is a user
 * preference and survives a reload.
 */

const EXPANDED_PREF_KEY = 'ww.operations.serviceStatusExpanded';
const OPEN_ROWS_PREF_KEY = 'ww.operations.serviceStatusOpenRows';

const SERVICE_DISPLAY = {
  db_server:           { label: 'DB Server',          icon: '🗄' },
  inference_server:    { label: 'Inference Server',   icon: '🤖' },
  tasks_server:        { label: 'Tasks Server',       icon: '⚙' },
  training_server:     { label: 'Training Server',    icon: '🏋' },
  relay_buffer_server: { label: 'Relay Buffer Server',icon: '📡' },
};

function _statusColor(status) {
  if (status === 'ok')          return 'var(--pico-ins-color,#4b8)';
  if (status === 'unreachable') return 'var(--pico-muted-color,#888)';
  return 'var(--pico-del-color,#e55)';
}

const SERVICE_CLICK_HANDLERS = {
  training_server: () => showTrainingServerModal(),
};

/**
 * The collapsed one-liner for a service.
 *
 * A service that returns its own `summary` owns the wording. The derived branch
 * is the fallback for services that haven't opted in — it reads the well-known
 * counter keys the way this panel always has, so an un-updated service (or an
 * old mock) renders unchanged.
 *
 * @param {object} svc
 * @returns {string}
 */
function _summaryLine(svc) {
  if (typeof svc.summary === 'string' && svc.summary) return svc.summary;

  const parts = [];
  if (svc.tasks)      parts.push(`running: ${svc.tasks.running}`);
  if (svc.models)     parts.push(`loaded: ${svc.models.loaded}/${svc.models.available ?? '?'}`);
  if (svc.streams)    parts.push(`streams: ${svc.streams.active}`);
  if (svc.hash_index) {
    const hi = svc.hash_index;
    parts.push(`index: ${hi.image_valid ?? 0} imgs${hi.needs_rebuild ? ' ⚠rebuild' : ''}`);
  }
  return parts.join('  ·  ') || 'ok';
}

/**
 * Render one `{title, rows:[{label,value}], empty}` section from a service's
 * `details`. Deliberately service-agnostic: adding a section is a backend-only
 * change.
 *
 * @param {object} sec
 * @returns {HTMLElement}
 */
function _renderDetailSection(sec) {
  const wrap = document.createElement('div');
  wrap.className = 'service-status-detail-section';

  if (sec.title) {
    const title = document.createElement('div');
    title.className = 'service-status-detail-section__title';
    title.textContent = sec.title;
    wrap.appendChild(title);
  }

  const rows = Array.isArray(sec.rows) ? sec.rows : [];
  if (!rows.length) {
    const empty = document.createElement('div');
    empty.className = 'service-status-detail-empty';
    empty.textContent = sec.empty ?? 'Nothing to show';
    wrap.appendChild(empty);
    return wrap;
  }

  for (const r of rows) {
    const line = document.createElement('div');
    line.className = 'service-status-detail-row';

    const label = document.createElement('span');
    label.className = 'service-status-detail-row__label';
    label.textContent = r.label ?? '';

    const value = document.createElement('span');
    value.className = 'service-status-detail-row__value';
    value.textContent = r.value ?? '';
    // The value can be long (a source URL, a task status message) and the panel
    // is narrow — the full text stays reachable on hover.
    value.title = `${r.label ?? ''}: ${r.value ?? ''}`;

    line.append(label, value);
    wrap.appendChild(line);
  }
  return wrap;
}

/**
 * A service row: the summary line, plus a disclosure holding `details` when the
 * service returned any.
 *
 * Rows with details expand in place. The `›` button is a separate control — it
 * opens the service's own modal (today only Training Server, whose modal holds
 * a per-job DELETE) and must not be swallowed by the disclosure toggle.
 *
 * @param {object} svc
 * @returns {HTMLElement}
 */
function _renderServiceRow(svc, { openServices, onToggle } = {}) {
  const meta = SERVICE_DISPLAY[svc.service] ?? { label: svc.service, icon: '●' };
  const onClick = SERVICE_CLICK_HANDLERS[svc.service];
  // Only a reachable service's details are meaningful; a stale payload from
  // before it went down would read as live.
  const sections = svc.status === 'ok' && Array.isArray(svc.details) ? svc.details : [];
  const expandable = sections.length > 0;

  const container = document.createElement(expandable ? 'details' : 'div');
  container.className = 'service-status-row';
  container.dataset.testid = `service-status-row-${svc.service}`;
  if (expandable) {
    // Every refresh rebuilds these rows from scratch, so which ones are open has
    // to live outside the DOM or a refresh silently collapses what you opened.
    container.open = openServices?.has(svc.service) === true;
    container.addEventListener('toggle', () => onToggle?.(svc.service, container.open));
  }

  const row = document.createElement(expandable ? 'summary' : 'div');
  row.className = 'service-status-row__line';
  const baseStyle = 'display:flex;gap:6px;align-items:center;padding:6px 0;';
  row.style.cssText = (expandable || onClick)
    ? baseStyle + 'cursor:pointer;border-radius:4px;transition:background 0.1s;'
    : baseStyle;
  if (expandable) {
    // A <summary> renders its own marker and adds it to the flex row; the
    // twisty below is ours so it can sit where the chevron always sat.
    row.style.listStyle = 'none';
  } else if (onClick) {
    row.addEventListener('click', onClick);
  }
  if (expandable || onClick) {
    row.addEventListener('mouseenter', () => { row.style.background = 'var(--pico-card-background-color,rgba(255,255,255,0.05))'; });
    row.addEventListener('mouseleave', () => { row.style.background = ''; });
  }

  const indicator = document.createElement('span');
  indicator.style.cssText = `color:${_statusColor(svc.status)};font-size:1.1em;flex-shrink:0;`;
  indicator.textContent = svc.status === 'ok' ? '●' : svc.status === 'unreachable' ? '○' : '✕';
  indicator.title = svc.status;

  const name = document.createElement('span');
  name.style.cssText = 'font-size:0.85em;font-weight:600;flex-shrink:0;min-width:140px;';
  name.textContent = `${meta.icon}  ${meta.label}`;

  const detail = document.createElement('span');
  detail.style.cssText = 'font-size:0.78em;color:var(--pico-muted-color,#888);flex:1;';

  if (svc.status === 'ok') {
    detail.textContent = _summaryLine(svc);
  } else {
    detail.textContent = svc.message ?? svc.status;
    detail.style.color = _statusColor(svc.status);
  }

  const twisty = document.createElement('span');
  twisty.className = 'service-status-row__twisty';
  twisty.dataset.testid = `service-status-twisty-${svc.service}`;
  twisty.style.cssText = `font-size:0.8em;color:var(--pico-muted-color,#888);flex-shrink:0;visibility:${expandable ? 'visible' : 'hidden'};`;
  twisty.textContent = '▸';

  row.append(indicator, name, detail, twisty);

  if (onClick) {
    // A real button: inside a <summary>, a plain click would also toggle the
    // disclosure, so it stops propagation and suppresses the default toggle.
    const openBtn = document.createElement('button');
    openBtn.type = 'button';
    openBtn.className = 'service-status-row__open';
    openBtn.dataset.testid = `service-status-open-${svc.service}`;
    openBtn.textContent = '›';
    openBtn.title = `Open ${meta.label}`;
    openBtn.setAttribute('aria-label', `Open ${meta.label}`);
    openBtn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      onClick();
    });
    row.appendChild(openBtn);
  }

  container.appendChild(row);

  if (expandable) {
    const body = document.createElement('div');
    body.className = 'service-status-row__details';
    body.dataset.testid = `service-status-details-${svc.service}`;
    for (const sec of sections) body.appendChild(_renderDetailSection(sec));
    container.appendChild(body);
  }

  return container;
}

/**
 * The one-line state shown while the panel is collapsed: overall health, plus
 * the handful of counters that are worth a permanent slot.
 *
 * @param {Array<object>} services
 * @returns {{ text: string, tone: 'ok' | 'warn' | 'bad' }}
 */
function _summarize(services) {
  if (!services.length) return { text: 'No services found', tone: 'warn' };

  const bad = services.filter(s => s.status !== 'ok' && s.status !== 'unreachable');
  const unreachable = services.filter(s => s.status === 'unreachable');

  const counters = [];
  for (const svc of services) {
    if (svc.tasks?.running) counters.push(`${svc.tasks.running} running`);
    if (svc.models?.loaded) counters.push(`${svc.models.loaded} model${svc.models.loaded === 1 ? '' : 's'} loaded`);
    if (svc.streams?.active) counters.push(`${svc.streams.active} stream${svc.streams.active === 1 ? '' : 's'}`);
    if (svc.hash_index?.needs_rebuild) counters.push('index needs rebuild');
  }

  // Name the broken services rather than just counting them — "1 service down"
  // sends you into the list to find out which, defeating the collapse.
  const nameOf = (s) => SERVICE_DISPLAY[s.service]?.label ?? s.service;
  let head;
  let tone;
  if (bad.length) {
    head = `${bad.map(nameOf).join(', ')} down`;
    tone = 'bad';
  } else if (unreachable.length) {
    head = `${unreachable.map(nameOf).join(', ')} unreachable`;
    tone = 'warn';
  } else {
    head = `All ${services.length} services OK`;
    tone = 'ok';
  }

  return { text: [head, ...counters].join('  ·  '), tone };
}

/**
 * Build the service-status panel element.
 *
 * @returns {HTMLElement} a detached <details>; the first refresh is already in
 *   flight so the caller can append immediately.
 */
export function createServiceStatusPanel() {
  const section = document.createElement('details');
  section.className = 'service-status-panel';
  section.dataset.testid = 'service-status-panel';
  section.open = prefs.read(EXPANDED_PREF_KEY, false) === true;
  section.addEventListener('toggle', () => prefs.write(EXPANDED_PREF_KEY, section.open));

  const header = document.createElement('summary');
  header.className = 'service-status-panel__header';

  const title = document.createElement('span');
  title.className = 'service-status-panel__title';
  title.textContent = 'Service Status';

  const summary = document.createElement('span');
  summary.className = 'service-status-panel__summary';
  summary.dataset.testid = 'service-status-summary';
  summary.textContent = 'checking…';

  const refreshBtn = document.createElement('button');
  refreshBtn.type = 'button';
  refreshBtn.textContent = '↻ Refresh';
  refreshBtn.className = 'panel-btn secondary';
  refreshBtn.dataset.testid = 'service-status-refresh';
  // Inside a <summary>, a click would also toggle the disclosure.
  refreshBtn.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); refresh(); });

  header.append(title, summary, refreshBtn);
  section.appendChild(header);

  const body = document.createElement('div');
  body.className = 'service-status-panel__body';
  section.appendChild(body);

  const stored = prefs.read(OPEN_ROWS_PREF_KEY, []);
  const openServices = new Set(Array.isArray(stored) ? stored : []);
  const onRowToggle = (service, open) => {
    if (open) openServices.add(service);
    else openServices.delete(service);
    prefs.write(OPEN_ROWS_PREF_KEY, [...openServices]);
  };

  async function refresh() {
    refreshBtn.disabled = true;
    refreshBtn.textContent = '↻ Refreshing…';
    body.innerHTML = '';
    try {
      const res = await apiFetch('/api/service-status');
      const data = await res.json();
      const services = data.services ?? [];
      for (const svc of services) {
        body.appendChild(_renderServiceRow(svc, { openServices, onToggle: onRowToggle }));
      }
      if (!services.length) {
        body.textContent = 'No services found.';
        body.style.color = 'var(--pico-muted-color,#888)';
      }
      const { text, tone } = _summarize(services);
      summary.textContent = text;
      summary.dataset.tone = tone;
    } catch (e) {
      body.textContent = `Error: ${e.message}`;
      body.style.color = 'var(--pico-del-color,#e55)';
      summary.textContent = `Error: ${e.message}`;
      summary.dataset.tone = 'bad';
    } finally {
      refreshBtn.disabled = false;
      refreshBtn.textContent = '↻ Refresh';
    }
  }

  // Fire-and-forget so the rest of the pane renders immediately.
  refresh();
  return section;
}

/**
 * Mount the service-status panel into a container element by id.
 *
 * @param {{target?: string}} opts
 */
export function renderServiceStatus({ target = 'service-status-box' } = {}) {
  const container = document.getElementById(target);
  if (!container) return;
  container.innerHTML = '';
  container.appendChild(createServiceStatusPanel());
}
