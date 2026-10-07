/**
 * Notices Bar — a fixed strip above the active-model bar that surfaces
 * advisory messages from backend services (e.g. "hash index rebuilding").
 *
 * Mount once at app startup:
 *   import { mountNoticesBar } from '/app-static/js/ui/noticesBar.js';
 *   mountNoticesBar('notices-bar');
 *
 * Subscribes to /api/notices/stream (SSE) for push updates. The bar is
 * hidden when there are no active notices and shown automatically when any arrive.
 *
 * Notice shape: { key: string, level: "info"|"warning"|"error", message: string,
 *                 actions?: [{ label: string, url: string }] }
 *
 * A notice carrying `actions` is one the user has to answer, not just read: each
 * action is a button that POSTs to its url, and the next SSE frame is what removes
 * the notice. Urls are restricted to relative `/api/` paths — a notice is backend
 * text, and the bar must not be a way to aim a POST somewhere else.
 *
 * The bar is fixed to the bottom above the whole app, so its height is capped
 * and the list scrolls inside it: an outage produces one notice per unreachable
 * service, which is exactly when an unbounded strip would grow over the app it
 * is reporting on. A header line summarises the notices and collapses them.
 */

import { suspendWhileHidden } from '/app-static/js/shared/visibility.js';
import { toast } from '/app-static/js/ui/utils/logging.js';

const LEVEL_ICONS = {
  info: 'ℹ',
  warning: '⚠',
  error: '✕',
};

// Worst-first: the header borrows the highest level present, since a collapsed
// bar reporting an error in info blue reads as "all fine".
const LEVEL_RANK = ['error', 'warning', 'info'];

const COLLAPSED_KEY = 'ww_notices_bar_collapsed';

/** Same-origin API paths only — see the action note in the module docstring. */
function _safeActionUrl(url) {
  return typeof url === 'string' && url.startsWith('/api/') && !url.startsWith('//') ? url : null;
}

function _readCollapsed() {
  try { return localStorage.getItem(COLLAPSED_KEY) === '1'; } catch { return false; }
}

function _writeCollapsed(collapsed) {
  try { localStorage.setItem(COLLAPSED_KEY, collapsed ? '1' : '0'); } catch { /* private mode */ }
}

function _worstLevel(notices) {
  const present = new Set(notices.map(n => n.level || 'info'));
  return LEVEL_RANK.find(level => present.has(level)) ?? 'info';
}

/**
 * @param {string} containerId  id of the mount <div>
 * @returns {{ destroy: () => void }}
 */
export function mountNoticesBar(containerId = 'notices-bar') {
  const container = document.getElementById(containerId);
  if (!container) return { destroy() {} };

  // Collapse lives outside the DOM because render() rebuilds the bar from
  // scratch on every frame — held in the markup it would be undone by the next
  // poll, and during an outage the next poll is always seconds away. It is
  // persisted for the same reason at a longer timescale: an outage outlives a
  // page load, and re-collapsing on every reload is the same annoyance as not
  // being able to collapse at all.
  let collapsed = _readCollapsed();
  let lastNotices = [];

  function buildHeader(notices) {
    const level = _worstLevel(notices);
    const header = document.createElement('div');
    // Takes the level *modifier* for its colour but not the base item class:
    // `.notices-bar__item` counts notices, and a header that answered to it
    // would report one row more than there are notices.
    header.className = `notices-bar__header notices-bar__item--${level}`;

    const icon = document.createElement('span');
    icon.className = 'notices-bar__icon';
    icon.textContent = LEVEL_ICONS[level] ?? 'ℹ';
    header.appendChild(icon);

    const summary = document.createElement('span');
    summary.className = 'notices-bar__summary';
    const count = `${notices.length} ${notices.length === 1 ? 'notice' : 'notices'}`;
    // Collapsed, this line is the whole bar, so it has to stay a status bar
    // rather than a chevron hiding an outage: it names the first notice too.
    summary.textContent = collapsed ? `${count} — ${notices[0]?.message ?? ''}` : count;
    header.appendChild(summary);

    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'notices-bar__toggle';
    toggle.textContent = collapsed ? '▴' : '▾';
    toggle.title = collapsed ? 'Show notices' : 'Hide notices';
    toggle.setAttribute('aria-label', toggle.title);
    toggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    toggle.addEventListener('click', () => {
      collapsed = !collapsed;
      _writeCollapsed(collapsed);
      render(lastNotices);
    });
    header.appendChild(toggle);

    return header;
  }

  function buildItem(notice) {
    const level = notice.level || 'info';
    const item = document.createElement('div');
    item.className = `notices-bar__item notices-bar__item--${level}`;

    const icon = document.createElement('span');
    icon.className = 'notices-bar__icon';
    icon.textContent = LEVEL_ICONS[level] ?? 'ℹ';
    item.appendChild(icon);

    const msg = document.createElement('span');
    msg.className = 'notices-bar__message';
    msg.textContent = notice.message || '';
    item.appendChild(msg);

    for (const action of notice.actions ?? []) {
      const url = _safeActionUrl(action?.url);
      if (!url) continue;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'notices-bar__action';
      btn.textContent = action.label || 'OK';
      btn.dataset.testid = 'notice-action';
      btn.addEventListener('click', async () => {
        // Disabled for the round trip, not removed: the notice is only gone once the
        // backend says so, and a button that vanishes on click leaves the row looking
        // answered while the write is still in flight.
        btn.disabled = true;
        try {
          const res = await fetch(url, { method: 'POST' });
          if (!res.ok) throw new Error(`${res.status}`);
          // The body decides, not the status code: these handlers answer a failed write
          // with 200 and `status: "failure"`, so trusting `res.ok` alone leaves the row
          // looking answered behind a button disabled for good.
          const data = await res.json().catch(() => null);
          if (data && data.status === 'failure') throw new Error(data.message || 'failed');
        } catch (err) {
          btn.disabled = false;
          toast(`Could not apply "${btn.textContent}"`, 4000);
        }
      });
      item.appendChild(btn);
    }

    return item;
  }

  function render(notices) {
    lastNotices = Array.isArray(notices) ? notices : [];

    // Rebuilding throws away where the user had scrolled to; during an outage
    // that would yank the list back to the top every time a frame arrives.
    const scrollTop = container.querySelector('.notices-bar__list')?.scrollTop ?? 0;
    container.innerHTML = '';

    if (lastNotices.length === 0) {
      container.style.display = 'none';
      container.classList.remove('is-collapsed');
      return;
    }

    container.style.display = '';
    container.classList.toggle('is-collapsed', collapsed);
    container.appendChild(buildHeader(lastNotices));

    if (collapsed) return;

    const list = document.createElement('div');
    list.className = 'notices-bar__list';
    for (const notice of lastNotices) list.appendChild(buildItem(notice));
    container.appendChild(list);
    list.scrollTop = scrollTop;
  }

  let _source = null;
  let _destroyed = false;

  function _connect() {
    if (_destroyed) return;
    _source?.close();
    _source = new EventSource('/api/notices/stream');
    _source.onmessage = (e) => {
      try { render(JSON.parse(e.data)); } catch {}
    };
    // EventSource reconnects automatically on error — no explicit handler needed.
  }

  function _disconnect() {
    _source?.close();
    _source = null;
  }

  _connect();

  // This bar is built once per page and never torn down, so without a gate its
  // stream is the one connection every open tab holds forever — the cheapest of
  // the lot individually, and the most numerous. A notice is state, not an event:
  // each frame carries the whole current list, so reconnecting repaints the bar
  // correctly no matter what was published while the tab was away.
  const _stopVisibilityGate = suspendWhileHidden({ open: _connect, close: _disconnect });

  return {
    destroy() {
      _destroyed = true;
      _stopVisibilityGate();
      _disconnect();
    },
  };
}
