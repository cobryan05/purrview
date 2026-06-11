/**
 * Notices Bar — a fixed strip above the active-model bar that surfaces
 * advisory messages from backend services (e.g. "hash index rebuilding").
 *
 * Mount once at app startup:
 *   import { mountNoticesBar } from '/app-static/js/ui/noticesBar.js';
 *   mountNoticesBar('notices-bar');
 *
 * Polls /api/notices every 5 seconds.  The bar is hidden when there are no
 * active notices and shown automatically when any arrive.
 *
 * Notice shape: { key: string, level: "info"|"warning"|"error", message: string }
 */

import { fetchNotices } from '/app-static/js/shared/api/notices.js';

const POLL_INTERVAL_MS = 5000;

const LEVEL_ICONS = {
  info: 'ℹ',
  warning: '⚠',
  error: '✕',
};

/**
 * @param {string} containerId  id of the mount <div>
 * @returns {{ destroy: () => void }}
 */
export function mountNoticesBar(containerId = 'notices-bar') {
  const container = document.getElementById(containerId);
  if (!container) return { destroy() {} };

  function render(notices) {
    container.innerHTML = '';

    if (!notices || notices.length === 0) {
      container.style.display = 'none';
      return;
    }

    container.style.display = '';
    for (const notice of notices) {
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

      container.appendChild(item);
    }
  }

  async function poll() {
    try {
      const { notices } = await fetchNotices();
      render(notices);
    } catch {
      // Silently ignore — notices are advisory only
    }
  }

  // Initial fetch, then poll on interval
  poll();
  const intervalId = setInterval(poll, POLL_INTERVAL_MS);

  return {
    destroy() {
      clearInterval(intervalId);
    },
  };
}
