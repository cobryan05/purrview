const DEBUG_ENABLED = true;
const WARN_ENABLED = true;
const ERROR_ENABLED = true;

export const Logger = {
  debug(...args) {
    if (DEBUG_ENABLED) {
      console.log('[DEBUG]', ...args);
    }
  },

  warn(...args) {
    if (WARN_ENABLED) {
      console.warn('[WARN]', ...args);
    }
  },

  notify(...args) {
    console.log('[NOTIFY]', ...args);
    if (args.length > 0) {
      toast(args.map(String).join(' '), 5000);
    }
  },

  error(...args) {
    if (!ERROR_ENABLED) {
      return;
    }
    args.forEach(arg => {
      if (arg instanceof Error) {
        console.error('[ERROR]', arg);
        // Print detail array if present
        const details = arg.data?.detail || arg.detail;
        if (details) console.error('[ERROR DETAILS]', details);
      } else {
        console.error('[ERROR]', arg);
      }
    });

    // Concise toast: show only messages
    const toastMessage = args
      .map(arg => arg instanceof Error ? arg.message : String(arg))
      .join(' ');

    if (toastMessage) {
      toast(toastMessage, 5000, 'error');
    }
  }
};


/** Toast kinds that carry a colour. Anything else renders as the neutral default. */
const TOAST_TYPES = new Set(['info', 'success', 'warning', 'error']);

/**
 * Transient message at the top of the screen.
 *
 * @param {string} message
 * @param {number|string} [duration=5000] - ms to stay up. **Also accepts a type
 *   string** for callers written as `toast(msg, 'error')` — see below.
 * @param {string} [type='info'] - 'info' | 'success' | 'warning' | 'error'
 * @param {number} [maxToasts=3]
 *
 * The `(message, duration, type)` order is easy to get wrong, and ~7 call sites
 * did: `toast('Failed to save', 'error')` put the type where the duration goes.
 * That failed twice over, silently. The toast rendered neutral grey instead of
 * red, because `type` kept its default; and `setTimeout(fn, 'error')` coerces
 * to NaN, which the spec clamps to 0 — so error messages appeared and vanished
 * in the same frame, which is worse than not showing them at all.
 *
 * Rather than only fixing the call sites and waiting for the next one, a string
 * in the duration slot is now read as the type. Callers stay free to pass
 * `(msg, 8000, 'error')` when they want a specific duration.
 */
export function toast(message, duration = 5000, type = "info", maxToasts = 3) {
  if (typeof duration === 'string') {
    // Only a recognised kind is treated as a misplaced type; an unrecognised
    // string is a bug worth surfacing rather than silently swallowing.
    if (TOAST_TYPES.has(duration)) {
      type = duration;
    } else {
      Logger.warn(`toast: ignoring unrecognised duration/type "${duration}"`);
    }
    duration = 5000;
  }
  if (typeof duration !== 'number' || !Number.isFinite(duration) || duration <= 0) {
    duration = 5000;
  }
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    Object.assign(container.style, {
      position: 'fixed',
      top: '20px',
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 100000,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      pointerEvents: 'none',
    });
    document.body.appendChild(container);
  }

  // Enforce max toast limit: remove oldest immediately
  while (container.children.length >= maxToasts) {
    container.firstElementChild?.remove();
  }

  const toast = document.createElement('div');
  toast.textContent = message;
  let background = '#555';
  if (type === 'error') {
    background = '#c0392b';
  } else if (type === 'warning') {
    background = '#b8620a';
  } else if (type === 'success') {
    // Call sites have been asking for 'success' all along; it had no colour, so
    // "Saved" and "Failed to save" were the same shade of grey.
    background = '#1e7a4a';
  }
  Object.assign(toast.style, {
    background,
    color: '#fff',
    padding: '8px 16px',
    marginTop: '6px',
    borderRadius: '6px',
    boxShadow: '0 2px 6px rgba(0,0,0,0.3)',
    fontSize: '14px',
    opacity: '0',
    transition: 'opacity 0.3s ease',
    pointerEvents: 'auto',
  });

  container.appendChild(toast);

  // Animate in
  requestAnimationFrame(() => {
    toast.style.opacity = '1';
  });

  // Remove after duration
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.addEventListener('transitionend', () => {
      toast.remove();
    }, { once: true });
  }, duration);
}

/**
 * A persistent, dismissible notice for content that needs to stay up until the
 * user acts on it — unlike toast(), which always fades after `duration` and only
 * ever holds plain text. Shares toast()'s fixed top-of-screen stack so the two
 * never overlap, but does not auto-remove itself.
 *
 * `buildContent(el, close)` populates `el` (already styled and attached) and
 * gets `close` to wire up its own dismiss/apply actions. Returns `close` too,
 * for a caller that wants to remove the notice programmatically (e.g. because
 * the thing it was about got deleted out from under it).
 */
export function actionNotice(buildContent) {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    Object.assign(container.style, {
      position: 'fixed',
      top: '20px',
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 100000,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      pointerEvents: 'none',
    });
    document.body.appendChild(container);
  }

  const notice = document.createElement('div');
  notice.dataset.testid = 'action-notice';
  Object.assign(notice.style, {
    background: '#b8620a',
    color: '#fff',
    padding: '10px 14px',
    marginTop: '6px',
    borderRadius: '6px',
    boxShadow: '0 2px 6px rgba(0,0,0,0.3)',
    fontSize: '14px',
    pointerEvents: 'auto',
    maxWidth: '420px',
  });
  container.appendChild(notice);

  const close = () => notice.remove();
  buildContent(notice, close);
  return close;
}

export function showPaneError(el, message) {
  el.innerHTML = '';
  const div = document.createElement('div');
  div.style.cssText = 'padding:1em; color:#c0392b; font-size:0.9em;';
  div.textContent = `⚠ ${message || 'Server is not responding'}`;
  el.appendChild(div);
}
