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


export function toast(message, duration = 5000, type = "info", maxToasts = 3) {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    Object.assign(container.style, {
      position: 'fixed',
      top: '20px',
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 1000,
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
  let background = '#333';
  if (type === 'error') {
    background = '#c0392b'; // reddish background for errors
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
