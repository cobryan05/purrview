import { LAYERS, nextModalZIndex, registerEscHandler, unregisterEscHandler } from '/app-static/js/ui/utils/modalZIndex.js';

/**
 * Floating context-menu engine.
 *
 * Exports:
 *   openMenu({ x, y, sections, onClose }) → { close }
 *   closeActiveMenu()
 *
 * A section is { title?, items: Item[] }.
 * An item is { label, icon?, danger?, disabled?, onSelect?(ctx), submenu?: () => Item[] | HTMLElement }.
 *
 * Only one menu can be open at a time.  Opening a new menu closes the existing one.
 *
 * Also exports:
 *   asyncPanel(minWidth, contentFn) → HTMLElement   — loading placeholder wrapper for async flyout content
 */

/** @type {{ close: () => void } | null} */
let _activeMenu = null;

/** Close the currently-open menu (no-op if none open). */
export function closeActiveMenu() {
  if (_activeMenu) {
    _activeMenu.close();
    _activeMenu = null;
  }
}

// ── Async panel helper ────────────────────────────────────────────────────────

/**
 * Build an async-loading wrapper div that will contain `contentFn()` once ready.
 * Used by providers to render flyout content that requires a fetch.
 *
 * @param {string} minWidth  CSS min-width value (e.g. '200px')
 * @param {() => Promise<HTMLElement>} contentFn
 * @returns {HTMLElement}
 */
export function asyncPanel(minWidth, contentFn) {
  const wrap = document.createElement('div');
  wrap.style.cssText = `min-width:${minWidth};padding:4px;`;
  wrap.textContent = 'Loading…';
  contentFn().then(el => {
    wrap.innerHTML = '';
    wrap.appendChild(el);
  }).catch(err => {
    wrap.textContent = 'Error loading';
    console.error('[contextMenu]', err);
  });
  return wrap;
}

// ── Styles ────────────────────────────────────────────────────────────────────

let _stylesInjected = false;

function _ensureStyles() {
  if (_stylesInjected) return;
  _stylesInjected = true;
  const style = document.createElement('style');
  style.id = 'ctx-menu-styles';
  style.textContent = `
    .ctx-menu {
      position: fixed;
      /* Overridden inline from the shared band on open; kept pointing at the
         token so this injected sheet can't drift out of the scale. */
      z-index: var(--layer-modal-base);
      min-width: 180px;
      background: var(--pico-card-background-color, #1a1a1a);
      border: 1px solid var(--pico-muted-border-color, #444);
      border-radius: var(--pico-border-radius, 4px);
      box-shadow: 0 4px 12px rgba(0,0,0,.4);
      padding: 4px 0;
      font-size: 0.875em;
      user-select: none;
      outline: none;
    }
    .ctx-menu-title {
      padding: 4px 12px 2px;
      color: var(--pico-muted-color, #aaa);
      font-size: 0.8em;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .ctx-menu-item {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      cursor: pointer;
      white-space: nowrap;
    }
    .ctx-menu-item:hover:not(.ctx-menu-item--disabled) {
      background: var(--pico-primary-background, rgba(255,255,255,0.08));
    }
    .ctx-menu-item--disabled {
      opacity: 0.45;
      cursor: default;
    }
    .ctx-menu-item--danger {
      color: var(--pico-del-color, #e05252);
    }
    .ctx-menu-sep {
      height: 1px;
      background: var(--pico-muted-border-color, #444);
      margin: 4px 0;
    }
    .ctx-submenu-arrow {
      margin-left: auto;
      padding-left: 8px;
      font-size: 0.75em;
      color: var(--pico-muted-color, #aaa);
    }
  `;
  document.head.appendChild(style);
}

// ── Item row builder ───────────────────────────────────────────────────────────

/**
 * Build item rows into `container`, sharing the same menu context.
 * Returns an array of `{ row, closeFlyout }` for sibling-flyout management.
 *
 * @param {HTMLElement} container
 * @param {Array<object>} items
 * @param {{ close: () => void }} menuCtx
 * @param {(el: HTMLElement) => void} registerEl   — register flyout in the all-elements list
 * @param {(el: HTMLElement) => void} unregisterEl — deregister flyout from the list
 * @returns {Array<{ row: HTMLElement, closeFlyout: () => void }>}
 */
function _buildItemRows(container, items, menuCtx, registerEl, unregisterEl) {
  const siblings = []; // { closeFlyout }

  const closeAllSiblings = () => siblings.forEach(s => s.closeFlyout());

  for (const item of items) {
    const row = document.createElement('div');
    row.className = 'ctx-menu-item';
    if (item.danger) row.classList.add('ctx-menu-item--danger');
    if (item.disabled) row.classList.add('ctx-menu-item--disabled');

    if (item.icon) {
      const ic = document.createElement('span');
      ic.textContent = item.icon;
      row.appendChild(ic);
    }
    const lbl = document.createElement('span');
    lbl.textContent = item.label;
    row.appendChild(lbl);

    let closeFlyout = () => {};

    if (!item.disabled) {
      if (item.submenu) {
        const arrow = document.createElement('span');
        arrow.className = 'ctx-submenu-arrow';
        arrow.textContent = '▸';
        row.appendChild(arrow);

        let _flyoutEl = null;
        let _closeTimer = null;
        let _innerSiblings = []; // siblings from within this flyout (for cascade close)

        const openFlyout = () => {
          clearTimeout(_closeTimer);
          closeAllSiblings(); // close any other open sibling flyout
          if (_flyoutEl) return; // already open

          const flyoutEl = document.createElement('div');
          flyoutEl.className = 'ctx-menu';
          // Sit one step above the menu this row belongs to, clamped so deep
          // submenu nesting can't walk out of the shared overlay band. The
          // value is threaded through menuCtx rather than read back off the
          // parent's inline style, so it survives .ctx-menu ever getting its
          // z-index from CSS.
          const _flyoutZ = Math.min(menuCtx.zIndex + 1, LAYERS.modalMax);
          flyoutEl.style.cssText = `position:fixed;z-index:${_flyoutZ};`;
          registerEl(flyoutEl);

          const sub = item.submenu();
          if (sub instanceof HTMLElement) {
            flyoutEl.appendChild(sub);
          } else if (Array.isArray(sub)) {
            _innerSiblings = _buildItemRows(flyoutEl, sub, menuCtx, registerEl, unregisterEl);
          }

          document.body.appendChild(flyoutEl);

          // Position flyout to the right of the row; flip left if overflow
          const rowRect = row.getBoundingClientRect();
          const fw = flyoutEl.offsetWidth;
          const fh = flyoutEl.offsetHeight;
          const ww = window.innerWidth;
          const wh = window.innerHeight;
          let fl = rowRect.right + 2;
          if (fl + fw > ww) fl = rowRect.left - fw - 2;
          const ft = Math.max(4, Math.min(rowRect.top, wh - fh - 4));
          flyoutEl.style.left = `${Math.max(0, fl)}px`;
          flyoutEl.style.top = `${ft}px`;

          _flyoutEl = flyoutEl;

          flyoutEl.addEventListener('mouseenter', () => clearTimeout(_closeTimer));
          flyoutEl.addEventListener('mouseleave', scheduledClose);
        };

        const closeFlyoutNow = () => {
          if (!_flyoutEl) return;
          // Don't auto-close if an input inside the flyout has focus (user is typing)
          if (_flyoutEl.contains(document.activeElement)) return;
          // Cascade: close any sub-flyouts opened from within this flyout first
          _innerSiblings.forEach(s => s.closeFlyout());
          _innerSiblings = [];
          unregisterEl(_flyoutEl);
          _flyoutEl.remove();
          _flyoutEl = null;
        };

        const scheduledClose = () => {
          _closeTimer = setTimeout(closeFlyoutNow, 150);
        };

        closeFlyout = closeFlyoutNow;

        row.addEventListener('mouseenter', openFlyout);
        row.addEventListener('click', openFlyout);
        row.addEventListener('mouseleave', scheduledClose);

      } else {
        // Non-submenu rows: close any open sibling flyout immediately on hover.
        // Submenu rows are covered by openFlyout → closeAllSiblings.
        row.addEventListener('mouseenter', closeAllSiblings);
        if (item.onSelect) {
          row.addEventListener('click', async () => {
            await item.onSelect(menuCtx);
            menuCtx.close();
          });
        }
      }
    }

    siblings.push({ row, closeFlyout });
    container.appendChild(row);
  }

  return siblings;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Open a floating context menu.
 *
 * @param {{
 *   x: number,
 *   y: number,
 *   sections: Array<{ title?: string, items: Array<object> }>,
 *   onClose?: () => void,
 * }} opts
 * @returns {{ close: () => void }}
 */
export function openMenu({ x, y, sections, onClose }) {
  closeActiveMenu();
  _ensureStyles();

  // Track all DOM elements (root + flyouts) for outside-click detection
  const _allEls = [];
  const registerEl = el => _allEls.push(el);
  const unregisterEl = el => {
    const i = _allEls.indexOf(el);
    if (i !== -1) _allEls.splice(i, 1);
  };

  let _closed = false;

  const close = () => {
    if (_closed) return;
    _closed = true;
    _allEls.forEach(el => el.parentNode && el.remove());
    _allEls.length = 0;
    document.removeEventListener('mousedown', _onMouseDown, true);
    unregisterEscHandler(close);
    _activeMenu = null;
    onClose?.();
  };

  // zIndex is threaded to flyouts through here (see _buildItemRows) so they
  // stack above their own menu without each hover burning a band slot.
  const menuCtx = { close, zIndex: nextModalZIndex() };

  const root = document.createElement('div');
  root.className = 'ctx-menu';
  root.style.cssText = `position:fixed;z-index:${menuCtx.zIndex};`;
  registerEl(root);

  let firstSection = true;
  for (const section of sections) {
    if (!firstSection) {
      const sep = document.createElement('div');
      sep.className = 'ctx-menu-sep';
      root.appendChild(sep);
    }
    firstSection = false;

    if (section.title) {
      const t = document.createElement('div');
      t.className = 'ctx-menu-title';
      t.textContent = section.title;
      root.appendChild(t);
    }

    _buildItemRows(root, section.items, menuCtx, registerEl, unregisterEl);
  }

  document.body.appendChild(root);

  // Clamp position after render (so offsetWidth/Height are available)
  const w = root.offsetWidth;
  const h = root.offsetHeight;
  const left = x + w > window.innerWidth ? x - w : x;
  const top = y + h > window.innerHeight ? y - h : y;
  root.style.left = `${Math.max(0, left)}px`;
  root.style.top = `${Math.max(0, top)}px`;

  const _onMouseDown = e => {
    const isInside = _allEls.some(el => el.contains(e.target));
    if (!isInside) close();
  };
  document.addEventListener('mousedown', _onMouseDown, true);
  // Shared ESC stack (PR 6.1) — a menu opened over a modal now closes before
  // the modal, instead of both racing on their own keydown listeners.
  registerEscHandler(close);

  _activeMenu = { close };
  return { close };
}
