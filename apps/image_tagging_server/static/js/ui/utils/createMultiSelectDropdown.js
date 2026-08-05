import { prefs } from '/app-static/js/shared/prefs.js';

/**
 * createMultiSelectDropdown — a compact trigger button that opens a dropdown
 * panel with optional "Any of / All of" mode toggle, an optional hierarchy
 * toggle, and a checkbox list with indentation for tree-shaped data.
 *
 * Reusable for any multi-select filter: tags, labels, datasets, etc.
 *
 * Usage:
 *   const dd = createMultiSelectDropdown({
 *     label: 'Labels',
 *     items: flattenedLabels,     // each item may have { depth } for indentation
 *     defaultSelected: [],
 *     defaultMode: 'any',
 *     hierarchyToggle: {
 *       label: 'Include child labels',
 *       default: false,
 *       onChange: (val) => { ... },
 *     },
 *     onChange: (selectedUuids, excludedUuids, mode) => { ... },
 *   });
 *   toolbar.appendChild(dd.element);
 *
 * @param {{
 *   label: string,
 *   items: Array<{uuid: string, name: string, color?: string|null, depth?: number}>,
 *   defaultSelected?: string[],
 *   defaultMode?: 'any'|'all',
 *   showModeToggle?: boolean,    // default true
 *   hierarchyToggle?: {          // optional "include descendants" toggle
 *     label?: string,            // checkbox label text
 *     default?: boolean,
 *     onChange: (val: boolean) => void,
 *   },
 *   onChange: (selected: string[], excluded: string[], mode: 'any'|'all') => void,
 *   filterKey?: string,          // data-filter-key attr + localStorage size key
 * }} opts
 *
 * @returns {{
 *   element: HTMLElement,
 *   getSelected: () => string[],
 *   getExcluded: () => string[],
 *   getMode: () => 'any'|'all',
 *   getHierarchyToggle: () => boolean,
 *   setState: (selected?: string[]|null, mode?: 'any'|'all'|null, hierarchyToggle?: boolean|null, excluded?: string[]|null) => void,
 *   destroy: () => void,
 * }}
 */
export function createMultiSelectDropdown({
  label,
  items,
  defaultSelected = [],
  defaultMode = 'any',
  showModeToggle = true,
  hierarchyToggle = null,
  onChange,
  filterKey,
  inline = false,
  defaultWidth = null,
  defaultHeight = null,
}) {
  let _selected = [...defaultSelected];
  let _excluded = new Set();
  let _mode = defaultMode;
  let _hierarchyToggle = hierarchyToggle?.default ?? false;
  let _open = false;
  /** @type {((e: MouseEvent) => void)|null} */
  let _outsideListener = null;
  /** @type {ResizeObserver|null} */
  let _resizeObserver = null;

  // In inline mode skip size persistence (panel is always visible, layout controls size).
  const _sizeKey = (!inline && filterKey) ? `ms-dropdown-size-${filterKey}` : null;

  // ── Wrapper (position:relative anchor for the panel) ──────────────────────
  const wrapper = document.createElement('div');
  wrapper.className = inline ? 'ms-dropdown ms-dropdown--inline' : 'ms-dropdown';
  if (filterKey) wrapper.dataset.filterKey = filterKey;

  // ── Trigger button (omitted in inline mode — panel is always visible) ──────
  const trigger = document.createElement('button');
  trigger.type = 'button';
  trigger.className = 'ms-dropdown__trigger';
  if (!inline) wrapper.appendChild(trigger);

  // ── Panel ──────────────────────────────────────────────────────────────────
  const panelEl = document.createElement('div');
  panelEl.className = 'ms-dropdown__panel';
  if (!inline) panelEl.style.display = 'none';
  wrapper.appendChild(panelEl);

  // Apply default size, then let localStorage override if the user has resized before.
  if (defaultWidth)  panelEl.style.width  = defaultWidth;
  if (defaultHeight) panelEl.style.height = defaultHeight;
  if (_sizeKey) {
    const saved = prefs.read(_sizeKey, null);
    if (saved?.width)  panelEl.style.width  = saved.width;
    if (saved?.height) panelEl.style.height = saved.height;
  }

  // ── Mode toggle (Any of / All of) ──────────────────────────────────────────
  let anyBtn, allBtn;
  if (showModeToggle) {
    const modeRow = document.createElement('div');
    modeRow.className = 'ms-mode-row';

    anyBtn = document.createElement('button');
    anyBtn.type = 'button';
    anyBtn.className = 'ms-mode-btn';
    anyBtn.dataset.mode = 'any';
    anyBtn.textContent = 'Any of';

    allBtn = document.createElement('button');
    allBtn.type = 'button';
    allBtn.className = 'ms-mode-btn';
    allBtn.dataset.mode = 'all';
    allBtn.textContent = 'All of';

    modeRow.append(anyBtn, allBtn);
    panelEl.appendChild(modeRow);

    anyBtn.addEventListener('mousedown', (e) => e.stopPropagation());
    anyBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      _mode = 'any';
      _syncMode();
      onChange([..._selected], [..._excluded], _mode);
    });

    allBtn.addEventListener('mousedown', (e) => e.stopPropagation());
    allBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      _mode = 'all';
      _syncMode();
      onChange([..._selected], [..._excluded], _mode);
    });
  }

  // ── Hierarchy toggle (optional) ────────────────────────────────────────────
  let _hierarchyCb = null;
  if (hierarchyToggle) {
    const htRow = document.createElement('label');
    htRow.className = 'ms-hierarchy-toggle';

    _hierarchyCb = document.createElement('input');
    _hierarchyCb.type = 'checkbox';
    _hierarchyCb.checked = _hierarchyToggle;

    htRow.appendChild(_hierarchyCb);
    htRow.appendChild(document.createTextNode(' ' + (hierarchyToggle.label ?? 'Include children')));
    panelEl.appendChild(htRow);

    htRow.addEventListener('mousedown', (e) => e.stopPropagation());
    _hierarchyCb.addEventListener('change', () => {
      _hierarchyToggle = _hierarchyCb.checked;
      _syncImplied();
      hierarchyToggle.onChange(_hierarchyToggle);
    });
  }

  // ── Select All / Select None ───────────────────────────────────────────────
  const selectAllRow = document.createElement('div');
  selectAllRow.className = 'ms-select-all-row';

  const selectAllBtn = document.createElement('button');
  selectAllBtn.type = 'button';
  selectAllBtn.className = 'ms-select-all-btn';
  selectAllBtn.textContent = 'Select All';

  const selectNoneBtn = document.createElement('button');
  selectNoneBtn.type = 'button';
  selectNoneBtn.className = 'ms-select-all-btn';
  selectNoneBtn.textContent = 'Select None';

  selectAllRow.append(selectAllBtn, selectNoneBtn);
  panelEl.appendChild(selectAllRow);

  selectAllBtn.addEventListener('mousedown', (e) => e.stopPropagation());
  selectAllBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    _excluded.clear();
    checkboxMap.forEach((cb) => { cb.checked = true; });
    _selected = [...checkboxMap.keys()];
    _syncTrigger();
    _syncExcluded();
    _syncImplied();
    onChange([..._selected], [..._excluded], _mode);
  });

  selectNoneBtn.addEventListener('mousedown', (e) => e.stopPropagation());
  selectNoneBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    _excluded.clear();
    checkboxMap.forEach((cb) => { cb.checked = false; });
    _selected = [];
    _syncTrigger();
    _syncExcluded();
    _syncImplied();
    onChange([..._selected], [..._excluded], _mode);
  });

  // ── Build parent→children map for "implied" highlight ─────────────────────
  // Items may carry a parentUuid field (e.g. label hierarchy).  We use it to
  // shade descendant checkboxes when the hierarchy toggle is on.
  const _childrenOf = new Map(); // parentUuid → uuid[]
  for (const item of items) {
    if (item.isSeparator) continue;
    if (item.parentUuid) {
      if (!_childrenOf.has(item.parentUuid)) _childrenOf.set(item.parentUuid, []);
      _childrenOf.get(item.parentUuid).push(item.uuid);
    }
  }
  const _hasHierarchy = _childrenOf.size > 0;

  // ── Item list (checkboxes with depth indentation) ──────────────────────────
  const itemsDiv = document.createElement('div');
  itemsDiv.className = 'ms-dropdown__items';
  panelEl.appendChild(itemsDiv);

  /** @type {Map<string, HTMLInputElement>} uuid → checkbox */
  const checkboxMap = new Map();
  /** @type {Map<string, HTMLButtonElement>} uuid → not-toggle button */
  const notBtnMap = new Map();
  /** @type {Map<string, HTMLLabelElement>} uuid → item label element */
  const itemElMap = new Map();

  for (const item of items) {
    if (item.isSeparator) {
      const sep = document.createElement('div');
      sep.className = 'ms-item-separator';
      sep.style.cssText = 'font-size:0.72em;font-weight:bold;text-transform:uppercase;letter-spacing:.06em;color:#666;padding:5px 8px 2px;user-select:none;pointer-events:none;';
      sep.textContent = item.label ?? '';
      itemsDiv.appendChild(sep);
      continue;
    }

    const lbl = document.createElement('label');
    lbl.className = 'ms-item';
    // Indent child items using the depth field from flattenHierarchy
    const depth = item.depth ?? 0;
    if (depth > 0) lbl.style.paddingLeft = `${0.4 + depth * 1.1}em`;

    if (item.color) {
      const swatch = document.createElement('span');
      swatch.className = 'ms-item__swatch';
      swatch.style.background = item.color;
      lbl.appendChild(swatch);
    }

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = item.uuid;
    cb.checked = defaultSelected.includes(item.uuid);
    checkboxMap.set(item.uuid, cb);
    itemElMap.set(item.uuid, lbl);
    lbl.appendChild(cb);

    const nameSpan = document.createElement('span');
    nameSpan.className = 'ms-item__name';
    nameSpan.textContent = ' ' + item.name;
    lbl.appendChild(nameSpan);

    // Per-row "not" toggle: excludes this item from results
    const notBtn = document.createElement('button');
    notBtn.type = 'button';
    notBtn.className = 'ms-item__not-btn';
    notBtn.title = 'Exclude this item';
    notBtn.textContent = '≠';
    notBtnMap.set(item.uuid, notBtn);
    lbl.appendChild(notBtn);

    itemsDiv.appendChild(lbl);

    // Prevent mousedown from closing panel before change fires
    lbl.addEventListener('mousedown', (e) => e.stopPropagation());
    cb.addEventListener('change', () => {
      if (cb.checked && _excluded.has(item.uuid)) {
        _excluded.delete(item.uuid);
      }
      _selected = [...checkboxMap.entries()]
        .filter(([, input]) => input.checked)
        .map(([uuid]) => uuid);
      _syncTrigger();
      _syncExcluded();
      _syncImplied();
      onChange([..._selected], [..._excluded], _mode);
    });

    notBtn.addEventListener('mousedown', (e) => e.stopPropagation());
    notBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (_excluded.has(item.uuid)) {
        _excluded.delete(item.uuid);
      } else {
        _excluded.add(item.uuid);
        // Uncheck include if it was on
        if (cb.checked) {
          cb.checked = false;
          _selected = _selected.filter(u => u !== item.uuid);
        }
      }
      _syncTrigger();
      _syncExcluded();
      _syncImplied();
      onChange([..._selected], [..._excluded], _mode);
    });
  }

  // ── Open / close ────────────────────────────────────────────────────────────
  function _openPanel() {
    _open = true;
    panelEl.style.display = '';

    // Wire up ResizeObserver to persist panel size when user drags the handle.
    if (_sizeKey && typeof ResizeObserver !== 'undefined' && !_resizeObserver) {
      let _saveTimer = null;
      _resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          clearTimeout(_saveTimer);
          _saveTimer = setTimeout(() => {
            const { width, height } = entry.contentRect;
            if (width > 0 && height > 0) {
              prefs.write(_sizeKey, {
                width: Math.round(width) + 'px',
                height: Math.round(height) + 'px',
              });
            }
          }, 150);
        }
      });
      _resizeObserver.observe(panelEl);
    }

    _outsideListener = (e) => {
      if (!wrapper.isConnected || !wrapper.contains(/** @type {Node} */(e.target))) {
        _closePanel();
      }
    };
    document.addEventListener('mousedown', _outsideListener);
  }

  function _closePanel() {
    _open = false;
    panelEl.style.display = 'none';
    if (_outsideListener) {
      document.removeEventListener('mousedown', _outsideListener);
      _outsideListener = null;
    }
  }

  trigger.addEventListener('mousedown', (e) => {
    e.preventDefault();
    e.stopPropagation();
    _open ? _closePanel() : _openPanel();
  });

  // ── Sync helpers ────────────────────────────────────────────────────────────
  function _syncMode() {
    if (!showModeToggle) return;
    anyBtn.classList.toggle('active', _mode === 'any');
    allBtn.classList.toggle('active', _mode === 'all');
  }

  function _syncTrigger() {
    const excCount = _excluded.size;
    let summary;
    if (_selected.length === 0 && excCount === 0) {
      summary = 'All';
    } else if (_selected.length === 1 && excCount === 0) {
      const item = items.find(i => i.uuid === _selected[0]);
      summary = item ? item.name : '1 selected';
    } else if (_selected.length === 0 && excCount === 1) {
      const excItem = items.find(i => i.uuid === [..._excluded][0]);
      summary = excItem ? `not ${excItem.name}` : '1 excluded';
    } else {
      const parts = [];
      if (_selected.length > 0) parts.push(`${_selected.length} incl`);
      if (excCount > 0) parts.push(`${excCount} excl`);
      summary = parts.join(', ');
    }
    trigger.textContent = `${label}: ${summary} ▾`;
    trigger.classList.toggle('ms-dropdown__trigger--active', _selected.length > 0 || excCount > 0);
  }

  function _syncExcluded() {
    for (const [uuid, lbl] of itemElMap) {
      const isExcluded = _excluded.has(uuid);
      lbl.classList.toggle('ms-item--excluded', isExcluded);
      const notBtn = notBtnMap.get(uuid);
      if (notBtn) notBtn.classList.toggle('active', isExcluded);
    }
  }

  /**
   * Mark items that are implied (descendants of selected items) when the
   * hierarchy toggle is on.  Adds/removes the `ms-item--implied` CSS class.
   */
  function _syncImplied() {
    if (!_hasHierarchy || !hierarchyToggle) return;
    if (!_hierarchyToggle) {
      // Toggle off — remove all implied styling
      for (const el of itemElMap.values()) el.classList.remove('ms-item--implied');
      return;
    }
    // BFS: collect all descendants of selected items
    const selectedSet = new Set(_selected);
    const implied = new Set();
    const queue = [..._selected];
    while (queue.length > 0) {
      const uuid = queue.shift();
      for (const childUuid of (_childrenOf.get(uuid) ?? [])) {
        // Only mark as implied if NOT explicitly selected
        if (!selectedSet.has(childUuid) && !implied.has(childUuid)) {
          implied.add(childUuid);
          queue.push(childUuid);
        }
      }
    }
    for (const [uuid, el] of itemElMap) {
      el.classList.toggle('ms-item--implied', implied.has(uuid));
    }
  }

  _syncMode();
  _syncTrigger();
  _syncExcluded();
  _syncImplied();

  // ── Public API ──────────────────────────────────────────────────────────────
  return {
    element: wrapper,
    getSelected: () => [..._selected],
    getExcluded: () => [..._excluded],
    getMode: () => _mode,
    getHierarchyToggle: () => _hierarchyToggle,

    /**
     * Programmatically update selection, mode, hierarchy toggle, and/or
     * exclusions without firing onChange. Pass null/undefined to leave a
     * field unchanged.
     */
    setState(selected, mode, includeDescendants, excluded) {
      if (selected != null) {
        _selected = [...selected];
        checkboxMap.forEach((cb, uuid) => { cb.checked = _selected.includes(uuid); });
      }
      if (excluded != null) {
        _excluded = new Set(excluded);
        // Clear include for anything now excluded
        for (const uuid of _excluded) {
          const cb = checkboxMap.get(uuid);
          if (cb?.checked) {
            cb.checked = false;
            _selected = _selected.filter(u => u !== uuid);
          }
        }
      }
      if (mode != null) {
        _mode = mode;
        _syncMode();
      }
      if (includeDescendants != null && _hierarchyCb) {
        _hierarchyToggle = includeDescendants;
        _hierarchyCb.checked = _hierarchyToggle;
      }
      _syncTrigger();
      _syncExcluded();
      _syncImplied();
    },

    /** Disconnect observers and remove listeners. Call when removing from DOM. */
    destroy() {
      if (!inline) _closePanel();
      if (_resizeObserver) {
        _resizeObserver.disconnect();
        _resizeObserver = null;
      }
    },
  };
}
