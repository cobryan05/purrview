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
 *   items: Array<{uuid: string, name: string, color?: string|null, depth?: number,
 *                 badge?: {text: string, title?: string, className?: string}|null}>,
 *   defaultSelected?: string[],
 *   defaultMode?: 'any'|'all',
 *   showModeToggle?: boolean,    // default true
 *   allowExclude?: boolean,      // default true; false hides the per-row "≠" exclude toggle
 *                                // (selection widgets pick items, they don't build queries)
 *   emptyText?: string,          // placeholder shown when `items` is empty
 *   searchable?: boolean,        // default false; adds a type-to-filter box above the list
 *                                // (worth it for label-sized vocabularies, noise for 5 tags)
 *   emptySummary?: string,       // trigger text when nothing is selected (default 'All' —
 *                                // a filter matches everything; a picker should say 'None selected')
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
  allowExclude = true,
  searchable = false,
  emptyText = null,
  emptySummary = 'All',
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

  // ── Optional type-to-filter box ────────────────────────────────────────────
  // Filters which rows are *visible* only. Select All / Select None keep
  // operating on the whole list, so a stale search term can never silently
  // shrink what those buttons do.
  let searchInput = null;
  if (searchable) {
    searchInput = document.createElement('input');
    searchInput.type = 'search';
    searchInput.className = 'ms-dropdown__search';
    searchInput.placeholder = 'Filter…';
    searchInput.autocomplete = 'off';
    panelEl.appendChild(searchInput);
    searchInput.addEventListener('mousedown', (e) => e.stopPropagation());
  }

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

  // Placeholder for an empty list — without it the panel opens on nothing and
  // reads as broken rather than as "there is nothing to pick yet".
  const emptyEl = document.createElement('div');
  emptyEl.className = 'ms-dropdown__empty';
  emptyEl.textContent = emptyText ?? 'Nothing to select';
  emptyEl.style.cssText = 'font-size:0.8em;color:var(--pico-muted-color,#888);font-style:italic;padding:4px 8px;';

  // Only the placeholder toggles. The Select All / Select None row stays put:
  // it is a no-op on an empty list, but callers (and tests) drive it whether or
  // not items have loaded, and hiding it makes the panel jump as they arrive.
  function _syncEmpty() {
    emptyEl.style.display = checkboxMap.size > 0 ? 'none' : '';
  }

  itemsDiv.appendChild(emptyEl);

  /** @type {HTMLElement[]} group headings, hidden while a search term is active */
  const separatorEls = [];

  function _buildItem(item) {
    if (item.isSeparator) {
      const sep = document.createElement('div');
      separatorEls.push(sep);
      sep.className = 'ms-item-separator';
      sep.style.cssText = 'font-size:0.72em;font-weight:bold;text-transform:uppercase;letter-spacing:.06em;color:#666;padding:5px 8px 2px;user-select:none;pointer-events:none;';
      sep.textContent = item.label ?? '';
      itemsDiv.appendChild(sep);
      return;
    }

    const lbl = document.createElement('label');
    lbl.className = 'ms-item';
    // Indent child items using the depth field from flattenHierarchy
    const depth = item.depth ?? 0;
    if (depth > 0) {
      lbl.style.paddingLeft = `${0.35 + depth * 0.9}rem`;
      lbl.classList.add('ms-item--child');
    }

    // Checkbox first, then swatch: the click targets line up in one column
    // instead of alternating position with whether an item has a colour.
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = item.uuid;
    cb.checked = defaultSelected.includes(item.uuid);
    checkboxMap.set(item.uuid, cb);
    itemElMap.set(item.uuid, lbl);
    lbl.appendChild(cb);

    if (item.color) {
      const swatch = document.createElement('span');
      swatch.className = 'ms-item__swatch';
      swatch.style.background = item.color;
      lbl.appendChild(swatch);
    }

    const nameSpan = document.createElement('span');
    nameSpan.className = 'ms-item__name';
    nameSpan.textContent = ' ' + item.name;
    lbl.appendChild(nameSpan);

    // Optional trailing pill — a per-row annotation the caller supplies
    // (e.g. "in training run"). Purely informational: it never affects
    // selection, so it must not swallow the click that toggles the row.
    if (item.badge?.text) {
      const badgeEl = document.createElement('span');
      badgeEl.className = 'ms-item__badge' + (item.badge.className ? ` ${item.badge.className}` : '');
      badgeEl.textContent = item.badge.text;
      if (item.badge.title) badgeEl.title = item.badge.title;
      lbl.appendChild(badgeEl);
    }

    // Per-row "not" toggle: excludes this item from results. Query-building
    // only — a plain "pick some items" widget passes allowExclude: false.
    let notBtn = null;
    if (allowExclude) {
      notBtn = document.createElement('button');
      notBtn.type = 'button';
      notBtn.className = 'ms-item__not-btn';
      // Spelled out rather than "Exclude this item": the glyph is the only clue this
      // control exists, so its tooltip has to say what the filter will end up meaning.
      notBtn.title = `Exclude — match everything except ${item.name}`;
      notBtn.setAttribute('aria-label', notBtn.title);
      notBtn.textContent = '≠';
      notBtnMap.set(item.uuid, notBtn);
      lbl.appendChild(notBtn);
    }

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

    if (!notBtn) return;
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

  /** Live copy of `items` — addItem() appends here so the trigger summary can name late arrivals. */
  const _items = [...items];
  for (const item of _items) _buildItem(item);
  _syncEmpty();

  // A one-line legend for the ≠ buttons. The glyph alone is not self-explanatory —
  // users read it as decoration or as "equals" — and a row that can be negated is worth
  // nothing if nobody knows it can be. Only where excluding is actually offered.
  if (allowExclude) {
    const hint = document.createElement('div');
    hint.className = 'ms-dropdown__exclude-hint';
    hint.textContent = '≠ excludes an item';
    panelEl.appendChild(hint);
  }

  /** Hide rows that don't match the search box; no-op when not searchable. */
  function _applySearch() {
    if (!searchInput) return;
    const term = searchInput.value.trim().toLowerCase();
    let visible = 0;
    for (const item of _items) {
      if (item.isSeparator) continue;
      const el = itemElMap.get(item.uuid);
      if (!el) continue;
      const match = !term || item.name.toLowerCase().includes(term);
      el.style.display = match ? '' : 'none';
      if (match) visible++;
    }
    // Group headings only make sense over an unfiltered list.
    for (const sep of separatorEls) sep.style.display = term ? 'none' : '';
    emptyEl.style.display = visible === 0 ? '' : 'none';
    emptyEl.textContent = term && checkboxMap.size > 0
      ? 'No matches'
      : (emptyText ?? 'Nothing to select');
  }

  searchInput?.addEventListener('input', _applySearch);

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
      summary = emptySummary;
    } else if (_selected.length === 1 && excCount === 0) {
      const item = _items.find(i => i.uuid === _selected[0]);
      summary = item ? item.name : '1 selected';
    } else if (_selected.length === 0 && excCount === 1) {
      const excItem = _items.find(i => i.uuid === [..._excluded][0]);
      summary = excItem ? `not ${excItem.name}` : '1 excluded';
    } else {
      const parts = [];
      if (_selected.length > 0) parts.push(`${_selected.length} incl`);
      if (excCount > 0) parts.push(`${excCount} excl`);
      summary = parts.join(', ');
    }
    // An empty `label` means the caller already draws a heading above the
    // control (schema forms do) — don't repeat it inside the button.
    trigger.textContent = label ? `${label}: ${summary} ▾` : `${summary} ▾`;
    trigger.classList.toggle('ms-dropdown__trigger--active', _selected.length > 0 || excCount > 0);
  }

  function _syncExcluded() {
    for (const [uuid, lbl] of itemElMap) {
      const isExcluded = _excluded.has(uuid);
      lbl.classList.toggle('ms-item--excluded', isExcluded);
      const notBtn = notBtnMap.get(uuid);
      if (notBtn) {
        notBtn.classList.toggle('active', isExcluded);
        notBtn.setAttribute('aria-pressed', String(isExcluded));
      }
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
     * Append an item to the open list — for widgets that can create a new
     * option inline (e.g. the "+ New tag" button on the tags field). Pass
     * `select: true` to check it immediately; that fires onChange, since the
     * selection really did change.
     * @param {{uuid: string, name: string, color?: string|null}} item
     * @param {{ select?: boolean }} [opts]
     */
    addItem(item, { select = false } = {}) {
      if (checkboxMap.has(item.uuid)) return;
      _items.push(item);
      _buildItem(item);
      _syncEmpty();
      _applySearch();
      if (select) {
        const cb = checkboxMap.get(item.uuid);
        if (cb) cb.checked = true;
        _selected = [..._selected, item.uuid];
        _syncTrigger();
        onChange([..._selected], [..._excluded], _mode);
      }
    },

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
