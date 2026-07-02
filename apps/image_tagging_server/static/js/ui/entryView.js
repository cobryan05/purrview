/**
 * Presentation-only shared scaffold for grid/list entry views.
 *
 * Owns the grid/list toggle, layout containers, view-mode persistence
 * (localStorage), and active-entry scroll/flash. Knows nothing about file
 * paths, DB entities, pagination, or selection — pages supply `renderRow` /
 * `renderCell` render-slot callbacks for their own entries and own all data
 * fetching, filtering, and actions.
 *
 * @param {{
 *   storageKey: string,
 *   defaultView?: 'list' | 'grid',
 *   showToggle?: boolean,
 *   listClass?: string,
 *   gridClass?: string,
 *   renderRow: (entry: any, index: number) => HTMLElement,
 *   renderCell: (entry: any, index: number) => HTMLElement,
 *   renderLeading?: (view: 'list' | 'grid') => HTMLElement | null,
 *   onToggle?: (view: 'list' | 'grid') => void,
 * }} opts
 * @returns {{
 *   element: HTMLElement,
 *   layout: HTMLElement,
 *   setEntries(entries: any[]): void,
 *   rerender(): void,
 *   getView(): 'list' | 'grid',
 *   setView(view: 'list' | 'grid'): void,
 *   scrollToActive(activeSelector: string): void,
 *   destroy(): void,
 * }}
 */
export function createEntryView({
  storageKey,
  defaultView = 'list',
  showToggle = true,
  listClass = 'entry-list',
  gridClass = 'entry-grid',
  renderRow,
  renderCell,
  renderLeading,
  onToggle,
  onBeforeRender,
}) {
  const stored = localStorage.getItem(storageKey);
  let _view = (stored === 'list' || stored === 'grid') ? stored : defaultView;
  let _entries = [];
  let _emptyText = '';

  const element = document.createElement('div');
  let listBtn = null;
  let gridBtn = null;

  if (showToggle) {
    element.className = 'fb-view-toggle';

    listBtn = document.createElement('button');
    listBtn.type = 'button';
    listBtn.textContent = '☰';
    listBtn.title = 'List view';
    listBtn.addEventListener('click', () => setView('list'));

    gridBtn = document.createElement('button');
    gridBtn.type = 'button';
    gridBtn.textContent = '⊞';
    gridBtn.title = 'Grid view';
    gridBtn.addEventListener('click', () => setView('grid'));

    element.append(listBtn, gridBtn);
    _updateToggleActive();
  }

  const layout = document.createElement('div');

  function _updateToggleActive() {
    if (!listBtn || !gridBtn) return;
    listBtn.classList.toggle('active', _view === 'list');
    gridBtn.classList.toggle('active', _view === 'grid');
  }

  function getView() {
    return _view;
  }

  function setView(view) {
    if (view !== 'list' && view !== 'grid') return;
    _view = view;
    localStorage.setItem(storageKey, view);
    _updateToggleActive();
    rerender();
    onToggle?.(view);
  }

  /**
   * Replace the full entry list and re-render. Accepts an optional emptyText to
   * avoid the ordering constraint of a separate setEmptyText() call.
   * @param {any[]} entries
   * @param {string} [emptyText] - if provided, replaces the current empty-state text
   */
  function setEntries(entries, emptyText = undefined) {
    _entries = entries;
    if (emptyText !== undefined) _emptyText = emptyText;
    rerender();
  }

  /**
   * Set the empty-state message shown when entries is []. Triggers a re-render
   * so callers can call setEmptyText in any order relative to setEntries.
   * @param {string} text
   */
  function setEmptyText(text) {
    _emptyText = text;
    if (_entries.length === 0) rerender();
  }

  /**
   * Append new entries to the current list without a full DOM rebuild.
   * Falls back to a full rerender if the layout has not been rendered yet.
   * @param {any[]} newEntries
   */
  function appendEntries(newEntries) {
    if (!newEntries.length) return;
    const wrapper = layout.firstElementChild;
    if (!wrapper) { _entries = newEntries; rerender(); return; }
    const startIndex = _entries.length;
    _entries = [..._entries, ...newEntries];
    const renderEntry = _view === 'grid' ? renderCell : renderRow;
    if (renderEntry) {
      newEntries.forEach((entry, i) => wrapper.appendChild(renderEntry(entry, startIndex + i)));
    }
    // Remove empty-state paragraph if it was showing (shouldn't normally happen, but be safe)
    const emptyP = wrapper.querySelector('p');
    if (emptyP && _entries.length > 0) emptyP.remove();
  }

  function rerender() {
    // Full rebuild ahead — give the owner a chance to release per-entry
    // resources (event subscriptions, observers) created by renderCell/
    // renderRow, or a view toggle duplicates them.
    onBeforeRender?.();
    layout.innerHTML = '';
    const wrapper = document.createElement(_view === 'grid' ? 'div' : 'ul');
    wrapper.className = _view === 'grid' ? gridClass : listClass;

    const leading = renderLeading?.(_view);
    if (leading) wrapper.appendChild(leading);

    const renderEntry = _view === 'grid' ? renderCell : renderRow;
    if (renderEntry) {
      _entries.forEach((entry, i) => wrapper.appendChild(renderEntry(entry, i)));
    }
    if (_entries.length === 0 && _emptyText) {
      const p = document.createElement('p');
      p.textContent = _emptyText;
      p.style.color = '#aaa';
      wrapper.appendChild(p);
    }

    layout.appendChild(wrapper);
  }

  /**
   * Scroll the active entry into view and briefly flash it.
   * @param {string} activeSelector - CSS selector matching the active entry's
   *   element, e.g. `.fb-entry--active, .fb-grid-cell--active`.
   */
  function scrollToActive(activeSelector) {
    const activeEl = layout.querySelector(activeSelector);
    if (!activeEl) return;
    activeEl.scrollIntoView({ block: 'center', behavior: 'smooth' });
    const activeClass = [...activeEl.classList].find(c => c.endsWith('--active'));
    if (!activeClass) return;
    const flashClass = activeClass.replace('--active', '--flash');
    // Remove then re-add the flash class so re-revealing the same entry re-triggers the animation.
    activeEl.classList.remove(flashClass);
    void activeEl.getBoundingClientRect(); // force reflow to restart animation
    activeEl.classList.add(flashClass);
  }

  function destroy() {
    element.innerHTML = '';
    layout.innerHTML = '';
    _entries = [];
  }

  return { element, layout, setEntries, setEmptyText, appendEntries, rerender, getView, setView, scrollToActive, destroy };
}
