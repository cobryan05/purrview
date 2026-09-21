import { showUnsavedConfigChangesModal } from '/app-static/js/ui/utils/confirmModal.js';
import { initSplitResize } from '/app-static/js/ui/utils/splitResize.js';
import { installUnsavedNavGuard } from '/app-static/js/ui/utils/unsavedNavGuard.js';

/**
 * Run a list/editor callback from a DOM listener.
 *
 * Every screen passes an `async` `_showForm` as `onNew`/`onClick`, and a DOM
 * event listener cannot await its handler — so a rejection inside one (a params
 * shape the schema doesn't cover, a failed fetch) became an unobserved promise
 * and the editor pane just stayed blank with nothing in the console. Observe it
 * here, once, instead of asking each caller to remember a `.catch`.
 */
function _invoke(fn) {
  try {
    Promise.resolve(fn()).catch((err) => console.error('List editor callback failed:', err));
  } catch (err) {
    console.error('List editor callback failed:', err);
  }
}

/**
 * Assemble the list-beside-editor screen shared by Datasets, Model Shapes,
 * Training Runs, Filter Chains, Sources and Task Configs.
 *
 * All of them hand-built the identical structure — a `panel-box <prefix>-list-panel`
 * with `__bar`/`__title`/`__body`/`__footer` beside a `panel-box <prefix>-editor`
 * with `__bar`/`__body` (+ optional `__footer`) — and then re-declared the same
 * three pieces of editor bookkeeping on top of it: the `_barHandle` /
 * `_currentDoSave` pair, the `_confirmIfDirty()` prompt that reads them, and the
 * `installUnsavedNavGuard` call that reads them again.
 *
 * ## Why every element carries BOTH a `le-*` class and a `<prefix>-*` one
 *
 * The prefixes were originally the only hook, so each screen had to re-declare
 * the list-item padding/hover/active rules under its own name. Two screens
 * (`mc`, `ds`) did; `fc` never did, and its rows rendered as unstyled bare divs
 * with no hover and no selected state — a defect invisible in code review
 * because the JS looked identical to its neighbours'. The `le-*` classes carry
 * the shared appearance so a new screen is styled the moment it exists; the
 * prefix classes remain for the genuinely per-screen bits (column width, panel
 * max-height) and for existing selectors. Do not remove one in favour of the
 * other — they answer different questions.
 *
 * @param {Object} opts
 * @param {HTMLElement} opts.container - emptied by the caller; the split is appended to it.
 * @param {string} opts.prefix - class prefix, e.g. `'ds'` → `.ds-list-panel` / `.ds-editor`.
 *   Training Runs and Model Stacks deliberately pass `'mc'`: they share Model Shapes' CSS.
 * @param {string} opts.title - list panel title.
 * @param {string} opts.itemUuidAttr - the `dataset.*` key list rows carry their uuid in,
 *   e.g. `'datasetUuid'` — read by `updateListSelection` and set by `createListItem`.
 * @param {boolean} [opts.withEditorFooter=false] - add a sticky editor footer for
 *   always-visible action buttons. Only the screens that have footer actions ask for one.
 * @param {boolean} [opts.searchable=false] - add a filter box under the list bar.
 *   Filtering is pure DOM (hide non-matching `.le-list-item`s), so it needs no
 *   cooperation from the caller's `refreshList` beyond calling `applyFilter()`.
 * @param {string} [opts.searchPlaceholder='Filter…']
 * @param {string} [opts.newLabel='+ New'] - footer button text.
 * @param {() => void} opts.onNew - click handler for the footer's `+ New` button.
 * @param {string} opts.ownTabId - this screen's sub-tab id, for the nav guard.
 * @param {AbortSignal} opts.navGuardSignal - the caller's re-render controller signal.
 * @returns {Object} the assembled elements plus the editor-state bookkeeping:
 *   `setBarHandle` / `setDoSave` (assigned at different points in a form build, so
 *   two setters rather than one), `confirmIfDirty()`, `updateListSelection(uuid)`,
 *   `createListItem(...)`, `setCount(n)` and `applyFilter()`.
 */
export function createListEditorScreen({
  container,
  prefix,
  title,
  itemUuidAttr,
  withEditorFooter = false,
  searchable = false,
  searchPlaceholder = 'Filter…',
  newLabel = '+ New',
  onNew,
  ownTabId,
  navGuardSignal,
}) {
  const split = document.createElement('div');
  split.className = 'list-editor-split';
  container.appendChild(split);

  // ── Left: list panel ────────────────────────────────────────────────────
  const listPanel = document.createElement('div');
  listPanel.className = `panel-box le-list-panel ${prefix}-list-panel`;
  split.appendChild(listPanel);

  const listBar = document.createElement('div');
  listBar.className = 'panel-box__bar';
  const listTitle = document.createElement('span');
  listTitle.className = 'panel-box__title';
  listTitle.textContent = title;
  listBar.appendChild(listTitle);

  // Item count lives in the bar rather than as a row in the body: it must stay
  // readable when the body is scrolled, and it is the cheapest way to tell
  // "empty" from "still loading" at a glance.
  const countEl = document.createElement('span');
  countEl.className = 'le-list-count';
  listBar.appendChild(countEl);
  listPanel.appendChild(listBar);

  let searchInput = null;
  if (searchable) {
    const searchWrap = document.createElement('div');
    searchWrap.className = 'le-list-search';
    searchInput = document.createElement('input');
    searchInput.type = 'search';
    searchInput.placeholder = searchPlaceholder;
    searchInput.setAttribute('aria-label', searchPlaceholder);
    searchInput.addEventListener('input', () => applyFilter());
    searchWrap.appendChild(searchInput);
    listPanel.appendChild(searchWrap);
  }

  const listBody = document.createElement('div');
  listBody.className = 'panel-box__body le-list-body';
  listPanel.appendChild(listBody);

  const listFooter = document.createElement('div');
  listFooter.className = 'panel-box__footer';
  listPanel.appendChild(listFooter);

  // Footer "+ New" button is static — created once here, not on every refreshList.
  const newBtn = document.createElement('button');
  newBtn.type = 'button';
  newBtn.textContent = newLabel;
  newBtn.className = 'panel-btn panel-btn--block';
  // Tour anchor, keyed on the screen prefix so every list editor's "+ New" is
  // addressable without each caller opting in (`list.src.new`, `list.mc.new`, …).
  newBtn.dataset.tour = `list.${prefix}.new`;
  newBtn.addEventListener('click', () => _invoke(onNew));
  listFooter.appendChild(newBtn);

  // ── Drag handle ─────────────────────────────────────────────────────────
  // Item names are the one thing the list exists to show, and a fixed column
  // ellipsised most of them ("Infer Timelapse L…"). Width is per-screen: a
  // Sources list of short names wants less room than one of task-config names.
  const resizeHandle = document.createElement('div');
  resizeHandle.className = 'split-resize-handle le-resize-handle';
  resizeHandle.setAttribute('role', 'separator');
  resizeHandle.setAttribute('aria-orientation', 'vertical');
  resizeHandle.setAttribute('aria-label', 'Resize list panel');
  split.appendChild(resizeHandle);

  initSplitResize(resizeHandle, listPanel, {
    storageKey: `ww.listEditor.${prefix}.width`,
    defaultWidth: 260,
    min: 170,
    max: 560,
    side: 'left',
    // Screens re-render on every tab switch; without this the document-level
    // mousemove/mouseup listeners stack up one set per visit.
    signal: navGuardSignal,
  });

  // ── Right: editor panel ─────────────────────────────────────────────────
  const editorPanel = document.createElement('div');
  editorPanel.className = `panel-box le-editor ${prefix}-editor`;
  split.appendChild(editorPanel);

  const editorBar = document.createElement('div');
  editorBar.className = 'panel-box__bar';
  editorPanel.appendChild(editorBar);

  const editorBody = document.createElement('div');
  editorBody.className = 'panel-box__body le-editor-body';
  editorPanel.appendChild(editorBody);

  // Sticky footer: always-visible action buttons regardless of scroll position
  let editorFooter = null;
  if (withEditorFooter) {
    editorFooter = document.createElement('div');
    editorFooter.className = 'panel-box__footer le-editor-footer';
    editorPanel.appendChild(editorFooter);
  }

  // ── List helpers ────────────────────────────────────────────────────────

  /**
   * Build one list row in the shared grammar: name line, optional meta line,
   * optional chips. Callers previously hand-rolled this per screen, which is how
   * the class names drifted apart in the first place.
   *
   * @param {{
   *   uuid: string,
   *   name: string,
   *   meta?: string,
   *   chips?: Array<{ text: string, title?: string, variant?: string }>,
   *   onClick?: () => any,
   * }} opts
   * @returns {HTMLElement}
   */
  function createListItem({ uuid, name, meta = '', chips = [], onClick }) {
    const item = document.createElement('div');
    item.className = `le-list-item ${prefix}-list-item`;
    item.dataset[itemUuidAttr] = uuid;
    // Searched against by applyFilter(); kept lowercase so filtering is a plain
    // substring test rather than a per-keystroke walk of the row's textContent.
    item.dataset.filterText = `${name} ${meta}`.toLowerCase();

    const nameLine = document.createElement('div');
    nameLine.className = `le-list-item__name ${prefix}-list-item__name`;
    nameLine.textContent = name;
    nameLine.title = name;
    item.appendChild(nameLine);

    if (meta || chips.length) {
      const metaLine = document.createElement('div');
      metaLine.className = `le-list-item__meta ${prefix}-list-item__meta`;
      if (meta) {
        const metaText = document.createElement('span');
        metaText.textContent = meta;
        metaLine.appendChild(metaText);
      }
      for (const chip of chips) {
        const chipEl = document.createElement('span');
        chipEl.className = `le-chip${chip.variant ? ` le-chip--${chip.variant}` : ''}`;
        chipEl.textContent = chip.text;
        if (chip.title) chipEl.title = chip.title;
        metaLine.appendChild(chipEl);
      }
      item.appendChild(metaLine);
    }

    if (onClick) item.addEventListener('click', () => _invoke(onClick));
    return item;
  }

  /** Section divider inside the list body, for screens that group their rows. */
  function createListSection(label) {
    const header = document.createElement('div');
    header.className = 'le-list-section';
    header.textContent = label;
    return header;
  }

  /** The italic "nothing here yet" line, in the shared style. */
  function createListEmpty(text) {
    const empty = document.createElement('p');
    empty.className = 'le-list-empty';
    empty.textContent = text;
    return empty;
  }

  /**
   * Hide rows that don't match the filter box. Also hides a section header when
   * every row beneath it is hidden, so filtering never leaves a heading floating
   * over nothing.
   */
  function applyFilter() {
    const q = (searchInput?.value ?? '').trim().toLowerCase();
    let visible = 0;
    let currentSection = null;
    let sectionHasVisible = false;

    const flushSection = () => {
      if (currentSection) currentSection.hidden = !sectionHasVisible;
    };

    for (const node of listBody.children) {
      if (node.classList.contains('le-list-section')) {
        flushSection();
        currentSection = node;
        sectionHasVisible = false;
        continue;
      }
      if (!node.classList.contains('le-list-item')) continue;
      const match = !q || (node.dataset.filterText ?? '').includes(q);
      node.hidden = !match;
      if (match) { visible++; sectionHasVisible = true; }
    }
    flushSection();

    // An empty *result* is a different state from an empty *list* — say so,
    // otherwise a typo in the filter box reads as "my configs are gone".
    // ...but only when there is something to have filtered out. An empty list
    // already shows its own "No X yet." line; adding "No matches." under it
    // renders both at once.
    const hasItems = !!listBody.querySelector('.le-list-item');
    let noMatch = listBody.querySelector('.le-list-nomatch');
    if (q && visible === 0 && hasItems) {
      if (!noMatch) {
        noMatch = createListEmpty('No matches.');
        noMatch.classList.add('le-list-nomatch');
        listBody.appendChild(noMatch);
      }
      noMatch.hidden = false;
    } else if (noMatch) {
      noMatch.hidden = true;
    }
  }

  function setCount(n) {
    countEl.textContent = n === null || n === undefined ? '' : String(n);
  }

  // ── Editor state ────────────────────────────────────────────────────────
  // The active form's bar handle (dirty flag) and its save-only fn. Both are
  // nulled while a form's async fetches are in flight, hence the optional calls.
  let _barHandle = null;
  let _doSave = null;

  const isDirty = () => !!_barHandle?.getDirty();

  /** Prompt when switching away from a dirty form. False = caller should abort. */
  async function confirmIfDirty() {
    if (!isDirty()) return true;
    const result = await showUnsavedConfigChangesModal();
    if (result === 'cancel') return false;
    if (result === 'save') return await (_doSave?.() ?? false);
    return true; // 'discard'
  }

  function updateListSelection(uuid) {
    // Selects on the shared class so a screen that builds rows by hand (rather
    // than via createListItem) still gets selection, as long as it uses it.
    for (const item of listBody.querySelectorAll(`.le-list-item, .${prefix}-list-item`)) {
      item.classList.toggle('active', item.dataset[itemUuidAttr] === uuid);
    }
  }

  installUnsavedNavGuard({
    ownTabId,
    signal: navGuardSignal,
    isDirty,
    save: () => _doSave?.(),
    markClean: () => _barHandle?.setDirty(false),
  });

  return {
    split,
    listPanel,
    listBar,
    listTitle,
    listBody,
    listFooter,
    newBtn,
    searchInput,
    resizeHandle,
    editorPanel,
    editorBar,
    editorBody,
    editorFooter,
    createListItem,
    createListSection,
    createListEmpty,
    applyFilter,
    setCount,
    setBarHandle: (handle) => { _barHandle = handle; },
    setDoSave: (fn) => { _doSave = fn; },
    isDirty,
    confirmIfDirty,
    updateListSelection,
  };
}
