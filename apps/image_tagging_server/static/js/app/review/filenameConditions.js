/**
 * createFilenameConditions(onChange, { matchMode })
 *
 * An editable list of filename filters for the Review filter panel. Each
 * condition is `{ pattern: string, negate: boolean }` — a case-insensitive
 * substring test against the image's FILES_ROOT-relative path, negated to
 * exclude instead of include.
 *
 * Why a list rather than the single input this replaced: the panel could only
 * ever hold one filename filter, so "add filter" a second time re-edited the
 * one that was already there. Excluding two cameras, or asking for either of
 * two, was not expressible at all.
 *
 * Include conditions are combined by `matchMode` ('any' — the default — is an
 * OR; 'all' requires every pattern in the same path); excludes always all
 * apply. The toggle only appears once there are two or more includes, since
 * with fewer the two modes are identical. See buildFilenameTerms in
 * shared/api/review.js for how this maps onto the server's filter terms.
 *
 * onChange is called with `{ conditions, matchMode }` on every change. Text
 * edits are debounced (the same 400ms as the input this replaced) so typing a
 * path doesn't re-run the search per keystroke.
 *
 * Returns: { element, setState({conditions, matchMode}), getState(), destroy() }
 */

const DEBOUNCE_MS = 400;

export function createFilenameConditions(onChange, { matchMode = 'any' } = {}) {
  /** @type {{pattern: string, negate: boolean}[]} */
  let _conditions = [];
  let _matchMode = matchMode;
  let _debounce = null;

  const root = document.createElement('div');
  root.className = 'itc-root fnc-root';
  root.dataset.testid = 'filename-conditions';

  const list = document.createElement('div');
  list.className = 'itc-list';
  root.appendChild(list);

  const addRow = document.createElement('div');
  addRow.className = 'itc-add-row fnc-add-row';
  root.appendChild(addRow);

  const addBtn = document.createElement('button');
  addBtn.type = 'button';
  addBtn.className = 'fnc-add';
  addBtn.dataset.testid = 'filename-condition-add';
  addBtn.textContent = '+ File filter';
  addBtn.title = 'Add another filename filter';
  addRow.appendChild(addBtn);

  // Any/All applies to the include conditions only — see the module comment.
  const matchModeSelect = document.createElement('select');
  matchModeSelect.className = 'itc-state-select fnc-match-mode';
  matchModeSelect.dataset.testid = 'filename-match-mode';
  matchModeSelect.title = 'How to combine the included filename patterns';
  for (const [value, text] of [['any', 'Match any'], ['all', 'Match all']]) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = text;
    matchModeSelect.appendChild(opt);
  }
  matchModeSelect.value = _matchMode;
  matchModeSelect.addEventListener('change', () => {
    _matchMode = matchModeSelect.value;
    _emit();
  });
  addRow.appendChild(matchModeSelect);

  function _emit() {
    onChange({ conditions: _conditions.map(c => ({ ...c })), matchMode: _matchMode });
  }

  function _flushPending() {
    if (_debounce === null) return;
    clearTimeout(_debounce);
    _debounce = null;
  }

  // Shown only once two includes exist; recomputed on every edit, including
  // debounced typing, so a row that only just became non-empty counts.
  function _syncMatchModeVisibility() {
    const includes = _conditions.filter(c => !c.negate && c.pattern.trim()).length;
    matchModeSelect.hidden = includes < 2;
  }

  function _renderRow(cond, idx) {
    const row = document.createElement('div');
    row.className = 'itc-row fnc-row';

    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'fnc-input';
    input.placeholder = 'substring…';
    input.value = cond.pattern;
    input.dataset.testid = 'filename-condition-input';
    input.title = 'Match images whose path contains this text (case-insensitive)';
    input.addEventListener('input', () => {
      // Mutated immediately but emitted on the debounce: getState() is read
      // synchronously by callers that aren't driven by onChange.
      _conditions[idx] = { ..._conditions[idx], pattern: input.value };
      _syncMatchModeVisibility();
      _flushPending();
      _debounce = setTimeout(() => {
        _debounce = null;
        _emit();
      }, DEBOUNCE_MS);
    });
    row.appendChild(input);

    const negateLabel = document.createElement('label');
    negateLabel.className = 'itc-negate-label';
    negateLabel.title = 'Exclude matches (path does NOT contain this)';
    const negateCb = document.createElement('input');
    negateCb.type = 'checkbox';
    negateCb.checked = !!cond.negate;
    negateCb.dataset.testid = 'filename-condition-negate';
    negateCb.addEventListener('change', () => {
      _conditions[idx] = { ..._conditions[idx], negate: negateCb.checked };
      _syncMatchModeVisibility();
      _flushPending();
      _emit();
    });
    negateLabel.append(negateCb, document.createTextNode('¬'));
    row.appendChild(negateLabel);

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'itc-remove';
    removeBtn.textContent = '×';
    removeBtn.title = 'Remove this filename filter';
    removeBtn.dataset.testid = 'filename-condition-remove';
    removeBtn.addEventListener('click', () => {
      const wasActive = !!_conditions[idx].pattern.trim();
      _conditions.splice(idx, 1);
      _flushPending();
      _renderAll();
      // Removing a row that was still blank changes nothing the search can
      // see, so don't re-run it.
      if (wasActive) _emit();
    });
    row.appendChild(removeBtn);

    return row;
  }

  // Rebuilds every row, so it must NOT be called from the text input handler —
  // that would drop focus mid-word. Add/remove/setState only.
  function _renderAll() {
    list.innerHTML = '';
    _conditions.forEach((cond, idx) => list.appendChild(_renderRow(cond, idx)));
    _syncMatchModeVisibility();
    matchModeSelect.value = _matchMode;
  }

  addBtn.addEventListener('click', () => {
    _conditions.push({ pattern: '', negate: false });
    _renderAll();
    // A blank new row is not a filter yet — no emit; focus it so the user can
    // type straight away.
    list.lastElementChild?.querySelector('.fnc-input')?.focus();
  });

  function setState({ conditions, matchMode: mode } = {}) {
    if (conditions !== undefined) {
      _conditions = (conditions ?? []).map(c => ({
        pattern: c?.pattern ?? '',
        negate: !!c?.negate,
      }));
    }
    if (mode !== undefined) _matchMode = mode ?? 'any';
    _flushPending();
    _renderAll();
  }

  function getState() {
    return { conditions: _conditions.map(c => ({ ...c })), matchMode: _matchMode };
  }

  function destroy() {
    _flushPending();
    root.remove();
  }

  _renderAll();

  return { element: root, setState, getState, destroy };
}
