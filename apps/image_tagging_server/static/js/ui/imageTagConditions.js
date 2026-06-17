/**
 * createImageTagConditions(tags, onChange, { allowUnset = true } = {})
 *
 * Per-tag condition list for filtering images by whole-image tag state.
 * Each condition: { tagUuid: string, state: 'present' | 'absent' | 'unset' }
 *
 * @param {Iterable} tags          - Visible tag objects with .uuid, .name, .color
 * @param {Function} onChange      - Called with conditions[] on every change
 * @param {boolean}  allowUnset    - Include the "Unset" state option (default true;
 *                                   disable for dataset editor where unset has no meaning)
 *
 * Returns: { element, setState(conditions), getState() → conditions,
 *            setAvailableTags(tags), destroy() }
 */
export function createImageTagConditions(tags, onChange, { allowUnset = true } = {}) {
  let _conditions = [];      // [{ tagUuid, state }]
  let _tags = [...tags];     // available visible tag objects

  const root = document.createElement('div');
  root.className = 'itc-root';

  const list = document.createElement('div');
  list.className = 'itc-list';
  root.appendChild(list);

  const addRow = document.createElement('div');
  addRow.className = 'itc-add-row';
  root.appendChild(addRow);

  // ── Helpers ──────────────────────────────────────────────────────────────

  function _getTag(uuid) {
    return _tags.find(t => t.uuid === uuid) ?? null;
  }

  function _makeStateSelect(currentState) {
    const sel = document.createElement('select');
    sel.className = 'itc-state-select';
    const opts = [
      ['present', 'Present'],
      ['absent',  'Absent'],
      ...(allowUnset ? [['unset', 'Unset']] : []),
    ];
    for (const [value, label] of opts) {
      const opt = document.createElement('option');
      opt.value = value;
      opt.textContent = label;
      if (value === currentState) opt.selected = true;
      sel.appendChild(opt);
    }
    return sel;
  }

  // ── Render ────────────────────────────────────────────────────────────────

  function _renderRow(cond, idx) {
    const row = document.createElement('div');
    row.className = 'itc-row';

    const tag = _getTag(cond.tagUuid);

    // Color swatch
    const swatch = document.createElement('span');
    swatch.className = 'itc-swatch';
    swatch.style.background = tag?.color || '#555';
    row.appendChild(swatch);

    // Tag name
    const nameEl = document.createElement('span');
    nameEl.className = 'itc-name';
    nameEl.textContent = tag?.name ?? cond.tagUuid;
    nameEl.title = tag?.name ?? cond.tagUuid;
    row.appendChild(nameEl);

    // State selector
    const stateSelect = _makeStateSelect(cond.state);
    stateSelect.addEventListener('change', () => {
      _conditions[idx] = { ..._conditions[idx], state: stateSelect.value };
      onChange([..._conditions]);
    });
    row.appendChild(stateSelect);

    // Remove button
    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'itc-remove';
    removeBtn.textContent = '×';
    removeBtn.title = 'Remove filter';
    removeBtn.addEventListener('click', () => {
      _conditions.splice(idx, 1);
      _renderAll();
      onChange([..._conditions]);
    });
    row.appendChild(removeBtn);

    return row;
  }

  function _renderPicker() {
    addRow.innerHTML = '';
    const usedUuids = new Set(_conditions.map(c => c.tagUuid));
    const available = _tags
      .filter(t => !usedUuids.has(t.uuid))
      .sort((a, b) => a.name.localeCompare(b.name));

    const picker = document.createElement('select');
    picker.className = 'itc-picker';
    picker.disabled = available.length === 0;

    const ph = document.createElement('option');
    ph.value = '';
    ph.textContent = available.length > 0
      ? 'Add image tag filter…'
      : (_tags.length === 0 ? 'No tags defined' : 'All tags added');
    picker.appendChild(ph);

    for (const tag of available) {
      const opt = document.createElement('option');
      opt.value = tag.uuid;
      opt.textContent = tag.name;
      picker.appendChild(opt);
    }

    picker.addEventListener('change', () => {
      const uuid = picker.value;
      if (!uuid) return;
      _conditions.push({ tagUuid: uuid, state: 'present' });
      _renderAll();
      onChange([..._conditions]);
    });

    addRow.appendChild(picker);
  }

  function _renderAll() {
    list.innerHTML = '';
    _conditions.forEach((cond, idx) => list.appendChild(_renderRow(cond, idx)));
    _renderPicker();
  }

  // ── Public API ────────────────────────────────────────────────────────────

  function setState(conditions) {
    _conditions = (conditions ?? []).map(c => ({ tagUuid: c.tagUuid, state: c.state ?? 'present' }));
    _renderAll();
  }

  function getState() {
    return [..._conditions];
  }

  /** Called when the visible tag list changes (tag created/renamed/deleted or kind visibility changed). */
  function setAvailableTags(newTags) {
    _tags = [...newTags];
    _renderAll();
  }

  function destroy() {
    // No external subscriptions — nothing to clean up.
  }

  _renderAll();

  return { element: root, setState, getState, setAvailableTags, destroy };
}
