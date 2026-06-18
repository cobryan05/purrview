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
function _createConditionPicker(items, onChange, {
  idKey,
  states,
  placeholder,
  emptyMsg,
} = {}) {
  let _conditions = [];
  let _items = [...items];

  const root = document.createElement('div');
  root.className = 'itc-root';

  const list = document.createElement('div');
  list.className = 'itc-list';
  root.appendChild(list);

  const addRow = document.createElement('div');
  addRow.className = 'itc-add-row';
  root.appendChild(addRow);

  function _getItem(uuid) {
    return _items.find(t => t.uuid === uuid) ?? null;
  }

  function _makeStateSelect(currentState) {
    const sel = document.createElement('select');
    sel.className = 'itc-state-select';
    for (const [value, label] of states) {
      const opt = document.createElement('option');
      opt.value = value;
      opt.textContent = label;
      if (value === currentState) opt.selected = true;
      sel.appendChild(opt);
    }
    return sel;
  }

  function _renderRow(cond, idx) {
    const row = document.createElement('div');
    row.className = 'itc-row';

    const item = _getItem(cond[idKey]);

    const swatch = document.createElement('span');
    swatch.className = 'itc-swatch';
    swatch.style.background = item?.color || '#555';
    row.appendChild(swatch);

    const nameEl = document.createElement('span');
    nameEl.className = 'itc-name';
    nameEl.textContent = item?.name ?? cond[idKey];
    nameEl.title = item?.name ?? cond[idKey];
    row.appendChild(nameEl);

    const stateSelect = _makeStateSelect(cond.state);
    stateSelect.addEventListener('change', () => {
      _conditions[idx] = { ..._conditions[idx], state: stateSelect.value };
      onChange([..._conditions]);
    });
    row.appendChild(stateSelect);

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
    const usedUuids = new Set(_conditions.map(c => c[idKey]));
    const available = _items
      .filter(t => !usedUuids.has(t.uuid))
      .sort((a, b) => a.name.localeCompare(b.name));

    const picker = document.createElement('select');
    picker.className = 'itc-picker';
    picker.disabled = available.length === 0;

    const ph = document.createElement('option');
    ph.value = '';
    ph.textContent = available.length > 0
      ? placeholder
      : (_items.length === 0 ? emptyMsg : 'All added');
    picker.appendChild(ph);

    for (const item of available) {
      const opt = document.createElement('option');
      opt.value = item.uuid;
      opt.textContent = item.name;
      picker.appendChild(opt);
    }

    picker.addEventListener('change', () => {
      const uuid = picker.value;
      if (!uuid) return;
      _conditions.push({ [idKey]: uuid, state: states[0][0] });
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

  function setState(conditions) {
    _conditions = (conditions ?? []).map(c => ({ [idKey]: c[idKey], state: c.state ?? states[0][0] }));
    _renderAll();
  }

  function getState() {
    return [..._conditions];
  }

  function setAvailableItems(newItems) {
    _items = [...newItems];
    _renderAll();
  }

  _renderAll();

  return { element: root, setState, getState, setAvailableItems, destroy() {} };
}

export function createImageTagConditions(tags, onChange, { allowUnset = true } = {}) {
  const stateOpts = [
    ['present', 'Present'],
    ['absent',  'Absent'],
    ...(allowUnset ? [['unset', 'Unset']] : []),
  ];
  const inner = _createConditionPicker(tags, onChange, {
    idKey: 'tagUuid',
    states: stateOpts,
    placeholder: 'Add image tag filter…',
    emptyMsg: 'No tags defined',
  });
  return {
    element: inner.element,
    setState: inner.setState,
    getState: inner.getState,
    setAvailableTags: inner.setAvailableItems,
    destroy: inner.destroy,
  };
}

export function createImageLabelConditions(labels, onChange) {
  const inner = _createConditionPicker(labels, onChange, {
    idKey: 'labelUuid',
    states: [['present', 'Present'], ['absent', 'Absent'], ['uncertain', 'Uncertain']],
    placeholder: 'Add image label filter…',
    emptyMsg: 'No image labels defined',
  });
  return {
    element: inner.element,
    setState: inner.setState,
    getState: inner.getState,
    setAvailableLabels: inner.setAvailableItems,
    destroy: inner.destroy,
  };
}
