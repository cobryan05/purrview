import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import { fetchModelsList } from '/app-static/js/shared/api/models.js';
import {
  fetchFilterChains, createFilterChain, updateFilterChain, deleteFilterChain,
} from '/app-static/js/shared/api/filterChains.js';
import { toast, makeFieldRow, buildPanelEditorBar } from '/app-static/js/ui/utils/index.js';
import { showUnsavedConfigChangesModal } from '/app-static/js/ui/utils/confirmModal.js';
import { DropDownField } from '/app-static/js/ui/utils/fields/index.js';

let _rendering = false;
let _navGuardController = null;

const ACTIONS = [
  { key: 'discard_image', text: 'Discard image (collect-time only)' },
  { key: 'tag_image', text: 'Tag image' },
  { key: 'strip_unverified_bboxes', text: 'Strip unverified bboxes' },
  { key: 'relabel_bbox', text: 'Relabel bbox' },
  { key: 'tag_bbox', text: 'Tag bbox' },
];

const MAX_RECOMMENDED_STEPS = 5;

function _actionParamsFor(action, params = {}) {
  // Normalize so every action only carries the params it actually uses.
  switch (action) {
    case 'tag_image':
    case 'tag_bbox':
      return { tag_uuid: params.tag_uuid ?? null };
    case 'relabel_bbox':
      return {
        target_label_uuid: params.target_label_uuid ?? null,
        only_if_label_uuid: params.only_if_label_uuid ?? null,
      };
    default:
      return {};
  }
}

// -------------------------------------------------------
// Step row — one FilterChainStep as an editable row.
// Returns { getStep, destroy, element }
// -------------------------------------------------------
function buildStepRow(step, { modelNames, tagItems, labelItems, onAnyChange, onDelete, onMove, isFirst, isLast }) {
  const row = document.createElement('div');
  row.className = 'panel-box fc-step-row';
  row.style.cssText = 'display: flex; flex-direction: column; gap: 0.4em; padding: 0.6em; margin-bottom: 0.5em;';

  let _action = step.action || 'tag_image';
  let _actionParams = _actionParamsFor(_action, step.action_params);

  const topRow = document.createElement('div');
  topRow.style.cssText = 'display: flex; gap: 0.5em; align-items: center; flex-wrap: wrap;';
  row.appendChild(topRow);

  const upBtn = document.createElement('button');
  upBtn.type = 'button';
  upBtn.textContent = '↑';
  upBtn.className = 'panel-btn secondary';
  upBtn.disabled = isFirst;
  upBtn.addEventListener('click', () => onMove(-1));
  topRow.appendChild(upBtn);

  const downBtn = document.createElement('button');
  downBtn.type = 'button';
  downBtn.textContent = '↓';
  downBtn.className = 'panel-btn secondary';
  downBtn.disabled = isLast;
  downBtn.addEventListener('click', () => onMove(1));
  topRow.appendChild(downBtn);

  // Placeholders preserve visual order (↑ ↓ Model Action Enabled Remove) even
  // though DropDownField.renderEdit() resolves its <select> asynchronously.
  const modelSlot = document.createElement('span');
  topRow.appendChild(modelSlot);
  const modelField = new DropDownField({
    value: step.model_name || null,
    editMode: true,
    options: modelNames,
    placeholder: 'Model…',
    onChange: () => onAnyChange?.(),
  });
  modelField.renderEdit().then(el => modelSlot.appendChild(el));

  const actionSlot = document.createElement('span');
  topRow.appendChild(actionSlot);
  const actionField = new DropDownField({
    value: ACTIONS.find(a => a.key === _action) ? { key: _action } : null,
    editMode: true,
    options: ACTIONS,
    placeholder: 'Action…',
    onChange: (value) => {
      _action = value;
      _actionParams = _actionParamsFor(_action, {});
      renderParamsRow();
      onAnyChange?.();
    },
  });
  actionField.renderEdit().then(el => actionSlot.appendChild(el));

  const enabledLabel = document.createElement('label');
  enabledLabel.style.cssText = 'display: flex; align-items: center; gap: 0.25em; font-size: 0.9em;';
  const enabledCheckbox = document.createElement('input');
  enabledCheckbox.type = 'checkbox';
  enabledCheckbox.checked = step.enabled !== false;
  enabledCheckbox.addEventListener('change', () => onAnyChange?.());
  enabledLabel.append(enabledCheckbox, document.createTextNode('Enabled'));
  topRow.appendChild(enabledLabel);

  const deleteBtn = document.createElement('button');
  deleteBtn.type = 'button';
  deleteBtn.textContent = 'Remove';
  deleteBtn.className = 'panel-btn secondary';
  deleteBtn.addEventListener('click', () => onDelete());
  topRow.appendChild(deleteBtn);

  const numbersRow = document.createElement('div');
  numbersRow.style.cssText = 'display: flex; gap: 1em;';
  row.appendChild(numbersRow);

  const confInput = document.createElement('input');
  confInput.type = 'number';
  confInput.min = '0';
  confInput.max = '1';
  confInput.step = '0.01';
  confInput.value = step.confidence ?? 0.5;
  confInput.style.width = '6em';
  confInput.addEventListener('input', () => onAnyChange?.());
  numbersRow.appendChild(makeFieldRow('Confidence', confInput));

  const iouInput = document.createElement('input');
  iouInput.type = 'number';
  iouInput.min = '0';
  iouInput.max = '1';
  iouInput.step = '0.01';
  iouInput.value = step.iou_threshold ?? 0.5;
  iouInput.style.width = '6em';
  iouInput.addEventListener('input', () => onAnyChange?.());
  numbersRow.appendChild(makeFieldRow('IoU threshold (detector-overlap steps)', iouInput));

  const paramsRow = document.createElement('div');
  row.appendChild(paramsRow);

  let _tagField = null;
  let _targetLabelField = null;
  let _onlyIfLabelField = null;

  function renderParamsRow() {
    paramsRow.innerHTML = '';
    if (_action === 'tag_image' || _action === 'tag_bbox') {
      _tagField = new DropDownField({
        value: _actionParams.tag_uuid ? tagItems.find(t => t.key === _actionParams.tag_uuid) ?? _actionParams.tag_uuid : null,
        editMode: true,
        options: tagItems,
        placeholder: 'Tag…',
        onChange: (value) => { _actionParams.tag_uuid = value; onAnyChange?.(); },
      });
      _tagField.renderEdit().then(el => paramsRow.appendChild(makeFieldRow('Tag', el)));
    } else if (_action === 'relabel_bbox') {
      _targetLabelField = new DropDownField({
        value: _actionParams.target_label_uuid ? labelItems.find(l => l.key === _actionParams.target_label_uuid) ?? _actionParams.target_label_uuid : null,
        editMode: true,
        options: labelItems,
        placeholder: 'Target label…',
        onChange: (value) => { _actionParams.target_label_uuid = value; onAnyChange?.(); },
      });
      _targetLabelField.renderEdit().then(el => paramsRow.appendChild(makeFieldRow('Relabel to', el)));

      _onlyIfLabelField = new DropDownField({
        value: _actionParams.only_if_label_uuid ? labelItems.find(l => l.key === _actionParams.only_if_label_uuid) ?? _actionParams.only_if_label_uuid : null,
        editMode: true,
        options: labelItems,
        placeholder: '(any label)',
        onChange: (value) => { _actionParams.only_if_label_uuid = value; onAnyChange?.(); },
      });
      _onlyIfLabelField.renderEdit().then(el => paramsRow.appendChild(makeFieldRow('Only if currently labeled', el)));
    }
  }
  renderParamsRow();

  function _resolveKey(field) {
    if (!field) return null;
    const v = field.getValue?.();
    return v ?? null;
  }

  return {
    element: row,
    getStep: () => ({
      model_name: modelField.getValue?.() ?? step.model_name ?? '',
      confidence: parseFloat(confInput.value) || 0,
      iou_threshold: parseFloat(iouInput.value) || 0,
      action: _action,
      action_params: {
        ..._actionParamsFor(_action, {
          tag_uuid: _resolveKey(_tagField) ?? _actionParams.tag_uuid,
          target_label_uuid: _resolveKey(_targetLabelField) ?? _actionParams.target_label_uuid,
          only_if_label_uuid: _resolveKey(_onlyIfLabelField) ?? _actionParams.only_if_label_uuid,
        }),
      },
      enabled: enabledCheckbox.checked,
    }),
  };
}

// -------------------------------------------------------
// Chain form — name/description + ordered steps
// -------------------------------------------------------
function buildChainForm(container, { name = '', description = '', steps = [] }, pickerData, { onAnyChange }) {
  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.value = name;
  nameInput.placeholder = 'Required';
  nameInput.addEventListener('input', () => onAnyChange?.());
  container.appendChild(makeFieldRow('Name', nameInput));

  const descInput = document.createElement('input');
  descInput.type = 'text';
  descInput.value = description;
  descInput.placeholder = 'Optional';
  descInput.addEventListener('input', () => onAnyChange?.());
  container.appendChild(makeFieldRow('Description', descInput));

  const warning = document.createElement('p');
  warning.style.cssText = 'color: var(--pico-form-element-invalid-border-color, #c0392b); font-size: 0.85em; display: none;';
  warning.textContent = `Chains longer than ${MAX_RECOMMENDED_STEPS} steps run every step's model per image — consider splitting this into multiple chains.`;
  container.appendChild(warning);

  const stepsBox = document.createElement('div');
  container.appendChild(stepsBox);

  let _stepData = steps.map(s => ({ ...s }));
  let _rows = [];

  function _updateWarning() {
    warning.style.display = _stepData.length > MAX_RECOMMENDED_STEPS ? 'block' : 'none';
  }

  function renderSteps() {
    // Callers mutate _stepData (via _rows.map(r => r.getStep())) themselves
    // before calling renderSteps() — don't re-snapshot here, or the initial
    // call (with _rows still empty) would clobber the seeded _stepData.
    stepsBox.innerHTML = '';
    _rows = _stepData.map((step, idx) => {
      const rowHandle = buildStepRow(step, {
        modelNames: pickerData.modelNames,
        tagItems: pickerData.tagItems,
        labelItems: pickerData.labelItems,
        onAnyChange,
        isFirst: idx === 0,
        isLast: idx === _stepData.length - 1,
        onDelete: () => {
          _stepData = _rows.map((r, i) => (i === idx ? null : (r ? r.getStep() : _stepData[i]))).filter(Boolean);
          renderSteps();
          onAnyChange?.();
        },
        onMove: (delta) => {
          _stepData = _rows.map(r => r.getStep());
          const newIdx = idx + delta;
          if (newIdx < 0 || newIdx >= _stepData.length) return;
          [_stepData[idx], _stepData[newIdx]] = [_stepData[newIdx], _stepData[idx]];
          renderSteps();
          onAnyChange?.();
        },
      });
      stepsBox.appendChild(rowHandle.element);
      return rowHandle;
    });
    _updateWarning();
  }
  renderSteps();

  const addBtn = document.createElement('button');
  addBtn.type = 'button';
  addBtn.textContent = '+ Add step';
  addBtn.className = 'panel-btn secondary';
  addBtn.addEventListener('click', () => {
    _stepData = _rows.map(r => r.getStep());
    _stepData.push({ model_name: '', confidence: 0.5, iou_threshold: 0.5, action: 'tag_image', action_params: {}, enabled: true });
    renderSteps();
    onAnyChange?.();
  });
  container.appendChild(addBtn);

  return {
    getName: () => nameInput.value.trim(),
    getDescription: () => descInput.value.trim(),
    getSteps: () => _rows.map((r, idx) => ({ ...r.getStep(), order: idx })),
    destroy: () => {},
  };
}

// -------------------------------------------------------
// Entry point
// -------------------------------------------------------
export async function renderFilterChainManager({ target = 'filter-chain-management-box' } = {}) {
  if (_rendering) return;
  _rendering = true;

  _navGuardController?.abort();
  _navGuardController = new AbortController();

  const container = document.getElementById(target);
  if (!container) { _rendering = false; return; }
  container.innerHTML = '';

  try {
    const [labelsMap, tagsMap, modelNames] = await Promise.all([fetchLabels(), fetchTags(), fetchModelsList()]);
    await fetchFilterChains();

    const labelItems = Array.from(labelsMap.values()).map(l => ({ key: l.uuid, text: l.name, color: l.color || '#cccccc' }));
    const tagItems = Array.from(tagsMap.values()).map(t => ({ key: t.uuid, text: t.name, color: t.color || '#cccccc' }));
    const pickerData = { modelNames, tagItems, labelItems };

    const split = document.createElement('div');
    split.style.cssText = 'display: flex; gap: 1em; align-items: flex-start;';
    container.appendChild(split);

    const listPanel = document.createElement('div');
    listPanel.className = 'panel-box fc-list-panel';
    split.appendChild(listPanel);

    const listBar = document.createElement('div');
    listBar.className = 'panel-box__bar';
    const listTitle = document.createElement('span');
    listTitle.className = 'panel-box__title';
    listTitle.textContent = 'Filter Chains';
    listBar.appendChild(listTitle);
    listPanel.appendChild(listBar);

    const listBody = document.createElement('div');
    listBody.className = 'panel-box__body';
    listPanel.appendChild(listBody);

    const listFooter = document.createElement('div');
    listFooter.className = 'panel-box__footer';
    listPanel.appendChild(listFooter);

    const newBtn = document.createElement('button');
    newBtn.type = 'button';
    newBtn.textContent = '+ New';
    newBtn.className = 'panel-btn';
    newBtn.style.cssText = 'width: 100%; margin-bottom: 0;';
    newBtn.addEventListener('click', () => _showForm(null));
    listFooter.appendChild(newBtn);

    const editorPanel = document.createElement('div');
    editorPanel.className = 'panel-box fc-editor';
    split.appendChild(editorPanel);

    const editorBar = document.createElement('div');
    editorBar.className = 'panel-box__bar';
    editorPanel.appendChild(editorBar);

    const editorBody = document.createElement('div');
    editorBody.className = 'panel-box__body';
    editorPanel.appendChild(editorBody);

    const editorFooter = document.createElement('div');
    editorFooter.className = 'panel-box__footer';
    editorPanel.appendChild(editorFooter);

    let _selectedUuid = null;
    let _barHandle = null;
    let _currentDoSave = null;
    let _switching = false;

    async function _confirmIfDirty() {
      if (!_barHandle?.getDirty()) return true;
      const result = await showUnsavedConfigChangesModal();
      if (result === 'cancel') return false;
      if (result === 'save') return await (_currentDoSave?.() ?? false);
      return true;
    }

    function _updateListSelection(uuid) {
      for (const item of listBody.querySelectorAll('.fc-list-item')) {
        item.classList.toggle('active', item.dataset.chainUuid === uuid);
      }
    }

    async function _showForm(chain, { force = false } = {}) {
      if (_switching && !force) return;
      _switching = true;
      try {
        if (!force && !await _confirmIfDirty()) return;
        _barHandle = null;
        _currentDoSave = null;
        _selectedUuid = chain?.uuid ?? null;

        editorBody.innerHTML = '';
        editorFooter.innerHTML = '';

        _barHandle = buildPanelEditorBar({
          container: editorBar,
          item: chain,
          newTitle: 'New Filter Chain',
          deleteTitle: 'Delete filter chain',
          onSave: () => _doSave(true),
          onDelete: chain ? async () => {
            await deleteFilterChain(chain.uuid);
            toast('Filter chain deleted', 3000);
            await refreshList();
            await _showForm(null, { force: true });
          } : undefined,
        });
        const { setDirty, setSaving, updateTitle } = _barHandle;

        const formHandle = buildChainForm(
          editorBody,
          {
            name: chain?.name ?? '',
            description: chain?.description ?? '',
            steps: (chain?.steps ?? []).slice().sort((a, b) => a.order - b.order).map(s => ({
              model_name: s.modelName,
              confidence: s.confidence,
              iou_threshold: s.iouThreshold,
              action: s.action,
              action_params: s.actionParams,
              enabled: s.enabled,
            })),
          },
          pickerData,
          { onAnyChange: () => setDirty(true) },
        );
        setDirty(false);

        async function _doSave(navigateAfterCreate = false) {
          const name = formHandle.getName();
          if (!name) { toast('Name is required', 3000, 'error'); return false; }
          setSaving(true);
          try {
            const steps = formHandle.getSteps();
            if (chain) {
              const updated = await updateFilterChain(chain.uuid, {
                name,
                description: formHandle.getDescription(),
                steps,
              });
              setDirty(false);
              setSaving(false);
              updateTitle(updated.name);
              toast('Filter chain saved', 3000);
              await refreshList();
            } else {
              const created = await createFilterChain({
                name,
                description: formHandle.getDescription(),
                steps,
              });
              setDirty(false);
              toast(`Filter chain "${created.name}" created`, 3000);
              await refreshList();
              if (navigateAfterCreate) {
                await _showForm(created, { force: true });
              } else {
                setSaving(false);
              }
            }
            return true;
          } catch (err) {
            toast(err.message, 5000, 'error');
            setSaving(false);
            return false;
          }
        }

        _currentDoSave = () => _doSave(false);
        _updateListSelection(chain?.uuid ?? null);
      } finally {
        _switching = false;
      }
    }

    async function refreshList() {
      listBody.innerHTML = '';
      const chainsMap = await fetchFilterChains();

      if (chainsMap.size === 0) {
        const empty = document.createElement('p');
        empty.style.cssText = 'color: var(--pico-muted-color, #888); font-style: italic; font-size: 0.85em; padding: 0.5em 0.75em; margin: 0;';
        empty.textContent = 'No filter chains yet.';
        listBody.appendChild(empty);
      }

      for (const chain of chainsMap.values()) {
        const item = document.createElement('div');
        item.className = 'fc-list-item';
        item.dataset.chainUuid = chain.uuid;

        const nameLine = document.createElement('div');
        nameLine.className = 'fc-list-item__name';
        nameLine.textContent = chain.name;
        item.appendChild(nameLine);

        const metaLine = document.createElement('div');
        metaLine.className = 'fc-list-item__meta';
        const stepCount = chain.steps?.length ?? 0;
        metaLine.textContent = `${stepCount} step${stepCount !== 1 ? 's' : ''}`;
        item.appendChild(metaLine);

        item.addEventListener('click', () => _showForm(chain));
        listBody.appendChild(item);
      }

      _updateListSelection(_selectedUuid);
    }

    const guardSignal = _navGuardController.signal;
    document.addEventListener('click', async (e) => {
      const btn = e.target.closest('.workspace-tab-button, .main-tab-button');
      if (!btn) return;
      const targetTab = btn.dataset.tab;
      if (targetTab === 'tab-filterchain-management' || targetTab === 'tab-configuration') return;
      if (!_barHandle?.getDirty()) return;

      e.stopPropagation();
      const result = await showUnsavedConfigChangesModal();
      if (result === 'cancel') return;
      if (result === 'save') {
        const ok = await _currentDoSave?.();
        if (!ok) return;
      }
      if (guardSignal.aborted) return;
      _barHandle?.setDirty(false);
      btn.click();
    }, { capture: true, signal: guardSignal });

    await refreshList();
    await _showForm(null, { force: true });
  } catch (err) {
    const errMsg = document.createElement('p');
    errMsg.style.color = 'red';
    errMsg.textContent = `Error loading filter chains: ${err.message}`;
    container.appendChild(errMsg);
  } finally {
    _rendering = false;
  }
}
