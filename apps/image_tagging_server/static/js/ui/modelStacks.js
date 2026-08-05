/**
 * Model Stacks management pane (Configuration → Model Stacks).
 *
 * A model stack is an ordered, reusable list of detectors, where a member is
 * either a raw model or another stack (expanded recursively at run time by
 * `shared/modelGroups/expand.js`). Until this pane existed they could only be
 * *created* — via the Inference panel's "Save as stack" prompt — and never
 * renamed, reordered, or deleted; the only other place they appear is a
 * filter-chain step's picker.
 *
 * The wire entity is still `model_groups` / `/api/model-groups`; "stack" is the
 * display noun. Members are `{order, model_name}` or `{order, child_group_uuid}`
 * on write, and camelCase `modelName` / `childGroupUuid` on read.
 *
 * Passes `prefix: 'mc'` to the list-editor primitive for the same reason
 * Training Runs does — it shares Model Shapes' list/editor CSS rather than
 * duplicating it under a prefix of its own.
 */

import { fetchModelsList } from '/app-static/js/shared/api/models.js';
import {
  fetchModelGroups, createModelGroup, updateModelGroup, deleteModelGroup,
} from '/app-static/js/shared/api/modelGroups.js';
import { toast, makeFieldRow, buildPanelEditorBar } from '/app-static/js/ui/utils/index.js';
import { createListEditorScreen } from '/app-static/js/ui/utils/listEditorScreen.js';

let _rendering = false;
let _navGuardController = null;

// -------------------------------------------------------
// Member row — one stack member (a model, or a nested stack).
// -------------------------------------------------------
function buildMemberRow(member, { modelNames, stackOptions, onAnyChange, onDelete, onMove, isFirst, isLast }) {
  const row = document.createElement('div');
  row.className = 'mstack-member-row';

  const select = document.createElement('select');
  select.className = 'mstack-member-row__select';

  const modelGroup = document.createElement('optgroup');
  modelGroup.label = 'Models';
  for (const name of modelNames) {
    const opt = document.createElement('option');
    opt.value = `model:${name}`;
    opt.textContent = name;
    modelGroup.appendChild(opt);
  }
  if (modelNames.length) select.appendChild(modelGroup);

  if (stackOptions.length) {
    const stackGroup = document.createElement('optgroup');
    stackGroup.label = 'Stacks';
    for (const s of stackOptions) {
      const opt = document.createElement('option');
      opt.value = `stack:${s.uuid}`;
      opt.textContent = s.name;
      stackGroup.appendChild(opt);
    }
    select.appendChild(stackGroup);
  }

  const current = member.childGroupUuid ? `stack:${member.childGroupUuid}` : `model:${member.modelName ?? ''}`;
  // A member can reference a model that has since been deleted, or a stack the
  // user can't nest here (itself). Keep it selectable rather than silently
  // rewriting the row to the first option on load — that would corrupt the
  // stack the moment the user saved anything else.
  if (!Array.from(select.options).some(o => o.value === current)) {
    const orphan = document.createElement('option');
    orphan.value = current;
    orphan.textContent = `${member.modelName ?? member.childGroupUuid} (missing)`;
    select.insertBefore(orphan, select.firstChild);
  }
  select.value = current;
  select.addEventListener('change', () => onAnyChange?.());
  row.appendChild(select);

  const upBtn = document.createElement('button');
  upBtn.type = 'button';
  upBtn.className = 'panel-btn';
  upBtn.textContent = '▲';
  upBtn.title = 'Move up';
  upBtn.disabled = isFirst;
  upBtn.addEventListener('click', () => onMove?.(-1));

  const downBtn = document.createElement('button');
  downBtn.type = 'button';
  downBtn.className = 'panel-btn';
  downBtn.textContent = '▼';
  downBtn.title = 'Move down';
  downBtn.disabled = isLast;
  downBtn.addEventListener('click', () => onMove?.(1));

  const delBtn = document.createElement('button');
  delBtn.type = 'button';
  delBtn.className = 'panel-btn';
  delBtn.textContent = '✕';
  delBtn.title = 'Remove from stack';
  delBtn.addEventListener('click', () => onDelete?.());

  row.append(upBtn, downBtn, delBtn);

  return {
    element: row,
    getMember: () => {
      const [kind, rest] = select.value.split(/:(.*)/s);
      return kind === 'stack' ? { child_group_uuid: rest } : { model_name: rest };
    },
  };
}

// -------------------------------------------------------
// Editor form
// -------------------------------------------------------
function buildStackForm(container, { name, description, members }, { modelNames, stackOptions }, { onAnyChange }) {
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

  const membersHdr = document.createElement('h5');
  membersHdr.textContent = 'Members (run in order)';
  container.appendChild(membersHdr);

  const membersBox = document.createElement('div');
  container.appendChild(membersBox);

  let _members = members.slice();
  let _rows = [];

  function renderMembers() {
    // Read current selections back before rebuilding, so a reorder or delete
    // doesn't discard edits made to rows the user hasn't saved yet.
    if (_rows.length) _members = _rows.map(r => r.getMember()).map(_toReadShape);
    membersBox.innerHTML = '';
    _rows = [];

    if (_members.length === 0) {
      const empty = document.createElement('p');
      empty.className = 'mstack-empty';
      empty.textContent = 'No members yet — add a model or another stack below, '
        + 'or save a selection from the Inference panel.';
      membersBox.appendChild(empty);
    }

    _members.forEach((m, idx) => {
      const row = buildMemberRow(m, {
        modelNames,
        stackOptions,
        onAnyChange,
        isFirst: idx === 0,
        isLast: idx === _members.length - 1,
        onDelete: () => {
          _members = _rows.map(r => r.getMember()).map(_toReadShape);
          _members.splice(idx, 1);
          _rows = [];
          renderMembers();
          onAnyChange?.();
        },
        onMove: (delta) => {
          _members = _rows.map(r => r.getMember()).map(_toReadShape);
          const target = idx + delta;
          if (target < 0 || target >= _members.length) return;
          [_members[idx], _members[target]] = [_members[target], _members[idx]];
          _rows = [];
          renderMembers();
          onAnyChange?.();
        },
      });
      _rows.push(row);
      membersBox.appendChild(row.element);
    });
  }

  /** Write shape (`model_name`) → read shape (`modelName`), so one render path serves both. */
  function _toReadShape(m) {
    return m.child_group_uuid
      ? { childGroupUuid: m.child_group_uuid, modelName: null }
      : { modelName: m.model_name, childGroupUuid: null };
  }

  const addBtn = document.createElement('button');
  addBtn.type = 'button';
  addBtn.className = 'panel-btn';
  addBtn.textContent = '+ Add member';
  addBtn.disabled = modelNames.length === 0 && stackOptions.length === 0;
  addBtn.addEventListener('click', () => {
    _members = _rows.map(r => r.getMember()).map(_toReadShape);
    _members.push(
      modelNames.length
        ? { modelName: modelNames[0], childGroupUuid: null }
        : { modelName: null, childGroupUuid: stackOptions[0].uuid },
    );
    _rows = [];
    renderMembers();
    onAnyChange?.();
  });
  container.appendChild(addBtn);

  renderMembers();

  return {
    getName: () => nameInput.value.trim(),
    getDescription: () => descInput.value.trim(),
    getMembers: () => _rows.map((r, order) => ({ ...r.getMember(), order })),
  };
}

// -------------------------------------------------------
// Entry point
// -------------------------------------------------------
export async function renderModelStackManager({ target = 'model-stack-management-box' } = {}) {
  if (_rendering) return;
  _rendering = true;

  _navGuardController?.abort();
  _navGuardController = new AbortController();

  const container = document.getElementById(target);
  if (!container) { _rendering = false; return; }
  container.innerHTML = '';

  try {
    const modelNames = await fetchModelsList();

    const screen = createListEditorScreen({
      container,
      prefix: 'mc',
      title: 'Model Stacks',
      itemUuidAttr: 'stackUuid',
      withEditorFooter: false,
      onNew: () => _showForm(null),
      ownTabId: 'tab-modelstack-management',
      navGuardSignal: _navGuardController.signal,
    });
    const { listBody, editorBar, editorBody } = screen;
    const _confirmIfDirty = screen.confirmIfDirty;
    const _updateListSelection = screen.updateListSelection;

    let _selectedUuid = null;
    let _switching = false;

    async function _showForm(stack, { force = false } = {}) {
      if (_switching && !force) return;
      _switching = true;
      try {
        if (!force && !await _confirmIfDirty()) return;
        screen.setBarHandle(null);
        screen.setDoSave(null);
        _selectedUuid = stack?.uuid ?? null;

        editorBody.innerHTML = '';

        const barHandle = buildPanelEditorBar({
          container: editorBar,
          item: stack,
          newTitle: 'New Model Stack',
          deleteTitle: 'Delete model stack',
          onSave: () => _doSave(true),
          onDelete: stack ? async () => {
            await deleteModelGroup(stack.uuid);
            toast('Model stack deleted', 3000);
            await refreshList();
            await _showForm(null, { force: true });
          } : undefined,
        });
        screen.setBarHandle(barHandle);
        const { setDirty, setSaving, updateTitle } = barHandle;

        // A stack can nest other stacks but never itself. The server also
        // rejects cycles on save; this just keeps the obvious one out of the picker.
        const stacksMap = await fetchModelGroups();
        const stackOptions = [...stacksMap.values()]
          .filter(s => s.uuid !== stack?.uuid)
          .sort((a, b) => a.name.localeCompare(b.name));

        const formHandle = buildStackForm(
          editorBody,
          {
            name: stack?.name ?? '',
            description: stack?.description ?? '',
            members: (stack?.members ?? []).slice().sort((a, b) => a.order - b.order),
          },
          { modelNames, stackOptions },
          { onAnyChange: () => setDirty(true) },
        );
        setDirty(false);

        async function _doSave(navigateAfterCreate = false) {
          const name = formHandle.getName();
          if (!name) { toast('Name is required', 3000, 'error'); return false; }
          setSaving(true);
          try {
            const members = formHandle.getMembers();
            const description = formHandle.getDescription();
            if (stack) {
              const updated = await updateModelGroup(stack.uuid, { name, description, members });
              setDirty(false);
              setSaving(false);
              updateTitle(updated.name);
              toast('Model stack saved', 3000);
              await refreshList();
            } else {
              const created = await createModelGroup({ name, description, members });
              setDirty(false);
              toast(`Model stack "${created.name}" created`, 3000);
              await refreshList();
              if (navigateAfterCreate) {
                await _showForm(created, { force: true });
              } else {
                setSaving(false);
              }
            }
            return true;
          } catch (err) {
            // Includes the server's cycle rejection — surface it verbatim.
            toast(err.message, 5000, 'error');
            setSaving(false);
            return false;
          }
        }

        screen.setDoSave(() => _doSave(false));
        _updateListSelection(stack?.uuid ?? null);
      } finally {
        _switching = false;
      }
    }

    async function refreshList() {
      listBody.innerHTML = '';
      const stacksMap = await fetchModelGroups();

      if (stacksMap.size === 0) {
        const empty = document.createElement('p');
        empty.className = 'mstack-empty';
        // Kept short: the list panel is narrow (it shares `.mc-list-panel`'s
        // width), so the longer "how do I make one" guidance lives in the editor.
        empty.textContent = 'No model stacks yet.';
        listBody.appendChild(empty);
      }

      for (const stack of stacksMap.values()) {
        const item = document.createElement('div');
        item.className = 'mc-list-item';
        item.dataset.stackUuid = stack.uuid;

        const nameLine = document.createElement('div');
        nameLine.className = 'mc-list-item__name';
        nameLine.textContent = stack.name;
        item.appendChild(nameLine);

        if (stack.description) {
          const descLine = document.createElement('div');
          descLine.className = 'mc-list-item__desc';
          descLine.textContent = stack.description;
          item.appendChild(descLine);
        }

        const metaLine = document.createElement('div');
        metaLine.className = 'mc-list-item__meta';
        const n = stack.members?.length ?? 0;
        metaLine.textContent = `${n} member${n !== 1 ? 's' : ''}`;
        item.appendChild(metaLine);

        item.addEventListener('click', () => _showForm(stack));
        listBody.appendChild(item);
      }

      _updateListSelection(_selectedUuid);
    }

    await refreshList();
    await _showForm(null, { force: true });
  } catch (err) {
    const errMsg = document.createElement('p');
    errMsg.style.color = 'red';
    errMsg.textContent = `Error loading model stacks: ${err.message}`;
    container.appendChild(errMsg);
  } finally {
    _rendering = false;
  }
}
