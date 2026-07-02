import { BboxInfoField, EditableField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, Logger, toast } from '/app-static/js/ui/utils/index.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { fetchLabels, fetchLabelByUuid } from '/app-static/js/shared/api/labels.js';
import { fetchTags, fetchTagByUuid } from '/app-static/js/shared/api/tags.js';
import { fetchImageTagStatuses, setImageTagStatus } from '/app-static/js/shared/api/imageTags.js';
import { UNVERIFIED_TAG_UUID, VERIFIED_TAG_UUID, VERIFICATION_TAG_UUIDS } from '/app-static/js/shared/api/review.js';
import { isDynamicLabelsVisible, isTagVisible } from '/app-static/js/shared/activeModel.js';
import { setTool } from '/app-static/js/app/image-tagging/canvas/tools.js';
import { applyVerifiedTag } from '/app-static/js/app/image-tagging/canvas/interaction.js';

/**
 * Opens the inspector tab in the sidebar and highlights it briefly.
 */
export function openInspectorTab() {
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;

  const tabs = sidebar.querySelectorAll('.sidebar-tab-button');
  const panes = sidebar.querySelectorAll('.sidebar-tab-pane');

  tabs.forEach(tab => {
    const tabName = tab.getAttribute('data-tab');
    tab.classList.toggle('active', tabName === 'tab-inspector');
  });

  panes.forEach(pane => {
    pane.classList.toggle('active', pane.id === 'tab-inspector');
  });

  document.getElementById('classInput')?.focus();

  const inspector = document.getElementById('tab-inspector');
  if (inspector) {
    inspector.style.outline = '2px solid #ff0033';
    setTimeout(() => inspector.style.outline = '', 1000);
  }
}


/**
 * @param {{ bboxView?: any, target?: string|HTMLElement, editable?: boolean, onSelectCallback?: any }} [opts]
 *   target – a string element ID *or* an HTMLElement; defaults to 'tab-inspector'
 */
export function renderBboxInspector({ bboxView = null, target = 'tab-inspector', editable = true, onSelectCallback = null, runtimeId = null, onAfterSave = null } = {}) {
  if (!bboxView) return;
  try {
    const targetElement = typeof target === 'string' ? document.getElementById(target) : target;
    if (!targetElement) {
      console.error('Class container not found');
      return;
    }
    targetElement.innerHTML = 'Inspector Loading...'
    const container = document.createElement('div');
    container.style.display = 'block';
    container.style.width = '100%';
    container.style.minWidth = '0';
    container.style.overflow = 'hidden';
    targetElement.appendChild(container);

    BboxInfoField.create({ bboxGroup: bboxView }).then(bboxInfoFieldInstance => {
      const row = createGenericRow({
        field: new EditableField({
          field: bboxInfoFieldInstance,
          editMode: false,
          buttonsLast: true,
          onChange: (val) => {
            Logger.warn(val);
          },
          onSave: async (val) => {
            const originalLabelUuid = bboxView.uiBBox.label?.uuid ?? null;
            const labelUuid = val.bbox_info.labelUuid ?? null;
            const labelChanged = labelUuid !== originalLabelUuid;

            // Build new tag list. Swap verification tag when the label changes.
            let savedTagUuids = val.bbox_info.tagUuids ?? [];
            if (labelChanged && labelUuid) {
              // Assigning a label: strip all verification tags and mark VERIFIED.
              savedTagUuids = [...savedTagUuids.filter(id => !VERIFICATION_TAG_UUIDS.has(id)), VERIFIED_TAG_UUID];
            } else if (labelChanged && !labelUuid) {
              // Removing a label: strip all verification tags and mark UNVERIFIED so it returns to the review queue.
              savedTagUuids = [...savedTagUuids.filter(id => !VERIFICATION_TAG_UUIDS.has(id)), UNVERIFIED_TAG_UUID];
            }

            // Immediate optimistic update using any icon data we already have.
            const existingIconsMap = new Map((bboxView.uiBBox.tags ?? []).map(t => [t.uuid, t.icon]));
            bboxView.uiBBox.tags = savedTagUuids.map(id => ({
              uuid: id,
              icon: existingIconsMap.get(id) ?? null,
            }));
            bboxView.refreshIcons(bboxView.uiBBox.tags);
            events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId });

            // Async: hydrate tag icons from cache/network and do a second pass.
            const tagsMap = await fetchTags();
            // Guard: if the inspector was re-rendered for a different bbox while we
            // were awaiting, bail out to avoid corrupting the new bbox's state.
            if (!container.isConnected) return;
            bboxView.uiBBox.tags = savedTagUuids.map(id => {
              const tag = tagsMap.get(id);
              return { uuid: id, icon: tag?.icon ?? null };
            });
            bboxView.refreshIcons(bboxView.uiBBox.tags);

            // Async: update label appearance only if it changed.
            if (labelChanged) {
              if (labelUuid) {
                const labelData = await fetchLabelByUuid(labelUuid);
                if (!container.isConnected) return;
                if (labelData) {
                  bboxView.uiBBox.label = { uuid: labelUuid, text: labelData.name, color: labelData.color };
                  bboxView.updateAppearance(labelData.name, labelData.color);
                }
              } else {
                bboxView.uiBBox.label = undefined;
                bboxView.updateAppearance('', 'grey');
              }
            }

            // Snapshot after all mutations are applied, then publish once so every
            // subscriber (canvas, label list, toolbar undo state) sees the final state.
            onAfterSave?.();
            events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId });
          },
          onCancel: () => {
            Logger.warn("Canceled");
          }
        }),
      });
      container.appendChild(row);
      targetElement.innerHTML = '';
      targetElement.appendChild(container);
    });
  } catch (err) {
    console.error('Failed to render bbox inspector:', err);
  }

}


/**
 * Bind the inspector to a specific canvas runtime so it only responds to
 * double-clicks from that canvas.
 *
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {{ container?: HTMLElement, openTab?: boolean }} [opts]
 *   container – render into this element instead of the default sidebar tab
 *   openTab   – switch the sidebar to the inspector tab on double-click (default true)
 * @returns {() => void} unsubscribe function
 */
export function initInspector(runtime, { container = null, openTab = true } = {}) {
  return events.subscribeToCanvas(
    runtime.id,
    EventTypes.CANVAS_BBOX_DOUBLE_CLICKED,
    ({ bboxView }) => {
      if (openTab) openInspectorTab();
      renderBboxInspector({
        bboxView,
        target: container ?? 'tab-inspector',
        runtimeId: runtime.id,
        onAfterSave: () => runtime.pushState(),
      });
    }
  );
}


// ─────────────────────────────────────────────────────────────────────────────
// Tag Panel — live bbox inspector that also serves as the tag-tool picker.
// Lives in the "Tags" sidebar tab.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create a persistent "Tags" panel in `container` for `runtime`.
 *
 * Two sections:
 *   1. Label dropdown — shows the selected bbox's class.  Changing the dropdown
 *      immediately updates the canvas appearance and pushes to the undo stack;
 *      the data only reaches the DB when the user hits the canvas Save button.
 *      Shown only when a bbox is selected (single- or double-click in select mode).
 *   2. Tag list — always visible.  Clicking a row selects `tag:<uuid>` as the
 *      active canvas tool (highlighted like the active label row).  The checkbox
 *      reflects whether that tag is on the selected bbox; toggling applies/removes
 *      the tag without switching tool.
 *
 * Single-click updates the panel content.
 * Double-click and tag-tool selection additionally switch the sidebar to this
 * tab (handled in sidebar.js).
 *
 * @param {HTMLElement} container
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @returns {{ destroy(): void }}
 */
export function createTagPanel(container, runtime) {
  /** @type {import('@canvas_types').BboxView|null} */
  let _selectedBbox = null;
  /** @type {Map<string, object>} */
  let _tagsMap = new Map();
  /** @type {Map<string, object>} */
  let _labelsMap = new Map();
  let _activeTool = runtime.tool ?? 'select';

  // ── DOM refs ──────────────────────────────────────────────────────────────
  let _labelContent = null;   // mutable child of the label section
  let _tagRows = new Map();   // uuid → { row, checkbox }

  // ── Whole-image tag tri-state state ───────────────────────────────────────
  let _currentImagePath = null;
  let _loadSeq = 0;
  /** @type {Map<string, 'present'|'absent'>} current server + user-interaction state */
  let _serverTagStatuses = new Map();
  /** @type {Map<string, string>} pending flushes not yet persisted to the server */
  let _pendingTagFlush = new Map();
  let _flushInFlight = false;
  /** @type {Map<string, (status: string) => void>} per-row button updaters */
  const _tagStatusUpdaters = new Map();

  // Full 3-state cycle — image-tag "present" is explicit, not derived from bboxes.
  const TAG_STATUS_CFG = Object.freeze({
    present:   { text: '✓', color: '#4c4' },
    absent:    { text: '✗', color: '#e55' },
    uncertain: { text: '?', color: '#888' },
  });
  const TAG_STATUS_CYCLE = Object.freeze({ uncertain: 'present', present: 'absent', absent: 'uncertain' });

  function _effectiveTagStatus(uuid) {
    return _serverTagStatuses.get(uuid) ?? 'uncertain';
  }

  function _setLocalTagStatus(uuid, status) {
    if (status === 'uncertain') {
      _serverTagStatuses.delete(uuid);
    } else {
      _serverTagStatuses.set(uuid, /** @type {'present'|'absent'} */ (status));
    }
    _pendingTagFlush.set(uuid, status);
  }

  function _refreshTagStatusButtons() {
    for (const [uuid, update] of _tagStatusUpdaters) {
      update(_effectiveTagStatus(uuid));
    }
  }

  async function _fetchAndApplyTagStatuses(path) {
    const seq = ++_loadSeq;
    try {
      const statuses = await fetchImageTagStatuses(path);
      if (seq !== _loadSeq) return; // superseded by a newer navigation
      _serverTagStatuses = statuses;
      _refreshTagStatusButtons();
    } catch (err) {
      console.error('Tags panel: failed to fetch image tag statuses:', err);
    }
  }

  // ── Styles ────────────────────────────────────────────────────────────────
  // NOTE: `container` is the shared `.sidebar-tab-pane` element itself (not an
  // inner wrapper) — its visibility is toggled via the `.active` class, which
  // sets `display`. Don't set `display` here: an inline style would win over
  // the class rule and make this pane visible on every tab.
  Object.assign(container.style, {
    overflowY: 'auto',
    padding: '4px',
  });

  // ── Label section ─────────────────────────────────────────────────────────

  function _buildLabelSection() {
    const section = document.createElement('div');
    section.style.cssText = 'padding: 4px 6px 8px; border-bottom: 1px solid #333; margin-bottom: 4px;';

    const hdr = document.createElement('div');
    hdr.textContent = 'Class';
    hdr.style.cssText = 'font-weight:bold; font-size:0.8em; text-transform:uppercase; letter-spacing:0.05em; color:#aaa; padding: 4px 0 6px;';
    section.appendChild(hdr);

    _labelContent = document.createElement('div');
    section.appendChild(_labelContent);

    return section;
  }

  function _refreshLabelSection() {
    if (!_labelContent) return;
    _labelContent.innerHTML = '';

    if (!_selectedBbox) {
      const hint = document.createElement('div');
      hint.style.cssText = 'font-size:0.82em; color:#666; font-style:italic;';
      hint.textContent = 'Click a bbox to inspect it';
      _labelContent.appendChild(hint);
      return;
    }

    const currentLabelUuid = _selectedBbox.uiBBox.label?.uuid ?? '';

    const select = document.createElement('select');
    select.style.cssText = 'width:100%; font-size:0.85em; padding:2px 4px;';

    const noneOpt = document.createElement('option');
    noneOpt.value = '';
    noneOpt.textContent = '— none —';
    select.appendChild(noneOpt);

    const showDynamic = isDynamicLabelsVisible();
    for (const label of _labelsMap.values()) {
      if (label.labelType === 'dynamic' && !showDynamic) continue;
      const opt = document.createElement('option');
      opt.value = label.uuid;
      opt.textContent = label.labelType === 'dynamic' ? `${label.name} [dynamic]` : label.name;
      if (label.uuid === currentLabelUuid) opt.selected = true;
      select.appendChild(opt);
    }

    // Immediate update on change — no Save button needed in the panel.
    // The canvas toolbar Save button is what persists to the DB.
    select.onchange = () => {
      if (!_selectedBbox) return;
      const newUuid = select.value || null;
      const oldUuid = _selectedBbox.uiBBox.label?.uuid ?? null;
      if (newUuid === oldUuid) return;

      if (newUuid) {
        const labelData = _labelsMap.get(newUuid);
        if (!labelData) return;
        _selectedBbox.uiBBox.label = { uuid: newUuid, text: labelData.name, color: labelData.color };
        _selectedBbox.updateAppearance(labelData.name, labelData.color);
        applyVerifiedTag(_selectedBbox.uiBBox);
        _selectedBbox.refreshIcons(_selectedBbox.uiBBox.tags);
      } else {
        _selectedBbox.uiBBox.label = undefined;
        _selectedBbox.updateAppearance('', 'grey');
        // Removing label → strip all verification tags and re-mark as UNVERIFIED.
        _selectedBbox.uiBBox.tags = [
          ...(_selectedBbox.uiBBox.tags ?? []).filter(t => !VERIFICATION_TAG_UUIDS.has(t.uuid)),
          { uuid: UNVERIFIED_TAG_UUID, icon: null },
        ];
        _selectedBbox.refreshIcons(_selectedBbox.uiBBox.tags);
      }
      runtime.pushState();
      events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
    };

    _labelContent.appendChild(select);
  }

  // ── Tag list ──────────────────────────────────────────────────────────────

  function _buildTagList() {
    const wrap = document.createElement('div');

    const hdr = document.createElement('div');
    hdr.textContent = 'Tags';
    hdr.style.cssText = 'font-weight:bold; font-size:0.8em; text-transform:uppercase; letter-spacing:0.05em; color:#aaa; padding:4px 6px 6px;';
    wrap.appendChild(hdr);

    _tagRows.clear();
    _tagStatusUpdaters.clear();

    if (_tagsMap.size === 0) {
      const empty = document.createElement('div');
      empty.style.cssText = 'font-size:0.82em; color:#666; font-style:italic; padding:4px 6px;';
      empty.textContent = 'No tags defined';
      wrap.appendChild(empty);
      return wrap;
    }

    for (const tag of _tagsMap.values()) {
      if (!isTagVisible(tag)) continue;
      const { uuid, name, icon, color } = tag;

      const row = document.createElement('div');
      row.dataset.uuid = uuid;
      row.className = 'tag-panel-row';
      Object.assign(row.style, {
        display: 'flex',
        alignItems: 'center',
        gap: '0.4em',
        padding: '4px 6px',
        borderRadius: '3px',
        cursor: 'pointer',
        userSelect: 'none',
      });

      // ── Whole-image tri-state button ─────────────────────────────────────
      const statusBtn = document.createElement('button');
      statusBtn.title = 'Whole-image tag: click to cycle (? uncertain → ✓ present → ✗ absent)';
      Object.assign(statusBtn.style, {
        flexShrink: '0',
        width: '22px',
        height: '20px',
        fontSize: '0.8em',
        border: '1px solid #444',
        borderRadius: '3px',
        padding: '0',
        lineHeight: '1',
        background: 'transparent',
        cursor: 'pointer',
        margin: '0',
      });

      function applyTagStatus(status) {
        const cfg = TAG_STATUS_CFG[status] ?? TAG_STATUS_CFG.uncertain;
        statusBtn.textContent = cfg.text;
        statusBtn.style.color = cfg.color;
      }

      _tagStatusUpdaters.set(uuid, applyTagStatus);

      statusBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!_currentImagePath) return;
        const next = TAG_STATUS_CYCLE[_effectiveTagStatus(uuid)] ?? 'uncertain';
        _setLocalTagStatus(uuid, next);
        applyTagStatus(_effectiveTagStatus(uuid));
        runtime?.markDirty?.();
      });

      // ── Bbox checkbox ────────────────────────────────────────────────────
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.style.cssText = 'margin:0; flex-shrink:0; cursor:pointer;';
      cb.title = 'Toggle tag on selected bbox';
      cb.addEventListener('click', async (e) => {
        e.stopPropagation();
        if (!_selectedBbox) return;
        await _toggleTagOnBbox(uuid);
      });

      const swatch = document.createElement('span');
      Object.assign(swatch.style, {
        width: '8px', height: '8px', borderRadius: '50%',
        flexShrink: '0', background: color ?? '#888', display: 'inline-block',
      });

      const iconEl = document.createElement('span');
      iconEl.textContent = icon ?? '';
      iconEl.style.cssText = 'font-size:0.9em; flex-shrink:0;';
      if (!icon) iconEl.style.display = 'none';

      const nameEl = document.createElement('span');
      nameEl.textContent = name;
      nameEl.style.cssText = 'font-size:0.88em; flex:1;';

      // Layout: [img-tag status] [swatch] [icon] [name…] [bbox checkbox]
      row.append(statusBtn, swatch, iconEl, nameEl, cb);

      row.addEventListener('click', () => {
        setTool(runtime, `tag:${uuid}`);
      });

      _tagRows.set(uuid, { row, checkbox: cb });
      wrap.appendChild(row);
    }

    return wrap;
  }

  async function _toggleTagOnBbox(tagUuid) {
    if (!_selectedBbox) return;
    const tag = _tagsMap.get(tagUuid) ?? await fetchTagByUuid(tagUuid);
    if (!tag) return;
    const tags = _selectedBbox.uiBBox.tags ?? [];
    const has = tags.some(t => t.uuid === tagUuid);
    let next;
    if (has) {
      next = tags.filter(t => t.uuid !== tagUuid);
    } else {
      const base = VERIFICATION_TAG_UUIDS.has(tagUuid)
        ? tags.filter(t => !VERIFICATION_TAG_UUIDS.has(t.uuid))
        : tags;
      next = [...base, { uuid: tagUuid, icon: tag.icon ?? null }];
    }
    _selectedBbox.uiBBox.tags = next;
    _selectedBbox.refreshIcons(next);
    runtime.pushState();
    events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
    _refreshCheckboxes();
  }

  // ── Refresh helpers ───────────────────────────────────────────────────────

  function _refreshCheckboxes() {
    const activeTags = new Set((_selectedBbox?.uiBBox?.tags ?? []).map(t => t.uuid));
    for (const [uuid, { checkbox }] of _tagRows) {
      checkbox.checked = activeTags.has(uuid);
      checkbox.disabled = !_selectedBbox;
    }
  }

  function _refreshHighlights() {
    const activeTagUuid = _activeTool.startsWith('tag:') ? _activeTool.slice(4) : null;
    for (const [uuid, { row }] of _tagRows) {
      const isActive = uuid === activeTagUuid;
      row.style.outline = isActive ? '2px solid #ff0033' : '';
      row.style.background = isActive ? '#ff003322' : '';
    }
  }

  // ── Full mount ────────────────────────────────────────────────────────────

  // Re-entrancy guard: TAGS_UPDATED / LABELS_UPDATED handlers call _mount()
  // without awaiting, so two synchronous events would interleave two mounts
  // and duplicate the panel. Coalesce into one trailing re-mount.
  let _mounting = false;
  let _remountQueued = false;

  async function _mount() {
    if (_mounting) {
      _remountQueued = true;
      return;
    }
    _mounting = true;
    try {
      do {
        _remountQueued = false;
        container.innerHTML = '';
        try {
          [_tagsMap, _labelsMap] = await Promise.all([fetchTags(), fetchLabels()]);
        } catch (err) {
          console.error('Tags panel: failed to load tags/labels', err);
          return;
        }

        container.appendChild(_buildLabelSection());
        container.appendChild(_buildTagList());

        _refreshLabelSection();
        _refreshCheckboxes();
        _refreshHighlights();
        // Apply any already-loaded statuses to the freshly built rows.
        _refreshTagStatusButtons();
      } while (_remountQueued);
    } finally {
      _mounting = false;
    }
  }

  // ── Event subscriptions ───────────────────────────────────────────────────

  // Shared handler: update panel content when a bbox is selected.
  // Used by both single-click and double-click; double-click fires click first,
  // so we skip the refresh when the bbox hasn't changed.
  function _onBboxSelected({ bboxView }) {
    if (!bboxView || bboxView === _selectedBbox) return;
    _selectedBbox = bboxView;
    _refreshLabelSection();
    _refreshCheckboxes();
  }

  const _unsubClick    = events.subscribeToCanvas(runtime.id, EventTypes.CANVAS_BBOX_CLICKED,        _onBboxSelected);
  // Double-click tab switch is handled by sidebar.js; content update is via _onBboxSelected.
  const _unsubDblClick = events.subscribeToCanvas(runtime.id, EventTypes.CANVAS_BBOX_DOUBLE_CLICKED, _onBboxSelected);

  const _unsubBboxChanged = events.subscribeToCanvas(
    runtime.id, EventTypes.CANVAS_BBOXES_CHANGED,
    () => _refreshCheckboxes()
  );

  const _unsubToolChanged = events.subscribeToCanvas(
    runtime.id, EventTypes.CANVAS_TOOL_CHANGED,
    ({ tool }) => {
      _activeTool = tool;
      _refreshHighlights();
    }
  );

  const _unsubImageLoaded = events.subscribeToCanvas(
    runtime.id, EventTypes.IMAGE_LOADED_ONTO_CANVAS,
    async ({ path }) => {
      _selectedBbox = null;
      // Best-effort flush for the departing image before we clear its state.
      // Fire-and-forget — we don't await, and don't re-queue on failure, because
      // we're already navigating away and the user can't act on a toast for a
      // different image.
      if (_currentImagePath && _pendingTagFlush.size) {
        const departingPath = _currentImagePath;
        const departingEntries = [..._pendingTagFlush];
        _pendingTagFlush.clear();
        Promise.allSettled(
          departingEntries.map(([uuid, status]) => setImageTagStatus(departingPath, uuid, status))
        ).then(results => {
          results.forEach((r, i) => {
            if (r.status === 'rejected')
              console.error('Failed to persist image tag status on navigate:', departingEntries[i], r.reason);
          });
        });
      } else {
        _pendingTagFlush.clear();
      }
      _currentImagePath = path;
      // Reset tri-state buttons immediately so the previous image's statuses
      // don't show while the fetch for the new image is in flight.
      _serverTagStatuses = new Map();
      _refreshTagStatusButtons();
      _refreshLabelSection();
      _refreshCheckboxes();
      await _fetchAndApplyTagStatuses(path);
    }
  );

  // Flush pending whole-image tag status changes when canvas annotations are saved.
  // Uses a while loop so that any entries added to _pendingTagFlush *during* an
  // await (e.g. the user cycled a button between two rapid saves) are picked up
  // in the next iteration rather than silently dropped.
  const _unsubSaved = events.subscribeToCanvas(
    runtime.id, EventTypes.CANVAS_ANNOTATIONS_SAVED,
    async () => {
      if (!_currentImagePath || !_pendingTagFlush.size || _flushInFlight) return;
      _flushInFlight = true;
      try {
        while (_pendingTagFlush.size && _currentImagePath) {
          const pathAtSave = _currentImagePath;
          const entries = [..._pendingTagFlush];
          _pendingTagFlush.clear();
          const results = await Promise.allSettled(
            entries.map(([uuid, status]) => setImageTagStatus(pathAtSave, uuid, status))
          );
          if (_currentImagePath !== pathAtSave) break; // navigated away mid-flush
          results.forEach((r, i) => {
            if (r.status === 'rejected') console.error('Failed to persist image tag status:', entries[i], r.reason);
          });
          const failed = entries.filter((_, i) => results[i].status === 'rejected');
          if (failed.length) {
            // Re-queue entries not superseded by newer user interaction.
            failed.forEach(([uuid, status]) => {
              if (!_pendingTagFlush.has(uuid)) _pendingTagFlush.set(uuid, status);
            });
            toast(`${failed.length} image tag state(s) failed to save`, 5000, 'error');
            break; // stop looping on persistent errors to avoid thrashing
          }
        }
      } finally {
        _flushInFlight = false;
      }
    }
  );

  const _unsubGlobal = events.subscribeMany([
    [EventTypes.DYNAMIC_LABELS_VISIBILITY_CHANGED, () => _refreshLabelSection()],
    [EventTypes.TAG_KIND_VISIBILITY_CHANGED,        () => _mount()],
    [EventTypes.TAGS_UPDATED,                       () => _mount()],
    [EventTypes.LABELS_UPDATED,                     () => _mount()],
  ]);

  // Initial render.
  _mount();

  // If an image was already loaded before this panel mounted (lazy init), we
  // missed the IMAGE_LOADED_ONTO_CANVAS event — fetch its statuses now so
  // tri-state buttons show the correct state immediately.
  const _alreadyLoadedPath = runtime?.state?.image?.image_path;
  if (_alreadyLoadedPath) {
    _currentImagePath = _alreadyLoadedPath;
    _fetchAndApplyTagStatuses(_alreadyLoadedPath);
  }

  return {
    destroy() {
      _unsubClick();
      _unsubDblClick();
      _unsubBboxChanged();
      _unsubToolChanged();
      _unsubImageLoaded();
      _unsubSaved();
      _unsubGlobal();
    },
  };
}
