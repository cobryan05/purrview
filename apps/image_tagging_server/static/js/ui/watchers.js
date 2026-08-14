import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchSourceList } from '/app-static/js/shared/api/sources.js';
import { fetchModelGroups } from '/app-static/js/shared/api/modelGroups.js';
import {
  fetchWatchProfiles, createWatchProfile, updateWatchProfile, deleteWatchProfile,
  validateWatchProfile,
} from '/app-static/js/shared/api/watchProfiles.js';
import { toast, buildPanelEditorBar } from '/app-static/js/ui/utils/index.js';
import { createListEditorScreen } from '/app-static/js/ui/utils/listEditorScreen.js';
import { openZonesModal } from '/app-static/js/ui/watchZonesModal.js';
import { takePendingSelection } from '/app-static/js/app/utils/tabs.js';

let _rendering = false;

// Relations, worded as a sentence fragment so a rule row reads as English. This is
// the only layout that makes "IoU vs containment" comprehensible to a non-expert.
// `iou` used to be worded "overlaps", which is the exact confusion it causes: IoU only
// reads as overlap for same-size boxes, and a cat lapping at a water bowl scores near
// zero on it. The honest wording sends that case to `overlaps` instead.
const RELATIONS = [
  { key: 'contains', text: 'is inside' },
  { key: 'overlaps', text: 'overlaps' },
  { key: 'centroid_in', text: 'has its centre in' },
  { key: 'iou', text: 'closely matches (same size)' },
  { key: 'distance', text: 'is near' },
  { key: 'present', text: 'is present' },
];

const ACTION_KINDS = [
  { key: 'publish_mqtt', text: 'Publish MQTT' },
  { key: 'record_clip', text: 'Record clip' },
  { key: 'save_snapshot', text: 'Save snapshot' },
];

const TRIGGERS = [
  { key: 'event_started', text: 'event start' },
  { key: 'event_ended', text: 'event end' },
];

/**
 * Read a numeric input, falling back when it is blank.
 *
 * `Number('')` is 0, which has already caused a silent-stop bug in this codebase —
 * an empty "Max duration" field would become a 0-second cap that closes every event
 * immediately.
 */
function num(input, fallback) {
  const raw = input?.value;
  if (raw === '' || raw == null) return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function select(options, value, onChange) {
  const sel = document.createElement('select');
  for (const opt of options) {
    const o = document.createElement('option');
    o.value = opt.key;
    o.textContent = opt.text;
    sel.appendChild(o);
  }
  sel.value = value ?? options[0]?.key;
  if (onChange) sel.addEventListener('change', onChange);
  return sel;
}

function numberInput(value, { step = '1', min = null, placeholder = '' } = {}) {
  const input = document.createElement('input');
  input.type = 'number';
  input.step = step;
  if (min != null) input.min = String(min);
  input.placeholder = placeholder;
  input.value = value ?? '';
  return input;
}

/**
 * A compact inline `label: control` pair.
 *
 * Deliberately not `makeFieldRow`: that helper pins its label to a fixed 100px column
 * and gives the control `flex: 1`, which is right for a stacked settings panel and
 * wrong here — it turned a four-field row into four full-width lines. These pairs size
 * to their content and flow along the row instead.
 */
function wfield(labelText, control, { width = null } = {}) {
  const wrap = el('label', 'watch-field');
  wrap.appendChild(el('span', 'watch-field__label', labelText));
  if (width) control.style.width = width;
  wrap.appendChild(control);
  return wrap;
}

/** The small ✕ that ends every row. Full-width "Remove" buttons cost a line each. */
function removeButton(onDelete, title) {
  const btn = el('button', 'watch-row-remove', '✕');
  btn.type = 'button';
  btn.title = title;
  btn.setAttribute('aria-label', title);
  btn.addEventListener('click', onDelete);
  return btn;
}

/**
 * A section heading with its "+ Add" button on the same line.
 *
 * The add button used to sit on its own line below the list, so every section spent
 * two lines on chrome before showing any content.
 */
function sectionHeader(text, addButton) {
  const head = el('div', 'watch-section-head');
  head.appendChild(el('span', 'panel-section-label', text));
  if (addButton) head.appendChild(addButton);
  return head;
}

// ── Source row ──────────────────────────────────────────────────────────────
function buildSourceRow(src, ctx) {
  const row = el('div', 'watch-source-row watch-row');
  row.dataset.testid = 'watch-source-row';

  const sourceSel = select(ctx.sourceOptions, src.sourceUuid, ctx.onAnyChange);
  sourceSel.dataset.testid = 'watch-source-select';
  row.appendChild(wfield('Source', sourceSel));

  const modelSel = select(
    [{ key: '', text: '(profile default)' }, ...ctx.modelGroupOptions],
    src.modelGroupUuid || '',
    ctx.onAnyChange,
  );
  row.appendChild(wfield('Models', modelSel));

  // The three rate fields are numeric and short; the units live in the label so the
  // inputs can stay narrow enough to sit on the same line as the selects.
  const rates = el('div', 'watch-row-rates');
  const fps = numberInput(src.analysisFps ?? '', { step: '0.5', min: 0, placeholder: 'auto' });
  fps.dataset.testid = 'watch-analysis-fps';
  const width = numberInput(src.analysisWidth ?? '', { step: '16', min: 0, placeholder: 'auto' });
  const retention = numberInput(src.recordRetentionS ?? '', { step: '5', min: 0, placeholder: 'auto' });
  fps.addEventListener('input', ctx.onAnyChange);
  width.addEventListener('input', ctx.onAnyChange);
  retention.addEventListener('input', ctx.onAnyChange);
  rates.appendChild(wfield('fps', fps, { width: '4.5rem' }));
  rates.appendChild(wfield('width px', width, { width: '5rem' }));
  rates.appendChild(wfield('buffer s', retention, { width: '4.5rem' }));
  row.appendChild(rates);

  // Zones live behind a modal because they need a video frame as a backdrop, which
  // is far too heavy to inline in a list row.
  let zones = (src.zones || []).map(z => ({ ...z }));
  const actions = el('div', 'watch-row-actions');
  const zonesBtn = el('button', 'panel-btn', `Zones (${zones.length})…`);
  zonesBtn.type = 'button';
  zonesBtn.dataset.testid = 'watch-zones-btn';
  zonesBtn.addEventListener('click', async () => {
    const result = await openZonesModal({
      sourceUuid: sourceSel.value,
      sourceName: ctx.sourceNameFor(sourceSel.value),
      zones,
    });
    if (result) {
      zones = result;
      zonesBtn.textContent = `Zones (${zones.length})…`;
      // The rules' zone dropdowns are derived from these zones, so they have to be
      // rebuilt here. Marking the editor dirty is not enough — nothing re-reads the
      // zone list until the profile is saved and reopened.
      ctx.onZonesChanged?.();
      ctx.onAnyChange();
    }
  });
  actions.appendChild(zonesBtn);
  actions.appendChild(removeButton(() => ctx.onDelete(), 'Remove source'));
  row.appendChild(actions);

  return {
    element: row,
    getZones: () => zones,
    getSource: () => ({
      sourceUuid: sourceSel.value,
      order: 0,
      enabled: true,
      modelGroupUuid: modelSel.value || null,
      analysisFps: fps.value === '' ? null : num(fps, null),
      analysisWidth: width.value === '' ? null : num(width, null),
      recordRetentionS: retention.value === '' ? null : num(retention, null),
      zones: zones.map(z => ({
        uuid: z.uuid,
        name: z.name,
        kind: z.kind,
        pointsJson: z.pointsJson,
        referenceWidth: z.referenceWidth ?? 0,
        referenceHeight: z.referenceHeight ?? 0,
      })),
    }),
  };
}

// ── Rule row ────────────────────────────────────────────────────────────────
function buildRuleRow(rule, ctx) {
  const row = el('div', 'watch-rule-row watch-row');
  row.dataset.testid = 'watch-rule-row';

  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.value = rule.name || '';
  nameInput.placeholder = 'e.g. cat using litter box';
  nameInput.dataset.testid = 'watch-rule-name';
  nameInput.addEventListener('input', ctx.onAnyChange);
  row.appendChild(wfield('Rule', nameInput));

  // The sentence: [subject A] [relation] [subject B].
  const sentence = el('div', 'watch-rule-sentence');
  const subjectA = select(ctx.labelOptions, rule.subjectARef || ctx.labelOptions[0]?.key, ctx.onAnyChange);
  const relation = select(RELATIONS, rule.relation || 'contains', () => {
    syncArity();
    ctx.onAnyChange();
  });
  relation.dataset.testid = 'watch-rule-relation';
  const subjectBKind = select(
    [{ key: 'zone', text: 'Zone' }, { key: 'label', text: 'Label' }],
    rule.subjectBKind === 'label' ? 'label' : 'zone',
    () => { syncSubjectB(); ctx.onAnyChange(); },
  );
  // Build the second subject from the option set its *own* kind implies. Seeding it
  // from the zone options regardless meant a label uuid never matched an <option>,
  // the select fell back to the first zone, and the later syncSubjectB() — which only
  // preserves a value still on offer — landed it on the first label instead: a saved
  // "is inside Litter Box" silently reopened, and re-saved, as "is inside Cat".
  const subjectBOptions = subjectBKind.value === 'label' ? ctx.labelOptions : ctx.zoneOptions;
  const subjectB = select(subjectBOptions, rule.subjectBRef || subjectBOptions[0]?.key, ctx.onAnyChange);
  subjectB.dataset.testid = 'watch-rule-subject-b';

  sentence.append(el('span', null, 'When'), subjectA, relation, subjectBKind, subjectB);
  row.appendChild(sentence);

  function syncSubjectB() {
    const options = subjectBKind.value === 'label' ? ctx.labelOptions : ctx.zoneOptions;
    const current = subjectB.value;
    subjectB.innerHTML = '';
    for (const opt of options) {
      const o = document.createElement('option');
      o.value = opt.key;
      o.textContent = opt.text;
      subjectB.appendChild(o);
    }
    if (options.some(o => o.key === current)) subjectB.value = current;
  }

  function syncArity() {
    // `present` is arity 1; showing a second subject for it would imply a
    // comparison that never happens.
    const isUnary = relation.value === 'present';
    subjectBKind.hidden = isUnary;
    subjectB.hidden = isUnary;
  }

  // Tuning keeps its four inputs in threshold / start / clear / max order — the
  // Playwright defaults test indexes `.watch-rule-tuning input` positionally.
  // The long labels ("Ends after clear (s)") became units-in-the-label plus a
  // tooltip: spelled out, the four of them alone filled a line.
  const tuning = el('div', 'watch-rule-tuning');
  const threshold = numberInput(rule.threshold ?? 0.6, { step: '0.05', min: 0 });
  const minDuration = numberInput(rule.minDurationS ?? 2, { step: '0.5', min: 0 });
  const rearm = numberInput(rule.rearmDelayS ?? 10, { step: '1', min: 0 });
  const maxDuration = numberInput(rule.maxDurationS ?? 300, { step: '10', min: 1 });
  for (const input of [threshold, minDuration, rearm, maxDuration]) {
    input.addEventListener('input', ctx.onAnyChange);
  }
  const tuningFields = [
    ['match', threshold, 'Match threshold — how strongly the relation must hold'],
    ['starts after', minDuration, 'Seconds the relation must hold before the event starts'],
    ['ends after', rearm, 'Seconds the relation must stay clear before the event ends'],
    ['max', maxDuration, 'Hard cap on event length, in seconds'],
  ];
  for (const [label, input, title] of tuningFields) {
    const field = wfield(label, input, { width: '4.5rem' });
    field.title = title;
    tuning.appendChild(field);
  }
  row.appendChild(tuning);

  row.appendChild(removeButton(() => ctx.onDelete(), 'Remove rule'));

  syncSubjectB();
  syncArity();

  return {
    element: row,
    /**
     * Adopt a new set of zone options.
     *
     * `refreshZoneOptions()` has always called this, but the row never returned it and
     * the `?.` on the call site swallowed the miss — so a zone drawn in the modal was
     * unreachable from any rule until the whole watcher was saved and reopened, which
     * is the wrong order to build a rule in.
     *
     * `syncSubjectB` re-selects the current value when it is still on offer, so a rule
     * already pointing at a zone keeps it; only a rule whose zone genuinely went away
     * falls back to the first option.
     */
    refreshZones: (options) => {
      ctx.zoneOptions = options;
      if (subjectBKind.value !== 'label') syncSubjectB();
    },
    getRule: () => {
      const isUnary = relation.value === 'present';
      return {
        uuid: rule.uuid || null,
        order: 0,
        name: nameInput.value.trim() || 'Unnamed rule',
        enabled: true,
        subjectAKind: 'label',
        subjectARef: subjectA.value,
        relation: relation.value,
        subjectBKind: isUnary ? 'none' : subjectBKind.value,
        subjectBRef: isUnary ? null : subjectB.value,
        threshold: num(threshold, 0.6),
        // These three defaults are not arbitrary: without them a raw threshold flaps
        // into an MQTT storm and a dozen clips for one visit.
        minDurationS: num(minDuration, 2),
        rearmDelayS: num(rearm, 10),
        maxDurationS: num(maxDuration, 300),
      };
    },
  };
}

// ── Action row ──────────────────────────────────────────────────────────────
function buildActionRow(action, ctx) {
  const row = el('div', 'watch-action-row watch-row');
  row.dataset.testid = 'watch-action-row';

  const kind = select(ACTION_KINDS, action.kind || 'publish_mqtt', () => { syncParams(); ctx.onAnyChange(); });
  kind.dataset.testid = 'watch-action-kind';
  const trigger = select(TRIGGERS, action.trigger || 'event_ended', ctx.onAnyChange);
  const ruleSel = select(
    [{ key: '', text: 'all rules' }, ...ctx.ruleOptions],
    action.ruleUuid || '',
    ctx.onAnyChange,
  );

  row.appendChild(wfield('Action', kind));
  row.appendChild(wfield('for', ruleSel));
  row.appendChild(wfield('on', trigger));

  const params = el('div', 'watch-action-params');
  row.appendChild(params);

  const stored = { ...(action.paramsJson || {}) };
  let readParams = () => ({});

  function syncParams() {
    params.innerHTML = '';
    if (kind.value === 'record_clip') {
      const preroll = numberInput(stored.preroll_s ?? 10, { step: '1', min: 0 });
      const postroll = numberInput(stored.postroll_s ?? 5, { step: '1', min: 0 });
      const maxLength = numberInput(stored.max_length_s ?? 180, { step: '10', min: 1 });
      for (const i of [preroll, postroll, maxLength]) i.addEventListener('input', ctx.onAnyChange);
      params.appendChild(wfield('Pre-roll s', preroll, { width: '4.5rem' }));
      params.appendChild(wfield('post-roll s', postroll, { width: '4.5rem' }));
      params.appendChild(wfield('max s', maxLength, { width: '4.5rem' }));
      readParams = () => ({
        preroll_s: num(preroll, 10),
        postroll_s: num(postroll, 5),
        max_length_s: num(maxLength, 180),
      });
    } else if (kind.value === 'publish_mqtt') {
      const retain = document.createElement('input');
      retain.type = 'checkbox';
      retain.checked = Boolean(stored.retain);
      retain.addEventListener('change', ctx.onAnyChange);
      params.appendChild(wfield('Retain', retain));
      // The topic pattern is reference material, not a control — a tooltip keeps it
      // available without spending a full line under every MQTT action.
      const hint = el('span', 'watch-hint', 'topic pattern');
      hint.title = 'Topic: <prefix>/<watcher>/<source>/event/<rule>. Broker is configured in Settings.';
      params.appendChild(hint);
      readParams = () => ({ retain: retain.checked, qos: 0 });
    } else {
      const dir = document.createElement('input');
      dir.type = 'text';
      dir.value = stored.output_dir || '';
      dir.placeholder = 'default snapshot directory';
      dir.addEventListener('input', ctx.onAnyChange);
      params.appendChild(wfield('Output dir', dir, { width: '14rem' }));
      readParams = () => (dir.value.trim() ? { output_dir: dir.value.trim() } : {});
    }
  }

  row.appendChild(removeButton(() => ctx.onDelete(), 'Remove action'));

  syncParams();

  return {
    element: row,
    getAction: () => ({
      uuid: action.uuid || null,
      ruleUuid: ruleSel.value || null,
      order: 0,
      enabled: true,
      kind: kind.value,
      trigger: trigger.value,
      paramsJson: readParams(),
    }),
  };
}

/**
 * Renders the Configuration → Watchers pane.
 *
 * Same list-beside-editor shape as Datasets / Filter Chains / Model Stacks, because a
 * Watch Profile is the same kind of thing: a curated definition that a task merely
 * executes.
 */
export async function renderWatchProfileManager({ target }) {
  if (_rendering) return;
  _rendering = true;
  try {
    const container = typeof target === 'string' ? document.getElementById(target) : target;
    if (!container) return;
    container.innerHTML = '';

    const screen = createListEditorScreen({
      container,
      prefix: 'wp',
      title: 'Watchers',
      itemUuidAttr: 'watchUuid',
      searchable: true,
      searchPlaceholder: 'Filter watchers…',
      newLabel: '+ New watcher',
      onNew: () => openEditor(null),
    });

    const [profilesMap, sourcesMap, labelsMap, modelGroups] = await Promise.all([
      fetchWatchProfiles().catch(() => new Map()),
      fetchSourceList().catch(() => new Map()),
      // fetchLabels() returns a Map, not an array — the fallback must match.
      fetchLabels().catch(() => new Map()),
      fetchModelGroups().catch(() => new Map()),
    ]);

    const sourceOptions = [...sourcesMap.values()].map(s => ({ key: s.uuid, text: s.name || s.uuid }));
    const labelOptions = [...labelsMap.values()].map(l => ({ key: l.uuid, text: l.name || l.uuid }));
    const modelGroupOptions = [...modelGroups.values()].map(g => ({ key: g.uuid, text: g.name || g.uuid }));
    const sourceNameFor = uuid => sourcesMap.get(uuid)?.name || uuid;

    function renderList(selectedUuid) {
      screen.listBody.innerHTML = '';
      const profiles = [...profilesMap.values()].sort((a, b) => a.name.localeCompare(b.name));
      screen.setCount(profiles.length);
      if (!profiles.length) {
        // createListItem/createListEmpty build a node; appending is the caller's job.
        screen.listBody.appendChild(screen.createListEmpty('No watchers yet.'));
        return;
      }
      for (const profile of profiles) {
        const item = screen.createListItem({
          uuid: profile.uuid,
          name: profile.name,
          meta: `${profile.sources?.length || 0} source(s), ${profile.rules?.length || 0} rule(s)`,
          onClick: () => openEditor(profile),
        });
        item.dataset.testid = 'watcher-list-item';
        screen.listBody.appendChild(item);
      }
      screen.updateListSelection(selectedUuid);
    }

    function openEditor(profile) {
      screen.editorBody.innerHTML = '';
      const sourceRows = [];
      const ruleRows = [];
      const actionRows = [];

      const bar = buildPanelEditorBar({
        container: screen.editorBar,
        item: profile,
        newTitle: 'New watcher',
        deleteTitle: 'Delete watcher',
        onSave: () => save(),
        onDelete: profile ? () => remove() : null,
      });
      const onAnyChange = () => bar.setDirty(true);

      // ── Identity ──
      // Four stacked full-width fields became one wrapping strip: the whole identity
      // block is now roughly the height a single one of them used to occupy.
      const identity = el('div', 'watch-identity');
      const nameInput = document.createElement('input');
      nameInput.type = 'text';
      nameInput.value = profile?.name || '';
      nameInput.placeholder = 'e.g. House cams';
      nameInput.dataset.testid = 'watcher-name';
      nameInput.addEventListener('input', onAnyChange);
      identity.appendChild(wfield('Name', nameInput));

      const enabled = document.createElement('input');
      enabled.type = 'checkbox';
      enabled.checked = profile ? profile.enabled !== false : true;
      enabled.addEventListener('change', onAnyChange);
      identity.appendChild(wfield('Enabled', enabled));

      const description = document.createElement('input');
      description.type = 'text';
      description.value = profile?.description || '';
      description.placeholder = 'optional';
      description.addEventListener('input', onAnyChange);
      identity.appendChild(wfield('Description', description));

      const defaultModels = select(
        [{ key: '', text: '(none)' }, ...modelGroupOptions],
        profile?.defaultModelGroupUuid || '',
        onAnyChange,
      );
      identity.appendChild(wfield('Default models', defaultModels));
      screen.editorBody.appendChild(identity);

      // ── Sources ──
      const addSourceBtn = el('button', 'panel-btn', '+ Add source');
      addSourceBtn.type = 'button';
      addSourceBtn.dataset.testid = 'watch-add-source';
      screen.editorBody.appendChild(sectionHeader('Sources', addSourceBtn));
      const sourcesBox = el('div', 'watch-rows');
      sourcesBox.dataset.testid = 'watch-sources-box';
      screen.editorBody.appendChild(sourcesBox);

      function addSource(src) {
        const entry = buildSourceRow(src, {
          sourceOptions,
          modelGroupOptions,
          sourceNameFor,
          onZonesChanged: () => refreshZoneOptions(),
          onAnyChange,
          onDelete: () => {
            entry.element.remove();
            sourceRows.splice(sourceRows.indexOf(entry), 1);
            refreshZoneOptions();
            onAnyChange();
          },
        });
        sourceRows.push(entry);
        sourcesBox.appendChild(entry.element);
        refreshZoneOptions();
      }

      addSourceBtn.addEventListener('click', () => {
        addSource({ sourceUuid: sourceOptions[0]?.key, zones: [] });
        onAnyChange();
      });

      // ── Rules ──
      const addRuleBtn = el('button', 'panel-btn', '+ Add rule');
      addRuleBtn.type = 'button';
      addRuleBtn.dataset.testid = 'watch-add-rule';
      screen.editorBody.appendChild(sectionHeader('Rules', addRuleBtn));
      const rulesBox = el('div', 'watch-rows');
      rulesBox.dataset.testid = 'watch-rules-box';
      screen.editorBody.appendChild(rulesBox);

      // Zone options are derived from the sources' zones, so a rule can only ever
      // reference a zone that actually exists in this profile.
      function zoneOptions() {
        const opts = [];
        for (const entry of sourceRows) {
          for (const zone of entry.getZones()) {
            opts.push({ key: zone.uuid, text: zone.name });
          }
        }
        return opts.length ? opts : [{ key: '', text: '(no zones defined)' }];
      }

      function refreshZoneOptions() {
        for (const entry of ruleRows) entry.refreshZones?.(zoneOptions());
      }

      function addRule(rule) {
        const entry = buildRuleRow(rule, {
          labelOptions: labelOptions.length ? labelOptions : [{ key: '', text: '(no labels)' }],
          zoneOptions: zoneOptions(),
          onAnyChange,
          onDelete: () => {
            entry.element.remove();
            ruleRows.splice(ruleRows.indexOf(entry), 1);
            onAnyChange();
          },
        });
        ruleRows.push(entry);
        rulesBox.appendChild(entry.element);
      }

      addRuleBtn.addEventListener('click', () => { addRule({}); onAnyChange(); });

      // ── Actions ──
      const addActionBtn = el('button', 'panel-btn', '+ Add action');
      addActionBtn.type = 'button';
      addActionBtn.dataset.testid = 'watch-add-action';
      screen.editorBody.appendChild(sectionHeader('Actions', addActionBtn));
      const actionsBox = el('div', 'watch-rows');
      actionsBox.dataset.testid = 'watch-actions-box';
      screen.editorBody.appendChild(actionsBox);

      function addAction(action) {
        const entry = buildActionRow(action, {
          ruleOptions: ruleRows.map(r => ({ key: r.getRule().uuid || '', text: r.getRule().name })),
          onAnyChange,
          onDelete: () => {
            entry.element.remove();
            actionRows.splice(actionRows.indexOf(entry), 1);
            onAnyChange();
          },
        });
        actionRows.push(entry);
        actionsBox.appendChild(entry.element);
      }

      addActionBtn.addEventListener('click', () => { addAction({}); onAnyChange(); });

      // ── Validation report ──
      const report = el('div', 'watch-validation');
      report.dataset.testid = 'watch-validation';
      screen.editorBody.appendChild(report);

      for (const src of profile?.sources || []) addSource(src);
      for (const rule of profile?.rules || []) addRule(rule);
      for (const action of profile?.actions || []) addAction(action);

      async function showValidation(uuid) {
        try {
          const result = await validateWatchProfile(uuid);
          report.innerHTML = '';
          for (const err of result.errors) {
            report.appendChild(el('div', 'notices-bar__item--error', `⚠ ${err}`));
          }
          for (const warn of result.warnings) {
            report.appendChild(el('div', 'notices-bar__item--warning', `• ${warn}`));
          }
          if (result.valid && !result.warnings.length) {
            report.appendChild(el('div', 'notices-bar__item--info', '✓ This watcher looks ready to run.'));
          }
        } catch (e) {
          report.textContent = '';
        }
      }

      function collect() {
        return {
          name: nameInput.value.trim(),
          description: description.value.trim(),
          enabled: enabled.checked,
          defaultModelGroupUuid: defaultModels.value || null,
          sources: sourceRows.map((entry, i) => ({ ...entry.getSource(), order: i })),
          rules: ruleRows.map((entry, i) => ({ ...entry.getRule(), order: i })),
          actions: actionRows.map((entry, i) => ({ ...entry.getAction(), order: i })),
        };
      }

      async function save() {
        const payload = collect();
        if (!payload.name) {
          toast('A watcher needs a name', 'error');
          return;
        }
        try {
          const saved = profile
            ? await updateWatchProfile(profile.uuid, payload)
            : await createWatchProfile(payload);
          profilesMap.set(saved.uuid, saved);
          toast(`Saved watcher "${saved.name}"`, 'success');
          bar.setDirty(false);
          renderList(saved.uuid);
          // Await the re-render: an unawaited one silently swallows its errors.
          await showValidation(saved.uuid);
          openEditor(saved);
        } catch (e) {
          toast(e.message || 'Failed to save watcher', 'error');
        }
      }

      async function remove() {
        try {
          await deleteWatchProfile(profile.uuid);
          profilesMap.delete(profile.uuid);
          toast(`Deleted watcher "${profile.name}"`, 'success');
          screen.editorBody.innerHTML = '';
          screen.editorBar.innerHTML = '';
          renderList(null);
        } catch (e) {
          toast(e.message || 'Failed to delete watcher', 'error');
        }
      }

      screen.setDoSave(save);
      if (profile) showValidation(profile.uuid);
    }

    // Arriving from a Runs launch card's ⚙ means "show me *this* watcher".
    const wantedUuid = takePendingSelection('tab-watcher-management');
    const wanted = wantedUuid ? profilesMap.get(wantedUuid) ?? null : null;
    renderList(wanted?.uuid ?? null);
    if (wanted) openEditor(wanted);
  } finally {
    _rendering = false;
  }
}
