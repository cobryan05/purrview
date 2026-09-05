import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchSourceIngestTiers, fetchSourceList } from '/app-static/js/shared/api/sources.js';
import { fetchModelGroups } from '/app-static/js/shared/api/modelGroups.js';
import { fetchModelsList } from '/app-static/js/shared/api/models.js';
import {
  fetchWatchProfiles, createWatchProfile, updateWatchProfile, deleteWatchProfile,
  validateWatchProfile, fetchTrackers, fetchWatchActionSchemas,
} from '/app-static/js/shared/api/watchProfiles.js';
import { toast, buildPanelEditorBar } from '/app-static/js/ui/utils/index.js';
import { createListEditorScreen } from '/app-static/js/ui/utils/listEditorScreen.js';
import { promptRestartAffectedTasks } from '/app-static/js/ui/utils/restartAffectedTasks.js';
import { openSourceRegionsModal } from '/app-static/js/ui/sourceRegionsModal.js';
import { fetchSourceRegions, REGION_ROLE_MASK, REGION_ROLE_ZONE } from '/app-static/js/shared/api/sourceRegions.js';
import { createMultiSelectDropdown } from '/app-static/js/ui/utils/createMultiSelectDropdown.js';
import { openFileBrowserModal } from '/app-static/js/ui/utils/openFileBrowserModal.js';
import { openAnchoredPopup } from '/app-static/js/ui/utils/anchoredPopup.js';
import { openOverlapPreviewModal } from '/app-static/js/ui/utils/overlapPreviewModal.js';
import { flattenHierarchy } from '/app-static/js/ui/labelList.js';
import { NEW_ITEM_SELECTION, navigateTabs, takePendingSelection } from '/app-static/js/app/utils/tabs.js';
import { fetchTrainingCollectors } from '/app-static/js/shared/api/trainingCollectors.js';
import { buildSchemaParams } from '/app-static/js/ui/utils/schemaParamsForm.js';

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
  // Reads as a whole sentence on its own because it has no subject: motion is the one
  // relation measured from pixels rather than from a detection, so there is no label to
  // put in front of it. `syncArity` hides the subject slot when this is chosen.
  { key: 'motion', text: 'motion is detected' },
];

// Whether the event spans the rule or is an instant at one of its edges. Worded as the
// tail of the row's sentence rather than as `while / on_enter / on_exit`, which name the
// implementation and not the choice.
//
// ⚠️ These say **event**, never "records". A rule does not record anything and cannot
// know whether it ever will — recording is one of several actions, scoped separately,
// and a rule may well have none attached. The wording used to promise "Records the whole
// time", which is wrong twice over: it claims an action the rule does not own, and for
// the two impulses it is not even true of the action, whose footage comes entirely from
// its own pre-roll and post-roll (see the tooltip, and `RuleSpec.trigger` in
// `apps/helpers/watch/rules.py`). What the user is picking is how long the *event* lasts.
const RULE_TRIGGERS = [
  { key: 'while', text: 'Event the whole time' },
  { key: 'on_enter', text: 'Event the moment it starts' },
  { key: 'on_exit', text: 'Event the moment it ends' },
];

/** Fraction of a region that must change for a motion rule to hold, by default.
 *  2% of the frame is roughly a person at mid-distance; the geometric relations' 0.6
 *  would demand most of the picture change and could never fire. */
const MOTION_DEFAULT_AREA = 0.02;

/** Per-pixel change that counts as movement, matching `watch.motion.DEFAULT_SENSITIVITY`. */
const MOTION_DEFAULT_SENSITIVITY = 0.1;

/**
 * What a rule's `threshold` actually means, per relation — the caption and the tooltip
 * for the same number.
 *
 * One field, six meanings. It was captioned `match` for all of them, which is a
 * geometric word for a number that under `present` is nothing of the kind: there is no
 * second box, and `evaluate_predicate` compares it against the *detection confidence*
 * of the best track. Captioning that `conf` is the difference between a number you can
 * reason about and one you can only fiddle with.
 *
 * `distance` gets its own caption for the opposite reason: its threshold is an upper
 * bound (a distance you must be *within*), so raising it loosens the rule while raising
 * every other relation's tightens it. `centroid_in` says out loud that it ignores the
 * number entirely — a centre is inside or it is not.
 *
 * Mirrors the branches in `evaluate_predicate` (`apps/helpers/watch/rules.py`).
 */
const THRESHOLD_HELP = {
  present: {
    label: 'conf',
    title: 'Minimum detection confidence, 0–1. The label must be detected at least this '
      + 'strongly to count as present — no geometry is measured for this relation. Raise it '
      + 'to ignore shaky detections, lower it to catch a half-hidden subject.',
  },
  contains: {
    label: 'match',
    title: 'How much of the subject must lie inside the target, as a fraction of the '
      + 'subject’s own area. 1 demands the whole box; 0.5 fires when half of it is in.',
  },
  overlaps: {
    label: 'match',
    // No markdown emphasis in a tooltip: a `title` renders as plain text, so asterisks
    // reach the user as asterisks. The wording carries the emphasis instead.
    title: 'How much the two must overlap, as a fraction of the smaller box’s area rather '
      + 'than of the pair — so a cat lapping at a bowl scores on the cat, not on the bowl.',
  },
  iou: {
    label: 'match',
    title: 'Intersection over union the pair must reach. Only reads as “overlap” for boxes '
      + 'of similar size; a small box inside a large one scores near zero.',
  },
  distance: {
    label: 'within',
    title: 'Greatest centre-to-centre distance that still counts, as a fraction of the '
      + 'frame diagonal. Higher is looser here — the opposite of every other relation.',
  },
  centroid_in: {
    label: 'match',
    title: 'Not used by this relation: the subject’s centre is either inside the target or '
      + 'it is not. Use the nearby margin to widen the target instead.',
  },
  motion: {
    label: 'area',
    title: 'How much of the watched area must change between frames, as a fraction of it. '
      + '0.02 is about a person at mid-distance; 0.2 is most of the view. Masked areas are '
      + 'left out of both what changed and what is measured, so a mask over a busy road '
      + 'does not just fail to trigger — it stops diluting this number for the rest of the '
      + 'frame. Pair it with the sensitivity beside it, which sets how big a change to a '
      + 'single pixel counts at all.',
  },
};

const ACTION_KINDS = [
  { key: 'publish_mqtt', text: 'Publish MQTT' },
  { key: 'publish_mqtt_image', text: 'Publish MQTT image' },
  { key: 'record_clip', text: 'Record clip' },
  // "Event timelapse", not "Record event timelapse": the kind dropdown sizes itself to
  // its widest option, and the longer label pushed the whole action row onto a second
  // line. It also reads better beside "Save timelapse" — the distinction between them is
  // per-event versus all-day, which is exactly what the first word says.
  { key: 'record_event_timelapse', text: 'Event timelapse' },
  { key: 'save_snapshot', text: 'Save snapshot' },
  { key: 'save_timelapse', text: 'Save timelapse' },
  { key: 'save_event', text: 'Save event' },
  // "Collect training", not "Collect training images": the dropdown sizes itself to its
  // widest option, the same reason "Event timelapse" is shortened above.
  { key: 'collect_training_images', text: 'Collect training' },
];

/**
 * What each section of the editor is for, shown on hover of its heading.
 *
 * The individual controls all explain themselves, but nothing said what the four
 * groups *are* or how they relate — and the relationship is the part you cannot infer
 * from any one field: sources are watched, tracking turns detections into objects that
 * persist between frames, rules ask questions about those objects, and actions are what
 * happens when a rule's answer changes. Read in that order the pane is a pipeline;
 * read as four lists of rows it is not.
 */
const SECTION_HELP = {
  sources: 'The cameras this watcher pulls frames from, and how fast. Everything below applies to '
    + 'all of them unless a rule or action is scoped to particular ones.',
  tracking: 'How detections on consecutive frames are joined into one object that persists. This is '
    + 'what lets a rule say "for 2 seconds" or "has left" — without it every frame is unrelated to '
    + 'the last.',
  rules: 'The questions asked of each frame: which labels, in what relation to which zone, for how '
    + 'long. A rule going true opens an event and going false closes it.',
  actions: 'What happens when a rule opens or closes an event — publish, record, collect training '
    + 'data. Each one can be scoped to particular rules and sources.',
};

const TRIGGERS = [
  { key: 'event_started', text: 'event start' },
  { key: 'event_ended', text: 'event end' },
  { key: 'frame_analysed', text: 'every analysed frame' },
];

/**
 * The trigger that fires per frame rather than per event, mirroring
 * `FRAME_ANALYSED` in `apps/helpers/watch/scoping.py`. A frame belongs to no rule, so
 * an action on it has no rule scope to show.
 */
const FRAME_ANALYSED_TRIGGER = 'frame_analysed';

/**
 * Triggers each action kind can actually do something on. Absent = every trigger.
 *
 * Only `record_clip` is constrained, and the constraint is real rather than a policy:
 * `RecordClipAction.run` returns early unless the phase is END, because the window to
 * cut is not known until the event closes — that is the whole point of recording into
 * a rewindable buffer. Offering the other two triggers let a watcher be saved with a
 * clip action that could never fire, with no error at save time and no clip at run
 * time. Keyed by kind so a second constrained action is one line here.
 */
const ACTION_TRIGGERS = {
  record_clip: ['event_ended'],
  record_event_timelapse: ['event_ended'],
  save_timelapse: ['frame_analysed'],
  save_event: ['event_started'],
};

/** Why an action kind's trigger list is shorter than the full set. */
const TRIGGER_LIMIT_HELP = {
  record_clip:
    'A clip can only be cut when the event ends. Its length is not known until then — '
    + 'the pre-roll is taken retroactively out of the rolling buffer, which is what lets '
    + 'you record the seconds before the rule fired. Use the pre-roll and post-roll '
    + 'settings to cover what happened either side.',
  record_event_timelapse:
    'The video is compiled when the event ends, because that is when its last frame is '
    + 'known. It does not only run then: while the event is open it quietly keeps a frame '
    + 'every few seconds, and those are what it compiles. Nothing is kept once the event '
    + 'is over, so an event that never opens costs nothing. For a timelapse that runs all '
    + 'day whether or not anything is happening, use Save timelapse instead.',
  save_timelapse:
    'A timelapse has to keep shooting through the hours when nothing happens, so it '
    + 'runs on the frame the watcher is already pulling from the camera rather than on '
    + 'an event. Its own interval decides how often a frame is kept — everything in '
    + 'between is discarded without being written. For a picture of one event, use '
    + 'Save snapshot instead.',
  save_event:
    'The event is filed when it opens, so it is visible in Events while it is still '
    + 'happening rather than only once it is over. There is nothing left to do at the '
    + 'end — the row is closed out with its duration either way.',
};

/**
 * Action kinds that put the event in the Events view. Mirrors `Action.logs_event()` in
 * `apps/helpers/actions/action.py`, which is what the watcher actually reads — this
 * copy only drives the hint below the action list.
 *
 * An event with none of these in scope still fires its actions and still publishes; it
 * simply leaves no row. That is the point: a rule like "a cat is present" matches for
 * hours a day, and filing every match buried the events that had a clip to watch.
 */
const EVENT_LOGGING_KINDS = new Set(['record_clip', 'record_event_timelapse', 'save_event']);

/** The triggers offered for one action kind. */
function triggersFor(kind) {
  const allowed = ACTION_TRIGGERS[kind];
  return allowed ? TRIGGERS.filter((t) => allowed.includes(t.key)) : TRIGGERS;
}

/**
 * Default image topic template, mirroring `DEFAULT_IMAGE_TOPIC_TEMPLATE` in
 * `apps/helpers/actions/mqttPublishImageAction.py`. Keyed by source rather than by
 * rule: the frame belongs to the camera, whatever happened to fire on it.
 */
const DEFAULT_IMAGE_TOPIC_TEMPLATE = '{watcher}/{source}/image';

/**
 * Default timelapse directory, relative to the image dataset root. Mirrors
 * `DEFAULT_TIMELAPSE_DIR` in `apps/helpers/actions/saveTimelapseAction.py`.
 */
const DEFAULT_TIMELAPSE_DIR = 'timelapse';

/**
 * Default snapshot folder, relative to the image dataset root. Mirrors
 * `DEFAULT_SNAPSHOT_DIR` in `apps/helpers/actions/saveSnapshotAction.py`.
 */
const DEFAULT_SNAPSHOT_DIR = 'watch_snapshots';

/**
 * Default MQTT topic template — reproduces the historical hard-coded scheme exactly, so
 * an action saved before templates existed publishes where it always did.
 * Mirrors `DEFAULT_TOPIC_TEMPLATE` in `apps/helpers/actions/mqttPublishAction.py`.
 */
const DEFAULT_TOPIC_TEMPLATE = '{watcher}/{source}/event/{rule}';

/**
 * The placeholder vocabulary, mirroring `apps/helpers/watch/templating.py`.
 * Kept in sync by hand — the backend is the authority; this list is for help only.
 *
 * Each entry carries its own one-line meaning. A bare list of names told you a token
 * existed but not what it resolved to, which for pairs like `{rule}` / `{rule_name}` or
 * `{source}` / `{source_config_name}` is the only thing worth knowing.
 */
const TEMPLATE_PLACEHOLDER_HELP = [
  { token: '{rule}', help: 'The rule\'s event label, or its name when it has none.' },
  { token: '{rule_name}', help: 'The rule\'s name, always — never the event label.' },
  { token: '{event_label}', help: 'The rule\'s event label. Empty when it has none.' },
  { token: '{rule_uuid}', help: 'Stable id of the rule. Survives a rename.' },
  { token: '{watcher}', help: 'This watcher\'s name.' },
  { token: '{watcher_uuid}', help: 'Stable id of the watcher.' },
  { token: '{source}', help: 'The source\'s MQTT name for this watcher, falling back to its own name.' },
  { token: '{source_config_name}', help: 'The source\'s name as the Sources pane shows it, ignoring any MQTT name.' },
  { token: '{source_uuid}', help: 'Stable id of the source. Survives a rename.' },
  { token: '{subject}', help: 'Which label actually matched — the cat, not the whole list.' },
  { token: '{subject_uuid}', help: 'Stable id of the matched subject.' },
  { token: '{target}', help: 'Which zone or label it matched against — the litter box that fired.' },
  { token: '{target_uuid}', help: 'Stable id of the matched target.' },
  { token: '{phase}', help: 'start, update or end.' },
  { token: '{event_uuid}', help: 'Stable id of this one event, for correlating topics.' },
  // Local time, not UTC: these name folders a human browses, and a UTC day boundary
  // files an evening under tomorrow. Dashes in {time} because it goes into filenames.
  { token: '{date}', help: 'The local date, as 2026-08-18.' },
  { token: '{time}', help: 'The local time, as 14-30-00. Dashes, so it is safe in a filename.' },
  { token: '{year}', help: 'Local year, as 2026.' },
  { token: '{month}', help: 'Local month, zero-padded, as 08.' },
  { token: '{day}', help: 'Local day of the month, zero-padded, as 18.' },
  { token: '{hour}', help: 'Local hour, 24-hour and zero-padded, as 14.' },
];

/** Names only, for the places that just need to list them. */
const TEMPLATE_PLACEHOLDERS = TEMPLATE_PLACEHOLDER_HELP.map(p => p.token);

/** Stand-in values for the topic preview, so the shape of the result is visible. */
const TEMPLATE_EXAMPLE = {
  rule: 'cat-using-litter-box',
  rule_name: 'cat-using-litter-box',
  event_label: 'litterbox',
  rule_uuid: 'rule-uuid',
  watcher: 'house-cams',
  watcher_uuid: 'watcher-uuid',
  // The watcher's MQTT name for the source when one is set, which is the whole reason
  // the override exists — a short, stable topic level rather than the descriptive name
  // the Sources list shows.
  source: 'back-room',
  source_config_name: 'back-room-reolink-upstairs',
  source_uuid: 'source-uuid',
  subject: 'tabby',
  subject_uuid: 'label-uuid',
  target: 'carpeted-litter-box',
  target_uuid: 'zone-uuid',
  phase: 'start',
  event_uuid: 'event-uuid',
  date: '2026-08-18',
  time: '14-30-00',
  year: '2026',
  month: '08',
  day: '18',
  hour: '14',
};

/**
 * Render a topic template against example values, for the live hint under the field.
 *
 * Mirrors `render_topic`: unknown placeholders are left verbatim so a typo is visible,
 * and empty levels collapse rather than emitting `a//b`.
 */
function renderTopicExample(template) {
  const rendered = template.replace(/\{([a-zA-Z][a-zA-Z0-9_]*)\}/g, (match, key) => (
    Object.prototype.hasOwnProperty.call(TEMPLATE_EXAMPLE, key) ? TEMPLATE_EXAMPLE[key] : match
  ));
  return rendered.split('/').filter(Boolean).join('/');
}

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

// ── Model selection ─────────────────────────────────────────────────────────
// A watcher picks *either* a Model Stack or one bare model. Two dropdowns would need a
// rule about which wins; one dropdown with two optgroups makes them exclusive by
// construction, which is exactly what the backend enforces. The option values are
// prefixed because a stack uuid and a model name share no namespace and a bare value
// could not say which it was.
const MODEL_CHOICE_GROUP = 'g:';
const MODEL_CHOICE_MODEL = 'm:';

/**
 * Encode a stored (groupUuid, modelName) pair as an option value.
 * @returns {string} '' when neither is set
 */
function modelChoiceValue(groupUuid, modelName) {
  if (groupUuid) return `${MODEL_CHOICE_GROUP}${groupUuid}`;
  if (modelName) return `${MODEL_CHOICE_MODEL}${modelName}`;
  return '';
}

/**
 * Decode an option value back into the two fields the API expects.
 *
 * Both are returned every time, one of them null — the payload has to *clear* the
 * other field, not omit it, or switching a source from a stack to a single model
 * would leave the old stack in place and the backend would reject the pair.
 *
 * @returns {{modelGroupUuid: ?string, modelName: ?string}}
 */
function readModelChoice(value) {
  if (value?.startsWith(MODEL_CHOICE_GROUP)) {
    return { modelGroupUuid: value.slice(MODEL_CHOICE_GROUP.length), modelName: null };
  }
  if (value?.startsWith(MODEL_CHOICE_MODEL)) {
    return { modelGroupUuid: null, modelName: value.slice(MODEL_CHOICE_MODEL.length) };
  }
  return { modelGroupUuid: null, modelName: null };
}

/**
 * A `<select>` of "(fallback) | stacks… | models…".
 *
 * @param {{groupOptions: Array, modelNames: string[]}} ctx
 * @param {string} emptyText  what "neither" means here — inherit, or nothing at all
 * @param {string} value      from modelChoiceValue()
 */
function modelChoiceSelect(ctx, emptyText, value, onChange, { testid = 'watch-model-choice' } = {}) {
  const sel = document.createElement('select');
  // The profile default and the per-source override are the same widget in different
  // places; distinct testids because the default renders first and would otherwise
  // capture every `.first` locator aimed at a source row.
  sel.dataset.testid = testid;

  const none = document.createElement('option');
  none.value = '';
  none.textContent = emptyText;
  sel.appendChild(none);

  const addGroup = (label, options) => {
    if (!options.length) return;
    const group = document.createElement('optgroup');
    group.label = label;
    for (const opt of options) {
      const o = document.createElement('option');
      o.value = opt.key;
      o.textContent = opt.text;
      group.appendChild(o);
    }
    sel.appendChild(group);
  };

  addGroup('Model stacks', ctx.groupOptions.map(g => ({ key: `${MODEL_CHOICE_GROUP}${g.key}`, text: g.text })));
  addGroup('Models', ctx.modelNames.map(name => ({ key: `${MODEL_CHOICE_MODEL}${name}`, text: name })));

  // A stack or model that has since been deleted would otherwise silently fall back to
  // the first option and be saved as that on the next edit. Keep it, marked.
  if (value && !sel.querySelector(`option[value="${CSS.escape(value)}"]`)) {
    const orphan = document.createElement('option');
    orphan.value = value;
    orphan.textContent = `${value.slice(2)} (missing)`;
    sel.appendChild(orphan);
  }

  sel.value = value || '';
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
 * A text input plus a `Browse…` button that opens the shared directory picker.
 *
 * Reuses `openFileBrowserModal` rather than growing a picker of its own, so creating a
 * directory that does not exist yet — which is the normal case for a new timelapse — is
 * the same "New Folder" affordance as everywhere else in the app.
 *
 * The input stays editable, unlike `DirBrowserField`'s read-only one: an action row is
 * a dense inline form, and typing `timelapse/front-door` is faster than three clicks
 * for a user who already knows where it goes.
 *
 * Paths are **dataset-root relative**, which is what the picker deals in and what the
 * `/api/images/list` and `/api/images/mkdir` endpoints behind it accept. Leading slashes
 * are cosmetic here; the action's `resolve_output_dir` accepts either spelling.
 */
function dirPickerField(value, onAnyChange, { placeholder = 'dataset root' } = {}) {
  const wrap = el('div', 'watch-dir-picker');
  const input = document.createElement('input');
  input.type = 'text';
  input.value = value ?? '';
  input.placeholder = placeholder;
  input.dataset.testid = 'watch-action-dir';
  const listeners = [];
  const changed = () => { listeners.forEach(fn => fn()); onAnyChange?.(); };
  input.addEventListener('input', changed);

  const browse = document.createElement('button');
  browse.type = 'button';
  browse.className = 'outline secondary watch-dir-picker__browse';
  browse.textContent = 'Browse…';
  browse.dataset.testid = 'watch-action-dir-browse';
  browse.title = 'Pick a folder under the image dataset root, or create one. You can also type '
    + 'the path directly, placeholders included.';
  browse.addEventListener('click', async () => {
    const picked = await openFileBrowserModal({
      initialPath: input.value.trim() ? `/${input.value.trim().replace(/^\/+/, '')}` : '/',
      allowCreateDir: true,
    });
    // null is "cancelled", which must leave the typed value alone.
    if (picked === null) return;
    input.value = picked.replace(/^\/+/, '');
    changed();
  });

  wrap.append(input, browse);
  return {
    element: wrap,
    // Exposed so a `(?)` can insert a token at the caret, which needs the input itself
    // rather than its value — see `templateHelp`.
    input,
    getValue: () => input.value,
    onChange: (fn) => listeners.push(fn),
  };
}

/**
 * Picks the training collector a `collect_training_images` action feeds.
 *
 * The action declares `collector_uuid` as its own schema type rather than `str`, so a
 * uuid is chosen from the collectors that exist instead of pasted — a pasted uuid that
 * names nothing makes the *watcher* fail at build time, far from the form that accepted
 * it. Mirrors `TrainingCollectorSelectField`'s rules (disabled collectors hidden unless
 * already selected) without reusing it: that class is an async `Field` for the stacked
 * task-config form, and this row is a synchronous flex line.
 *
 * The list arrives after the widget does. The stored uuid is therefore rendered as its
 * own option immediately, so a save made before the fetch lands keeps the selection
 * rather than reading back an empty select.
 */
function collectorPickerWidget(value, onChange) {
  const wrap = el('div', 'watch-collector-picker');
  const control = document.createElement('select');
  control.dataset.testid = 'watch-action-collector';
  const stored = value ?? '';

  control.title = 'Which collector decides what to keep and how to annotate it. Its own settings — '
    + 'models, confidence bands, dedup, output folder — live in Configuration → Collectors.';

  const placeholder = document.createElement('option');
  placeholder.value = stored;
  placeholder.textContent = stored ? 'Loading…' : 'Select a collector…';
  control.appendChild(placeholder);
  control.value = stored;
  control.addEventListener('change', onChange);

  fetchTrainingCollectors().then((collectorsMap) => {
    const options = [...(collectorsMap?.values?.() || [])]
      .filter(c => c.enabled !== false || c.uuid === stored);
    if (!options.length) {
      // Short: the option's own text sizes the <select>, and the full sentence made it
      // three times the width of every other control in the row. The where-to-fix-it
      // half lives in the title and in the Edit… link beside it, which goes there.
      placeholder.textContent = 'No collectors yet';
      return;
    }
    control.replaceChildren();
    if (!stored) {
      const none = document.createElement('option');
      none.value = '';
      none.textContent = 'Select a collector…';
      control.appendChild(none);
    }
    for (const c of options) {
      const option = document.createElement('option');
      option.value = c.uuid;
      option.textContent = c.enabled === false ? `${c.name} (disabled)` : c.name;
      control.appendChild(option);
    }
    // A stored uuid whose collector has since been deleted would otherwise silently
    // select the first option and rewrite the action on the next save.
    if (stored && !options.some(c => c.uuid === stored)) {
      const missing = document.createElement('option');
      missing.value = stored;
      missing.textContent = `${stored} (missing)`;
      control.appendChild(missing);
    }
    control.value = stored;
  }).catch(() => {
    placeholder.textContent = stored || 'Could not load collectors';
  });

  const link = document.createElement('a');
  link.href = '#';
  // NOT `.watch-hint`: that class is dotted-underlined with `cursor: help`, which says
  // "hover me for an explanation" — the opposite of what a navigation link does.
  link.className = 'watch-inline-link';
  link.textContent = 'Edit…';
  link.title = 'Open Configuration → Collectors, where a collector’s ~30 knobs live.';
  link.dataset.testid = 'watch-action-collector-edit';
  link.addEventListener('click', (e) => {
    e.preventDefault();
    navigateTabs('tab-configuration', 'tab-collector-management', { selectUuid: control.value || null });
  });

  wrap.append(control, link);
  return { element: wrap, control, read: () => control.value.trim() };
}

/**
 * A compact inline `label: control` pair.
 *
 * Deliberately not `makeFieldRow`: that helper pins its label to a fixed 100px column
 * and gives the control `flex: 1`, which is right for a stacked settings panel and
 * wrong here — it turned a four-field row into four full-width lines. These pairs size
 * to their content and flow along the row instead.
 *
 * `help` goes on the wrapping `<label>`, not the control: a `title` is inherited by
 * descendants, so one assignment covers hovering the caption *and* the input. Putting
 * it here rather than at each call site is what makes "every settable thing explains
 * itself" a property of the form instead of a list of places to remember.
 */
function wfield(labelText, control, { width = null, help = null } = {}) {
  const wrap = el('label', 'watch-field');
  wrap.appendChild(el('span', 'watch-field__label', labelText));
  if (width) control.style.width = width;
  if (help) wrap.title = help;
  wrap.appendChild(control);
  return wrap;
}

/**
 * A checkbox with its name *beside* it rather than above it.
 *
 * `wfield` stacks the label over the control, which costs a line — and the rule and
 * camera rows are both held to one control line (`test_the_source_row_stays_one_control_line`,
 * `test_source_and_action_rows_are_single_line`). That constraint is why these boxes
 * were briefly unlabelled, and unlabelled was the wrong trade: two adjacent checkboxes
 * on a rule row are unreadable without hovering, and a lone one on a camera row reads as
 * "enable this camera" — which is not what it does, and the camera row has no enable box
 * of its own to disambiguate it.
 *
 * Hover `title` stays as the long explanation, but it is no longer load-bearing: a
 * tooltip is invisible until you already suspect it is there, and it does not exist at
 * all on touch.
 *
 * @param {HTMLInputElement} box   the checkbox itself; callers keep reading `.checked`
 * @param {string} text            two or three characters, not a sentence
 */
function checkTag(box, text, { className = '' } = {}) {
  const wrap = el('label', `watch-row-tag${className ? ` ${className}` : ''}`);
  // The title lives on the wrapper so the text is hoverable too, not just the 13px box.
  wrap.title = box.title;
  wrap.appendChild(box);
  wrap.appendChild(el('span', 'watch-row-tag__text', text));
  return wrap;
}

/**
 * The on/off box that leads a rule or action row.
 *
 * A real switch, not a note to yourself: `StreamWatcherTask` builds its rule list from
 * `profile.rules if r.enabled` and skips any action row whose flag is false, so an
 * unticked row is never evaluated. That is the opposite of the watcher-level "Enabled"
 * box in the identity strip, which nothing reads at run time and says so.
 *
 * It exists because the alternative was deleting a rule to silence it for an evening and
 * rebuilding it from memory afterwards — losing the thresholds, the hysteresis and the
 * scope along with it.
 *
 * Dims the row rather than hiding or collapsing anything. A disabled rule still has to be
 * readable and editable (turning it off without losing how it was configured is the whole
 * point), and a row that changed size on toggle would reflow the editor under the cursor.
 */
function enabledToggle(row, current, onChange, { testid, title }) {
  const box = document.createElement('input');
  box.type = 'checkbox';
  // `!== false` rather than truthiness: a row being built from scratch has no flag at
  // all, and a new rule defaults to enabled.
  box.checked = current !== false;
  box.className = 'watch-row-enabled';
  box.dataset.testid = testid;
  box.title = title;
  const sync = () => row.classList.toggle('watch-row--disabled', !box.checked);
  box.addEventListener('change', () => { sync(); onChange(); });
  sync();
  return box;
}

/**
 * The "publish to Home Assistant" box, beside the enabled box on rules and cameras.
 *
 * Deliberately *not* the same control as `enabledToggle`, and the distinction is the
 * whole reason it exists: unticking `enabled` stops the rule being evaluated, so there
 * are no events, no clips and no history. This stops only the Home Assistant entities.
 * The row keeps working; it just goes quiet in HA.
 *
 * Untinting the row is intentional too — a row that is publishing nothing is still fully
 * operational, so dimming it the way a disabled row is dimmed would misreport it.
 *
 * The cost this answers is entity count: HA holds a registry entry per entity whether or
 * not anyone enables it, and the total is a product of cameras, rules and labels. See the
 * app device's "HA entities published" sensor for the running figure.
 */
function publishToggle(current, onChange, { testid, title }) {
  const box = document.createElement('input');
  box.type = 'checkbox';
  // `!== false` for the same reason as `enabledToggle`: a row built from scratch has no
  // flag, and publishing is the default.
  box.checked = current !== false;
  box.className = 'watch-row-publish';
  box.dataset.testid = testid;
  box.title = title;
  box.addEventListener('change', onChange);
  return box;
}

/**
 * A `(?)` beside a topic field, opening the placeholder vocabulary.
 *
 * The tokens used to be reachable only as the topic input's `title`, which is not
 * discoverable at all: a hover tooltip is invisible until you already suspect it is
 * there, so the templating feature read as a free-text box you had to guess at.
 *
 * Clicking a row inserts that token at the cursor rather than only describing it —
 * the list is then the fastest way to *write* a template, not merely documentation
 * about one, and it removes any chance of a typo in a token name.
 *
 * @param {HTMLInputElement} input the topic field tokens are inserted into
 */
function templateHelp(input, { intro, testid = 'watch-template-help', label } = {}) {
  return helpPopover({
    testid,
    label: label || 'Placeholders you can use here',
    build: (panel, close) => {
      panel.appendChild(el(
        'p',
        'watch-template-help__intro',
        intro
        || 'Substituted per topic level and slugified. Click one to insert it. An unknown '
        + 'placeholder is left as-is, so a typo shows up on the broker rather than '
        + 'silently collapsing a level.',
      ));

      for (const { token, help } of TEMPLATE_PLACEHOLDER_HELP) {
        const item = el('button', 'watch-template-help__item');
        item.type = 'button';
        item.appendChild(el('code', 'watch-template-help__token', token));
        item.appendChild(el('span', 'watch-template-help__desc', help));
        item.addEventListener('click', () => {
          // At the cursor, not appended: a template is usually edited in the middle, and
          // an insert that always lands at the end would have to be cut and re-pasted.
          const start = input.selectionStart ?? input.value.length;
          const end = input.selectionEnd ?? start;
          input.value = input.value.slice(0, start) + token + input.value.slice(end);
          const caret = start + token.length;
          input.setSelectionRange(caret, caret);
          input.focus();
          // The field's own listeners drive the live example and the dirty flag, and a
          // programmatic value change fires neither on its own.
          input.dispatchEvent(new Event('input', { bubbles: true }));
        });
        panel.appendChild(item);
      }
    },
  });
}

/**
 * A `(?)` that opens a short static explanation.
 *
 * Shares `templateHelp`'s classes and open/close behaviour deliberately: the two read
 * as the same affordance because they are, and a second set of styles for a popover
 * that differs only in having nothing clickable inside it would drift apart.
 *
 * Exists because a *removed* option explains itself least of all — a trigger list that
 * silently has one entry looks like the feature was never built, so the reason it is
 * short has to be reachable from where the choice would have been.
 *
 * @param {string} text the explanation
 * @param {{testid?: string, label?: string}} [opts]
 */
function noteHelp(text, { testid = 'watch-note-help', label = 'Show more information' } = {}) {
  return helpPopover({
    testid,
    label,
    build: (panel) => panel.appendChild(el('p', 'watch-template-help__intro', text)),
  });
}

/**
 * The `(?)` affordance both help flavours are built from.
 *
 * Positioning is delegated to `openAnchoredPopup` rather than done in CSS, which is the
 * whole reason this exists as one function. The panel used to be `position: absolute;
 * right: 0` inside the 1.35em button's wrapper, so a 24rem panel hung ~23rem to the left
 * of the button — fine for a field near the right of a wide window, and most of the way
 * off the left edge of the screen for every other case. `openAnchoredPopup` measures the
 * panel, clamps it into the viewport on both axes, flips it above or below depending on
 * the room available, and brings the outside-click and ESC dismissal with it.
 *
 * The panel therefore lives on `document.body` while open and is removed on close, which
 * also frees it from the action row's `overflow` — the other reason a popover anchored
 * inside a dense row is hard to see.
 *
 * @param {{testid: string, label: string, build: (panel: HTMLElement, close: () => void) => void}} opts
 */
function helpPopover({ testid, label, build }) {
  const wrap = el('span', 'watch-template-help');

  const btn = el('button', 'watch-template-help__btn', '?');
  btn.type = 'button';
  btn.dataset.testid = `${testid}-btn`;
  btn.title = label;
  btn.setAttribute('aria-label', label);
  btn.setAttribute('aria-expanded', 'false');

  let popup = null;
  function close() {
    popup?.close();
    popup = null;
    btn.setAttribute('aria-expanded', 'false');
  }
  btn.addEventListener('click', () => {
    if (popup) {
      close();
      return;
    }
    const panel = el('div', 'watch-template-help__panel');
    panel.dataset.testid = `${testid}-panel`;
    build(panel, close);
    popup = openAnchoredPopup(btn, panel);
    btn.setAttribute('aria-expanded', 'true');
  });

  wrap.appendChild(btn);
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
function sectionHeader(text, addButton, help = null) {
  const head = el('div', 'watch-section-head');
  const label = el('span', 'panel-section-label', text);
  // On the label, not the strip: a title on the strip also fires over the "+ Add"
  // button inside it, replacing that button's own more specific tooltip with the
  // section's general one.
  if (help) label.title = help;
  head.appendChild(label);
  if (addButton) head.appendChild(addButton);
  return head;
}

/**
 * Wrap a section's heading and its list in one bordered panel.
 *
 * The editor was a flat run of headings and bordered rows in a single column, and
 * Sources, Rules and Actions all render as near-identical rows — so the only thing
 * marking the boundary between two sections was a half-rem of margin, and the three
 * read as one long list. The panel gives each section an edge and a surface of its own.
 *
 * @param {HTMLElement} head A `sectionHeader`, already built (Tracking inserts controls
 *   into its own before wrapping).
 * @param {...HTMLElement} bodies
 */
function sectionPanel(head, ...bodies) {
  const section = el('div', 'watch-section');
  section.dataset.testid = 'watch-section';
  section.append(head, ...bodies);
  return section;
}

// ── Tracker params ──────────────────────────────────────────────────────────
/**
 * Renders one tracker's `params_schema()` into a form, and reads it back.
 *
 * Schema-driven rather than hand-written per tracker: that is the whole point of the
 * registry, and it means adding a tracker needs no change in this file. Field *keys*
 * arrive snake_case (they are data, not model fields, so they are not camelised) and
 * are stored back exactly as declared — the Python side looks them up by those names.
 *
 * @param {object} schema `paramsSchema` for the selected tracker.
 * @param {object} stored Saved `trackerParamsJson`, which may be from another tracker.
 * @param {() => void} onAnyChange
 */
function buildTrackerParams(schema, stored, onAnyChange) {
  // The shared renderer, which grew out of this function when the action forms were
  // moved onto it. Trackers need none of its escape hatches — every tracker parameter
  // is a plain scalar with a help string, which is exactly the case it handles bare.
  const form = buildSchemaParams(schema, stored, onAnyChange, {
    testidPrefix: 'watch-tracker-param',
    emptyText: 'This tracker has no options.',
    decorator: { className: 'watch-tracker-params' },
    fieldWrapper: wfield, numberInput, el, num,
  });
  form.element.dataset.testid = 'watch-tracker-params';
  return form;
}

// ── Label lifetime row ──────────────────────────────────────────────────────
/**
 * One per-label tracking lifetime: how long that label's track survives without a
 * detection, and whether it latches.
 *
 * In **seconds, not frames**, matching the tracker: a frame count means 1 s on a 5 fps
 * camera and 0.2 s on a 25 fps one. This is what makes "a litter box survives 30 s of
 * misses while a cat survives 1 s" expressible.
 */
function buildLifetimeRow(lifetime, ctx) {
  const row = el('div', 'watch-lifetime-row watch-row');
  row.dataset.testid = 'watch-lifetime-row';

  // The same hierarchy-aware multi-select the rule slots use, for the same two reasons:
  // a flat alphabetical list of every cat, breed and species is unnavigable, and "every
  // cat survives 1 s" is one statement, not one row per breed. `Include child labels`
  // is resolved when the watcher starts, so a breed added later is covered without
  // editing the watcher.
  let labelRefs = [...(lifetime.labelUuidsJson || [])];
  let includeDescendants = Boolean(lifetime.includeDescendants);
  const labelHost = el('span', 'watch-lifetime-labels');
  labelHost.dataset.testid = 'watch-lifetime-label';
  const labelDropdown = createMultiSelectDropdown({
    label: '',
    items: ctx.labelItems,
    defaultSelected: labelRefs,
    showModeToggle: false,
    allowExclude: false,
    searchable: true,
    emptySummary: 'None selected',
    emptyText: '(no labels defined)',
    filterKey: 'watch-lifetime-labels',
    hierarchyToggle: {
      label: 'Include child labels',
      default: includeDescendants,
      onChange: (value) => {
        includeDescendants = value;
        ctx.onAnyChange();
      },
    },
    onChange: (selected) => {
      labelRefs = selected;
      ctx.onAnyChange();
    },
  });
  labelHost.appendChild(labelDropdown.element);
  row.appendChild(wfield('Labels', labelHost, {
    help: 'The labels this lifetime applies to. Labels with no row here use the tracker’s own '
      + 'defaults. A label named outright by one row wins over one it only inherits from a '
      + 'parent via “Include child labels”.',
  }));

  const ttl = numberInput(lifetime.ttlS ?? 2.0, { step: '0.5', min: 0 });
  ttl.dataset.testid = 'watch-lifetime-ttl';
  ttl.title = 'Seconds the track survives with no detection before it is assumed gone.';
  ttl.addEventListener('input', ctx.onAnyChange);
  row.appendChild(wfield('TTL s', ttl, { width: '5rem', help: ttl.title }));

  const anchor = document.createElement('input');
  anchor.type = 'checkbox';
  anchor.checked = lifetime.isAnchor === true;
  anchor.dataset.testid = 'watch-lifetime-anchor';
  // Says what it changes *and* what it does not: the flag is about the box, and the
  // most expensive misreading of it is "anchored things stay alive longer", which it
  // has never done — the TTL beside it is the only thing that decides that.
  anchor.title =
    'Pins the box in place. Once this label is detected confidently (at or above the '
    + 'tracker’s Anchor confidence), later weak detections keep refreshing the track but '
    + 'no longer move its box — so a stationary object stops jittering in and out of a '
    + 'zone. A confident detection still moves it.\n\n'
    + 'It does not extend the lifetime: an anchored track still dies after the TTL beside '
    + 'it with no detections at all, so pair this with a long TTL.\n\n'
    + 'For fixtures — a bin, a parked vehicle, a door. Not for subjects that really do move.';
  anchor.addEventListener('change', ctx.onAnyChange);
  row.appendChild(wfield('Anchor', anchor, { help: anchor.title }));

  row.appendChild(removeButton(ctx.onDelete, 'Remove lifetime'));

  return {
    element: row,
    destroy: () => labelDropdown.destroy(),
    getLifetime: () => ({
      labelUuidsJson: [...labelRefs],
      includeDescendants,
      // ttl_s is rejected server-side at <= 0, so an emptied box must fall back rather
      // than send 0 and fail the whole save.
      ttlS: num(ttl, 2.0) || 2.0,
      isAnchor: anchor.checked,
    }),
  };
}

/**
 * The four-state buffer badge, from a source's ingest tier.
 *
 * Wording and glyphs match `WatchPipeline._buffer_label()`, which renders the same
 * four states on the running task's status card. Two places saying different things
 * about the same source is how a user learns to trust neither.
 *
 * @param {?{tier: string, reason: string, buffered: boolean, transcoding: boolean}} plan
 * @returns {{text: string, title: string, buffered: boolean}}
 */
function bufferBadgeFor(plan) {
  if (!plan) return { text: '', title: '', buffered: true };
  const suffix = plan.transcoding ? ' — transcoding (per-camera CPU)' : '';
  switch (plan.tier) {
    case 'native':
      return { text: '● rewind (native)', title: plan.reason + suffix, buffered: true };
    case 'ffmpeg':
      return { text: '● rewind (ffmpeg ingest)', title: plan.reason + suffix, buffered: true };
    case 'ring':
      return { text: '◐ frame ring (no rewind)', title: plan.reason, buffered: false };
    default:
      return { text: '⚠ no buffer', title: plan.reason, buffered: false };
  }
}

/**
 * The `buffer s` field's help. A constant because `refreshBadge` rewrites the input's
 * own title to explain an unbuffered source, and has to be able to put this back.
 */
// Deliberately opens by saying what this is *not*. The previous wording — "so a clip can
// start before the rule fired" — described this field by naming the pre-roll setting's
// job, and the reasonable conclusion was that this is where you choose your lead-in.
// Shortening it to a preferred pre-roll then quietly starves the clip drain, and the only
// symptom is that long events lose their opening.
const BUFFER_SECONDS_HELP =
  'How much video this camera keeps rewindable, in seconds. Blank uses the default, which is '
  + 'almost always right. This is NOT the pre-roll setting: how much lead-in a clip gets is set '
  + 'on the Record clip action, and it can be shorter than this. Two things need the room — '
  + 'pre-roll is cut from it, and an event that outlasts it is copied out to disk from it as it '
  + 'runs, which needs several seconds of headroom to keep up. Making it smaller buys nothing '
  + 'and risks both. The badge beside it says whether this camera can keep a rewindable buffer '
  + 'at all.';

// ── Source row ──────────────────────────────────────────────────────────────
function buildSourceRow(src, ctx) {
  const row = el('div', 'watch-source-row watch-row');
  row.dataset.testid = 'watch-source-row';

  // Repointing a row at a different camera changes the uuid the actions' source scopes
  // are written against, so they have to be rebuilt — not just marked dirty.
  const sourceSel = select(ctx.sourceOptions, src.sourceUuid, () => {
    ctx.onSourceChanged?.();
    ctx.onAnyChange();
  });
  sourceSel.dataset.testid = 'watch-source-select';
  row.appendChild(wfield('Source', sourceSel, {
    help: 'The camera this row configures. Repointing it moves the row’s settings to the new '
      + 'camera and drops anything that belonged to the old one — its mask opt-outs, and any '
      + 'action or rule scoped to it.',
  }));

  // Tagged "HA", not left bare: this row has no enable/disable box of its own, so an
  // unlabelled checkbox here reads as one — and unticking it would then look like it
  // stops the camera being watched, which is the opposite of what it does.
  const publishBox = publishToggle(src.publishToHa, ctx.onAnyChange, {
    testid: 'watch-source-publish',
    title: 'Publish this camera to Home Assistant. Unticked, the watcher still analyses it and '
      + 'still records — only its HA device and that device’s entities go away, which is worth '
      + 'roughly (rules × 6) + 9 entities. Its rules keep contributing to the Rules device, so '
      + 'automations written against a rule rather than a camera are unaffected.',
  });
  row.appendChild(checkTag(publishBox, 'HA'));

  // What `{source}` publishes as, for this watcher only. The placeholder tracks the
  // selected camera so "blank means the source's own name" needs no help text — and so
  // it keeps reading correctly after the row is repointed at a different camera.
  const mqttName = document.createElement('input');
  mqttName.type = 'text';
  mqttName.value = src.mqttName || '';
  mqttName.dataset.testid = 'watch-source-mqtt-name';
  mqttName.title =
    'Name this source publishes under in MQTT topics — what {source} renders as. '
    + 'Leave blank to use the source\'s own name. Set it when the name that reads well '
    + 'in the Sources list is not the short, stable topic level you want to subscribe to.';
  function syncMqttPlaceholder() {
    mqttName.placeholder = ctx.sourceNameFor(sourceSel.value) || 'source name';
  }
  syncMqttPlaceholder();
  sourceSel.addEventListener('change', syncMqttPlaceholder);
  mqttName.addEventListener('input', ctx.onAnyChange);
  row.appendChild(wfield('MQTT as', mqttName, { width: '8rem', help: mqttName.title }));

  const modelSel = modelChoiceSelect(
    ctx,
    '(profile default)',
    modelChoiceValue(src.modelGroupUuid, src.modelName),
    ctx.onAnyChange,
  );
  row.appendChild(wfield('Models', modelSel, {
    help: 'Which model or model stack runs on this camera. Leave at “(profile default)” to '
      + 'use the watcher’s Default models — set it here only when one camera needs something '
      + 'the others do not.',
  }));

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
  rates.appendChild(wfield('fps', fps, {
    width: '4.5rem',
    help: 'Frames per second pulled from this camera for analysis. Blank uses the watcher’s '
      + 'own default. This is the main cost knob: every frame is a full inference pass, and '
      + 'the tracker only needs enough of them to follow an object between detections.',
  }));
  rates.appendChild(wfield('width px', width, {
    width: '5rem',
    help: 'Downscale each frame to this width before inference. Blank uses the default. '
      + 'Smaller is faster and loses small or distant objects first; it does not affect what a '
      + 'recorded clip looks like.',
  }));
  // "rewind s", not "buffer s": the old caption named the mechanism, which left the
  // field looking like the generic place to configure anything clip-shaped. "Rewind" is
  // what it actually gives you, and it does not read as a synonym for pre-roll.
  rates.appendChild(wfield('rewind s', retention, {
    width: '4.5rem',
    help: BUFFER_SECONDS_HELP,
  }));

  // The honesty affordance: "buffer s" above is a request for pre-roll, and a snapshot
  // camera cannot honour it at any value. Without this the mistake is invisible until
  // an event fires and the clip that was supposed to appear simply doesn't.
  const badge = el('span', 'watch-buffer-badge');
  badge.dataset.testid = 'watch-buffer-badge';
  rates.appendChild(badge);

  // Frame-ring bounds. Shown only for a source that has no rewind buffer, because that
  // is the only case where a ring is kept — but always *sent*, see getSource below.
  const ringFields = el('span', 'watch-ring-fields');
  ringFields.dataset.testid = 'watch-ring-fields';
  const ringSeconds = numberInput(src.ringSeconds ?? '', { step: '5', min: 0, placeholder: '20' });
  ringSeconds.dataset.testid = 'watch-ring-seconds';
  const ringMaxMb = numberInput(src.ringMaxMb ?? '', { step: '8', min: 0, placeholder: '32' });
  ringMaxMb.dataset.testid = 'watch-ring-max-mb';
  ringSeconds.addEventListener('input', ctx.onAnyChange);
  ringMaxMb.addEventListener('input', ctx.onAnyChange);
  ringFields.appendChild(wfield('ring s', ringSeconds, {
    width: '4.5rem',
    help: 'Seconds of analysed frames kept in memory for a camera that cannot rewind. It is '
      + 'what gives such a source any pre-roll at all — a clip cut from it is built from these '
      + 'frames, not from video.',
  }));
  ringFields.appendChild(wfield('ring MB', ringMaxMb, {
    width: '4.5rem',
    help: 'Memory ceiling for that ring, per camera. Whichever limit is hit first wins, so a '
      + 'high-resolution camera keeps fewer seconds than the number beside this asks for.',
  }));
  rates.appendChild(ringFields);

  function refreshBadge() {
    const info = bufferBadgeFor(ctx.ingestTierFor?.(sourceSel.value));
    badge.textContent = info.text;
    badge.title = info.title;
    badge.classList.toggle('watch-buffer-badge--unbuffered', !info.buffered);
    // Not disabled — the value is still stored and still meaningful if the source is
    // later repointed at a stream. Marked instead, so the field explains itself.
    retention.classList.toggle('watch-input--inert', !info.buffered);
    // The base help when it works, the reason it cannot when it does not — never `''`,
    // which means "no tooltip" and blocks the wrapper's from showing through.
    retention.title = info.buffered
      ? BUFFER_SECONDS_HELP
      : `Pre-roll cannot be honoured: ${info.title}`;
    // The ring is what gives an unbuffered source any pre-roll at all, so its bounds
    // are only worth showing there; on a rewindable source they would be dead controls.
    ringFields.hidden = info.buffered;
  }
  refreshBadge();
  sourceSel.addEventListener('change', refreshBadge);

  row.appendChild(rates);

  // Regions live behind a modal because they need a video frame as a backdrop, which is
  // far too heavy to inline in a list row. They belong to the *source*, so editing them
  // saves immediately rather than waiting for this watcher to be saved — the camera's
  // geometry is not this watcher's to hold hostage.
  const actions = el('div', 'watch-row-actions');
  const regionsBtn = el('button', 'panel-btn', 'Regions…');
  regionsBtn.type = 'button';
  regionsBtn.dataset.testid = 'watch-zones-btn';
  regionsBtn.title =
    'Draw zones and masks on this camera’s picture. Zones are what a rule matches against; '
    + 'masks hide part of the frame from every detection. They belong to the camera, so '
    + 'editing them saves at once rather than waiting for this watcher to be saved.';
  regionsBtn.addEventListener('click', async () => {
    const result = await openSourceRegionsModal({
      sourceUuid: sourceSel.value,
      sourceName: ctx.sourceNameFor(sourceSel.value),
      // Opened from a watcher, so each mask also offers "applies to this watcher".
      // Opened from the Sources pane there is no watcher to scope it to, and the modal
      // renders the geometry alone.
      maskOptOut: {
        disabled: disabledMasks,
        onChange: (next) => { disabledMasks = next; ctx.onAnyChange(); },
      },
    });
    if (result) {
      // Rules offer zones from every source, and the mask checkboxes below are derived
      // from this source's masks, so both have to be rebuilt. Marking the editor dirty
      // is not enough — nothing re-reads regions until the pane is reopened.
      await ctx.onRegionsChanged?.();
      ctx.onAnyChange();
    }
  });
  actions.appendChild(regionsBtn);
  actions.appendChild(removeButton(() => ctx.onDelete(), 'Remove source'));
  row.appendChild(actions);

  // ── Mask opt-out ──
  // Masks act on every detection, so a source owning them is right for the common case
  // ("ignore the driveway, always") and wrong for the rare one (a people-watcher masking
  // a street a car-watcher needs). All ticked by default: a mask added to a camera
  // tomorrow must protect the watchers written today without anyone revisiting them.
  // The per-watcher mask opt-out is edited inside the Regions modal, beside the masks it
  // refers to, rather than as another control in this row. Two reasons: the row is
  // already a full line and a permanent control here costs everyone a line, and this is
  // a *rare* setting — the whole point of the default is that it needs no configuration.
  let disabledMasks = new Set((src.disabledMaskUuidsJson || []).map(String));
  function renderMasks() {
    // Masks the camera no longer has cannot stay opted out: the uuid is rejected on save
    // and the checkbox it came from no longer exists to clear it.
    const live = new Set(ctx.masksForSource(sourceSel.value).map(m => m.uuid));
    disabledMasks = new Set([...disabledMasks].filter(u => live.has(u)));
  }
  renderMasks();
  sourceSel.addEventListener('change', () => {
    // A different camera has different masks; carrying the old source's opt-outs across
    // would disable masks by uuid that this source has never heard of, which the backend
    // rejects on save.
    disabledMasks = new Set();
    renderMasks();
  });

  return {
    element: row,
    getSourceUuid: () => sourceSel.value,
    refreshMasks: renderMasks,
    getSource: () => ({
      sourceUuid: sourceSel.value,
      order: 0,
      enabled: true,
      publishToHa: publishBox.checked,
      // null, not '', so "cleared" and "never set" are one state. The backend
      // normalises blanks too; doing it here as well keeps the payload honest.
      mqttName: mqttName.value.trim() || null,
      ...readModelChoice(modelSel.value),
      analysisFps: fps.value === '' ? null : num(fps, null),
      analysisWidth: width.value === '' ? null : num(width, null),
      recordRetentionS: retention.value === '' ? null : num(retention, null),
      // Always sent, including while hidden, and `bufferMode` even though nothing
      // renders it: a watch profile is saved as a whole-list replace, so a field the
      // editor omits is a field the save silently resets to its default. These three
      // were harmless to drop while nothing read them — the watcher reads all three
      // now, so dropping them would quietly undo a configured ring on every save.
      bufferMode: src.bufferMode ?? 'auto',
      ringSeconds: ringSeconds.value === '' ? 20.0 : num(ringSeconds, 20.0),
      ringMaxMb: ringMaxMb.value === '' ? 32.0 : num(ringMaxMb, 32.0),
      // Only masks that still exist on this source — a stale uuid is rejected on save,
      // and a mask deleted in the editor must not keep a dead opt-out alive.
      disabledMaskUuidsJson: ctx
        .masksForSource(sourceSel.value)
        .filter(m => disabledMasks.has(m.uuid))
        .map(m => m.uuid),
    }),
  };
}

// ── Rule row ────────────────────────────────────────────────────────────────
function buildRuleRow(rule, ctx) {
  const row = el('div', 'watch-rule-row watch-row');
  row.dataset.testid = 'watch-rule-row';

  const enabledBox = enabledToggle(row, rule.enabled, ctx.onAnyChange, {
    testid: 'watch-rule-enabled',
    title: 'Whether the watcher evaluates this rule at all. Unticked it is skipped entirely — '
      + 'no events, no actions, nothing recorded — but everything you configured here is kept, '
      + 'so this is the way to silence a rule for an evening without rebuilding it afterwards. '
      + 'Takes effect on the watcher’s next start.',
  });
  // "On" and "HA" rather than bare boxes: they sit next to each other and mean very
  // different things — only the first stops the rule being evaluated at all.
  row.appendChild(checkTag(enabledBox, 'On', { className: 'watch-row-tag--enabled' }));

  const publishBox = publishToggle(rule.publishToHa, ctx.onAnyChange, {
    testid: 'watch-rule-publish',
    title: 'Publish to Home Assistant. Unticked, the rule still runs — events are still raised, '
      + 'recorded and browsable here — but its entities are removed from HA, on the cameras, the '
      + 'subjects and the Rules device alike. Use it to keep a noisy or experimental rule out of '
      + 'your automations without losing its history.',
  });
  row.appendChild(checkTag(publishBox, 'HA'));

  // Independent of publishBox, and off by default unlike it - a snapshot is a
  // meaningfully bigger MQTT payload than everything else this rule publishes, so
  // it's opt-in rather than inherited "free" the way the rest of the entities are.
  const snapshotBox = document.createElement('input');
  snapshotBox.type = 'checkbox';
  snapshotBox.checked = rule.publishSnapshotToHa === true;
  snapshotBox.className = 'watch-row-publish';
  snapshotBox.dataset.testid = 'watch-rule-publish-snapshot';
  snapshotBox.title = 'Also publish a snapshot of the rule\'s last event as a Home Assistant camera '
    + 'entity - the same picture the events grid shows, updated each time the rule fires. Has no '
    + 'effect while "HA" above is unticked, since there is no device for it to appear on.';
  snapshotBox.addEventListener('change', ctx.onAnyChange);
  row.appendChild(checkTag(snapshotBox, 'Photo'));

  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.value = rule.name || '';
  nameInput.placeholder = 'e.g. cat using litter box';
  nameInput.dataset.testid = 'watch-rule-name';
  nameInput.addEventListener('input', ctx.onAnyChange);
  // `change` fires on commit (blur / Enter), not per keystroke — the actions' scope
  // pickers name this rule, and rebuilding them mid-word would steal focus.
  nameInput.addEventListener('change', () => ctx.onNameCommitted?.());
  // Explicitly narrow so the name and the sentence share a line. At its natural width
  // the pair overflowed the row by a handful of pixels and cost a whole extra line.
  row.appendChild(wfield('Rule', nameInput, {
    width: '10rem',
    help: 'What this rule is called — how it reads in an action’s Rules picker, and what '
      + '{rule_name} renders as in a topic or folder template. Scoping follows the rule itself, '
      + 'not its name, so renaming it keeps every action pointed at it. A rule added here can '
      + 'only be picked in an action once the watcher has been saved.',
  }));

  // The sentence: [subject A] [relation] [subject B]. Each subject is a *list* — "any of
  // these three cats is inside any of these two litter boxes" is one rule, not six, and
  // the engine reports which members actually matched so an action can name them.
  const sentence = el('div', 'watch-rule-sentence');

  let subjectARefs = [...(rule.subjectARefsJson || [])];
  let subjectBRefs = [...(rule.subjectBRefsJson || [])];
  // "…and everything under it in the label tree". Stored as a flag and expanded when the
  // watcher starts, so a label added to the tree later is covered without editing the
  // rule — which also means it takes effect on the watcher's next restart, not instantly.
  let subjectADescendants = Boolean(rule.subjectAIncludeDescendants);
  let subjectBDescendants = Boolean(rule.subjectBIncludeDescendants);

  const subjectAHost = el('span', 'watch-rule-slot');
  subjectAHost.dataset.testid = 'watch-rule-subject-a';
  subjectAHost.title =
    'The labels this rule watches for. Picking several means “any one of these”, not all '
    + 'of them together — one rule for three cats rather than three rules.';
  let subjectADropdown = null;

  function buildSubjectA() {
    subjectADropdown?.destroy();
    subjectAHost.innerHTML = '';
    subjectADropdown = createMultiSelectDropdown({
      // No "Labels:" prefix — the sentence around it already says what this slot is, and
      // the prefix is wide enough to push the tuning fields onto another line.
      label: '',
      items: ctx.labelItems,
      defaultSelected: subjectARefs,
      // A rule slot picks things; it does not build a query. Mode and exclusion are
      // filter concepts and would read here as though a rule could match "all of" three
      // cats at once, which is not what the engine does.
      showModeToggle: false,
      allowExclude: false,
      searchable: true,
      emptySummary: 'None selected',
      emptyText: '(no labels defined)',
      filterKey: 'watch-rule-subject-a',
      hierarchyToggle: {
        label: 'Include child labels',
        default: subjectADescendants,
        onChange: (value) => {
          subjectADescendants = value;
          ctx.onAnyChange();
        },
      },
      onChange: (selected) => {
        subjectARefs = selected;
        ctx.onAnyChange();
      },
    });
    subjectAHost.appendChild(subjectADropdown.element);
  }
  buildSubjectA();

  // Where a motion rule measures. Its own slot rather than reusing subject A, which is a
  // *label* picker: motion looks for no label, and the two mean different things by an
  // empty selection — an empty label slot can never fire, an empty motion slot is the
  // common case ("the whole frame").
  let motionZoneRefs = rule.relation === 'motion' ? [...(rule.subjectARefsJson || [])] : [];
  let motionZoneDropdown = null;
  const motionZoneHost = el('span', 'watch-rule-slot');
  motionZoneHost.dataset.testid = 'watch-rule-motion-zones';
  motionZoneHost.title =
    'Which drawn zones to watch for movement. Leave it empty to watch the whole frame. '
    + 'Masks are always excluded either way — a mask is how you tell this rule to ignore a '
    + 'road, a tree or a timestamp overlay.';

  function buildMotionZones() {
    const items = ctx.zoneItems();
    const available = new Set(items.map(i => i.uuid));
    motionZoneRefs = motionZoneRefs.filter(r => available.has(r));
    motionZoneDropdown?.destroy();
    motionZoneHost.innerHTML = '';
    motionZoneDropdown = createMultiSelectDropdown({
      label: '',
      items,
      defaultSelected: motionZoneRefs,
      showModeToggle: false,
      allowExclude: false,
      // "Anywhere", not "None selected": an empty motion slot watches the whole frame,
      // so the empty state must not read as the opposite of what it does.
      emptySummary: 'Anywhere',
      emptyText: '(no zones drawn on this watcher’s sources)',
      filterKey: 'watch-rule-motion-zones',
      onChange: (selected) => { motionZoneRefs = [...selected]; ctx.onAnyChange(); },
    });
    motionZoneHost.appendChild(motionZoneDropdown.element);
  }
  buildMotionZones();

  // The threshold means something different in kind for motion (a changed-area
  // fraction, ~0.02) than for the geometric relations (~0.6). Carrying one across the
  // switch leaves a rule that silently cannot fire, so each side remembers its own.
  let lastGeometricThreshold = rule.relation === 'motion' ? 0.6 : (rule.threshold ?? 0.6);
  let lastMotionThreshold = rule.relation === 'motion' ? (rule.threshold ?? MOTION_DEFAULT_AREA) : MOTION_DEFAULT_AREA;
  let previousRelation = rule.relation || 'contains';

  const relation = select(RELATIONS, rule.relation || 'contains', () => {
    const wasMotion = previousRelation === 'motion';
    const isMotion = relation.value === 'motion';
    if (wasMotion !== isMotion) {
      if (wasMotion) {
        lastMotionThreshold = num(threshold, MOTION_DEFAULT_AREA);
        threshold.value = String(lastGeometricThreshold);
      } else {
        lastGeometricThreshold = num(threshold, 0.6);
        threshold.value = String(lastMotionThreshold);
      }
    }
    previousRelation = relation.value;
    syncArity();
    ctx.onAnyChange();
  });
  relation.dataset.testid = 'watch-rule-relation';
  relation.title =
    'How the subject has to relate to the target for the rule to hold. “is inside” asks how '
    + 'much of the subject is within it; “overlaps” measures against the smaller of the two; '
    + '“closely matches” is IoU and only suits boxes of similar size; “has its centre in” is '
    + 'the cheapest robust test; “is near” compares centre points; “is present” needs no '
    + 'target at all. “motion is detected” needs no subject either — it compares pixels '
    + 'between frames, so it fires with no model and no detection. Use Preview… to see any '
    + 'of the geometric ones on boxes you can drag.';

  const subjectBKind = select(
    [{ key: 'zone', text: 'Zone' }, { key: 'label', text: 'Label' }],
    rule.subjectBKind === 'label' ? 'label' : 'zone',
    () => {
      // Switching kind changes what the uuids *mean*, so keeping them would leave label
      // uuids selected in a zone picker — the shape of the bug that used to silently
      // rewrite "is inside Litter Box" as "is inside Cat" on the next save.
      subjectBRefs = [];
      // Zones have no hierarchy, and the backend rejects the flag on a zone slot, so a
      // leftover true would make the whole save fail with a message about a control the
      // user can no longer see.
      subjectBDescendants = false;
      buildSubjectB();
      ctx.onAnyChange();
    },
  );
  subjectBKind.dataset.testid = 'watch-rule-subject-b-kind';
  subjectBKind.title =
    'Whether the subject is compared against a zone you have drawn on a camera, or against '
    + 'another detected label. Switching this clears the picker beside it — the two kinds '
    + 'do not share ids.';

  const subjectBHost = el('span', 'watch-rule-slot');
  subjectBHost.dataset.testid = 'watch-rule-subject-b';
  subjectBHost.title =
    'What the subject is compared against. Several entries mean “any one of these” — one '
    + 'rule can span two litter boxes. Zones come from the sources this watcher uses; draw '
    + 'them with Regions… on a source row.';
  let subjectBDropdown = null;

  function buildSubjectB() {
    subjectBDropdown?.destroy();
    subjectBHost.innerHTML = '';
    const isLabel = subjectBKind.value === 'label';
    const items = isLabel ? ctx.labelItems : ctx.zoneItems();
    // Drop refs that are no longer on offer — a zone deleted in the region editor must
    // not linger as an invisible selection that fails validation on save.
    const available = new Set(items.map(i => i.uuid));
    subjectBRefs = subjectBRefs.filter(r => available.has(r));
    subjectBDropdown = createMultiSelectDropdown({
      // Unprefixed for the same reason as subject A: the `Zone`/`Label` select sitting
      // immediately to its left already names the kind.
      label: '',
      items,
      defaultSelected: subjectBRefs,
      showModeToggle: false,
      allowExclude: false,
      searchable: isLabel,
      emptySummary: 'None selected',
      emptyText: isLabel ? '(no labels defined)' : '(no zones drawn on this watcher’s sources)',
      filterKey: 'watch-rule-subject-b',
      // Zones have no hierarchy, so the toggle only exists for a label-kind subject B —
      // and the backend rejects the flag on a zone slot rather than ignoring it.
      hierarchyToggle: isLabel
        ? {
          label: 'Include child labels',
          default: subjectBDescendants,
          onChange: (value) => {
            subjectBDescendants = value;
            ctx.onAnyChange();
          },
        }
        : null,
      onChange: (selected) => {
        subjectBRefs = selected;
        ctx.onAnyChange();
      },
    });
    subjectBHost.appendChild(subjectBDropdown.element);
  }
  buildSubjectB();

  // Which cameras this rule is evaluated on. Empty means all of them, exactly as an
  // action's scope does — and it is the *sharper* of the two: scoping the action leaves
  // the event raised on every camera and only silences the side effect, while scoping
  // the rule stops the event existing. "A cat is present" is the case that needs this
  // one, being true in every room all day.
  let sourceScope = [...(rule.sourceUuidsJson || [])];
  let sourceScopeDropdown = null;
  const sourceScopeHost = el('span', 'watch-rule-slot');
  sourceScopeHost.dataset.testid = 'watch-rule-source-scope';
  sourceScopeHost.title =
    'Which cameras this rule is evaluated on. Empty means all of them. Sharper than scoping '
    + 'an action: an unscoped rule still raises its event everywhere and only the side effect '
    + 'is silenced, while scoping the rule stops the event existing.';

  function buildSourceScope() {
    const items = ctx.sourceScopeItems();
    // A source removed or repointed in the rows above must not survive here as a uuid
    // this watcher no longer has — the backend rejects it, from a control the user can
    // no longer see.
    sourceScope = sourceScope.filter(uuid => items.some(i => i.uuid === uuid));
    sourceScopeDropdown?.destroy();
    sourceScopeHost.innerHTML = '';
    sourceScopeDropdown = createMultiSelectDropdown({
      label: '',
      items,
      defaultSelected: sourceScope,
      showModeToggle: false,
      allowExclude: false,
      // "All sources" rather than "None selected": an unscoped rule really does run
      // everywhere, so the empty state must not read as the opposite of what it does.
      emptySummary: 'All sources',
      emptyText: 'This watcher has no sources yet',
      onChange: (selected) => { sourceScope = [...selected]; ctx.onAnyChange(); },
    });
    sourceScopeHost.appendChild(sourceScopeDropdown.element);
  }
  buildSourceScope();

  // *When* the rule fires, as opposed to what it compares.
  //
  // It lives on the summary line rather than in the sentence above, and that is a
  // layout decision as much as an editorial one: the sentence was already full at this
  // pane's width, so a sixth control wrapped it onto a second line. Putting the choice
  // at the head of the sentence that explains its consequence costs one line instead of
  // two, and reads better than either half alone — "Event the moment it ends — fires
  // when this stops being true, if it had held for 2 seconds."
  const ruleTrigger = select(RULE_TRIGGERS, rule.trigger || 'while', () => {
    syncTrigger();
    ctx.onAnyChange();
  });
  ruleTrigger.dataset.testid = 'watch-rule-trigger';
  ruleTrigger.title =
    'Whether the event covers the whole time the rule holds, or is a single instant at one '
    + 'edge of it. An instant has no length, so what gets recorded around it comes entirely '
    + 'from an action’s pre-roll and post-roll — which is how “the last 15 seconds before the '
    + 'car leaves” is expressed. “the moment it ends” also waits out the clear delay before '
    + 'believing the subject really left, and only fires if it had been there for the start '
    + 'delay first.';

  const motionInWord = el('span', null, 'in');
  sentence.append(el('span', null, 'When'), subjectAHost, relation, motionInWord, motionZoneHost, subjectBKind, subjectBHost);
  sentence.append(el('span', null, 'on'), sourceScopeHost);
  row.appendChild(sentence);

  function syncArity() {
    // Motion is arity 0 — no subject at all. It reads pixels, so a label slot in front
    // of it would promise a filter the engine never applies.
    const isMotion = relation.value === 'motion';
    subjectAHost.hidden = isMotion;
    motionInWord.hidden = !isMotion;
    motionZoneHost.hidden = !isMotion;
    // `present` is arity 1; showing a second subject for it would imply a
    // comparison that never happens.
    const isUnary = relation.value === 'present' || isMotion;
    subjectBKind.hidden = isUnary;
    subjectBHost.hidden = isUnary;
    // The margin widens boxes before they are compared, so it means nothing to a
    // relation that compares no boxes (`present`) or only their centres (`distance`) —
    // and the backend rejects a nonzero one there rather than ignoring it. Hidden
    // rather than disabled: `getRule` sends 0 for both, so a value typed under an
    // earlier relation cannot survive as a payload the save would bounce.
    const usesMargin = !isUnary && relation.value !== 'distance';
    for (const field of marginField) field.hidden = !usesMargin;
    // Sensitivity is meaningless without pixels to difference.
    for (const field of sensitivityField) field.hidden = !isMotion;
    // Nothing to drag for `present`, and nothing to *draw* for motion: the preview
    // modal is two boxes and a score, and motion has neither.
    previewBtn.hidden = isUnary;
    // The threshold field is renamed, not just re-explained: under `present` it is a
    // detection confidence and calling it `match` was simply wrong. See THRESHOLD_HELP.
    const help = THRESHOLD_HELP[relation.value] || THRESHOLD_HELP.contains;
    thresholdLabel.textContent = help.label;
    thresholdField.title = help.title;
  }

  // Tuning keeps its four inputs in threshold / start / clear / max order — the
  // Playwright defaults test indexes `.watch-rule-tuning input` positionally.
  // The long labels ("Ends after clear (s)") became units-in-the-label plus a
  // tooltip: spelled out, the four of them alone filled a line.
  const tuning = el('div', 'watch-rule-tuning');
  const threshold = numberInput(rule.threshold ?? 0.6, { step: '0.05', min: 0 });
  threshold.dataset.testid = 'watch-rule-threshold';
  // Bounds mirror `InteractionRuleWrite.margin`: past -1 a box erodes to nothing and
  // the rule can never fire, and +2 already reaches two subject-widths away.
  const margin = numberInput(rule.margin ?? 0, { step: '0.1', min: -1 });
  margin.max = '2';
  // 0 is a real value here and means "use the backend default", so it must not be
  // coerced away — the floor is 0 and `num`'s fallback only fires on a blank field.
  const sensitivity = numberInput(rule.motionSensitivity ?? MOTION_DEFAULT_SENSITIVITY, { step: '0.02', min: 0 });
  sensitivity.max = '1';
  sensitivity.dataset.testid = 'watch-rule-motion-sensitivity';
  const minDuration = numberInput(rule.minDurationS ?? 2, { step: '0.5', min: 0 });
  const rearm = numberInput(rule.rearmDelayS ?? 10, { step: '1', min: 0 });
  // min 0, not 1: zero is the "never expire" value, for a rule whose event should last
  // exactly as long as its subject is there. See the caption's tooltip.
  const maxDuration = numberInput(rule.maxDurationS ?? 300, { step: '10', min: 0 });
  maxDuration.dataset.testid = 'watch-rule-max-duration';
  for (const input of [threshold, margin, sensitivity, minDuration, rearm, maxDuration]) {
    input.addEventListener('input', ctx.onAnyChange);
  }
  const tuningFields = [
    // Caption and tooltip are placeholders: `syncArity` rewrites both from
    // THRESHOLD_HELP for the relation in force, and runs before this row is shown.
    ['match', threshold, THRESHOLD_HELP.contains.title],
    [
      'nearby',
      margin,
      'Widens both boxes before matching, in units of the smaller box’s width: 1 fires '
      + 'when they are within about one subject-width. Sized to the object, not the frame, '
      + 'so it means the same thing near the camera and far from it. Negative demands a '
      + 'deeper overlap. 0 matches the boxes exactly as detected.',
    ],
    // 'starts'/'ends' rather than 'starts after'/'ends after': the row gained a fifth
    // field and the preview button, and the two long labels alone were what pushed the
    // button onto a line of its own. Both keep the sentence-length tooltip.
    // Both read differently under an impulse trigger, so the tooltips name the guard
    // rather than the transition: the same number is the event's start delay, or the
    // dwell an instant demands before it counts.
    [
      'starts',
      minDuration,
      'Seconds the relation must hold before it counts — the event’s start when recording the '
      + 'whole time, and the dwell required first when recording an instant.',
    ],
    [
      'ends',
      rearm,
      'Seconds the relation must stay clear — before the event ends when recording the whole '
      + 'time, and before a departure is believed when recording the moment it ends.',
    ],
    [
      'max',
      maxDuration,
      'Hard cap on event length, in seconds, so a stuck detection cannot hold an event open '
      + 'forever. Reaching it closes the event, and nothing opens again until the subject '
      + 'has been gone for the clear delay — so a car parked in the garage under a 30s cap '
      + 'gives you one 30s clip of it arriving, not one every thirty seconds until it '
      + 'leaves. 0 disables the cap, for an event that should last exactly as long as its '
      + 'subject does.',
    ],
    // Appended last on purpose. The four fields above are indexed positionally by the
    // Playwright defaults test, so a motion-only field inserted beside the threshold it
    // belongs with would renumber every one of them.
    [
      'sens',
      sensitivity,
      'How big a change to a single pixel counts as movement, 0–1. Raise it on a noisy or '
      + 'infrared camera that reports motion with nothing in front of it; lower it to catch '
      + 'a dark subject against a dark floor. This is the other half of “area” beside it: '
      + 'sensitivity decides which pixels changed, area decides how many is enough. '
      + '0 means use the default (0.1).',
    ],
  ];
  const marginField = [];
  const sensitivityField = [];
  const maxDurationField = [];
  let thresholdField = null;
  let thresholdLabel = null;
  for (const [label, input, title] of tuningFields) {
    // 3.8rem, not the 4.5 these fields used to get: there are five of them plus the
    // preview button now, and at the old width the button wrapped onto a second line —
    // which costs the whole row 33px and pushes a 1/1/1 profile into a scrollbar.
    const field = wfield(label, input, { width: '3.8rem' });
    field.title = title;
    if (input === margin) {
      field.dataset.testid = 'watch-rule-margin-field';
      marginField.push(field);
    }
    if (input === threshold) {
      field.dataset.testid = 'watch-rule-threshold-field';
      thresholdField = field;
      thresholdLabel = field.querySelector('.watch-field__label');
    }
    if (input === sensitivity) {
      field.dataset.testid = 'watch-rule-sensitivity-field';
      sensitivityField.push(field);
    }
    if (input === maxDuration) {
      field.dataset.testid = 'watch-rule-max-duration-field';
      maxDurationField.push(field);
    }
    tuning.appendChild(field);
  }

  // The threshold is the one number in this editor with no intuition behind it — a
  // fraction of an area nobody sees. The preview is where it becomes a thing you drag.
  const previewBtn = el('button', 'panel-btn watch-rule-preview', 'Preview…');
  previewBtn.type = 'button';
  previewBtn.dataset.testid = 'watch-rule-preview';
  previewBtn.title = 'Try this relation on two boxes you can drag, and see the score it '
    + 'produces. Apply sends the relation, match and nearby values back into this row.';
  previewBtn.addEventListener('click', () => {
    const labelName = uuid => ctx.labelItems.find(i => i.uuid === uuid)?.name || null;
    const subjectAUuid = subjectARefs[0] || null;
    const subjectBUuid = subjectBRefs[0] || null;
    const isZoneTarget = subjectBKind.value === 'zone';
    openOverlapPreviewModal({
      relation: relation.value,
      threshold: num(threshold, 0.6),
      margin: num(margin, 0),
      // Only the first ref of each slot: a rule spanning three cats still has one
      // geometry, and picking a sample per member would say nothing extra.
      subjectA: subjectAUuid ? { uuid: subjectAUuid, name: labelName(subjectAUuid) || 'Subject' } : null,
      subjectB: !isZoneTarget && subjectBUuid ? { uuid: subjectBUuid, name: labelName(subjectBUuid) || 'Target' } : null,
      subjectBName: isZoneTarget
        ? (ctx.zoneItems().find(z => z.uuid === subjectBUuid)?.name || 'Zone')
        : 'Target',
      onApply: ({ relation: appliedRelation, threshold: appliedThreshold, margin: appliedMargin }) => {
        relation.value = appliedRelation;
        threshold.value = String(Number(appliedThreshold.toFixed(2)));
        margin.value = String(Number(appliedMargin.toFixed(2)));
        syncArity();
        ctx.onAnyChange();
      },
    });
  });
  tuning.appendChild(previewBtn);
  row.appendChild(tuning);

  // What the six numbers above actually *do*, in a sentence, recomputed as they change.
  //
  // Worth the space because the row without it is five 3.8rem boxes labelled "match /
  // nearby / starts / ends / max / sens", with every explanation hidden in a tooltip.
  // That is how a 30 s cap on a `present` rule came to file an event every thirty
  // seconds for an evening: the behaviour was on screen the whole time, spelled as two
  // numbers whose interaction nothing stated.
  const summary = el('div', 'watch-rule-summary');
  summary.append(ruleTrigger);
  const summaryText = el('span', 'watch-rule-summary__text');
  summaryText.dataset.testid = 'watch-rule-summary';
  summary.append(summaryText);

  function renderSummary() {
    const start = num(minDuration, 2);
    const clear = num(rearm, 10);
    const cap = num(maxDuration, 300);
    // Terse on purpose. It has to fit one line beside the select at this pane's width,
    // and the row is already at the height the density tests bound — the tooltips carry
    // the long form. Seconds are "2s" for the same reason.
    if (ruleTrigger.value === 'on_enter') {
      summaryText.textContent =
        `— fires ${start}s after this becomes true, then not again until ${clear}s clear. `
        + 'Length comes from the action.';
    } else if (ruleTrigger.value === 'on_exit') {
      summaryText.textContent =
        `— fires when this stops, if it had held ${start}s; ${clear}s clear confirms it. `
        + 'Length comes from the action.';
    } else {
      // The cap's whole subtlety is what happens *after* it fires, which is the part
      // that was invisible — and the reason a parked car recorded all evening.
      summaryText.textContent = cap > 0
        ? `— starts after ${start}s true, ends after ${clear}s clear, at most ${cap}s `
          + 'then waits for the subject to leave.'
        : `— starts after ${start}s true, ends after ${clear}s clear. No length limit.`;
    }
  }

  for (const input of [minDuration, rearm, maxDuration]) {
    input.addEventListener('input', renderSummary);
  }

  // Remove button first, summary last, and the order matters: the summary takes a whole
  // flex line, so anything appended after it is pushed onto a *third* line. Appending it
  // before the button cost the row 48px instead of 25 — one line for the summary and
  // another for a button that had been sharing the tuning line quite happily.
  row.appendChild(removeButton(() => ctx.onDelete(), 'Remove rule'));
  row.appendChild(summary);

  /**
   * Show only the tuning fields the chosen trigger actually uses.
   *
   * The cap is hidden rather than disabled for an impulse, and `getRule` sends the
   * stored value untouched: an impulse has no span to cap, and leaving the box on
   * screen would invite someone to set the very thing that would break the headline
   * case — a 15-minute dwell under a 300 s cap.
   */
  function syncTrigger() {
    const isImpulse = ruleTrigger.value !== 'while';
    for (const field of maxDurationField) field.hidden = isImpulse;
    renderSummary();
  }

  syncArity();
  syncTrigger();

  return {
    element: row,
    /**
     * Adopt a newly-drawn set of zones.
     *
     * Called after the region editor saves. Rebuilding preserves any selection still on
     * offer and drops the rest, so a rule already pointing at a zone keeps it and only a
     * rule whose zone genuinely went away loses its reference.
     */
    refreshZones: () => {
      if (subjectBKind.value !== 'label') buildSubjectB();
      // The motion slot is always zones, so it always needs the new list — including
      // when subject B happens to be a label picker.
      buildMotionZones();
    },
    /** Adopt the watcher's current source list — see `buildScopes` on the action row. */
    refreshScopes: () => buildSourceScope(),
    destroy: () => {
      subjectADropdown?.destroy();
      subjectBDropdown?.destroy();
      motionZoneDropdown?.destroy();
      sourceScopeDropdown?.destroy();
    },
    getRule: () => {
      const isMotion = relation.value === 'motion';
      const isUnary = relation.value === 'present' || isMotion;
      return {
        uuid: rule.uuid || null,
        order: 0,
        name: nameInput.value.trim() || 'Unnamed rule',
        enabled: enabledBox.checked,
        publishToHa: publishBox.checked,
        publishSnapshotToHa: snapshotBox.checked,
        // Always sent: a watch profile saves as a whole-list replace, so a field the
        // editor omits is a field the save resets to its default.
        sourceUuidsJson: [...sourceScope],
        // For motion, subject A is *where* to measure, not what to look for: zones if
        // any are picked, otherwise `any`, which the engine reads as the whole frame.
        // The backend rejects `label` here, so this must not fall through.
        subjectAKind: isMotion ? (motionZoneRefs.length ? 'zone' : 'any') : 'label',
        subjectARefsJson: isMotion ? [...motionZoneRefs] : subjectARefs,
        // Zones have no hierarchy and the backend rejects the flag on a non-label slot.
        subjectAIncludeDescendants: isMotion ? false : subjectADescendants,
        relation: relation.value,
        subjectBKind: isUnary ? 'none' : subjectBKind.value,
        subjectBRefsJson: isUnary ? [] : subjectBRefs,
        // Never sent true for a unary or zone-kind slot — the backend rejects the flag
        // where there is no hierarchy to descend.
        subjectBIncludeDescendants:
          !isUnary && subjectBKind.value === 'label' && subjectBDescendants,
        threshold: num(threshold, isMotion ? MOTION_DEFAULT_AREA : 0.6),
        // Zero unless the relation actually measures boxes — see `syncArity`.
        margin: isUnary || relation.value === 'distance' ? 0 : num(margin, 0),
        // Zero for every other relation, which is also the backend's "use the default"
        // value — so a sensitivity typed under motion cannot leak into a geometric rule.
        motionSensitivity: isMotion ? num(sensitivity, MOTION_DEFAULT_SENSITIVITY) : 0,
        trigger: ruleTrigger.value,
        // These three defaults are not arbitrary: without them a raw threshold flaps
        // into an MQTT storm and a dozen clips for one visit.
        minDurationS: num(minDuration, 2),
        rearmDelayS: num(rearm, 10),
        // 0 is a real value here ("never expire"), so it must not be coerced to the
        // default — `num` only falls back on a blank or unparseable field.
        maxDurationS: num(maxDuration, 300),
      };
    },
  };
}

// ── Action row ──────────────────────────────────────────────────────────────
function buildActionRow(action, ctx) {
  const row = el('div', 'watch-action-row watch-row');
  row.dataset.testid = 'watch-action-row';

  const kind = select(ACTION_KINDS, action.kind || 'publish_mqtt', () => {
    syncTriggers();
    syncParams();
    ctx.onAnyChange();
  });
  kind.dataset.testid = 'watch-action-kind';
  kind.title =
    'What this action does. Changing it replaces the settings to its right, and may narrow '
    + 'the trigger beside it — a clip can only be cut when an event ends.';
  const trigger = select(TRIGGERS, action.trigger || 'event_ended', () => { syncScopeVisibility(); ctx.onAnyChange(); });
  trigger.dataset.testid = 'watch-action-trigger';
  trigger.title =
    'When this action runs. “event start” and “event end” fire once each per event; “every '
    + 'analysed frame” fires per frame regardless of whether any rule is holding, which is why '
    + 'it has no rule scope.';

  // Sits beside the trigger, and only when that kind's list is short — a `(?)` next to
  // an unconstrained dropdown is a promise of information that isn't there.
  const triggerHelp = el('span', 'watch-action-trigger-help');
  triggerHelp.dataset.testid = 'watch-action-trigger-help';

  /**
   * Rebuild the trigger options for the current action kind.
   *
   * Rebuilt rather than disabled: a greyed-out option still reads as "this ought to
   * work", and the honest thing for a trigger the action can never fire on is not to
   * offer it. A stored value that is no longer offered falls back to the first allowed
   * one, so switching an existing per-frame action to "Record clip" leaves a
   * configuration that actually runs instead of one that silently does nothing.
   */
  function syncTriggers() {
    const allowed = triggersFor(kind.value);
    const previous = trigger.value;
    trigger.innerHTML = '';
    for (const opt of allowed) {
      const o = document.createElement('option');
      o.value = opt.key;
      o.textContent = opt.text;
      trigger.appendChild(o);
    }
    trigger.value = allowed.some((t) => t.key === previous) ? previous : allowed[0].key;

    triggerHelp.innerHTML = '';
    const help = TRIGGER_LIMIT_HELP[kind.value];
    if (help && allowed.length < TRIGGERS.length) {
      triggerHelp.appendChild(noteHelp(help, {
        testid: 'watch-action-trigger-help',
        label: 'Why this action only has one trigger',
      }));
    }
    // The trigger drives whether a rule scope makes sense, so a coerced value has to
    // take the scope picker with it.
    syncScopeVisibility();
  }

  // Scope: two lists, both empty-means-all, mirroring `helpers/watch/scoping.py`.
  // Deliberately *not* a rules × sources grid — that grid is almost entirely empty and
  // stops fitting on screen at about four cameras, while two "All" chips keep the
  // common case a thing you never open.
  let ruleScope = [...(action.ruleUuidsJson || [])];
  let sourceScope = [...(action.sourceUuidsJson || [])];
  let ruleDropdown = null;
  let sourceDropdown = null;

  const ruleScopeField = el('div', 'watch-action-scope');
  ruleScopeField.dataset.testid = 'watch-action-rule-scope';
  ruleScopeField.title =
    'Which rules’ events this action responds to. Empty means all of them, including rules you '
    + 'have just added. Only saved rules can be picked individually — a new rule is listed here '
    + 'greyed out, and becomes selectable once you save the watcher.';
  const sourceScopeField = el('div', 'watch-action-scope');
  sourceScopeField.dataset.testid = 'watch-action-source-scope';
  sourceScopeField.title =
    'Which cameras this action runs for. Empty means all of them. The event is still raised '
    + 'on the others — only this side effect is skipped. To stop the event itself, scope the '
    + 'rule instead.';

  /**
   * Rebuild both scope pickers from the current rules and sources.
   *
   * Rebuilt rather than mutated for the same reason `buildRuleRow` rebuilds its zone
   * dropdown in `refreshZones`: the multi-select takes its items at construction, and
   * a watcher's rules and sources are edited in the same form, right above this row.
   * Selections survive, minus anything that no longer exists — keeping a uuid whose
   * rule was just deleted would fail validation on save with no visible cause.
   */
  function buildScopes() {
    const ruleItems = ctx.ruleScopeItems();
    const sourceItems = ctx.sourceScopeItems();
    ruleScope = ruleScope.filter(uuid => ruleItems.some(i => i.uuid === uuid));
    sourceScope = sourceScope.filter(uuid => sourceItems.some(i => i.uuid === uuid));

    ruleDropdown?.destroy();
    sourceDropdown?.destroy();
    ruleScopeField.innerHTML = '';
    sourceScopeField.innerHTML = '';

    ruleDropdown = createMultiSelectDropdown({
      label: 'Rules',
      items: ruleItems,
      defaultSelected: ruleScope,
      showModeToggle: false,
      allowExclude: false,
      // "All" is the honest summary of an empty selection here: an unscoped action
      // really does fire for every rule, so "None selected" would read as the opposite
      // of what it does.
      emptySummary: 'All',
      // Only reached when the watcher has no rules at all — an unsaved rule still gets a
      // greyed row of its own, which says more than this line can.
      emptyText: 'No rules yet — add one above',
      onChange: (selected) => { ruleScope = [...selected]; ctx.onAnyChange(); },
    });
    sourceDropdown = createMultiSelectDropdown({
      label: 'Sources',
      items: sourceItems,
      defaultSelected: sourceScope,
      showModeToggle: false,
      allowExclude: false,
      emptySummary: 'All',
      emptyText: 'This watcher has no sources yet',
      onChange: (selected) => { sourceScope = [...selected]; ctx.onAnyChange(); },
    });
    ruleScopeField.appendChild(ruleDropdown.element);
    sourceScopeField.appendChild(sourceDropdown.element);
  }

  /** A per-frame action has no rule to be scoped against, so the picker is hidden. */
  function syncScopeVisibility() {
    ruleScopeField.hidden = trigger.value === FRAME_ANALYSED_TRIGGER;
  }

  // A label for humans, never read by the engine. Three "Save snapshot" rows scoped to
  // different cameras are otherwise indistinguishable in the list, and the parameters
  // that tell them apart are spread across a dense row. Placeholdered rather than
  // required: an unnamed action still says what kind it is.
  //
  // Deliberately label-less and narrow. The row's single-line density is an invariant
  // with a test behind it, and a "Name" caption plus a comfortable input was a third
  // line on its own — the placeholder carries the same information for a field that is
  // empty exactly when it needs explaining.
  const enabledBox = enabledToggle(row, action.enabled, ctx.onAnyChange, {
    testid: 'watch-action-enabled',
    title: 'Whether this action runs. Unticked the watcher never constructs it, so it costs '
      + 'nothing and publishes nothing, but its parameters and scope are kept — the way to stop '
      + 'recording for a week without losing the setup. Takes effect on the watcher’s next start.',
  });
  // Left bare, unlike the rule row's. Tagging it cost this row a whole line (measured:
  // 90px -> 114px, past `test_source_and_action_rows_are_single_line`), and it does not
  // buy the same thing: an action row has exactly one checkbox, so there is no second
  // box to confuse it with and nothing it could be mistaken for.
  enabledBox.classList.add('watch-row-tag--enabled');
  row.appendChild(enabledBox);

  const name = document.createElement('input');
  name.type = 'text';
  name.value = action.name || '';
  name.placeholder = 'note…';
  name.dataset.testid = 'watch-action-name';
  name.className = 'watch-action-name';
  name.title = 'A note for you — e.g. "timelapse of the back garden every 30s". Not used by the watcher.';
  name.addEventListener('input', ctx.onAnyChange);
  row.appendChild(name);

  row.appendChild(wfield('Action', kind, { help: kind.title }));
  row.appendChild(wfield('on', trigger, { help: trigger.title }));
  row.appendChild(triggerHelp);
  // Appended bare, not through `wfield`: the dropdown's trigger already reads "Rules" /
  // "Sources", so a field label repeats the word and spends the horizontal budget that
  // keeps this row on one line. `wfield` also wraps its control in a `<label>`, which
  // around a panel of checkboxes means clicking the caption toggles the first one.
  row.appendChild(ruleScopeField);
  row.appendChild(sourceScopeField);

  buildScopes();
  // Before syncScopeVisibility's own call site: syncTriggers may coerce the trigger,
  // and the scope picker follows from whatever value survives that.
  syncTriggers();

  const params = el('div', 'watch-action-params');
  row.appendChild(params);

  const stored = { ...(action.paramsJson || {}) };
  let readParams = () => ({});

  function syncParams() {
    params.innerHTML = '';
    if (kind.value === 'record_clip') {
      const form = buildSchemaParams(ctx.schemaFor?.('record_clip') || {}, stored, ctx.onAnyChange, {
        testidPrefix: 'watch-action-param',
        emptyText: 'This action has no options.',
        fieldWrapper: wfield, numberInput, el, num,
        container: params,
        decorator: {
          // `output_dir` for the same reason as the timelapse's: the clip is a recording
          // attached to its event, swept by the recording retention settings, and a
          // typed path takes it out of the tree that sweep walks.
          //
          // `save_overlay` because there is no good reason to turn it off. The sidecar
          // is a few tens of KB, it re-encodes nothing, and the player can already hide
          // the boxes — so the checkbox only ever offered a way to permanently lose
          // information for no saving. Hidden rather than removed: the action still
          // honours the parameter, and omitting it here means the action's own default
          // (on) applies, including for rows that had previously stored it off.
          hidden: ['output_dir', 'save_overlay'],
          labels: { preroll_s: 'Pre-roll s', postroll_s: 'post-roll s', max_length_s: 'max s' },
          widths: { preroll_s: '4.5rem', postroll_s: '4.5rem', max_length_s: '4.5rem' },
          steps: { preroll_s: '1', postroll_s: '1', max_length_s: '10' },
          mins: { max_length_s: 1 },
        },
      });
      readParams = () => form.getParams();
    } else if (kind.value === 'publish_mqtt' || kind.value === 'publish_mqtt_image') {
      // One branch for both: they differ in their default topic and in which way round
      // two checkboxes default, and everything else about them is now declared rather
      // than written out twice.
      const isImage = kind.value === 'publish_mqtt_image';
      const defaultTopic = isImage ? DEFAULT_IMAGE_TOPIC_TEMPLATE : DEFAULT_TOPIC_TEMPLATE;
      let topicInput = null;
      const form = buildSchemaParams(ctx.schemaFor?.(kind.value) || {}, stored, ctx.onAnyChange, {
        testidPrefix: 'watch-action-param',
        emptyText: 'This action has no options.',
        fieldWrapper: wfield, numberInput, el, num,
        container: params,
        makeWidget: (type, field, value, onChange) => {
          if (type !== 'str') return null;
          // The topic is a template, not a fixed path: a rule matching several cats and
          // several litter boxes knows which ones fired, and routing on them is the whole
          // point of exposing those values. Blank means "unset", not "publish to the
          // empty topic", so the read falls back to the default rather than sending ''.
          const input = document.createElement('input');
          input.type = 'text';
          input.value = value || defaultTopic;
          input.placeholder = defaultTopic;
          input.addEventListener('input', onChange);
          topicInput = input;
          return { element: input, control: input, read: () => input.value.trim() || defaultTopic };
        },
        decorator: {
          // Declared by the action and left at its default, exactly as the hand-written
          // form did. Not an oversight carried forward: the action row is a single line
          // by design (there is a density test on it), and a QoS control costs that line
          // — it wraps the row and adds ~110px of editor scroll for a setting nobody has
          // needed to change. Offering it is a UX decision with a price, not a freebie.
          hidden: ['qos'],
          order: ['topic_template'],
          labels: { topic_template: 'Topic', max_width: 'max width', max_fps: 'max fps', annotate: 'Draw boxes' },
          testids: {
            topic_template: isImage ? 'watch-action-image-topic' : 'watch-action-topic',
            annotate: 'watch-action-image-annotate',
            max_fps: 'watch-action-image-max-fps',
          },
          widths: { topic_template: '13rem', quality: '4.5rem', max_width: '5rem', max_fps: '4.5rem', qos: '4.5rem' },
          steps: { quality: '5', max_width: '80', max_fps: '0.5', qos: '1' },
          mins: { quality: 1 },
          after: {
            topic_template: () => {
              if (!topicInput) return null;
              // The placeholder vocabulary lives behind this, not in the input's
              // tooltip — see `templateHelp`.
              const help = templateHelp(topicInput);
              // Ellipsised rather than wrapped: a resolved topic is long, and letting it
              // push the action row onto a second line is what made a small watcher
              // scroll. The full value stays available as a tooltip.
              const example = el('span', 'watch-hint watch-hint--truncate', '');
              example.dataset.testid = isImage ? 'watch-action-image-topic-example' : 'watch-action-topic-example';
              const sync = () => {
                example.textContent = `e.g. ${renderTopicExample(topicInput.value || defaultTopic)}`;
                example.title = example.textContent;
              };
              topicInput.addEventListener('input', sync);
              sync();
              return [help, example];
            },
          },
        },
      });
      readParams = () => form.getParams();
    } else if (kind.value === 'record_event_timelapse') {
      // The first kind rendered from the action's own `params_schema()` rather than
      // from a hand-written list of fields. What is left here is only what the schema
      // genuinely cannot say: which parameter this form must refuse to offer, and the
      // two live previews that are computed from more than one field at once.
      const form = buildSchemaParams(ctx.schemaFor?.('record_event_timelapse') || {}, stored, ctx.onAnyChange, {
        testidPrefix: 'watch-action-param',
        emptyText: 'This action has no options.',
        fieldWrapper: wfield, numberInput, el, num,
        container: params,
        decorator: {
          // Offered by the action, deliberately not by this form: the action defaults to
          // the recordings root, which is where the retention sweep looks. A typed path
          // puts the mp4 somewhere nothing will ever delete it, and the row it hangs off
          // the event would point outside the tree the file endpoint serves. Hidden
          // rather than sent as the default, so the action keeps owning the value.
          //
          // `save_overlay` is hidden for a different reason: it should simply always be
          // on. The sidecar costs a few tens of KB, re-encodes nothing, and the boxes
          // can already be switched off in the player — so the checkbox only offered a
          // way to lose the detections permanently in exchange for nothing. Omitting it
          // lets the action's default (on) apply, including to rows that stored it off.
          hidden: ['output_dir', 'save_overlay'],
          labels: {
            interval_s: 'Every s',
            output_fps: 'play fps',
            min_frames: 'min frames',
            max_frames: 'max frames',
            max_width: 'max width',
            full_resolution: 'Full res',
            preroll_frames: 'pre-roll',
          },
          // Kept exactly as they were before this form was generated, so the browser
          // tests keep asserting behaviour rather than being rewritten to match a
          // generated naming scheme.
          testids: {
            interval_s: 'watch-action-event-timelapse-interval',
            output_fps: 'watch-action-event-timelapse-fps',
            min_frames: 'watch-action-event-timelapse-min-frames',
            max_frames: 'watch-action-event-timelapse-max-frames',
            full_resolution: 'watch-action-event-timelapse-full-res',
            preroll_frames: 'watch-action-event-timelapse-preroll',
          },
          widths: {
            interval_s: '4.5rem', output_fps: '4.5rem', quality: '4.5rem',
            min_frames: '5rem', max_frames: '5rem', max_width: '5rem', preroll_frames: '4.5rem',
          },
          // Steps sized to how each number is actually adjusted — a 0.05 step on a
          // seconds-between-frames field means forty clicks to get from 10 to 12.
          steps: {
            interval_s: '5', output_fps: '1', min_frames: '1', max_frames: '100',
            max_width: '80', quality: '5', preroll_frames: '1',
          },
          mins: { interval_s: 1, output_fps: 1, min_frames: 1, max_frames: 1, quality: 1 },
          after: {
            // The speed-up is the product of the two fields above it, and it is the
            // number the user actually has in mind ("I want the two hours in ten
            // seconds"). Neither field means much alone, and working it out by hand is
            // exactly the arithmetic that makes people leave a timelapse at its defaults
            // and never find out what it does.
            output_fps: (controls) => {
              const speed = el('span', 'watch-hint watch-hint--truncate', '');
              speed.dataset.testid = 'watch-action-event-timelapse-speed';
              const every = controls.interval_s?.control;
              const fps = controls.output_fps?.control;
              if (!every || !fps) return null;
              const sync = () => {
                const interval = Math.max(1, num(every, 10));
                const rate = Math.max(1, Math.min(60, num(fps, 12)));
                const factor = interval * rate;
                // A half-hour visit: long enough to be the case worth previewing, short
                // enough that the answer stays in seconds at every sane setting.
                const plays = (30 * 60) / factor;
                const shown = plays >= 10 ? Math.round(plays) : Math.round(plays * 10) / 10;
                speed.textContent = `→ ${Math.round(factor)}× — a 30 min visit plays in ${shown} s`;
                speed.title = speed.textContent;
              };
              for (const i of [every, fps]) i.addEventListener('input', sync);
              sync();
              return speed;
            },
            // Frames, but the user is thinking in seconds — and the answer depends on
            // the interval field further up, so it cannot be a per-field help string.
            preroll_frames: (controls) => {
              const hint = el('span', 'watch-hint watch-hint--truncate', '');
              hint.dataset.testid = 'watch-action-event-timelapse-preroll-hint';
              const preroll = controls.preroll_frames?.control;
              const every = controls.interval_s?.control;
              if (!preroll || !every) return null;
              const sync = () => {
                const frames = Math.max(0, num(preroll, 0));
                hint.textContent = frames
                  ? `→ the ${frames * Math.max(1, num(every, 10))} s before it opened`
                  : '→ starts when the event does';
                hint.title = hint.textContent;
              };
              for (const i of [preroll, every]) i.addEventListener('input', sync);
              sync();
              return hint;
            },
          },
        },
      });
      readParams = () => form.getParams();
    } else if (kind.value === 'save_timelapse') {
      // Browsed rather than typed, unlike `save_snapshot`'s: a timelapse directory is
      // one the user makes for the purpose (and the picker can make it), where a
      // snapshot directory is normally left at its default. The path is dataset-root
      // relative — the only space the browser and /api/images/mkdir speak — and
      // `resolve_output_dir` in the action accepts that spelling.
      let dirPicker = null;
      const form = buildSchemaParams(ctx.schemaFor?.('save_timelapse') || {}, stored, ctx.onAnyChange, {
        testidPrefix: 'watch-action-param',
        emptyText: 'This action has no options.',
        fieldWrapper: wfield, numberInput, el, num,
        container: params,
        makeWidget: (type, field, value, onChange) => {
          if (type !== 'directory') return null;
          dirPicker = dirPickerField(value ?? DEFAULT_TIMELAPSE_DIR, onChange);
          return {
            element: dirPicker.element,
            // The picker's own text input, so the test id and the schema's help land on
            // the thing the user types into rather than on its wrapper.
            control: dirPicker.input,
            read: () => dirPicker.getValue().trim() || DEFAULT_TIMELAPSE_DIR,
          };
        },
        decorator: {
          // Declared order puts full_resolution before max_width; the numbers read
          // better together, with the two checkboxes after them.
          order: ['output_dir', 'interval_s', 'quality', 'max_width', 'full_resolution', 'annotate'],
          labels: { output_dir: 'Output dir', interval_s: 'Every s', max_width: 'max width', full_resolution: 'Full res', annotate: 'Draw boxes' },
          testids: {
            output_dir: 'watch-action-dir',
            interval_s: 'watch-action-timelapse-interval',
            full_resolution: 'watch-action-timelapse-full-res',
            annotate: 'watch-action-timelapse-annotate',
          },
          widths: { output_dir: '15rem', interval_s: '4.5rem', quality: '4.5rem', max_width: '5rem' },
          steps: { interval_s: '5', quality: '5', max_width: '80' },
          mins: { interval_s: 1, quality: 1 },
          after: {
            output_dir: () => {
              if (!dirPicker) return null;
              const help = templateHelp(dirPicker.input, {
                label: 'Placeholders you can use in this folder',
                intro: 'Substituted per path segment and slugified. One action covers every camera in '
                  + 'the watcher, so use {source} to give each its own folder — without it they all '
                  + 'share one. Must start with a real folder name before any placeholder. A YYYY-MM-DD '
                  + 'folder is added automatically unless you place {date} yourself.',
              });

              // The per-day subdirectory is created by the action, not picked here, so
              // the hint is the only place the user learns the frames are not flat.
              const layout = el('span', 'watch-hint watch-hint--truncate', '');
              layout.dataset.testid = 'watch-action-timelapse-layout';
              const sync = () => {
                const root = (dirPicker.getValue() || DEFAULT_TIMELAPSE_DIR).replace(/^\/+|\/+$/g, '') || DEFAULT_TIMELAPSE_DIR;
                const rendered = renderTopicExample(root);
                const day = root.includes('{date}') ? '' : `${TEMPLATE_EXAMPLE.date}/`;
                layout.textContent = `→ ${rendered}/${day}back-room_14-30-00.jpg`;
                layout.title = root.includes('{source}')
                  ? 'A folder per camera, then one per day.'
                  : 'Every camera in this watcher writes into this one folder — add {source} to split them.';
              };
              dirPicker.onChange(sync);
              sync();
              return [help, layout];
            },
          },
        },
      });
      readParams = () => form.getParams();
    } else {
      // The generic path: whatever the selected kind declares, rendered from its schema
      // with no branch of its own. `save_snapshot` is the kind that lands here today and
      // the decorator below is shaped for it, but an action kind added tomorrow gets a
      // working form here without touching this file.
      //
      // This also fixes `save_event`, which declares no parameters at all and used to
      // fall through to the snapshot form — it was asked for an output directory it does
      // not have, and saved one.
      let dirPicker = null;
      const form = buildSchemaParams(ctx.schemaFor?.(kind.value) || {}, stored, ctx.onAnyChange, {
        testidPrefix: 'watch-action-param',
        emptyText: 'This action has no options.',
        fieldWrapper: wfield, numberInput, el, num,
        container: params,
        makeWidget: (type, field, value, onChange) => {
          if (type === 'training_collector_uuid') return collectorPickerWidget(value, onChange);
          if (type !== 'directory') return null;
          dirPicker = dirPickerField(value ?? DEFAULT_SNAPSHOT_DIR, onChange, { placeholder: DEFAULT_SNAPSHOT_DIR });
          return {
            element: dirPicker.element,
            control: dirPicker.input,
            read: () => dirPicker.getValue().trim() || DEFAULT_SNAPSHOT_DIR,
          };
        },
        decorator: {
          labels: { output_dir: 'Output dir', collector_uuid: 'Collector' },
          testids: { output_dir: 'watch-action-dir', collector_uuid: 'watch-action-collector' },
          // The collector picker is a select plus a link, so it is sized as a unit here
          // and the select flexes inside it (see `.watch-collector-picker`). Without a
          // width the longest collector name sets it, and one long name pushed the row
          // onto a second line.
          widths: { output_dir: '15rem', quality: '4.5rem', collector_uuid: '15rem', min_interval_s: '4.5rem' },
          steps: { quality: '5' },
          mins: { quality: 1 },
          after: {
            output_dir: () => {
              if (!dirPicker) return null;
              // A template, not a fixed path — the same vocabulary the topics use, so "a
              // folder per camera" or "a folder per rule" is a thing you can write rather
              // than a feature request.
              const help = templateHelp(dirPicker.input, {
                label: 'Placeholders you can use in this folder',
                intro: 'Substituted per path segment and slugified, so a camera named "Kitchen / Cam" '
                  + 'cannot turn one folder into two. Click one to insert it. A YYYY-MM-DD folder is '
                  + 'added automatically unless you place {date} yourself.',
              });

              const example = el('span', 'watch-hint watch-hint--truncate', '');
              example.dataset.testid = 'watch-action-snapshot-example';
              const sync = () => {
                const root = (dirPicker.getValue().trim() || DEFAULT_SNAPSHOT_DIR).replace(/^\/+|\/+$/g, '');
                // `renderTopicExample` substitutes and collapses empty segments, which is
                // exactly what `render_path` does to a folder template — the two differ
                // only in what the result is fed to, so there is no second renderer to
                // keep in step.
                const rendered = renderTopicExample(root);
                // The automatic day level, shown only when it is actually added —
                // otherwise the preview claims a folder the action will not create.
                const day = root.includes('{date}') ? '' : `${TEMPLATE_EXAMPLE.date}/`;
                example.textContent = `→ ${rendered}/${day}14-30-00_back-room_cat-using-litter-box.jpg`;
                example.title = example.textContent;
              };
              dirPicker.onChange(sync);
              sync();
              return [help, example];
            },
          },
        },
      });
      readParams = () => form.getParams();
    }
  }

  row.appendChild(removeButton(() => ctx.onDelete(), 'Remove action'));

  syncParams();

  return {
    element: row,
    refreshScopes: buildScopes,
    /**
     * Relabel one rule in this action's scope picker, without rebuilding it.
     *
     * A rename is not a structural change, and rebuilding for one costs a swallowed
     * click: committing the new name happens on blur, so the rebuild fires *during* the
     * very click that moved focus away — often the click on this dropdown — detaching
     * its trigger between mousedown and mouseup. Returns false if the rule is unknown
     * here, leaving the caller to rebuild.
     */
    renameScopeItem: (uuid, name) => !!ruleDropdown?.renameItem(uuid, name),
    destroy: () => {
      ruleDropdown?.destroy();
      sourceDropdown?.destroy();
    },
    getAction: () => ({
      uuid: action.uuid || null,
      // A per-frame action has no rule, so its rule scope is not sent even if one was
      // picked before the trigger was switched — the backend ignores it, and saving it
      // anyway would leave a stale scope to reappear if the trigger changed back.
      ruleUuidsJson: trigger.value === FRAME_ANALYSED_TRIGGER ? [] : [...ruleScope],
      sourceUuidsJson: [...sourceScope],
      order: 0,
      enabled: enabledBox.checked,
      // Blank means unnamed, not an empty label — the backend stores null either way.
      name: name.value.trim() || null,
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

    const [profilesMap, sourcesMap, labelsMap, modelGroups, modelNames, trackers, ingestTiers, actionSchemas] = await Promise.all([
      fetchWatchProfiles().catch(() => new Map()),
      fetchSourceList().catch(() => new Map()),
      // fetchLabels() returns a Map, not an array — the fallback must match.
      fetchLabels().catch(() => new Map()),
      fetchModelGroups().catch(() => new Map()),
      // fetchModelsList() is the array of names, unlike its Map-returning neighbours.
      fetchModelsList().catch(() => []),
      fetchTrackers().catch(() => []),
      // Its own fetch rather than a field on the source list: the tier is derived, not
      // stored, and the sources list is served from a cache this must not inherit.
      fetchSourceIngestTiers().catch(() => new Map()),
      // Falls back to an empty list rather than failing the pane: without it the action
      // rows render their fields from nothing, which is bad, but a Watchers pane that
      // will not open at all is worse.
      fetchWatchActionSchemas().catch(() => []),
    ]);

    // kind → paramsSchema, which is what the action rows render from.
    const actionSchemaByKind = new Map((actionSchemas || []).map(a => [a.kind, a.paramsSchema || {}]));

    const sourceOptions = [...sourcesMap.values()].map(s => ({ key: s.uuid, text: s.name || s.uuid }));
    // Indented tree rather than a flat list: a vocabulary of cats under "Cat" under
    // "Animal" is unnavigable alphabetically, and the rule slots are multi-select so the
    // user needs to *find* three siblings, not just one label.
    const labelItems = flattenHierarchy(labelsMap).map(l => ({
      uuid: l.uuid,
      name: l.name || l.uuid,
      color: l.color || null,
      depth: l.depth,
      parentUuid: l.parentUuid ?? null,
    }));

    // Regions are keyed by source and fetched in one call rather than per source: a
    // watcher with eight cameras would otherwise open the editor with eight requests.
    let regionsBySource = new Map();
    async function loadRegions() {
      try {
        const all = await fetchSourceRegions(null);
        const next = new Map();
        for (const region of all) {
          if (!next.has(region.sourceUuid)) next.set(region.sourceUuid, []);
          next.get(region.sourceUuid).push(region);
        }
        regionsBySource = next;
      } catch {
        // An empty map degrades to "no zones on offer", which the pickers already say
        // out loud. Failing the whole pane over it would be worse.
        regionsBySource = new Map();
      }
    }
    await loadRegions();

    const regionsFor = (sourceUuid, role) =>
      (regionsBySource.get(sourceUuid) || []).filter(r => r.role === role);
    const modelGroupOptions = [...modelGroups.values()].map(g => ({ key: g.uuid, text: g.name || g.uuid }));
    const sourceNameFor = uuid => sourcesMap.get(uuid)?.name || uuid;
    // Undefined (not null) for an unknown source, so the badge renders nothing rather
    // than asserting "no buffer" about a source it simply has no answer for.
    const ingestTierFor = uuid => ingestTiers.get(uuid);

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
      const lifetimeRows = [];

      const bar = buildPanelEditorBar({
        container: screen.editorBar,
        item: profile,
        newTitle: 'New watcher',
        deleteTitle: 'Delete watcher',
        onSave: () => save(),
        onDelete: profile ? () => remove() : null,
      });
      // Assigned once the Actions section exists. Declared here because every edit goes
      // through `onAnyChange`, including edits made before that point — the hint is a
      // no-op until there is something to sync.
      let syncEventHint = () => {};
      const onAnyChange = () => { bar.setDirty(true); syncEventHint(); };

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
      identity.appendChild(wfield('Name', nameInput, {
        help: 'What this watcher is called, in the list on the left and in the Jobs tab. It is '
          + 'also what {watcher} renders as in a topic or folder template, so a rename changes '
          + 'where things are published — use {watcher_uuid} where that would matter.',
      }));

      const enabled = document.createElement('input');
      enabled.type = 'checkbox';
      enabled.checked = profile ? profile.enabled !== false : true;
      enabled.addEventListener('change', onAnyChange);
      // Deliberately modest wording: nothing reads this flag at run time today — the task
      // filters on each *row's* enabled flag, not the profile's — so promising that
      // unticking it stops the watcher would be a lie the user would find out the hard way.
      identity.appendChild(wfield('Enabled', enabled, {
        help: 'Your own on/off marker for this watcher. It does not stop a run that is already '
          + 'going — stop that from the Jobs tab.',
      }));

      const description = document.createElement('input');
      description.type = 'text';
      description.value = profile?.description || '';
      description.placeholder = 'optional';
      description.addEventListener('input', onAnyChange);
      identity.appendChild(wfield('Description', description, {
        help: 'A note for you. Never read by the watcher and never used in a template.',
      }));

      const defaultModels = modelChoiceSelect(
        { groupOptions: modelGroupOptions, modelNames },
        '(none)',
        modelChoiceValue(profile?.defaultModelGroupUuid, profile?.defaultModelName),
        onAnyChange,
        { testid: 'watch-default-model-choice' },
      );
      identity.appendChild(wfield('Default models', defaultModels, {
        help: 'The model or model stack every source in this watcher uses unless its own row '
          + 'overrides it. A stack and a single model are exclusive — picking one clears the '
          + 'other.',
      }));
      screen.editorBody.appendChild(identity);

      // ── Sources ──
      const addSourceBtn = el('button', 'panel-btn', '+ Add source');
      addSourceBtn.type = 'button';
      addSourceBtn.dataset.testid = 'watch-add-source';
      addSourceBtn.title = 'Add a camera for this watcher to analyse. Every rule and action below '
        + 'applies to all of them unless you scope it to some.';
      const sourcesBox = el('div', 'watch-rows');
      sourcesBox.dataset.testid = 'watch-sources-box';
      screen.editorBody.appendChild(sectionPanel(sectionHeader('Sources', addSourceBtn, SECTION_HELP.sources), sourcesBox));

      function addSource(src) {
        const entry = buildSourceRow(src, {
          sourceOptions,
          groupOptions: modelGroupOptions,
          modelNames,
          sourceNameFor,
          ingestTierFor,
          masksForSource: uuid => regionsFor(uuid, REGION_ROLE_MASK),
          onRegionsChanged: () => refreshRegions(),
          onSourceChanged: () => { refreshScopes(); for (const rule of ruleRows) rule.refreshZones?.(); },
          onAnyChange,
          onDelete: () => {
            entry.element.remove();
            sourceRows.splice(sourceRows.indexOf(entry), 1);
            // Removing a source removes its zones from the rules' options.
            for (const rule of ruleRows) rule.refreshZones?.();
            // ...and removes it from every action that was scoped to it, which would
            // otherwise be saved as a scope pointing at a source the watcher no
            // longer has and rejected by the backend with no visible cause.
            refreshScopes();
            onAnyChange();
          },
        });
        sourceRows.push(entry);
        sourcesBox.appendChild(entry.element);
        for (const rule of ruleRows) rule.refreshZones?.();
        refreshScopes();
      }

      addSourceBtn.addEventListener('click', () => {
        addSource({ sourceUuid: sourceOptions[0]?.key });
        onAnyChange();
      });

      // ── Tracking ──
      // The tracker turns per-frame detections into objects with identity over time,
      // which is what every rule below is written against — so it belongs above Rules.
      const addLifetimeBtn = el('button', 'panel-btn', '+ Add label lifetime');
      addLifetimeBtn.type = 'button';
      addLifetimeBtn.dataset.testid = 'watch-add-lifetime';
      addLifetimeBtn.title = 'Override how long one label’s track survives without a detection — '
        + 'a litter box can outlast 30s of misses where a cat should not.';

      const trackerOptions = trackers.map(t => ({ key: t.typename, text: t.displayName || t.typename }));
      const trackerSel = select(
        trackerOptions.length ? trackerOptions : [{ key: 'IouTracker', text: 'IoU / SORT-lite' }],
        profile?.trackerTypename || 'IouTracker',
        () => { renderTrackerParams(); onAnyChange(); },
      );
      trackerSel.dataset.testid = 'watch-tracker';
      trackerSel.title =
        'How per-frame detections are joined into objects with an identity over time. Every '
        + 'rule below is written against those objects, not against raw detections — which is '
        + 'what lets a rule say "for 2 seconds". Its options are under Tuning.';

      // The picker rides the section's own heading line and the params collapse behind a
      // closed twisty, so the whole section costs no height beyond the heading it needed
      // anyway. A profile with one source, rule and action must still fit without the
      // editor scrolling — there is a Playwright test that says so.
      // Reviewing what the tracker got wrong belongs beside the tracker that got it
      // wrong, not in its own section — it is read as a property of tracking, and it is
      // one checkbox.
      const analyseBlips = document.createElement('input');
      analyseBlips.type = 'checkbox';
      analyseBlips.dataset.testid = 'watch-analyse-blips';
      analyseBlips.checked = profile ? profile.blipAnalysisEnabled !== false : true;
      analyseBlips.addEventListener('change', onAnyChange);
      analyseBlips.title =
        'After each clip is recorded, compare the track against itself to find where the '
        + 'detector contradicted itself — an identity that flipped and flipped back, a subject '
        + 'lost while the tracker carried on. Costs no inference: it reads the boxes already '
        + 'saved beside the clip. Findings show on the event, and the frames can be sent to '
        + 'training.';

      const trackingHead = sectionHeader('Tracking', addLifetimeBtn, SECTION_HELP.tracking);
      trackingHead.insertBefore(wfield('Tracker', trackerSel, { help: trackerSel.title }), addLifetimeBtn);
      trackingHead.insertBefore(
        wfield('Find tracking mistakes', analyseBlips, { help: analyseBlips.title }),
        addLifetimeBtn,
      );

      const trackerParamsHost = el('div', 'watch-tracker-params-host');
      const tuning = document.createElement('details');
      tuning.className = 'watch-tracker-tuning';
      tuning.dataset.testid = 'watch-tracker-tuning';
      const tuningSummary = document.createElement('summary');
      tuningSummary.textContent = 'Tuning';
      tuningSummary.title = 'The selected tracker’s own options. Collapsed by default — the '
        + 'defaults are sane, and each field explains itself on hover.';
      tuning.append(tuningSummary, trackerParamsHost);
      trackingHead.insertBefore(tuning, addLifetimeBtn);

      const lifetimesBox = el('div', 'watch-rows');
      lifetimesBox.dataset.testid = 'watch-lifetimes-box';
      screen.editorBody.appendChild(sectionPanel(trackingHead, lifetimesBox));

      let trackerParams = null;
      function renderTrackerParams() {
        const chosen = trackers.find(t => t.typename === trackerSel.value);
        trackerParamsHost.innerHTML = '';
        trackerParams = buildTrackerParams(
          chosen?.paramsSchema || {},
          // Only reuse saved params while the selected tracker is the saved one; another
          // tracker's keys are meaningless to this one.
          trackerSel.value === profile?.trackerTypename ? profile?.trackerParamsJson : null,
          onAnyChange,
        );
        trackerParamsHost.appendChild(trackerParams.element);
      }
      renderTrackerParams();


      function addLifetime(lifetime) {
        const entry = buildLifetimeRow(lifetime, {
          labelItems,
          onAnyChange,
          onDelete: () => {
            entry.destroy();
            entry.element.remove();
            lifetimeRows.splice(lifetimeRows.indexOf(entry), 1);
            onAnyChange();
          },
        });
        lifetimeRows.push(entry);
        lifetimesBox.appendChild(entry.element);
      }

      addLifetimeBtn.addEventListener('click', () => { addLifetime({}); onAnyChange(); });

      // ── Rules ──
      const addRuleBtn = el('button', 'panel-btn', '+ Add rule');
      addRuleBtn.type = 'button';
      addRuleBtn.dataset.testid = 'watch-add-rule';
      addRuleBtn.title = 'Add a condition that raises an event while it holds. Rules decide when '
        + 'something happened; the actions below decide what is done about it.';
      const rulesBox = el('div', 'watch-rows');
      rulesBox.dataset.testid = 'watch-rules-box';
      // A rule has no uuid until the backend gives it one, and an action's rule scope is
      // a list of uuids — so a rule typed in just now cannot be scoped to until the
      // watcher is saved. Said here, at the moment the rule is added, rather than leaving
      // the user to notice the gap in the Actions pane's picker.
      const unsavedRuleHint = el('span', 'watch-hint', '⚠ save to use in actions');
      unsavedRuleHint.dataset.testid = 'watch-rules-unsaved-hint';
      unsavedRuleHint.title =
        'Some rules here have not been saved yet. An action is scoped to rules by their saved '
        + 'id, so a new rule is listed but greyed out in an action’s Rules picker until you '
        + 'save the watcher. Until then, an unscoped action still applies to it — "All" means '
        + 'every rule, including the new ones.';
      unsavedRuleHint.hidden = true;
      const rulesHead = sectionHeader('Rules', addRuleBtn, SECTION_HELP.rules);
      rulesHead.appendChild(unsavedRuleHint);
      screen.editorBody.appendChild(sectionPanel(rulesHead, rulesBox));

      // Zones on offer come from the sources this watcher actually uses, so a rule can
      // only ever reference a zone the watcher will have loaded at runtime. Prefixed
      // with the camera name because two cameras may each have a "Doorway".
      function zoneItems() {
        const items = [];
        const multipleSources = sourceRows.length > 1;
        for (const entry of sourceRows) {
          const sourceUuid = entry.getSourceUuid();
          for (const zone of regionsFor(sourceUuid, REGION_ROLE_ZONE)) {
            items.push({
              uuid: zone.uuid,
              name: multipleSources ? `${sourceNameFor(sourceUuid)} — ${zone.name}` : zone.name,
            });
          }
        }
        return items;
      }

      /**
       * The rules an action can be scoped to.
       *
       * Only *saved* rules are selectable: a rule gets its uuid from the backend on
       * first save, and offering an unsaved one would let the user pick a scope that
       * cannot be stored.
       *
       * Unsaved ones are still listed, greyed and unpickable. Dropping them silently was
       * the whole confusion this addresses: a rule the user had just typed in simply was
       * not in the list, which reads as a bug rather than as "not yet". A row that is
       * present, named and marked "unsaved" says which rules exist, which can be scoped,
       * and what to do about the difference.
       */
      function ruleScopeItems() {
        return ruleRows.map((entry, i) => {
          const rule = entry.getRule();
          // `getRule` already substitutes "Unnamed rule" for a blank name, so every row
          // reads as something.
          const name = rule.name;
          if (rule.uuid) return { uuid: rule.uuid, name };
          return {
            // Synthetic key: unsaved rules have no uuid, and the dropdown needs a unique
            // one per row. It never reaches a payload — the row cannot be ticked, and
            // `buildScopes` drops any selected uuid that is not in this list.
            uuid: `unsaved:${i}`,
            name,
            disabled: true,
            badge: { text: 'unsaved' },
            disabledTitle: 'This rule has not been saved yet, so nothing can be scoped to it. '
              + 'Save the watcher and it becomes selectable here.',
          };
        });
      }

      /**
       * Show or hide the Rules-header warning that some rule is not yet scopeable.
       *
       * Deliberately in the section header, beside the existing hints, rather than as its
       * own line: the editor's "a small profile does not scroll" budget is spent (see the
       * event hint below), and this one appears exactly when the user is mid-edit.
       */
      function syncUnsavedRuleHint() {
        unsavedRuleHint.hidden = !ruleRows.some(entry => !entry.getRule().uuid);
      }

      /** The sources an action can be scoped to — the ones this watcher actually uses. */
      function sourceScopeItems() {
        return sourceRows
          .map(entry => entry.getSourceUuid())
          .filter(Boolean)
          .map(uuid => ({ uuid, name: sourceNameFor(uuid) }));
      }

      /**
       * Push the current rules and sources into every scope picker — the actions' pair
       * and each rule's source scope.
       *
       * Called on structural changes only (a rule or source added, removed or
       * repointed), never from `onAnyChange` — that fires on every keystroke, and
       * rebuilding a dropdown under the cursor would drop focus mid-word.
       */
      function refreshScopes() {
        for (const entry of ruleRows) entry.refreshScopes?.();
        for (const entry of actionRows) entry.refreshScopes?.();
        syncUnsavedRuleHint();
      }

      /**
       * A rule was renamed: relabel it wherever it is offered, in place.
       *
       * Not `refreshScopes`. A name is committed on blur, so the rebuild that used to
       * follow ran *inside* the click that moved focus — and when that click was aimed
       * at an action's Rules picker, it landed on a button the rebuild then replaced
       * before mouseup, so no click event was dispatched and the picker stayed shut. The
       * user's first click after renaming a rule simply did nothing.
       *
       * Falls back to a full rebuild if any action does not know this rule yet — better
       * a lost click than a picker showing a name that no longer exists.
       */
      function renameRuleInScopes(entry) {
        const rule = entry.getRule();
        // Unsaved rules are keyed by row position (see `ruleScopeItems`), which a rename
        // does not change.
        const key = rule.uuid || `unsaved:${ruleRows.indexOf(entry)}`;
        let allKnown = true;
        for (const action of actionRows) {
          if (!action.renameScopeItem?.(key, rule.name)) allKnown = false;
        }
        if (!allKnown) refreshScopes();
      }

      /** Re-read regions from the server and push them into every row that shows them. */
      async function refreshRegions() {
        await loadRegions();
        for (const entry of ruleRows) entry.refreshZones?.();
        for (const entry of sourceRows) entry.refreshMasks?.();
      }

      function addRule(rule) {
        const entry = buildRuleRow(rule, {
          labelItems,
          zoneItems,
          sourceScopeItems,
          onAnyChange,
          onDelete: () => {
            entry.destroy?.();
            entry.element.remove();
            ruleRows.splice(ruleRows.indexOf(entry), 1);
            // A deleted rule must leave the scope of every action that named it.
            refreshScopes();
            onAnyChange();
          },
          // A renamed rule should read by its new name in the scope pickers. On
          // `change` (commit), not `input` — relabelling on every keystroke is wasted
          // work, and the rebuild this used to do fought the user for focus.
          onNameCommitted: () => renameRuleInScopes(entry),
        });
        ruleRows.push(entry);
        rulesBox.appendChild(entry.element);
        refreshScopes();
      }

      addRuleBtn.addEventListener('click', () => { addRule({}); onAnyChange(); });

      // ── Actions ──
      const addActionBtn = el('button', 'panel-btn', '+ Add action');
      addActionBtn.type = 'button';
      addActionBtn.dataset.testid = 'watch-add-action';
      addActionBtn.title = 'Add something to do when an event starts or ends, or on every analysed '
        + 'frame — publish MQTT, record a clip, save a snapshot or a timelapse frame.';
      const actionsBox = el('div', 'watch-rows');
      actionsBox.dataset.testid = 'watch-actions-box';
      // The immediate counterpart of the backend's "Nothing files events" warning, which
      // only appears once the watcher has been saved and validated. Worth saying at the
      // moment the actions are being chosen: a watcher that publishes but files nothing
      // is a legitimate setup, and an empty Events list is otherwise indistinguishable
      // from a camera that saw nothing all day.
      //
      // In the section *header*, not above the rows, and terse with the explanation on
      // the tooltip. The editor's "a 1/1/1 profile does not scroll" budget is 30px and
      // spent; a full sentence on its own line costs ~20 of them for a watcher that is
      // configured perfectly correctly, which is the wrong thing to charge for a hint.
      const eventHint = el('span', 'watch-hint', '⚠ files no events');
      eventHint.dataset.testid = 'watch-actions-event-hint';
      eventHint.title =
        'None of these actions records the event, so these rules will not appear in Events. '
        + 'They still publish, and still show in the watcher’s log — only the Events list is '
        + 'empty. Add a Save event action to list them there.';
      const actionsHead = sectionHeader('Actions', addActionBtn, SECTION_HELP.actions);
      actionsHead.appendChild(eventHint);
      screen.editorBody.appendChild(sectionPanel(actionsHead, actionsBox));

      syncEventHint = () => {
        eventHint.hidden = actionRows.some((entry) => {
          const a = entry.getAction();
          return a.enabled && EVENT_LOGGING_KINDS.has(a.kind) && a.trigger !== FRAME_ANALYSED_TRIGGER;
        });
      };

      function addAction(action) {
        const entry = buildActionRow(action, {
          ruleScopeItems,
          sourceScopeItems,
          schemaFor: kind => actionSchemaByKind.get(kind) || {},
          onAnyChange,
          onDelete: () => {
            entry.destroy?.();
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

      for (const lifetime of profile?.labelLifetimes || []) addLifetime(lifetime);
      for (const src of profile?.sources || []) addSource(src);
      for (const rule of profile?.rules || []) addRule(rule);
      for (const action of profile?.actions || []) addAction(action);
      syncEventHint();

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
          // Both keys, always: switching away from a stack has to null it out, or the
          // backend sees a stack *and* a model and rejects the pair.
          defaultModelGroupUuid: readModelChoice(defaultModels.value).modelGroupUuid,
          defaultModelName: readModelChoice(defaultModels.value).modelName,
          trackerTypename: trackerSel.value,
          trackerParamsJson: trackerParams?.getParams() || {},
          blipAnalysisEnabled: analyseBlips.checked,
          // Preserved rather than collected: the thresholds have no editor yet, and
          // sending `{}` from a form that cannot show them would silently wipe tuning
          // set through the API.
          blipThresholdsJson: profile?.blipThresholdsJson || {},
          // A row with nothing selected would fail server-side validation and take the
          // whole save with it — an unfinished row is dropped rather than fatal.
          labelLifetimes: lifetimeRows
            .map(entry => entry.getLifetime())
            .filter(l => l.labelUuidsJson.length),
          sources: sourceRows.map((entry, i) => ({ ...entry.getSource(), order: i })),
          rules: ruleRows.map((entry, i) => ({ ...entry.getRule(), order: i })),
          actions: actionRows.map((entry, i) => ({ ...entry.getAction(), order: i })),
        };
      }

      async function save() {
        const payload = collect();
        if (!payload.name) {
          toast('A watcher needs a name', 5000, 'error');
          return;
        }
        try {
          const saved = profile
            ? await updateWatchProfile(profile.uuid, payload)
            : await createWatchProfile(payload);
          profilesMap.set(saved.uuid, saved);
          toast(`Saved watcher "${saved.name}"`, 5000, 'success');
          bar.setDirty(false);
          renderList(saved.uuid);
          // Await the re-render: an unawaited one silently swallows its errors.
          await showValidation(saved.uuid);
          openEditor(saved);
          // A running watcher loaded its rules at start and will not see these
          // edits until it restarts. No-op when nothing is running on it.
          await promptRestartAffectedTasks(saved.uuid, { label: 'watcher' });
        } catch (e) {
          toast(e.message || 'Failed to save watcher', 5000, 'error');
        }
      }

      async function remove() {
        try {
          await deleteWatchProfile(profile.uuid);
          profilesMap.delete(profile.uuid);
          toast(`Deleted watcher "${profile.name}"`, 5000, 'success');
          screen.editorBody.innerHTML = '';
          screen.editorBar.innerHTML = '';
          renderList(null);
        } catch (e) {
          toast(e.message || 'Failed to delete watcher', 5000, 'error');
        }
      }

      screen.setDoSave(save);
      if (profile) showValidation(profile.uuid);
    }

    // Arriving from a Jobs launch card's ⚙ means "show me *this* watcher"; from the
    // Watch pane's add button, "start me a new one".
    const wantedUuid = takePendingSelection('tab-watcher-management');
    if (wantedUuid === NEW_ITEM_SELECTION) {
      renderList(null);
      openEditor(null);
      return;
    }
    const wanted = wantedUuid ? profilesMap.get(wantedUuid) ?? null : null;
    renderList(wanted?.uuid ?? null);
    if (wanted) openEditor(wanted);
  } finally {
    _rendering = false;
  }
}
