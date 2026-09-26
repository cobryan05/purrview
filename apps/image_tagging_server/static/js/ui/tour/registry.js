/**
 * Every tour the app can run, and when each offers itself.
 *
 * Tours nest the way the app does. `overview` visits each main tab and branches into
 * that tab's tour; `configuration` in turn visits each of its screens and branches
 * into that screen's tour. A branch returns, so finishing the Watch Rules walk puts
 * Configure back on the step after the one that offered it, and finishing Configure
 * puts the overview back on Annotate.
 *
 * `pages` says where the ? offers a tour: a main tab id for a whole tab, or
 * `mainTab:subTab` for one screen inside it. Configuration has eleven screens and one
 * tour across all of them told a user on Watch Rules nothing about watch rules.
 *
 * Only the overview sets `autoStart` — a page tour exists to be asked for, and a
 * workspace that has finished the overview should not be ambushed by nine more.
 * `eligible` then asks "is there anything here yet?" rather than "is this the first
 * visit?", because the first visit to an install someone else set up needs no
 * walkthrough, and a wiped workspace needs one again.
 *
 * Anchors are `data-tour` ids: the navigation ones live in index.html, the in-pane
 * ones on the control they name. A step's `navigate` is the tab path that gets the
 * anchor on screen — main tab first, then one id per nested tier.
 */
import { EventTypes } from '/app-static/js/shared/events/index.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchSourceList } from '/app-static/js/shared/api/sources.js';

/** Offered on every branching step. */
const DIVE_IN = 'Tour this tab →';
const DIVE_IN_SCREEN = 'Tour this screen →';
/** What Next means on a branching step: skip the depth, carry on along. */
const NEXT_TAB = 'Next tab →';
const NEXT_SCREEN = 'Next screen →';

const CONFIG = 'tab-configuration';

/** @type {import('./engine.js').Tour} */
export const LABELS = {
  id: 'labels',
  title: 'Labels & Tags',
  pages: [`${CONFIG}:tab-tag-management`],
  steps: [
    {
      id: 'intro',
      target: 'config.labelsAndTags',
      navigate: [CONFIG, 'tab-tag-management'],
      title: 'Labels are what you detect',
      body: 'A label is a thing a model can find in a frame: a particular cat, a food bowl, '
        + 'a person. Labels nest, so a broad "Cat" can hold one child per cat you want told '
        + 'apart. Tags are separate: notes you attach to an image or a box, like "needs '
        + 'review" or "night", and they never train anything.',
    },
    {
      id: 'create',
      target: 'config.labels.newObject',
      title: 'Create the top of the hierarchy',
      body: 'Type a name here and press Enter. Start broad — "Cat", "Person", "Vehicle" — '
        + 'because a label with children becomes the group and the children are the variants '
        + 'inside it. A label that never gets children is just a label, so this works either way.',
      advanceOn: { event: EventTypes.LABELS_UPDATED },
    },
    {
      id: 'addChild',
      target: 'config.labels.addChild',
      navigate: [CONFIG, 'tab-tag-management'],
      requireTarget: true,
      title: 'Add a variant under it',
      body: 'This button adds a child to the label. Nesting is optional: a flat list of labels '
        + 'trains perfectly well. Use it when you want variants told apart — a name for each cat '
        + 'under "Cat". Children are their own training classes either way, so later you can '
        + 'train one flat model over all of them, or let a coarse model hand off to a finer one.',
      advanceOn: { event: EventTypes.LABELS_UPDATED },
    },
    {
      id: 'variants',
      target: 'config.labels.variants',
      navigate: [CONFIG, 'tab-tag-management'],
      requireTarget: true,
      title: 'Exclusive Variants, on the parent',
      body: 'Turn this on when a parent\'s children are competing guesses about the same object '
        + '— which cat this is, not how many cats there are. Overlapping boxes from the group '
        + 'then compete as one class, so an ambiguous detection collapses to its best guess '
        + 'instead of drawing one box per candidate. How much overlap counts is a threshold you '
        + 'set here, kept stricter than the model\'s general one so two siblings standing close '
        + 'together are not merged into one.',
    },
    {
      id: 'ha',
      target: 'config.labels.ha',
      navigate: [CONFIG, 'tab-tag-management'],
      requireTarget: true,
      title: 'Home Assistant, per label',
      body: 'This is where a label earns its own Home Assistant device with a "detected now" '
        + 'sensor, optionally broken down per camera. Off by default so the vocabulary does not '
        + 'become a wall of devices — turn it on for the few you automate against.',
    },
    {
      id: 'classMap',
      target: 'config.labels.classMap',
      navigate: [CONFIG, 'tab-tag-management'],
      title: 'The class map connects models to labels',
      body: 'A trained model emits numbered classes, not names. This column maps those numbers '
        + 'onto your labels — which is also how you adopt an off-the-shelf model without '
        + 'retraining it: point its "man" at your Person and its "cat" at your Cat, and it '
        + 'detects into your vocabulary instead of its author\'s.',
    },
    {
      id: 'download',
      target: 'config.labels.classMap',
      navigate: [CONFIG, 'tab-tag-management'],
      title: 'Ready-made models are imported here too',
      body: 'You do not have to train anything to begin. This same panel downloads and imports a '
        + 'curated base model, or takes an .onnx file you already have — then the class map above '
        + 'points its classes at your labels. Enough to have something detecting on day one.',
    },
  ],
};

/** @type {import('./engine.js').Tour} */
export const SOURCES = {
  id: 'sources',
  title: 'Sources',
  pages: [`${CONFIG}:tab-source-management`],
  steps: [
    {
      id: 'intro',
      target: 'config.sources',
      navigate: [CONFIG, 'tab-source-management'],
      title: 'Sources are where frames come from',
      body: 'A source is a camera stream, a folder of images, a snapshot URL or an MQTT topic. '
        + 'Watchers run against any of them — a camera that only publishes a still every 30 '
        + 'seconds is watched the same way a live stream is — and annotation and training draw '
        + 'on all of them.',
    },
    {
      id: 'create',
      target: 'list.src.new',
      title: 'Add a source',
      body: 'Press + New Source and point it at a camera, a folder, a URL or a topic. Each kind '
        + 'asks only for what it needs, and a source can be tested here before anything is built '
        + 'on top of it.',
    },
  ],
};

/** @type {import('./engine.js').Tour} */
export const MODEL_SHAPES = {
  id: 'model-shapes',
  title: 'Model Shapes',
  pages: [`${CONFIG}:tab-model-management`],
  steps: [
    {
      id: 'intro',
      target: 'config.modelShapes',
      navigate: [CONFIG, 'tab-model-management'],
      title: 'A shape is the list of labels a model recognises',
      body: 'Nothing more than that: which labels this model can find, named and nested. It says '
        + 'nothing about how the model is trained — Training Configs, further down the rail, are '
        + 'where you connect a shape to an image dataset. Keeping the two apart is what lets one '
        + 'shape be trained several different ways.',
    },
    {
      id: 'nesting',
      target: 'config.modelShapes',
      title: 'Shapes nest, so models can hand off',
      body: 'A coarse "cat" detector can hand off to a finer one that tells your cats apart, '
        + 'along the same hierarchy you built under Labels. Nesting is a choice, not a '
        + 'requirement — one flat shape covering every label is a perfectly good setup.',
    },
  ],
};

/** @type {import('./engine.js').Tour} */
export const WATCH_RULES = {
  id: 'watch-rules',
  title: 'Watch Rules',
  pages: [`${CONFIG}:tab-watcher-management`],
  steps: [
    {
      id: 'intro',
      target: 'config.watchRules',
      navigate: [CONFIG, 'tab-watcher-management'],
      title: 'This is where most of the customisation lives',
      body: 'Everything else you have configured exists to feed this screen. A watcher attaches '
        + 'image sources to conditions — which labels, in which zones, for how long — and then '
        + 'says what to do when those conditions fire. The rest of this walk is the three parts '
        + 'of that sentence.',
    },
    {
      id: 'new',
      target: 'list.wp.new',
      title: 'A watcher is one running configuration',
      body: 'Press + New to build one. Everything below lives inside it, so a watcher is the '
        + 'unit you start, stop and reason about — one per camera group, or one for the whole '
        + 'house, whichever matches how you think about it.',
    },
    {
      id: 'sources',
      target: 'watchRules.addSource',
      requireTarget: true,
      title: 'First: which cameras, and how often',
      body: 'Add the sources this watcher pulls frames from, each with its own ingest rate and '
        + 'optional masked-out regions. Everything further down applies to all of them unless '
        + 'you scope it to some.',
    },
    {
      id: 'rules',
      target: 'watchRules.addRule',
      requireTarget: true,
      title: 'Then: what counts as something happening',
      body: 'A rule is a relation between two things — a label inside a zone, one label near '
        + 'another — plus how long it must hold and how soon it may fire again. Duration and '
        + 're-arm are what separate "a cat is at the bowl" from a box flickering for one frame.',
    },
    {
      id: 'actions',
      target: 'watchRules.addAction',
      requireTarget: true,
      title: 'Finally: what to do about it',
      body: 'Actions are the point of the whole app: record a clip, flip a Home Assistant '
        + 'sensor, save the frame for annotation, call a webhook. Each can be scoped to '
        + 'particular rules and sources, so one watcher can record everything while notifying '
        + 'you about only one thing.',
    },
  ],
};

/** @type {import('./engine.js').Tour} */
export const CONFIG_OVERVIEW = {
  id: 'config-overview',
  title: 'Configuration Overview',
  pages: [`${CONFIG}:tab-config-overview`],
  steps: [
    {
      id: 'map',
      target: 'config.overview',
      navigate: [CONFIG, 'tab-config-overview'],
      title: 'Every screen, with a count',
      body: 'Each entry is one screen on the rail beside it, with a live count of what it holds. '
        + 'A count of 0 is the fastest way to find what is still missing on a half-built install, '
        + 'and each entry takes you straight there.',
    },
    {
      id: 'order',
      target: 'config.labelsAndTags',
      title: 'They depend on each other in rail order',
      body: 'Labels first, because everything names them. Then Sources and Datasets, which supply '
        + 'images; Model Shapes and Training, which produce models; and Watch Rules, which spends '
        + 'all of it. Every screen has its own ? once you are on it.',
    },
  ],
};

/** @type {import('./engine.js').Tour} */
export const DATASETS = {
  id: 'datasets',
  title: 'Datasets',
  pages: [`${CONFIG}:tab-dataset-management`],
  steps: [
    {
      id: 'intro',
      target: 'config.datasets',
      navigate: [CONFIG, 'tab-dataset-management'],
      title: 'A dataset is a saved filter, not a folder',
      body: 'Images are not copied into it. A dataset stores the same filter the Review tab uses — '
        + 'these labels, these tags, this verification state — so it re-evaluates as you annotate '
        + 'and a model trained tomorrow sees everything added since.',
    },
    {
      id: 'create',
      target: 'list.ds.new',
      title: 'Build one from a filter',
      body: 'Press + New and set the filter here, or build it visually in Review and save the '
        + 'filter you already have. Training Configs then point at the dataset by name.',
    },
  ],
};

/** @type {import('./engine.js').Tour} */
export const MODEL_STACKS = {
  id: 'model-stacks',
  title: 'Model Stacks',
  pages: [`${CONFIG}:tab-modelstack-management`],
  steps: [
    {
      id: 'intro',
      target: 'config.modelStacks',
      navigate: [CONFIG, 'tab-modelstack-management'],
      title: 'A stack is an ordered list of detectors',
      body: 'Run several models over the same frame as one unit. A member is either a model or '
        + 'another stack, expanded recursively at run time, so a stack you already trust can be '
        + 'reused inside a bigger one rather than restated.',
    },
    {
      id: 'use',
      target: 'config.modelStacks',
      title: 'Anywhere a model is chosen, a stack will do',
      body: 'Inference, collectors and filter-chain steps all accept one. Naming the combination '
        + 'once means changing it in one place instead of everywhere it is used.',
    },
  ],
};

/** @type {import('./engine.js').Tour} */
export const TRAINING = {
  id: 'training',
  title: 'Training',
  pages: [`${CONFIG}:tab-training-management`],
  steps: [
    {
      id: 'intro',
      target: 'config.training',
      navigate: [CONFIG, 'tab-training-management'],
      title: 'Training joins a dataset to a model shape',
      body: 'This is the screen that connects the two halves: the images to learn from (a dataset) '
        + 'and the labels to learn (a model shape), plus which trainer and any hyperparameters. '
        + 'Saving one defines the run; starting it is a job.',
    },
    {
      id: 'readiness',
      target: 'config.training',
      title: 'Readiness tells you whether it is worth starting',
      body: 'The panel counts how many of the dataset\'s images have every target label explicitly '
        + 'verified, against how many are still unreviewed. Training on unverified boxes teaches '
        + 'the model your mistakes, so this number is the one to watch.',
    },
    {
      id: 'run',
      target: 'config.training',
      title: 'Starting a run',
      body: 'A saved config is started from Jobs, where its progress and the resulting model turn '
        + 'up. The model then needs a class map under Labels & Tags before anything can use it.',
    },
  ],
};

/** @type {import('./engine.js').Tour} */
export const FILTER_CHAINS = {
  id: 'filter-chains',
  title: 'Filter Chains',
  pages: [`${CONFIG}:tab-filterchain-management`],
  steps: [
    {
      id: 'intro',
      target: 'config.filterChains',
      navigate: [CONFIG, 'tab-filterchain-management'],
      title: 'Rules applied to images in bulk',
      body: 'A chain is an ordered set of steps that act on an image or its boxes: discard the '
        + 'image, tag it, strip unverified boxes, relabel or tag a box, add a box for a detection '
        + 'that overlaps nothing. Each step decides for itself whether it applies.',
    },
    {
      id: 'use',
      target: 'list.fc.new',
      title: 'Where a chain runs',
      body: 'Collectors run one as frames arrive, so only images worth keeping are stored, and '
        + 'Review can run one over a filtered set to fix a mistake across thousands of images at '
        + 'once instead of by hand.',
    },
  ],
};

/** @type {import('./engine.js').Tour} */
export const COLLECTORS = {
  id: 'collectors',
  title: 'Collectors',
  pages: [`${CONFIG}:tab-collector-management`],
  steps: [
    {
      id: 'intro',
      target: 'config.collectors',
      navigate: [CONFIG, 'tab-collector-management'],
      title: 'A collector gathers training images for you',
      body: 'It is the whole collect-images parameter set saved as one reusable thing: which '
        + 'models to run, the per-label confidence bands worth keeping, the filter chain, '
        + 'deduplication, and where frames land.',
    },
    {
      id: 'bands',
      target: 'config.collectors',
      title: 'Confidence bands are the useful part',
      body: 'Keeping frames a model was *unsure* about is what improves it. Confident detections '
        + 'teach it nothing it does not already know, so the bands usually want the middle of the '
        + 'range rather than the top.',
    },
    {
      id: 'use',
      target: 'list.col.new',
      title: 'Named once, used twice',
      body: 'A job names a collector and supplies the sources and pacing; a watcher can name one '
        + 'too, so frames are gathered while it watches. That reuse is why this is its own screen '
        + 'rather than thirty knobs on a task.',
    },
  ],
};

/** @type {import('./engine.js').Tour} */
export const TASKS = {
  id: 'tasks',
  title: 'Tasks',
  pages: [`${CONFIG}:tab-task-management`],
  steps: [
    {
      id: 'intro',
      target: 'config.tasks',
      navigate: [CONFIG, 'tab-task-management'],
      title: 'Task configs are saved job settings',
      body: 'A task config is one job\'s parameters kept under a name — an inference pass over a '
        + 'folder, a collection run, a timelapse. Defining it here separates deciding what a job '
        + 'does from deciding when to run it.',
    },
    {
      id: 'run',
      target: 'list.tc.new',
      title: 'Running one',
      body: 'Jobs starts a saved config and reports its progress. The same config can be run again '
        + 'whenever you like, which is what makes a repeated pass over new images one click.',
    },
  ],
};

/** @type {import('./engine.js').Tour} */
export const SETTINGS = {
  id: 'settings',
  title: 'Settings',
  pages: ['tab-settings'],
  steps: [
    {
      id: 'hotkeys',
      target: 'settings.hotkeys',
      navigate: ['tab-settings', 'tab-settings-hotkeys'],
      title: 'Keyboard shortcuts are worth five minutes',
      body: 'Annotating is the part of this you will do most, and it is far faster from the '
        + 'keyboard: switch tool, next image, save, verify — all one key. Every binding is listed '
        + 'here and every one can be rebound.',
    },
    {
      id: 'labelKeys',
      target: 'settings.hotkeys',
      title: 'Your labels get keys of their own',
      body: 'Once labels exist they appear in this list too, so a key can select one directly '
        + 'while you draw. If the list looks short, it is because the vocabulary is — come back '
        + 'after creating labels and assign the ones you use most.',
    },
    {
      id: 'interface',
      target: 'settings.interface',
      navigate: ['tab-settings', 'tab-settings-interface'],
      title: 'Interface holds the tour controls',
      body: 'Guided tours can be switched off here, restarted, or forgotten so they offer '
        + 'themselves again — useful when showing someone else around.',
    },
  ],
};

/** @type {import('./engine.js').Tour} */
export const CONFIGURATION = {
  id: 'configuration',
  title: 'Configure',
  pages: [CONFIG],
  steps: [
    {
      id: 'overview',
      target: 'config.overview',
      navigate: [CONFIG, 'tab-config-overview'],
      title: 'Overview is the map',
      body: 'Every screen on the left is listed here with a live count of what it holds. When a '
        + 'count reads 0 you know exactly what is still missing. This walk visits the four that '
        + 'matter first, in the order they depend on each other.',
      nextLabel: 'Start with Labels →',
    },
    {
      id: 'labels',
      target: 'config.labelsAndTags',
      navigate: [CONFIG, 'tab-tag-management'],
      title: 'Labels — the vocabulary',
      body: 'Nothing can be detected, trained or watched for until it has a name here. This is '
        + 'the first screen on a new install.',
      branches: [{ label: DIVE_IN_SCREEN, tour: 'labels' }],
      nextLabel: NEXT_SCREEN,
    },
    {
      id: 'sources',
      target: 'config.sources',
      navigate: [CONFIG, 'tab-source-management'],
      title: 'Sources — where frames come from',
      body: 'Cameras, folders, snapshot URLs and MQTT topics. Anything that can produce a frame '
        + 'can be a source, and watchers run against all of them.',
      branches: [{ label: DIVE_IN_SCREEN, tour: 'sources' }],
      nextLabel: NEXT_SCREEN,
    },
    {
      id: 'modelShapes',
      target: 'config.modelShapes',
      navigate: [CONFIG, 'tab-model-management'],
      title: 'Model Shapes — which labels a model knows',
      body: 'A shape is the list of labels a model recognises. Training Configs, further down the '
        + 'rail, are where a shape is connected to an image dataset and actually trained.',
      branches: [{ label: DIVE_IN_SCREEN, tour: 'model-shapes' }],
      nextLabel: NEXT_SCREEN,
    },
    {
      id: 'training',
      target: 'config.training',
      navigate: [CONFIG, 'tab-training-management'],
      title: 'Training — turn a dataset into a model',
      body: 'Where a model shape and a dataset are joined and actually trained. The readiness '
        + 'count here is what tells you whether a run is worth starting yet.',
      branches: [{ label: DIVE_IN_SCREEN, tour: 'training' }],
      nextLabel: NEXT_SCREEN,
    },
    {
      id: 'watchRules',
      target: 'config.watchRules',
      navigate: [CONFIG, 'tab-watcher-management'],
      title: 'Watch Rules — where it all comes together',
      body: 'Sources, conditions and actions in one place. This is where most of the '
        + 'customisation happens, and the screen worth the most of your time.',
      branches: [{ label: DIVE_IN_SCREEN, tour: 'watch-rules' }],
      nextLabel: NEXT_SCREEN,
    },
    {
      id: 'rest',
      target: 'config.overview',
      navigate: [CONFIG, 'tab-config-overview'],
      title: 'Every screen has a tour of its own',
      body: 'Datasets save a filter over your images; Model Stacks run several detectors as one; '
        + 'Filter Chains fix boxes in bulk; Collectors gather training frames; Tasks are saved job '
        + 'settings. Open any of them and the ? in the corner offers that screen\'s own walk — you '
        + 'never have to sit through this one again to reach it.',
    },
  ],
};

/** @type {import('./engine.js').Tour} */
export const ANNOTATE = {
  id: 'annotate',
  title: 'Annotate',
  pages: ['tab-image-tagging'],
  steps: [
    {
      id: 'files',
      target: 'annotate.files',
      navigate: ['tab-image-tagging', 'tab-annotate-workspace', 'tab-files'],
      title: 'Files is the image browser',
      body: 'Every image the workspace knows about is here, grouped by source. Frames a watcher '
        + 'saved for a closer look land here too, which is where most training data comes from '
        + 'once something is actually running.',
    },
    {
      id: 'import',
      target: 'annotate.toolbar.import',
      navigate: ['tab-image-tagging', 'tab-annotate-workspace', 'tab-canvas'],
      requireTarget: true,
      title: 'Import brings your own images in',
      body: 'Pick files from this computer and they open straight onto the canvas. Useful for '
        + 'seeding a label with a handful of examples before any camera is watching.',
    },
    {
      id: 'bbox',
      target: 'annotate.toolbar.bbox',
      requireTarget: true,
      title: 'Draw a box around the thing',
      body: 'BBox is the drawing tool: drag a rectangle around what you want the model to learn. '
        + 'Select switches back to moving and resizing boxes you already drew.',
    },
    {
      id: 'labels',
      target: 'annotate.sidebar.labels',
      navigate: ['tab-image-tagging', 'tab-annotate-workspace', 'tab-labels-tool'],
      title: 'A box means nothing until it has a label',
      body: 'The sidebar holds the same hierarchy you set up under Configure. Pick a label here '
        + 'before you draw and every new box gets it; pick one with a box selected to change it.',
    },
    {
      id: 'inference',
      target: 'annotate.sidebar.inference',
      navigate: ['tab-image-tagging', 'tab-annotate-workspace', 'tab-inference'],
      title: 'Let a model do the first pass',
      body: 'Run a model over the image and it fills in boxes for you. Correcting its mistakes is '
        + 'far faster than drawing from scratch, and the corrections are exactly the examples the '
        + 'next round of training needs most.',
    },
    {
      id: 'save',
      target: 'annotate.toolbar.save',
      requireTarget: true,
      title: 'Save writes the boxes back',
      body: 'Annotations live on the image, alongside it in a sidecar file. Nothing is written '
        + 'until you save, so a wrong box is one Undo away from never having happened.',
    },
    {
      id: 'review',
      target: 'annotate.review',
      navigate: ['tab-image-tagging', 'tab-review'],
      title: 'Review is annotation in bulk',
      body: 'A grid of every box matching a filter, verified or rejected a screen at a time. '
        + 'Training draws on verified boxes, so this is the step that turns a pile of guesses '
        + 'into a dataset.',
    },
  ],
};

/** @type {import('./engine.js').Tour} */
export const WATCH = {
  id: 'watch',
  title: 'Watch',
  pages: ['tab-nvr'],
  steps: [
    {
      id: 'live',
      target: 'nvr.live',
      navigate: ['tab-nvr', 'tab-nvr-live'],
      title: 'Live is every camera at once',
      body: 'One feed per running watcher, fanned out from a single connection to each camera so '
        + 'watching here costs the camera nothing extra.',
    },
    {
      id: 'events',
      target: 'nvr.events',
      navigate: ['tab-nvr', 'tab-nvr-events'],
      title: 'Events are what was actually seen',
      body: 'Each event is a detection with the clip around it — including the seconds before it '
        + 'triggered. This is the history you go back through when you want to know what happened.',
    },
    {
      id: 'addWatcher',
      target: 'nvr.addWatcher',
      navigate: ['tab-nvr', 'tab-nvr-live'],
      requireTarget: true,
      title: 'Add a watcher',
      body: 'A watcher is a watch rule that is running. This takes you to the rule editor under '
        + 'Configure, and once a rule is saved you start it from here or from Jobs.',
    },
  ],
};

/** @type {import('./engine.js').Tour} */
export const RUNS = {
  id: 'runs',
  title: 'Jobs',
  pages: ['tab-runs'],
  steps: [
    {
      id: 'launch',
      target: 'runs.launch',
      navigate: ['tab-runs', 'tab-runs-launch'],
      title: 'Start a job',
      body: 'Training, inference passes over a folder, collectors that pull frames in — anything '
        + 'that takes longer than a click starts here, from a task config you set up under '
        + 'Configure.',
    },
    {
      id: 'active',
      target: 'runs.active',
      navigate: ['tab-runs', 'tab-runs-active'],
      title: 'Running shows progress',
      body: 'Live progress for everything in flight, wherever it was started from. The badge on '
        + 'the tab is how you know something is still going while you are on another page. '
        + 'Results are handed over once and then cleared, so collect a finished job before '
        + 'starting the next.',
    },
  ],
};

/** @type {import('./engine.js').Tour} */
export const OVERVIEW = {
  id: 'overview',
  title: 'Getting started',
  autoStart: true,
  async eligible() {
    const [labels, sources] = await Promise.all([fetchLabels(), fetchSourceList()]);
    // Image-scope labels (Corrupt, Blurry) are seeded on every install and say
    // nothing about whether anyone has set the workspace up.
    const objectLabels = [...labels.values()].filter(l => l.scope !== 'image');
    return objectLabels.length === 0 && sources.size === 0;
  },
  steps: [
    {
      id: 'welcome',
      target: null,
      title: 'Welcome to PurrView',
      body: 'PurrView watches camera streams and tells you what it saw. Four tabs, in the order '
        + 'you need them. This is a minute-long lap of all four — at any stop you can dive into '
        + 'that tab instead, and you will come back here when you are done.',
      nextLabel: 'Show me round →',
    },
    {
      id: 'configure',
      target: 'nav.configuration',
      navigate: [CONFIG, 'tab-config-overview'],
      title: 'Configure — what to look for',
      body: 'What to look for, where to look, and which model does the looking. Nothing else in '
        + 'the app has anything to do until something is set up here, so this is the first stop '
        + 'on a new install.',
      branches: [{ label: DIVE_IN, tour: 'configuration' }],
      nextLabel: NEXT_TAB,
    },
    {
      id: 'annotate',
      target: 'nav.annotate',
      navigate: ['tab-image-tagging'],
      title: 'Annotate — browse, upload and label images',
      body: 'Browse every image the workspace holds, upload your own from this computer, and draw '
        + 'labelled boxes on them. A model can take the first pass and you correct what it got '
        + 'wrong, and Review verifies boxes in bulk once there are more than a few.',
      branches: [{ label: DIVE_IN, tour: 'annotate' }],
      nextLabel: NEXT_TAB,
    },
    {
      id: 'watch',
      target: 'nav.nvr',
      navigate: ['tab-nvr'],
      title: 'Watch — live feeds and recorded events',
      body: 'Two halves: Live shows the feed of every watcher currently running, and Events is the '
        + 'history of what they saw, with the clip around each one. "+ Add Watcher" in the corner '
        + 'starts a new one — a watcher is what links image sources to rules, and rules to the '
        + 'actions that fire when they match.',
      branches: [{ label: DIVE_IN, tour: 'watch' }],
      nextLabel: NEXT_TAB,
    },
    {
      id: 'runs',
      target: 'nav.runs',
      navigate: ['tab-runs'],
      title: 'Jobs — everything that takes a while',
      body: 'Training, inference passes and collectors run here. Start them, watch them, pick '
        + 'their results up here.',
      branches: [{ label: DIVE_IN, tour: 'runs' }],
      nextLabel: NEXT_TAB,
    },
    {
      id: 'settings',
      target: 'nav.settings',
      navigate: ['tab-settings'],
      title: 'Settings — and the keyboard shortcuts',
      body: 'Worth one stop before you start: annotating is much faster from the keyboard, and '
        + 'every shortcut is listed and rebindable here. Your own labels join that list once they '
        + 'exist, so a key can pick one while you draw.',
      branches: [{ label: 'Show me the shortcuts →', tour: 'settings' }],
      nextLabel: 'Finish →',
    },
    {
      id: 'done',
      target: 'nav.settings',
      title: 'That is the loop',
      body: 'Configure what to look for, annotate examples, train on them, watch. The ? in the '
        + 'bottom left opens the tour for whichever page you are on — including each screen '
        + 'inside Configure — and picks up where you left off.',
    },
  ],
};

/**
 * @type {ReadonlyArray<import('./engine.js').Tour>} in the order they are offered.
 * The overview is first because Settings runs `TOURS[0]` as "the" tour.
 */
export const TOURS = Object.freeze([
  OVERVIEW,
  CONFIGURATION,
  ANNOTATE,
  WATCH,
  RUNS,
  SETTINGS,
  CONFIG_OVERVIEW,
  LABELS,
  DATASETS,
  SOURCES,
  MODEL_SHAPES,
  MODEL_STACKS,
  TRAINING,
  FILTER_CHAINS,
  COLLECTORS,
  TASKS,
  WATCH_RULES,
]);

/** @param {string} id */
export function tourById(id) {
  return TOURS.find(t => t.id === id) ?? null;
}
