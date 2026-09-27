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
      body: 'Actions are the point of the whole app: record a clip, save a snapshot or a '
        + 'timelapse, collect the frame for annotation, publish to MQTT. Each can be scoped to '
        + 'particular rules and sources, so one watcher can record everything while announcing '
        + 'only one thing. Home Assistant needs no action of its own — a rule publishes its own '
        + 'entities over MQTT discovery, which is a switch on the rule rather than something you '
        + 'wire up here.',
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
      body: 'An image trains only once every label in the shape is accounted for — either a '
        + 'verified box on it, or an explicit "not in this picture". Nothing else goes in: an '
        + 'unverified box is not trained on badly, it is not trained on at all, which is why the '
        + 'ready count can sit at zero beside thousands of images. Confirming absence is real '
        + 'work here, not bookkeeping, because those images are the negatives — and the panel '
        + 'splits ready into positives and background so a set that is almost all empty frames '
        + 'cannot read as more signal than it holds.',
    },
    {
      id: 'run',
      target: 'config.training',
      title: 'Starting a run',
      body: 'A saved config is started from Jobs, where its progress and the resulting model turn '
        + 'up. A model trained here is ready to use immediately — it was trained against your '
        + 'own labels, so it records which ones its classes are and needs no mapping. The class '
        + 'map on Labels & Tags is for models you bring in from elsewhere, whose class names mean '
        + 'nothing here until you say what they correspond to.',
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
      title: 'Files browses the image folder',
      body: 'A folder tree, not a list: this is the image directory on the server, so what you '
        + 'see here is what is on disk. Click through the directories to find an image, or bring '
        + 'your own in from this computer. Frames a watcher saved for a closer look land here '
        + 'too, which is where most training data comes from once something is running.',
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
      id: 'contextMenu',
      target: 'annotate.canvasStage',
      navigate: ['tab-image-tagging', 'tab-annotate-workspace', 'tab-canvas'],
      requireTarget: true,
      title: 'Right-click a box for everything else',
      body: 'Right-clicking a box is where the rest of it lives: change its label, edit its tags, '
        + 'mark it Verified, Unverified or a Rejection, copy it, delete it, or search the whole '
        + 'workspace for boxes that look like it. With several boxes selected the menu acts on all '
        + 'of them at once. Right-clicking empty canvas is a different gesture — it toggles '
        + 'between Select and BBox, so you can keep drawing without going back to the toolbar.',
    },
    {
      id: 'inference',
      target: 'annotate.sidebar.inference',
      navigate: ['tab-image-tagging', 'tab-annotate-workspace', 'tab-inference'],
      title: 'Let a model do the first pass',
      body: 'Run a model over the image and it fills in boxes for you. Correcting its mistakes is '
        + 'far faster than drawing from scratch.',
    },
    {
      id: 'save',
      target: 'annotate.toolbar.save',
      requireTarget: true,
      title: 'Save writes the boxes back',
      body: 'Annotations live beside the image in a sidecar file, and nothing is written until you '
        + 'save — so a wrong box is one Undo away from never having happened. You will almost '
        + 'certainly stop using this button: save is {key:action:save}, {key:action:prevImage} and '
        + '{key:action:nextImage} step between images, and annotating a folder is much faster with '
        + 'a hand on the keyboard. Settings lists the rest.',
    },
    {
      id: 'sameCanvasElsewhere',
      target: 'annotate.canvasStage',
      navigate: ['tab-image-tagging', 'tab-annotate-workspace', 'tab-canvas'],
      requireTarget: true,
      title: 'This canvas turns up in modals too',
      body: 'Once there is real data, most of the drawing you do will not happen on this tab: '
        + 'Review opens the same canvas in a modal on double-click, and so does an event when you '
        + 'annotate a frame of its clip. It is the same editor with the same tools, the same '
        + 'right-click menu and the same hotkeys, so everything above still applies — this tab is '
        + 'where you go to browse the folder and work through images one at a time.',
    },
    {
      id: 'review',
      target: 'annotate.review',
      navigate: ['tab-image-tagging', 'tab-review'],
      title: 'Review is annotation in bulk',
      body: 'A grid of every box matching a filter, judged a screen at a time. Training draws '
        + 'on verified boxes, so this is the step that turns a pile of guesses into a dataset — '
        + 'and it has far more in it than one step can say.',
      branches: [{ label: DIVE_IN_SCREEN, tour: 'review' }],
    },
  ],
};

/**
 * @type {import('./engine.js').Tour}
 *
 * The longest tour in here, and deliberately so. Review is where a pile of guesses
 * becomes a dataset, and almost none of it is discoverable: the verification marks are
 * on a right-click, the editor is on a double-click, and half the toolbar operates on
 * the whole filter rather than on what is on screen — which is the difference between
 * relabelling twelve boxes and relabelling nine thousand.
 */
export const REVIEW = {
  id: 'review',
  title: 'Review',
  pages: ['tab-image-tagging:tab-review'],
  steps: [
    {
      id: 'intro',
      target: 'review.grid',
      navigate: ['tab-image-tagging', 'tab-review'],
      title: 'Review is annotation in bulk',
      body: 'A grid of everything matching a filter, judged a screen at a time instead of one '
        + 'image at a time. Training draws on boxes you have verified, so this is the screen '
        + 'that turns a pile of model guesses into a dataset worth training on.',
    },
    {
      id: 'viewToggle',
      target: 'review.viewToggle',
      title: 'Whole images, or individual boxes',
      body: 'Images shows one card per picture. Bboxes shows one card per box, cropped to it — '
        + 'which is the view for judging labels quickly, because you see the thing itself '
        + 'rather than hunting for it in the frame. All interleaves both.',
    },
    {
      id: 'filters',
      target: 'review.filters',
      title: 'Filters decide what you are looking at',
      body: 'Labels, tags, dates, filename — and Box status, which is the one you will live in. '
        + 'Verified is what you have already checked, Unverified is what a model proposed and '
        + 'nobody has confirmed, and No boxes is everything still untouched. Those three are '
        + 'your done pile, your review queue and your not-started pile. Filters also decide '
        + 'what every button further along acts on.',
    },
    {
      id: 'filterNegation',
      target: 'review.filters',
      title: 'Filters can be turned inside out',
      body: 'Every row in the label, tag and source pickers has a ≠ toggle, so "images that do '
        + 'NOT contain a cat" is a filter rather than something you eyeball. Dates and '
        + 'filenames negate too. This is how you find what a model is missing — everything it '
        + 'never put a box on — and how you audit a tag you believe you applied everywhere. '
        + 'The chips below say which way round each one is set.',
    },
    {
      id: 'filterChips',
      target: 'review.filterChips',
      requireTarget: true,
      title: 'Active filters stay visible',
      body: 'Each filter you add becomes a chip here, so a grid that looks emptier than it '
        + 'should is explained by something you can see rather than something you forgot. '
        + 'Drop one by its ×, or clear the lot.',
    },
    {
      id: 'views',
      target: 'review.views',
      requireTarget: true,
      title: 'Save a filter you will want again',
      body: 'A filter combination worth rebuilding is worth naming. Save it as a View and it '
        + 'comes back from this dropdown — "cat boxes still unverified" is a View, not '
        + 'something to reassemble every session.',
    },
    {
      id: 'openEditor',
      target: 'review.grid',
      title: 'Double-click to fix it properly',
      body: 'A single click on a card ticks it for a bulk action; a double-click opens the image '
        + 'on a full canvas in a modal, with the drawing tools, the right-click menu and the same '
        + 'keyboard shortcuts as the Annotate tab — so a box that is simply in the wrong place '
        + 'gets dragged rather than rejected. This is where most of the real annotating happens '
        + 'once there is data to work through. {key:action:prevImage} and {key:action:nextImage} '
        + 'move through the grid without closing it.',
    },
    {
      id: 'contextMenu',
      target: 'review.grid',
      title: 'Right-click is the fast path',
      body: 'Most judgements need no editor at all. Right-click a box to mark it Verified, '
        + 'Unverified or a Rejection, change its label, edit its tags, or search the whole '
        + 'workspace for boxes that look like it. On a whole image you also get present / '
        + 'absent, which is how you record that a label is genuinely not in a picture. '
        + 'Right-clicking one of several ticked cards acts on the whole selection, so the menu '
        + 'doubles as the bulk tool. The same menu is on the canvas inside the editor.',
    },
    {
      id: 'bulkApply',
      // The real control first, the checkbox that summons it second: "Apply to Selected"
      // lives in a bar that only appears with a selection, and a step about acting on
      // many boxes is worth more with a spotlight on the way in than passed over.
      target: ['review.bulkApply', 'review.select'],
      title: 'Select many, change them together',
      body: 'Tick cards — shift-click takes a range, and Select all takes the page — and an '
        + 'action bar appears along the bottom. From there one label or tag change applies to '
        + 'every card at once. Before it commits you are shown the first three and last three '
        + 'it will touch, so you can see the range is consistent end to end rather than '
        + 'trusting the count. On whole images the actions include present and absent, which '
        + 'is what you want straight after adding a label: mark it absent across everything '
        + 'you already have, so the model learns where it is not as well as where it is.',
    },
    {
      id: 'bulkApplyAll',
      target: 'review.bulkApplyAll',
      requireTarget: true,
      title: 'Or change everything the filter matches',
      body: 'Not just this page — every result behind the current filter, however many pages '
        + 'that is. The count in the button is the real total. This is the one button here '
        + 'that can rewrite thousands of boxes on one click, so read the preview it shows you.',
    },
    {
      id: 'runModel',
      target: 'review.runModel',
      title: 'Let a model pre-fill the whole queue',
      body: 'Runs a model across everything the filter matches and holds the results as '
        + 'proposals rather than writing them. You then walk the proposals and keep the good '
        + 'ones — far faster than drawing from scratch on every image.',
    },
    {
      id: 'savedPreviews',
      target: 'review.savedPreviews',
      title: 'A big triage does not have to finish today',
      body: 'Those proposals are saved, so a run over ten thousand images can be picked up '
        + 'tomorrow instead of abandoned. The count on the button is how many runs are waiting '
        + 'for you.',
    },
    {
      id: 'verifyModel',
      target: 'review.verifyModel',
      title: 'Ask how good a model actually is',
      body: 'Scores a model — or two, side by side — against boxes you have already verified, '
        + 'and reports how often it agrees with you. It writes nothing. This is how you find '
        + 'out whether the last training round helped, rather than guessing from how the '
        + 'overlays look.',
    },
    {
      id: 'uniqueFirst',
      target: 'review.uniqueFirst',
      requireTarget: true,
      title: 'Annotate the most different images first',
      body: 'Frames from a camera are mostly near-duplicates, and labelling fifty views of the '
        + 'same empty hallway teaches a model almost nothing. This orders the queue so each '
        + 'image looks as unlike the ones before it as possible — the fastest way to spend an '
        + 'hour of annotating well.',
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
      id: 'openEvent',
      target: 'nvr.eventGrid',
      navigate: ['tab-nvr', 'tab-nvr-events'],
      requireTarget: true,
      title: 'Open one to see, fix and harvest it',
      body: 'A single click opens an event on its clip — the Review grid next door takes a '
        + 'double-click, because there a click ticks the card instead. Inside you get what the '
        + 'watcher decided and why, its detection boxes drawn over the video as it plays, a '
        + 'correction for a label it got wrong, and any frame of it saved into the annotation '
        + 'queue as training data. Shift- or ctrl-click a card to tick it for a bulk action '
        + 'without opening anything.',
    },
    {
      id: 'eventTour',
      target: 'nvr.eventGrid',
      navigate: ['tab-nvr', 'tab-nvr-events'],
      requireTarget: true,
      title: 'That modal has its own tour',
      body: 'More than this step can hold, so it carries its own ? beside Close. A tour cannot '
        + 'open a modal for you: open an event, press that ?, and it walks you through the clip, '
        + 'the overlays, the corrections and the frame capture.',
    },
    {
      id: 'addWatcher',
      target: 'nvr.addWatcher',
      navigate: ['tab-nvr', 'tab-nvr-live'],
      requireTarget: true,
      title: 'Add a watcher',
      body: 'A watcher is the thing that actually watches. It gathers up the cameras to look at, '
        + 'how to track what moves through them, which events are worth noticing, and what to do '
        + 'when one happens — then runs as a unit you start and stop. This takes you to its editor '
        + 'under Configure; once saved, you start it from here or from Jobs.',
    },
  ],
};

/**
 * @type {import('./engine.js').Tour}
 *
 * The one tour that belongs to a modal rather than a page: `pages` is empty, so the
 * page ? never lists it, and it is started from the ? in the event modal's own header.
 * The engine can navigate between tabs but cannot open a modal, so no page tour
 * branches into this one either — they point at the ? instead.
 */
export const EVENT_DETAIL = {
  id: 'event-detail',
  title: 'Inside an event',
  pages: [],
  steps: [
    {
      id: 'media',
      target: 'eventDetail.media',
      title: 'The clip, and the moment it fired',
      body: 'The recording of the event, including the seconds before the watcher decided '
        + 'anything — the mark on the timeline is where it triggered, so you can see what led '
        + 'up to it. ‹ Previous and Next › walk the events behind this modal without closing '
        + 'it, and {key:action:prevImage} / {key:action:nextImage} do the same from the keyboard.',
    },
    {
      id: 'mediaTabs',
      target: 'eventDetail.mediaTabs',
      requireTarget: true,
      title: 'Clip or timelapse',
      body: 'An event that ran for hours has a timelapse as well as a clip. The clip is the '
        + 'literal record and the timelapse is a summary of it — a second of timelapse covers '
        + 'much more than a second of real time, which is why frames are captured from the '
        + 'clip when there is one.',
    },
    {
      id: 'overlays',
      target: 'eventDetail.overlays',
      requireTarget: true,
      title: 'See what the watcher saw',
      body: 'Draws the boxes the detector had at each frame, over the video as it plays. This '
        + 'is how you tell a genuine detection from a tracker that latched onto a shadow — and '
        + 'the boxes are drawn above the video, never burnt into it, so a frame you save is '
        + 'still a clean picture.',
    },
    {
      id: 'frameTools',
      target: 'eventDetail.frameTools',
      requireTarget: true,
      title: 'Turn a frame into training data',
      body: 'Pause anywhere and this saves that frame into the annotation queue, pre-seeded '
        + 'with the boxes the watcher had at that instant — so correcting one is a drag rather '
        + 'than a redraw. Annotate Image saves it and opens it on a canvas straight away; '
        + 'closing that canvas brings you back here, at the same moment of the clip. This is '
        + 'the loop that turns "the camera got it wrong" into the example that fixes it.',
    },
    {
      id: 'details',
      target: 'eventDetail.details',
      title: 'What the watcher decided',
      body: 'Which camera, which rule matched, what was acting and what it acted on, how long '
        + 'it ran and how confident the detector ever got. An event that ended because the '
        + 'watcher stopped, or because it hit its length cap, says so here rather than leaving '
        + 'you to infer it from the duration.',
    },
    {
      id: 'blips',
      target: 'eventDetail.blips',
      requireTarget: true,
      title: 'Where the tracking went wrong',
      body: 'The analyser\'s own findings for this clip: where the identity flipped from one '
        + 'label to another, where the track dropped out, where two animals merged into one. '
        + 'Each one seeks the clip to the moment it happened. Confirming that a flip was a '
        + 'real change rather than a mistake carries the correction straight into the dialog '
        + 'below, so you do not pick the label twice.',
    },
    {
      id: 'relabel',
      target: 'eventDetail.relabel',
      requireTarget: true,
      title: 'Correct the label across the whole event',
      body: '"Wrong label…" is for the event the watcher got wrong: say which label should '
        + 'have been which, and it harvests frames from across the clip with the corrected '
        + 'boxes, saved unverified for review. It can fix the event\'s own row at the same '
        + 'time, so the headline and the grouping stop being wrong too. Frames are only ever '
        + 'taken from moments the recording actually contains — a clip shorter than its event '
        + 'yields fewer frames rather than eight copies of its last picture.',
    },
    {
      id: 'keep',
      target: 'eventDetail.pin',
      title: 'Pin what you want kept',
      body: 'Events are pruned as the store fills, and a pin is a promise this one will not be '
        + 'removed — by the cleanup or by hand. Pin the events you are keeping as evidence or '
        + 'as a test case before you go anywhere near a bulk delete.',
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
  REVIEW,
  WATCH,
  EVENT_DETAIL,
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
