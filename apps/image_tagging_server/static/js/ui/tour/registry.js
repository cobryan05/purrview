/**
 * Every tour the app can run, and when each offers itself.
 *
 * A tour auto-starts only when its `eligible` check says the workspace looks like
 * one that needs it — the getting-started tour asks "is there anything here yet?"
 * rather than "is this the first visit?", because the first visit to an install
 * someone else set up needs no walkthrough, and a wiped workspace needs one again.
 *
 * Anchors are `data-tour` ids: the navigation ones live in index.html, the
 * in-pane ones on the control they name. A step's `navigate` is the pair of tab
 * ids that get the anchor on screen.
 */
import { EventTypes } from '/app-static/js/shared/events/index.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchSourceList } from '/app-static/js/shared/api/sources.js';

/** @type {import('./engine.js').Tour} */
export const GETTING_STARTED = {
  id: 'getting-started',
  title: 'Getting started',
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
      target: 'nav.configuration',
      navigate: ['tab-configuration', 'tab-config-overview'],
      title: 'Welcome to PurrView',
      body: 'Everything a watcher needs — what to look for, where to look, and which model does '
        + 'the looking — is set up under Configuration. This tour walks through the pieces in '
        + 'the order they depend on each other.',
    },
    {
      id: 'overview',
      target: 'config.overview',
      title: 'Overview is the map',
      body: 'Every section on the left is listed here with a live count of what it holds. '
        + 'When a count reads 0 you know exactly what is still missing.',
    },
    {
      id: 'labels',
      target: 'config.labelsAndTags',
      navigate: ['tab-configuration', 'tab-tag-management'],
      title: 'Labels are what you detect',
      body: 'A label is a thing a model can find in a frame: a particular cat, a food bowl, '
        + 'a person. Tags are notes you attach to an image or a box — "needs review", '
        + '"night" — and never train anything.',
    },
    {
      id: 'labels.create',
      target: 'config.labels.newObject',
      title: 'Create your first label',
      body: 'Type a name here and press Enter. Start with the subject you most want to watch for.',
      advanceOn: { event: EventTypes.LABELS_UPDATED },
    },
    {
      id: 'sources',
      target: 'config.sources',
      navigate: ['tab-configuration', 'tab-source-management'],
      title: 'Sources are where frames come from',
      body: 'A source is a camera stream, a folder of images, or a URL. Watchers run against '
        + 'stream sources; annotation and training draw on any of them.',
    },
    {
      id: 'sources.create',
      target: 'list.src.new',
      title: 'Add a source',
      body: 'Press + New Source and point it at a camera (an RTSP URL) or a folder of images. '
        + 'You can come back to this later — the rest of the tour does not need one.',
    },
    {
      id: 'modelShapes',
      target: 'config.modelShapes',
      navigate: ['tab-configuration', 'tab-model-management'],
      title: 'A model shape says what a model detects',
      body: 'A shape maps a model\'s output classes to your labels. One shape per model, and '
        + 'shapes can nest: a coarse "cat" detector can hand off to a finer one that tells '
        + 'your cats apart.',
    },
    {
      id: 'annotate',
      target: 'nav.annotate',
      title: 'Annotate to build training data',
      body: 'Draw and correct boxes on images, then verify them in Review. Frames a watcher '
        + 'saves for annotation queue up here too.',
    },
    {
      id: 'runs',
      target: 'nav.runs',
      title: 'Jobs run everything long',
      body: 'Training, inference passes and collectors run as jobs. Their progress and results '
        + 'live here, and the badge on the tab shows when something is running.',
    },
    {
      id: 'watch',
      target: 'nav.nvr',
      title: 'Watch is the day job',
      body: 'Once a source, a label and a model are in place, a watch rule turns them into '
        + 'events: live feeds, clips and a history of what was seen. This tour can be re-run '
        + 'any time from Settings › Interface.',
    },
  ],
};

/** @type {ReadonlyArray<import('./engine.js').Tour>} in the order they are offered */
export const TOURS = Object.freeze([GETTING_STARTED]);

/** @param {string} id */
export function tourById(id) {
  return TOURS.find(t => t.id === id) ?? null;
}
