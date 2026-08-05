/**
 * Configuration → Overview landing.
 *
 * A wayfinding surface grouping the Configuration sub-tabs by purpose, each
 * entry annotated with a **live** count fetched from the API.
 *
 * ⚠️ Counts are always queried, never hardcoded — the design mockup's numbers
 * were fabricated filler (UI_REDESIGN §33.1). Every count here comes from the
 * same cached fetchers the destination panes use, so the landing costs no
 * extra round-trips once a pane has been visited.
 *
 * Several entries deliberately share one destination: "Inference Class Map" is
 * a column inside the Labels & Tags sub-tab, and Views live in the Dataset
 * Management pane alongside Datasets (decision #27, PR 5.3). The counts do the
 * wayfinding; the navigation target is whatever pane actually holds the thing.
 */
import { fetchDatasetsAndViews } from '/app-static/js/shared/api/datasets.js';
import { fetchFilterChains } from '/app-static/js/shared/api/filterChains.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchModelConfigs } from '/app-static/js/shared/api/modelConfigs.js';
import { fetchModelsList } from '/app-static/js/shared/api/models.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import { fetchTrainingConfigs } from '/app-static/js/shared/api/trainingConfigs.js';
import { Logger } from '/app-static/js/ui/utils/index.js';

/** Count of a fetcher result that may be an Array or a Map (fetchLabels returns a Map). */
function sizeOf(result) {
  if (!result) return 0;
  if (result instanceof Map) return result.size;
  if (Array.isArray(result)) return result.length;
  return 0;
}

/**
 * Resolve every count in parallel. A single failing fetcher must not blank the
 * whole landing, so each resolves independently and failures render as "—".
 * @returns {Promise<Record<string, number|null>>}
 */
async function fetchCounts() {
  const settle = async (fn, map = sizeOf) => {
    try {
      return map(await fn());
    } catch (err) {
      Logger.warn('configLanding: count fetch failed', err);
      return null;
    }
  };

  const [labels, tags, models, rows, blueprints, trainingRuns, filterChains] = await Promise.all([
    settle(fetchLabels),
    settle(fetchTags),
    settle(fetchModelsList),
    // ⚠️ Returns a Map keyed by uuid, not an array (same shape trap as
    // fetchLabels — see CLAUDE.md "Key Gotchas").
    settle(fetchDatasetsAndViews, m => [...m.values()]),
    settle(fetchModelConfigs),
    settle(fetchTrainingConfigs),
    settle(fetchFilterChains),
  ]);

  return {
    labels,
    tags,
    models,
    datasets: rows ? rows.filter(r => r.promoted).length : null,
    views: rows ? rows.filter(r => !r.promoted).length : null,
    blueprints,
    trainingRuns,
    filterChains,
  };
}

/**
 * @param {Record<string, number|null>} counts
 * @returns {Array<{title: string, entries: Array<{name: string, count: number|null, tab: string, hint: string}>}>}
 */
function buildGroups(counts) {
  return [
    {
      title: 'Vocabulary',
      entries: [
        {
          name: 'Labels', count: counts.labels, tab: 'tab-tag-management',
          hint: 'What you can annotate — hierarchical, object- or image-scoped',
        },
        {
          name: 'Tags', count: counts.tags, tab: 'tab-tag-management',
          hint: 'Free-form and system markers applied to boxes',
        },
        {
          name: 'Inference Class Map', count: counts.models, tab: 'tab-tag-management',
          hint: "Maps each model's output class IDs onto labels",
        },
      ],
    },
    {
      title: 'Data',
      entries: [
        {
          name: 'Datasets', count: counts.datasets, tab: 'tab-dataset-management',
          hint: 'Promoted image collections that training consumes',
        },
        {
          name: 'Views', count: counts.views, tab: 'tab-dataset-management',
          hint: 'Saved searches — promote one to make it a Dataset',
        },
      ],
    },
    {
      title: 'Models',
      entries: [
        {
          name: 'Model Blueprints', count: counts.blueprints, tab: 'tab-model-management',
          hint: 'Which model handles which label, hierarchically',
        },
        {
          name: 'Training Runs', count: counts.trainingRuns, tab: 'tab-training-management',
          hint: 'Reusable training definitions and their datasets',
        },
      ],
    },
    {
      title: 'Automation',
      entries: [
        {
          name: 'Filter Chains', count: counts.filterChains, tab: 'tab-filterchain-management',
          hint: 'Ordered filter steps runnable over search results',
        },
      ],
    },
  ];
}

function buildEntry(entry, activateTab) {
  const el = document.createElement('button');
  el.type = 'button';
  el.className = 'config-landing__entry';
  el.dataset.testid = `config-landing-entry-${entry.tab}-${entry.name.toLowerCase().replace(/\s+/g, '-')}`;

  const count = document.createElement('span');
  count.className = 'config-landing__count';
  count.textContent = entry.count === null ? '—' : String(entry.count);
  if (entry.count === null) count.classList.add('config-landing__count--error');
  el.appendChild(count);

  const text = document.createElement('span');
  text.className = 'config-landing__entry-text';

  const name = document.createElement('span');
  name.className = 'config-landing__entry-name';
  name.textContent = entry.name;
  text.appendChild(name);

  // The hint is permanently visible, so no title attribute — it would both
  // duplicate the tooltip and make screen readers announce the string twice.
  const hint = document.createElement('span');
  hint.className = 'config-landing__entry-hint';
  hint.textContent = entry.hint;
  text.appendChild(hint);

  el.appendChild(text);
  el.addEventListener('click', () => activateTab(entry.tab));
  return el;
}

function buildGroupCard(group, activateTab) {
  const card = document.createElement('section');
  card.className = 'config-landing__group';
  card.dataset.testid = `config-landing-group-${group.title.toLowerCase()}`;

  const heading = document.createElement('h4');
  heading.className = 'config-landing__group-title';
  heading.textContent = group.title;
  card.appendChild(heading);

  group.entries.forEach(entry => card.appendChild(buildEntry(entry, activateTab)));
  return card;
}

// Re-entrancy guard (CLAUDE.md "Key Gotchas") — the container is cleared before
// an await, so a fast tab bounce could otherwise interleave two renders.
let _rendering = false;

/**
 * Render the Configuration overview landing.
 * @param {Object} [options]
 * @param {string} [options.target] - id of the container element
 * @param {(tabName: string) => void} options.activateTab - the owning tab
 *   controller's activator, from setupWorkspaceTabs. Passed in rather than
 *   re-deriving the tab DOM here, so navigation stays the controller's job.
 */
export async function renderConfigLanding({ target = 'config-overview-box', activateTab } = {}) {
  const container = document.getElementById(target);
  if (!container || _rendering) return;
  _rendering = true;

  try {
    const counts = await fetchCounts();

    container.innerHTML = '';
    container.className = 'config-landing';
    container.dataset.testid = 'config-landing';

    const grid = document.createElement('div');
    grid.className = 'config-landing__grid';
    buildGroups(counts).forEach(group => grid.appendChild(buildGroupCard(group, activateTab)));
    container.appendChild(grid);
  } finally {
    _rendering = false;
  }
}
