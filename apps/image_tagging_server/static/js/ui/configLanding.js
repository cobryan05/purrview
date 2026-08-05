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
import { Logger, showPaneError } from '/app-static/js/ui/utils/index.js';

/** Count of a fetcher result that may be an Array or a Map (fetchLabels returns a Map). */
function sizeOf(result) {
  if (!result) return 0;
  if (result instanceof Map || result instanceof Set) return result.size;
  if (Array.isArray(result)) return result.length;
  return 0;
}

/**
 * Resolve every count in parallel. A single failing fetcher must not blank the
 * whole landing, so each resolves independently and failures render as "—".
 * @returns {Promise<Record<string, number|null>>}
 */
async function fetchCounts() {
  const settle = async (fn) => {
    try {
      return sizeOf(await fn());
    } catch (err) {
      Logger.warn('configLanding: count fetch failed', err);
      return null;
    }
  };

  const [labels, tags, models, datasetsAndViews, blueprints, trainingRuns, filterChains] = await Promise.all([
    settle(fetchLabels),
    settle(fetchTags),
    settle(fetchModelsList),
    (async () => {
      try {
        // ⚠️ Returns a Map keyed by uuid, not an array (same shape trap as
        // fetchLabels — see CLAUDE.md "Key Gotchas").
        const rows = [...(await fetchDatasetsAndViews()).values()];
        return {
          datasets: rows.filter(r => r.promoted).length,
          views: rows.filter(r => !r.promoted).length,
        };
      } catch (err) {
        Logger.warn('configLanding: dataset/view count fetch failed', err);
        return { datasets: null, views: null };
      }
    })(),
    settle(fetchModelConfigs),
    settle(fetchTrainingConfigs),
    settle(fetchFilterChains),
  ]);

  return {
    labels,
    tags,
    models,
    datasets: datasetsAndViews.datasets,
    views: datasetsAndViews.views,
    blueprints,
    trainingRuns,
    filterChains,
  };
}

/**
 * @param {Record<string, number|null>} counts
 * @returns {Array<{title: string, entries: Array<{name: string, count: number|null, unit: string, tab: string, hint: string}>}>}
 */
function buildGroups(counts) {
  return [
    {
      title: 'Vocabulary',
      entries: [
        {
          name: 'Labels', count: counts.labels, unit: 'labels',
          tab: 'tab-tag-management', hint: 'What you can annotate — hierarchical, object- or image-scoped',
        },
        {
          name: 'Tags', count: counts.tags, unit: 'tags',
          tab: 'tab-tag-management', hint: 'Free-form and system markers applied to boxes',
        },
        {
          name: 'Inference Class Map', count: counts.models, unit: 'models',
          tab: 'tab-tag-management', hint: "Maps each model's output class IDs onto labels",
        },
      ],
    },
    {
      title: 'Data',
      entries: [
        {
          name: 'Datasets', count: counts.datasets, unit: 'datasets',
          tab: 'tab-dataset-management', hint: 'Promoted image collections that training consumes',
        },
        {
          name: 'Views', count: counts.views, unit: 'views',
          tab: 'tab-dataset-management', hint: 'Saved searches — promote one to make it a Dataset',
        },
      ],
    },
    {
      title: 'Models',
      entries: [
        {
          name: 'Model Blueprints', count: counts.blueprints, unit: 'blueprints',
          tab: 'tab-model-management', hint: 'Which model handles which label, hierarchically',
        },
        {
          name: 'Training Runs', count: counts.trainingRuns, unit: 'runs',
          tab: 'tab-training-management', hint: 'Reusable training definitions and their datasets',
        },
      ],
    },
    {
      title: 'Automation',
      entries: [
        {
          name: 'Filter Chains', count: counts.filterChains, unit: 'chains',
          tab: 'tab-filterchain-management', hint: 'Ordered filter steps runnable over search results',
        },
      ],
    },
  ];
}

/** Navigate to a Configuration sub-tab by clicking its workspace tab button. */
function goToTab(tabName) {
  document.querySelector(`#tab-configuration .workspace-tab-button[data-tab="${tabName}"]`)?.click();
}

function buildEntry(entry) {
  const el = document.createElement('button');
  el.type = 'button';
  el.className = 'config-landing__entry';
  el.dataset.testid = `config-landing-entry-${entry.tab}-${entry.name.toLowerCase().replace(/\s+/g, '-')}`;
  el.title = entry.hint;

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

  const hint = document.createElement('span');
  hint.className = 'config-landing__entry-hint';
  hint.textContent = entry.hint;
  text.appendChild(hint);

  el.appendChild(text);
  el.addEventListener('click', () => goToTab(entry.tab));
  return el;
}

function buildGroupCard(group) {
  const card = document.createElement('section');
  card.className = 'config-landing__group';
  card.dataset.testid = `config-landing-group-${group.title.toLowerCase()}`;

  const heading = document.createElement('h4');
  heading.className = 'config-landing__group-title';
  heading.textContent = group.title;
  card.appendChild(heading);

  group.entries.forEach(entry => card.appendChild(buildEntry(entry)));
  return card;
}

/**
 * Render the Configuration overview landing.
 * @param {{target?: string}} [options]
 */
export async function renderConfigLanding({ target = 'config-overview-box' } = {}) {
  const container = document.getElementById(target);
  if (!container) return;

  container.innerHTML = '';
  container.className = 'config-landing';
  container.dataset.testid = 'config-landing';

  let counts;
  try {
    counts = await fetchCounts();
  } catch (err) {
    Logger.error('configLanding: failed to build overview', err);
    showPaneError(container, 'Could not load the configuration overview.');
    return;
  }

  const grid = document.createElement('div');
  grid.className = 'config-landing__grid';
  buildGroups(counts).forEach(group => grid.appendChild(buildGroupCard(group)));
  container.appendChild(grid);
}
