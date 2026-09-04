/**
 * Model tree summary panel — read-only visualisation of a model config's
 * sub-model hierarchy (the NOTES.md train-time contract):
 *
 *   ◉ Top model — trained on all assigned labels, with labels that live under
 *     a sub-model boundary rolled up into that boundary label.
 *   ● One sub-model per boundary label — distinguishes that label's assigned
 *     descendants from each other.
 *
 * Pure render: derives everything from a `nodes` array (saved config or a
 * live computeNodes() result) + labelsMap. Callers re-render on change.
 */

import { parseNodes, specDisplay } from '/app-static/js/ui/modelSpec.js';

/**
 * Nearest proper ancestor of `uuid` that is a sub-model boundary, or null.
 */
function _boundaryAncestor(uuid, labelsMap, submodels) {
  let p = labelsMap.get(uuid)?.parentUuid ?? null;
  while (p !== null) {
    if (submodels.has(p)) return p;
    p = labelsMap.get(p)?.parentUuid ?? null;
  }
  return null;
}

function _labelChip(label, uuid) {
  const chip = document.createElement('span');
  chip.className = 'mtp-chip';
  const swatch = document.createElement('span');
  swatch.className = 'mtp-chip__swatch';
  swatch.style.background = label?.color ?? '#888';
  chip.appendChild(swatch);
  chip.appendChild(document.createTextNode(label?.name ?? uuid));
  return chip;
}

function _chipRow(prefixText, uuids, labelsMap, compact, compactNoun) {
  const row = document.createElement('div');
  row.className = 'mtp-detail';
  if (compact) {
    row.textContent = `${prefixText} ${uuids.length} ${compactNoun}${uuids.length === 1 ? '' : 's'}`;
    return row;
  }
  row.appendChild(document.createTextNode(`${prefixText} `));
  const sorted = [...uuids].sort((a, b) =>
    (labelsMap.get(a)?.name ?? a).localeCompare(labelsMap.get(b)?.name ?? b));
  for (const uuid of sorted) row.appendChild(_labelChip(labelsMap.get(uuid), uuid));
  return row;
}

/**
 * "trained" / "not trained yet" marker for one node of the tree.
 *
 * An untrained part is the failure this panel exists to make visible: the hierarchy
 * still runs, the top model still detects, and the only symptom is that detections stop
 * at the parent label — indistinguishable from a model that simply wasn't sure. Stated
 * per node rather than as one summary line so it names WHICH part is missing.
 */
function _trainedChip(isTrained) {
  const chip = document.createElement('div');
  // Its own modifier rather than the shared --warn: "not trained" and "nothing to
  // distinguish" are different problems with different fixes, and one selector matching
  // both makes each impossible to assert on independently.
  chip.className = 'mtp-detail' + (isTrained ? ' mtp-detail--ok' : ' mtp-detail--untrained');
  chip.dataset.testid = isTrained ? 'model-tree-trained' : 'model-tree-untrained';
  chip.textContent = isTrained ? 'trained ✓' : '⚠ not trained yet — this part will not run';
  return chip;
}

/**
 * Render the model tree summary.
 *
 * @param {object} opts
 * @param {Array<object>} opts.nodes  Model config nodes (saved or computeNodes() output).
 * @param {Map<string, object>} opts.labelsMap
 * @param {boolean} [opts.compact]  Counts instead of chip lists (Configuration → Training preview).
 * @param {?string} [opts.highlightNodeUuid]  Node uuid to mark as the training target.
 * @param {?{top: boolean, subLabelUuids: Set<string>}} [opts.trained]
 *   Which parts have a trained artifact. Omit (the default) to say nothing about
 *   training state — every caller that only describes the *shape* of the tree should,
 *   since "no marker" and "not trained" must not look the same.
 * @returns {HTMLElement}
 */
export function renderModelTreePanel({
  nodes,
  labelsMap,
  compact = false,
  highlightNodeUuid = null,
  trained = null,
}) {
  const panel = document.createElement('div');
  panel.className = 'model-tree-panel' + (compact ? ' model-tree-panel--compact' : '');

  const { rootSpec, selected, submodels } = parseNodes(nodes, labelsMap);
  // node uuids only exist on saved configs; used for training-target highlight
  const nodeUuidByLabel = new Map((nodes ?? []).map(n => [n.labelUuid, n.uuid]));

  if (selected.size === 0) {
    const empty = document.createElement('div');
    empty.className = 'mtp-empty';
    empty.textContent = 'No labels assigned yet — the tree will appear as labels are selected.';
    panel.appendChild(empty);
    return panel;
  }

  // Top-model classes: each assigned label rolls up to its nearest sub-model
  // boundary ancestor (if any); boundary labels are classes themselves.
  const topClasses = new Set();
  for (const uuid of selected) {
    topClasses.add(_boundaryAncestor(uuid, labelsMap, submodels) ?? uuid);
  }

  // ── Root node ─────────────────────────────────────────────────────────────
  const rootRow = document.createElement('div');
  rootRow.className = 'mtp-node mtp-node--root';
  rootRow.dataset.testid = 'model-tree-root-node';
  if (highlightNodeUuid && nodeUuidByLabel.get(null) === highlightNodeUuid) {
    rootRow.classList.add('mtp-node--target');
  }
  const rootTitle = document.createElement('div');
  rootTitle.className = 'mtp-node__title';
  rootTitle.textContent = `◉ Top model — ${specDisplay(rootSpec)}`;
  if (rootRow.classList.contains('mtp-node--target')) {
    rootTitle.textContent += '  (training target)';
  }
  rootRow.appendChild(rootTitle);
  if (trained) rootRow.appendChild(_trainedChip(trained.top));
  rootRow.appendChild(_chipRow('detects:', [...topClasses], labelsMap, compact, 'class'));
  panel.appendChild(rootRow);

  // ── Sub-model nodes ───────────────────────────────────────────────────────
  const subRows = [...submodels.entries()].sort(([a], [b]) =>
    (labelsMap.get(a)?.name ?? a).localeCompare(labelsMap.get(b)?.name ?? b));

  for (const [uuid, spec] of subRows) {
    const row = document.createElement('div');
    row.className = 'mtp-node mtp-node--sub';
    row.dataset.testid = 'model-tree-sub-node';
    if (highlightNodeUuid && nodeUuidByLabel.get(uuid) === highlightNodeUuid) {
      row.classList.add('mtp-node--target');
    }

    const title = document.createElement('div');
    title.className = 'mtp-node__title';
    title.appendChild(_labelChip(labelsMap.get(uuid), uuid));
    title.appendChild(document.createTextNode(` sub-model — ${specDisplay(spec)}`));
    if (row.classList.contains('mtp-node--target')) {
      title.appendChild(document.createTextNode('  (training target)'));
    }
    row.appendChild(title);
    if (trained) row.appendChild(_trainedChip(trained.subLabelUuids.has(uuid)));

    // Labels this sub-model distinguishes: assigned labels that roll up to it.
    const children = [...selected].filter(
      l => l !== uuid && _boundaryAncestor(l, labelsMap, submodels) === uuid);
    if (children.length === 0) {
      const warn = document.createElement('div');
      warn.className = 'mtp-detail mtp-detail--warn';
      warn.textContent = '⚠ no assigned child labels — this sub-model has nothing to distinguish';
      row.appendChild(warn);
    } else {
      row.appendChild(_chipRow('distinguishes:', children, labelsMap, compact, 'child label'));
    }
    panel.appendChild(row);
  }

  // ── Artifact summary ──────────────────────────────────────────────────────
  // One training config trains exactly one node (TrainModelTask resolves a single
  // model_config_node_uuid), so this counts *runs*, not artifacts of one run. The
  // wording here used to describe a whole-tree run, left over from the removed
  // "Train all → bundle" path, and read as a promise that training the root would
  // produce the sub-models too — it doesn't, and the tree silently stops one level
  // short at inference with no error to point at.
  const artifactCount = 1 + submodels.size;
  const footer = document.createElement('div');
  footer.className = 'mtp-footer';
  footer.textContent = submodels.size === 0
    ? 'Training this tree is 1 training run, producing 1 model artifact.'
    : `Training this tree takes ${artifactCount} training runs — one per part ` +
      `(1 top model + ${submodels.size} sub-model${submodels.size === 1 ? '' : 's'}). ` +
      '“Train hierarchy” (top of this editor) runs them all in order; ' +
      'Configuration → Training retrains one part on its own.';
  panel.appendChild(footer);

  return panel;
}
