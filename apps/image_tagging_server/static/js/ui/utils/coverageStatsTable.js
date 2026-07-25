/**
 * Shared "dataset coverage by label" table — renders a training config's
 * label-stats breakdown with clickable cells that open a review modal
 * scoped to that label/status. Used by both the full Training Runs page
 * (ui/trainingConfigs.js) and the bottom-bar coverage popup
 * (ui/activeModelBar.js) so the two stay in sync rather than drifting.
 */

import { VERIFIED_TAG_UUID, UNVERIFIED_TAG_UUID } from '/app-static/js/shared/api/review.js';
import {
  fetchTrainingConfigImagesByLabelStatus,
  fetchTrainingConfigNeedsReview,
} from '/app-static/js/shared/api/trainingConfigs.js';

/**
 * @param {{uuid: string, name: string}} config
 */
export async function openLabelModal(config, labelUuid, labelStatus, modalTitle, rollupChildUuids = [], missingLabelUuid = null) {
  const { showReviewModal } = await import('/app-static/js/app/review/main.js');
  await showReviewModal({
    title: modalTitle,
    fetchImages: async ({ afterUuid = null, afterSortValue = null, limit }) => {
      return fetchTrainingConfigImagesByLabelStatus(config.uuid, {
        labelUuid,
        rollupLabelUuids: rollupChildUuids.length > 0 ? rollupChildUuids : undefined,
        status: labelStatus,
        missingLabelUuid,
        afterUuid,
        afterSortValue,
        limit,
      });
    },
  });
}

export async function openVerifiedBboxModal(config, effectiveLabelUuids, modalTitle) {
  const { showReviewModal } = await import('/app-static/js/app/review/main.js');
  const uuidSet = effectiveLabelUuids instanceof Set ? effectiveLabelUuids : new Set(effectiveLabelUuids);
  await showReviewModal({
    title: modalTitle,
    initialViewFilter: 'bboxes',
    bboxFilterFn: (b) => uuidSet.has(b.labelUuid) && (b.tags || []).some(t => t.uuid === VERIFIED_TAG_UUID),
    fetchImages: async ({ afterUuid = null, afterSortValue = null, limit }) => {
      return fetchTrainingConfigImagesByLabelStatus(config.uuid, {
        labelUuid: [...uuidSet][0],
        rollupLabelUuids: uuidSet.size > 1 ? [...uuidSet].slice(1) : undefined,
        status: 'has_verified_bbox',
        afterUuid,
        afterSortValue,
        limit,
      });
    },
  });
}

export async function openBboxNeedsReviewModal(config, labelUuid, modalTitle, rollupChildUuids = []) {
  const { showReviewModal } = await import('/app-static/js/app/review/main.js');
  const effectiveUuids = new Set([labelUuid, ...rollupChildUuids]);
  await showReviewModal({
    title: modalTitle,
    initialViewFilter: 'bboxes',
    bboxFilterFn: (b) => effectiveUuids.has(b.labelUuid) && (b.tags || []).some(t => t.uuid === UNVERIFIED_TAG_UUID),
    fetchImages: async ({ afterUuid = null, afterSortValue = null, limit }) => {
      return fetchTrainingConfigNeedsReview({
        uuid: config.uuid,
        labelUuids: [labelUuid, ...rollupChildUuids],
        afterUuid,
        afterSortValue,
        limit,
      });
    },
  });
}

const _clickableCellCss = 'text-align: right; padding: 2px 6px; cursor: pointer; text-decoration: underline dotted; text-underline-offset: 2px;';
const _clickableLabelCss = 'padding: 2px 6px; cursor: pointer; text-decoration: underline dotted; text-underline-offset: 2px;';

/**
 * Appends the "N images across dataset(s)" line plus Image/Object label
 * tables (with clickable cells) into `content`. Caller owns fetching the
 * stats and clearing/loading states around this call.
 *
 * @param {HTMLElement} content
 * @param {{
 *   config: {uuid: string, name: string, datasetUuids?: string[]},
 *   labelStats: object[],
 *   totalImages: number,
 *   labelsMap: Map<string, object>,
 *   onNavigate?: () => void,
 * }} opts
 * @param {() => void} [opts.onNavigate] - called right before any cell click opens the
 *   review modal, so a caller rendering this table inside its own floating panel
 *   (e.g. activeModelBar.js's coverage popup) can close itself first — otherwise a
 *   high-z-index panel would sit on top of the review modal it just opened.
 */
export function renderCoverageStats(content, { config, labelStats, totalImages, labelsMap, onNavigate }) {
  const totalNote = document.createElement('div');
  totalNote.style.cssText = 'color: var(--pico-muted-color, #888); margin-bottom: 0.5em; font-size: 0.9em;';
  totalNote.textContent = `${totalImages.toLocaleString()} image${totalImages === 1 ? '' : 's'} across selected dataset${(config.datasetUuids?.length ?? 0) === 1 ? '' : 's'}`;
  content.appendChild(totalNote);

  if (labelStats.length === 0) {
    const none = document.createElement('div');
    none.style.cssText = 'color: var(--pico-muted-color, #888); font-style: italic;';
    none.textContent = 'No relevant labels found for this model blueprint / dataset combination.';
    content.appendChild(none);
    return;
  }

  const imageLabels = labelStats.filter(s => s.labelScope === 'image');
  const objectLabels = labelStats.filter(s => s.labelScope === 'object');

  function _labelName(uuid) {
    return labelsMap?.get(uuid)?.name ?? uuid;
  }

  function _labelDepth(uuid) {
    let depth = 0;
    let current = labelsMap?.get(uuid);
    while (current?.parentUuid) {
      depth++;
      current = labelsMap?.get(current.parentUuid);
      if (depth > 20) break;
    }
    return depth;
  }

  // Returns the child label UUIDs that roll up into parentLabelUuid.
  function _rollupChildUuids(parentLabelUuid) {
    return labelStats
      .filter(s => s.rollUpParentUuid === parentLabelUuid)
      .map(s => s.labelUuid);
  }

  function _rollupTooltip(parentUuid, allStats) {
    const children = allStats
      .filter(s => s.rollUpParentUuid === parentUuid)
      .map(s => _labelName(s.labelUuid));
    if (children.length === 0) return 'Roll-up active: counts aggregate all child labels';
    return `Roll-up active — counts aggregate:\n${children.map(n => `  • ${n}`).join('\n')}`;
  }

  function _makeRollupBadge(tooltip) {
    const badge = document.createElement('span');
    badge.style.cssText = 'display:inline-block; font-size:0.75em; background:#2a4a6a; color:#8cf; border-radius:3px; padding:0 4px; margin-left:5px; cursor:default; vertical-align:middle; white-space:nowrap;';
    badge.title = tooltip;
    badge.textContent = '⟳ roll-up';
    return badge;
  }

  function _makeChildBadge(parentName) {
    const badge = document.createElement('span');
    badge.style.cssText = 'display:inline-block; font-size:0.75em; color:var(--pico-muted-color,#888); margin-left:5px; cursor:default; vertical-align:middle; white-space:nowrap;';
    badge.title = `Rolled up into: ${parentName}`;
    badge.textContent = '↑';
    return badge;
  }

  function _makeTable(stats, columns) {
    const table = document.createElement('table');
    table.style.cssText = 'width: 100%; border-collapse: collapse; font-size: 0.9em;';

    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    const labelTh = document.createElement('th');
    labelTh.textContent = 'Label';
    labelTh.style.cssText = 'text-align: left; padding: 2px 6px; border-bottom: 1px solid var(--pico-border-color, #444);';
    headerRow.appendChild(labelTh);
    for (const col of columns) {
      const th = document.createElement('th');
      th.textContent = col.header;
      th.title = col.title ?? '';
      th.style.cssText = `text-align: right; padding: 2px 6px; border-bottom: 1px solid var(--pico-border-color, #444); white-space: nowrap;${col.title ? ' cursor: help;' : ''}`;
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const totalCols = columns.length + 1;

    function _makeBreakdownRow(stat, col) {
      const breakdown = stat[col.breakdownKey] ?? [];
      const detailTr = document.createElement('tr');
      const detailTd = document.createElement('td');
      detailTd.colSpan = totalCols;
      detailTd.style.cssText = 'padding: 4px 6px 8px 24px; background: rgba(255,255,255,0.03);';

      const heading = document.createElement('div');
      heading.style.cssText = 'font-size: 0.85em; color: var(--pico-muted-color, #888); margin-bottom: 2px;';
      heading.textContent = `Labels driving these gaps for ${_labelName(stat.labelUuid)}:`;
      detailTd.appendChild(heading);

      const list = document.createElement('div');
      list.style.cssText = 'display: flex; flex-direction: column; gap: 1px;';
      for (const entry of breakdown) {
        const row = document.createElement('div');
        row.style.cssText = 'display: flex; justify-content: space-between; gap: 1em; padding: 1px 4px; cursor: pointer; text-decoration: underline dotted; text-underline-offset: 2px;';
        row.title = `Browse images with a verified box for ${_labelName(stat.labelUuid)} where ${entry.labelName} is still unresolved`;

        const nameSpan = document.createElement('span');
        nameSpan.textContent = `↳ ${entry.labelName}`;
        row.appendChild(nameSpan);

        const countSpan = document.createElement('span');
        countSpan.textContent = entry.count.toLocaleString();
        row.appendChild(countSpan);

        row.addEventListener('click', () => {
          onNavigate?.();
          openLabelModal(
            config, stat.labelUuid, 'co_label_gaps',
            `${_labelName(stat.labelUuid)} — gaps missing ${entry.labelName} (${config.name})`,
            stat.hasRollup ? _rollupChildUuids(stat.labelUuid) : [],
            entry.labelUuid
          );
        });
        list.appendChild(row);
      }
      detailTd.appendChild(list);

      detailTr.appendChild(detailTd);
      return detailTr;
    }

    const tbody = document.createElement('tbody');
    for (const stat of stats) {
      const tr = document.createElement('tr');

      const nameTd = document.createElement('td');
      const dot = labelsMap?.get(stat.labelUuid)?.color;
      if (dot) {
        const dotEl = document.createElement('span');
        dotEl.style.cssText = `display:inline-block;width:7px;height:7px;border-radius:50%;background:${dot};margin-right:4px;vertical-align:middle;`;
        nameTd.appendChild(dotEl);
      }
      nameTd.appendChild(document.createTextNode(_labelName(stat.labelUuid)));

      if (stat.hasRollup) {
        nameTd.appendChild(_makeRollupBadge(_rollupTooltip(stat.labelUuid, stats)));
      } else if (stat.rollUpParentUuid) {
        nameTd.appendChild(_makeChildBadge(_labelName(stat.rollUpParentUuid)));
      }

      const depth = _labelDepth(stat.labelUuid);
      const indentPx = 6 + depth * 14;

      if (stat.labelScope === 'image') {
        nameTd.style.cssText = _clickableLabelCss;
        nameTd.style.paddingLeft = `${indentPx}px`;
        nameTd.title = 'Browse all images in this training run for this label';
        nameTd.addEventListener('click', () => {
          onNavigate?.();
          openLabelModal(
            config, stat.labelUuid, 'all',
            `${_labelName(stat.labelUuid)} — all images (${config.name})`,
            stat.hasRollup ? _rollupChildUuids(stat.labelUuid) : []
          );
        });
      } else {
        nameTd.style.cssText = _clickableLabelCss;
        nameTd.style.paddingLeft = `${indentPx}px`;
        nameTd.title = 'Browse images with unverified boxes for this label';
        nameTd.addEventListener('click', () => {
          onNavigate?.();
          openBboxNeedsReviewModal(
            config, stat.labelUuid,
            `${_labelName(stat.labelUuid)} — needs review (${config.name})`,
            stat.hasRollup ? _rollupChildUuids(stat.labelUuid) : []
          );
        });
      }
      tr.appendChild(nameTd);

      let detailTr = null;
      for (const col of columns) {
        const td = document.createElement('td');
        const val = stat[col.key] ?? 0;

        const isClickable = (col.browseStatus || col.browseFn) && val > 0;
        const breakdown = col.breakdownKey ? (stat[col.breakdownKey] ?? []) : [];
        if (isClickable) {
          td.style.cssText = _clickableCellCss;
          if (col.warnIf?.(stat)) td.style.color = '#f90';
          if (col.goodIf?.(stat)) td.style.color = '#4c4';
          td.title = `Browse ${val.toLocaleString()} image${val === 1 ? '' : 's'} — click to open`;
          td.addEventListener('click', () => {
            onNavigate?.();
            if (col.browseFn) {
              col.browseFn(stat, col);
            } else {
              openLabelModal(
                config, stat.labelUuid, col.browseStatus,
                `${_labelName(stat.labelUuid)} — ${col.header.toLowerCase()} (${config.name})`,
                stat.hasRollup ? _rollupChildUuids(stat.labelUuid) : []
              );
            }
          });
        } else {
          td.style.cssText = 'text-align: right; padding: 2px 6px;';
          if (col.warnIf?.(stat)) td.style.color = '#f90';
          if (col.goodIf?.(stat)) td.style.color = '#4c4';
        }

        if (breakdown.length > 0) {
          const countSpan = document.createElement('span');
          countSpan.textContent = val.toLocaleString();
          td.appendChild(countSpan);

          const toggle = document.createElement('span');
          toggle.textContent = ' ▾';
          toggle.title = 'Show which labels are driving these gaps';
          toggle.style.cssText = 'cursor: pointer; text-decoration: none; opacity: 0.75;';
          toggle.addEventListener('click', (ev) => {
            ev.stopPropagation();
            if (detailTr) {
              detailTr.remove();
              detailTr = null;
              toggle.textContent = ' ▾';
            } else {
              detailTr = _makeBreakdownRow(stat, col);
              tr.after(detailTr);
              toggle.textContent = ' ▴';
            }
          });
          td.appendChild(toggle);
        } else {
          td.textContent = val.toLocaleString();
        }

        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    return table;
  }

  if (imageLabels.length > 0) {
    const sectionHead = document.createElement('div');
    sectionHead.style.cssText = 'font-weight: 600; margin: 0.5em 0 0.25em; font-size: 0.9em;';
    sectionHead.textContent = 'Image labels';
    content.appendChild(sectionHead);
    content.appendChild(_makeTable(imageLabels, [
      { header: 'Positive', key: 'presentImageCount', title: 'Images explicitly marked Present for this label', goodIf: s => s.presentImageCount > 0, browseStatus: 'present' },
      { header: 'Negative', key: 'absentImageCount', title: 'Images explicitly marked Absent for this label', goodIf: s => s.absentImageCount > 0, browseStatus: 'absent' },
      { header: 'Unverified', key: 'unverifiedImageCount', title: 'Images with no status set for this label — click to review', warnIf: s => s.unverifiedImageCount > 0, browseStatus: 'uncertain' },
    ]));
  }

  if (objectLabels.length > 0) {
    const sectionHead = document.createElement('div');
    sectionHead.style.cssText = 'font-weight: 600; margin: 0.6em 0 0.25em; font-size: 0.9em;';
    sectionHead.textContent = 'Object labels';
    content.appendChild(sectionHead);
    content.appendChild(_makeTable(objectLabels, [
      { header: 'Verified', key: 'verifiedBboxCount', title: 'Verified boxes — bounding boxes marked Verified — click to browse', goodIf: s => s.unverifiedBboxCount === 0 && s.verifiedBboxCount > 0,
        browseFn: (stat) => {
          const effectiveUuids = new Set([stat.labelUuid, ...(stat.hasRollup ? _rollupChildUuids(stat.labelUuid) : [])]);
          return openVerifiedBboxModal(config, effectiveUuids, `${_labelName(stat.labelUuid)} — verified boxes (${config.name})`);
        }},
      { header: 'Unverified', key: 'unverifiedBboxCount', title: 'Unverified boxes — bounding boxes still marked Unverified — click to review images', warnIf: s => s.unverifiedBboxCount > 0,
        browseFn: (stat) => openBboxNeedsReviewModal(config, stat.labelUuid, `${_labelName(stat.labelUuid)} — unverified boxes (${config.name})`, stat.hasRollup ? _rollupChildUuids(stat.labelUuid) : []) },
      { header: 'All clear', key: 'coLabelsCoveredCount',
        title: 'Imgs all clear — images with a verified box for this label where every other label in the model blueprint is accounted for (verified present or marked absent) — click to browse',
        goodIf: s => s.coLabelGapsCount === 0 && s.coLabelsCoveredCount > 0,
        browseStatus: 'co_labels_covered' },
      { header: 'Gaps', key: 'coLabelGapsCount', breakdownKey: 'coLabelGapBreakdown',
        title: 'Co-label gaps — images with a verified box for this label but where one or more other model-blueprint labels are not yet verified or marked absent — click the number to review all, or ▾ to see which labels are driving the gaps',
        warnIf: s => s.coLabelGapsCount > 0,
        browseStatus: 'co_label_gaps' },
    ]));
  }
}
