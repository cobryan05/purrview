import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import {
  fetchReviewQueue,
  updateImageBboxes,
  setImageLabel,
  swapVerificationTag,
  UNVERIFIED_TAG_UUID,
  VERIFIED_TAG_UUID,
  FALSE_POS_TAG_UUID,
} from '/app-static/js/shared/api/review.js';

/**
 * Initialize the Review tab. Called once on first tab focus.
 */
export async function init() {
  const root = document.getElementById('review-root');
  if (!root) return;
  root.innerHTML = '';

  // ── Controls bar ──────────────────────────────────────────────────
  const controls = document.createElement('div');
  controls.style.cssText = 'display:flex;align-items:center;gap:1em;padding:0.75em 0;flex-wrap:wrap;';

  const labelSelect = document.createElement('select');
  labelSelect.style.minWidth = '12em';

  const allOption = document.createElement('option');
  allOption.value = '';
  allOption.textContent = '— All labels —';
  labelSelect.appendChild(allOption);

  const labels = await fetchLabels().catch(() => new Map());
  for (const lbl of labels.values()) {
    const opt = document.createElement('option');
    opt.value = lbl.uuid;
    opt.textContent = lbl.name;
    labelSelect.appendChild(opt);
  }

  const refreshBtn = document.createElement('button');
  refreshBtn.textContent = 'Refresh';
  refreshBtn.style.cssText = 'padding:0.3em 0.8em;cursor:pointer;';

  const statusMsg = document.createElement('span');
  statusMsg.style.cssText = 'color:#aaa;font-size:0.9em;';

  controls.appendChild(labelSelect);
  controls.appendChild(refreshBtn);
  controls.appendChild(statusMsg);
  root.appendChild(controls);

  // ── Table container ───────────────────────────────────────────────
  const tableContainer = document.createElement('div');
  tableContainer.style.overflowX = 'auto';
  root.appendChild(tableContainer);

  // ── Render logic ──────────────────────────────────────────────────
  let _rendering = false;
  async function render() {
    if (_rendering) return;
    _rendering = true;
    refreshBtn.disabled = true;
    statusMsg.textContent = 'Loading…';
    tableContainer.innerHTML = '';

    let images;
    try {
      images = await fetchReviewQueue({
        tagUuid: UNVERIFIED_TAG_UUID,
        labelUuid: labelSelect.value || undefined,
      });
    } catch (err) {
      statusMsg.textContent = `Error: ${err.message}`;
      refreshBtn.disabled = false;
      _rendering = false;
      return;
    }

    statusMsg.textContent = `${images.length} image(s) with unverified annotations`;

    if (images.length === 0) {
      const empty = document.createElement('p');
      empty.textContent = 'No unverified annotations found.';
      empty.style.color = '#aaa';
      tableContainer.appendChild(empty);
      refreshBtn.disabled = false;
      _rendering = false;
      return;
    }

    const table = document.createElement('table');
    table.style.width = '100%';
    table.style.borderCollapse = 'collapse';

    // Header
    const thead = document.createElement('thead');
    thead.innerHTML = `<tr>
      <th style="text-align:left;padding:0.4em 0.6em;">Image</th>
      <th style="text-align:left;padding:0.4em 0.6em;">Label</th>
      <th style="text-align:left;padding:0.4em 0.6em;">Confidence</th>
      <th style="text-align:left;padding:0.4em 0.6em;">Actions</th>
    </tr>`;
    table.appendChild(thead);

    const tbody = document.createElement('tbody');

    for (const img of images) {
      const unverifiedBboxes = (img.bboxes || []).filter(
        b => (b.tags || []).some(t => t.uuid === UNVERIFIED_TAG_UUID)
      );
      // All bboxes on this image (needed when we POST back the full list)
      const allBboxes = img.bboxes || [];

      const labelUuids = [...new Set(unverifiedBboxes.map(b => b.labelUuid).filter(Boolean))];

      // Per-image separator row
      const imgRow = document.createElement('tr');
      imgRow.innerHTML = `<td colspan="4" style="padding:0.5em 0.6em;background:#111;font-size:0.85em;color:#ccc;word-break:break-all;">
        📄 ${img.filename}
      </td>`;
      tbody.appendChild(imgRow);

      // One row per unverified bbox
      for (const bbox of unverifiedBboxes) {
        const label = bbox.labelUuid ? labels.get(bbox.labelUuid) : null;
        const labelName = label?.name ?? bbox.labelUuid ?? '(unknown)';
        const labelColor = label?.color ?? '#888';
        const confidence = bbox.metadataJson?.confidence
          ? `${(parseFloat(bbox.metadataJson.confidence) * 100).toFixed(0)}%`
          : '—';

        const tr = document.createElement('tr');
        tr.style.borderBottom = '1px solid #333';

        // Image path cell (empty for subsequent bboxes of same image)
        const pathTd = document.createElement('td');
        pathTd.style.padding = '0.3em 0.6em';
        tr.appendChild(pathTd);

        // Label cell
        const labelTd = document.createElement('td');
        labelTd.style.cssText = `padding:0.3em 0.6em;`;
        const labelChip = document.createElement('span');
        labelChip.textContent = labelName;
        labelChip.style.cssText = `background:${labelColor}22;border:1px solid ${labelColor};
          border-radius:3px;padding:0.1em 0.4em;font-size:0.85em;`;
        labelTd.appendChild(labelChip);
        tr.appendChild(labelTd);

        // Confidence cell
        const confTd = document.createElement('td');
        confTd.style.padding = '0.3em 0.6em';
        confTd.textContent = confidence;
        tr.appendChild(confTd);

        // Actions cell
        const actionsTd = document.createElement('td');
        actionsTd.style.cssText = 'padding:0.3em 0.6em;display:flex;gap:0.4em;flex-wrap:wrap;align-items:center;';

        const acceptBtn = makeButton('✓ Accept', '#2a6', async () => {
          await applyTagSwap(img.filename, allBboxes, bbox.uuid, VERIFIED_TAG_UUID);
          await render();
        });

        const fpBtn = makeButton('✗ False Positive', '#a22', async () => {
          await applyTagSwap(img.filename, allBboxes, bbox.uuid, FALSE_POS_TAG_UUID);
          await render();
        });

        const relabelBtn = makeButton('✎ Change Label', '#666', async () => {
          const newLabelUuid = await promptLabelChange(labels, bbox.labelUuid);
          if (newLabelUuid === null) return;
          await applyTagSwapAndRelabel(img.filename, allBboxes, bbox.uuid, VERIFIED_TAG_UUID, newLabelUuid);
          await render();
        });

        actionsTd.appendChild(acceptBtn);
        actionsTd.appendChild(fpBtn);
        actionsTd.appendChild(relabelBtn);
        tr.appendChild(actionsTd);
        tbody.appendChild(tr);
      }

      // "Mark absent" row — one per distinct label in unverified bboxes
      for (const lUuid of labelUuids) {
        const lbl = labels.get(lUuid);
        const lblName = lbl?.name ?? lUuid;

        const absentRow = document.createElement('tr');
        absentRow.style.cssText = 'background:#1a1a1a;';
        const absentTd = document.createElement('td');
        absentTd.colSpan = 4;
        absentTd.style.padding = '0.25em 0.6em 0.4em 1.5em';

        const absentBtn = makeButton(`Mark "${lblName}" absent from this image`, '#555', async () => {
          await setImageLabel(img.filename, lUuid, 'absent');
          await render();
        });
        absentBtn.style.fontSize = '0.8em';

        absentTd.appendChild(absentBtn);
        absentRow.appendChild(absentTd);
        tbody.appendChild(absentRow);
      }
    }

    table.appendChild(tbody);
    tableContainer.appendChild(table);
    refreshBtn.disabled = false;
    _rendering = false;
  }

  refreshBtn.addEventListener('click', render);
  labelSelect.addEventListener('change', render);
  render();
}

// ── Helpers ───────────────────────────────────────────────────────────

function makeButton(text, color, onClick) {
  const btn = document.createElement('button');
  btn.textContent = text;
  btn.style.cssText = `padding:0.2em 0.6em;cursor:pointer;background:${color}33;
    border:1px solid ${color};border-radius:3px;font-size:0.85em;white-space:nowrap;`;
  btn.addEventListener('click', async () => {
    btn.disabled = true;
    try { await onClick(); } catch (err) { alert(`Error: ${err.message}`); }
    btn.disabled = false;
  });
  return btn;
}

/**
 * Convert a BBoxRead from the API into the payload shape for update_image_metadata.
 * @param {object} bbox - BBoxRead object from the review queue response
 */
function bboxToPayload(bbox) {
  return {
    uuid: bbox.uuid,
    labelUuid: bbox.labelUuid ?? bbox.label?.uuid,
    x: bbox.x,
    y: bbox.y,
    width: bbox.width,
    height: bbox.height,
    tagUuids: (bbox.tags || []).map(t => t.uuid),
    extra: bbox.metadataJson ?? {},
  };
}

/**
 * Swap the verification tag on one bbox and POST the full bbox list back.
 */
async function applyTagSwap(imagePath, allBboxes, targetBboxUuid, newTagUuid) {
  const updated = allBboxes.map(b => {
    const payload = bboxToPayload(b);
    if (b.uuid === targetBboxUuid) {
      payload.tagUuids = swapVerificationTag(payload.tagUuids, newTagUuid);
    }
    return payload;
  });
  await updateImageBboxes(imagePath, updated);
}

/**
 * Swap verification tag AND change the label UUID on one bbox.
 */
async function applyTagSwapAndRelabel(imagePath, allBboxes, targetBboxUuid, newTagUuid, newLabelUuid) {
  const updated = allBboxes.map(b => {
    const payload = bboxToPayload(b);
    if (b.uuid === targetBboxUuid) {
      payload.tagUuids = swapVerificationTag(payload.tagUuids, newTagUuid);
      payload.labelUuid = newLabelUuid;
    }
    return payload;
  });
  await updateImageBboxes(imagePath, updated);
}

/**
 * Show a simple in-page modal to pick a new label. Returns the chosen label UUID or null.
 * @param {Array} labels - all available labels
 * @param {string|null} currentLabelUuid
 * @returns {Promise<string|null>}
 */
function promptLabelChange(labels, currentLabelUuid) {
  return new Promise(resolve => {
    const overlay = document.createElement('div');
    overlay.style.cssText = `position:fixed;inset:0;background:#0008;display:flex;
      align-items:center;justify-content:center;z-index:1000;`;

    const dialog = document.createElement('div');
    dialog.style.cssText = `background:#222;border:1px solid #555;border-radius:6px;
      padding:1.5em;min-width:260px;display:flex;flex-direction:column;gap:0.8em;`;

    const title = document.createElement('strong');
    title.textContent = 'Choose new label';

    const select = document.createElement('select');
    for (const lbl of labels.values()) {
      const opt = document.createElement('option');
      opt.value = lbl.uuid;
      opt.textContent = lbl.name;
      if (lbl.uuid === currentLabelUuid) opt.selected = true;
      select.appendChild(opt);
    }

    const btnRow = document.createElement('div');
    btnRow.style.cssText = 'display:flex;gap:0.5em;justify-content:flex-end;';

    const cancelBtn = document.createElement('button');
    cancelBtn.textContent = 'Cancel';
    cancelBtn.addEventListener('click', () => { overlay.remove(); resolve(null); });

    const okBtn = document.createElement('button');
    okBtn.textContent = 'OK';
    okBtn.addEventListener('click', () => { overlay.remove(); resolve(select.value || null); });

    btnRow.appendChild(cancelBtn);
    btnRow.appendChild(okBtn);
    dialog.append(title, select, btnRow);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    overlay.addEventListener('click', e => { if (e.target === overlay) { overlay.remove(); resolve(null); } });
  });
}
