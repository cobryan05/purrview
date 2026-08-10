import { createModalShell } from '/app-static/js/ui/utils/index.js';

/**
 * Zone drawing modal.
 *
 * A user-drawn box named "litter box area" is more reliable than a litter-box
 * detector, needs no training data, and makes `cat ∩ zone` work today — which is why
 * zones are part of the first pass rather than a later refinement.
 *
 * Two traps this respects:
 *
 * * Coordinates are stored **normalised** (0..1) together with the resolution they
 *   were drawn against, so a camera resolution change doesn't silently move every
 *   zone somewhere else.
 * * The canvas container is attached to the DOM **with explicit dimensions before**
 *   anything measures it. A container measured at zero yields a permanent scale of 0.
 */

const MIN_NORMALISED_SIZE = 0.01;

function uuidv4() {
  if (crypto?.randomUUID) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * @returns {Promise<Array|null>} the edited zone list, or null if cancelled.
 */
export async function openZonesModal({ sourceUuid, sourceName, zones = [] }) {
  return new Promise((resolve) => {
    let settled = false;
    const shell = createModalShell({
      onClose: () => { if (!settled) { settled = true; resolve(null); } },
    });

    const working = zones.map(z => ({ ...z, pointsJson: (z.pointsJson || []).map(p => [...p]) }));

    shell.dialog.classList.add('watch-zones-dialog');
    shell.dialog.appendChild(el('h3', null, `Zones — ${sourceName || sourceUuid}`));
    shell.dialog.appendChild(el('p', 'panel-hint',
      'Drag on the frame to draw a zone. Coordinates are stored relative to the frame, '
      + 'so they survive a camera resolution change. '
      + 'A live still as a backdrop is not wired up yet — zones drawn on the blank '
      + 'frame are still correct, since they are normalised.'));

    const body = el('div', 'watch-zones-body');
    shell.dialog.appendChild(body);

    // Explicit dimensions BEFORE anything measures this container — a zero-sized
    // parent produces a scale of 0 that never recovers.
    const stageWrap = el('div', 'watch-zones-stage');
    stageWrap.style.width = '640px';
    stageWrap.style.height = '360px';
    stageWrap.style.position = 'relative';
    stageWrap.dataset.testid = 'watch-zones-stage';
    body.appendChild(stageWrap);

    // Blank backdrop for now. Zones are normalised, so drawing against a plain
    // frame produces exactly the same stored coordinates a live still would.
    const backdrop = el('div', 'watch-zones-backdrop');
    backdrop.style.width = '100%';
    backdrop.style.height = '100%';
    backdrop.style.background = 'var(--pico-card-background-color, #222)';
    stageWrap.appendChild(backdrop);

    const overlay = document.createElement('canvas');
    overlay.width = 640;
    overlay.height = 360;
    overlay.style.position = 'absolute';
    overlay.style.inset = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.cursor = 'crosshair';
    overlay.dataset.testid = 'watch-zones-canvas';
    stageWrap.appendChild(overlay);

    const side = el('div', 'watch-zones-list');
    side.dataset.testid = 'watch-zones-list';
    body.appendChild(side);

    const ctx = overlay.getContext('2d');
    let referenceWidth = 0;
    let referenceHeight = 0;

    function draw() {
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      for (const zone of working) {
        const pts = (zone.pointsJson || []).map(([x, y]) => [x * overlay.width, y * overlay.height]);
        if (pts.length < 2) continue;
        ctx.strokeStyle = '#4ade80';
        ctx.lineWidth = 2;
        ctx.fillStyle = 'rgba(74, 222, 128, 0.15)';
        ctx.beginPath();
        if (zone.kind === 'box') {
          const [[x1, y1], [x2, y2]] = pts;
          ctx.rect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
        } else {
          ctx.moveTo(pts[0][0], pts[0][1]);
          for (const [x, y] of pts.slice(1)) ctx.lineTo(x, y);
          ctx.closePath();
        }
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = '#4ade80';
        ctx.font = '13px sans-serif';
        ctx.fillText(zone.name, pts[0][0] + 4, pts[0][1] + 16);
      }
    }

    function renderList() {
      side.innerHTML = '';
      if (!working.length) {
        side.appendChild(el('div', 'panel-hint', 'No zones yet — drag on the frame to add one.'));
        return;
      }
      working.forEach((zone, index) => {
        const row = el('div', 'watch-zone-row');
        row.dataset.testid = 'watch-zone-row';
        const nameInput = document.createElement('input');
        nameInput.type = 'text';
        nameInput.value = zone.name;
        nameInput.dataset.testid = 'watch-zone-name';
        nameInput.addEventListener('input', () => { zone.name = nameInput.value; draw(); });
        row.appendChild(nameInput);
        const del = el('button', 'panel-btn panel-btn--danger', '×');
        del.type = 'button';
        del.addEventListener('click', () => {
          working.splice(index, 1);
          renderList();
          draw();
        });
        row.appendChild(del);
        side.appendChild(row);
      });
    }

    // ── Drawing ──
    let dragStart = null;
    function toNormalised(event) {
      const rect = overlay.getBoundingClientRect();
      // Guard against a zero-sized rect: dividing by it yields Infinity and would
      // write a zone that can never match anything.
      if (!rect.width || !rect.height) return null;
      return [
        Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
        Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
      ];
    }

    overlay.addEventListener('pointerdown', (event) => {
      dragStart = toNormalised(event);
      if (dragStart) overlay.setPointerCapture(event.pointerId);
    });

    overlay.addEventListener('pointermove', (event) => {
      if (!dragStart) return;
      const current = toNormalised(event);
      if (!current) return;
      draw();
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = 2;
      ctx.strokeRect(
        dragStart[0] * overlay.width,
        dragStart[1] * overlay.height,
        (current[0] - dragStart[0]) * overlay.width,
        (current[1] - dragStart[1]) * overlay.height,
      );
    });

    overlay.addEventListener('pointerup', (event) => {
      if (!dragStart) return;
      const end = toNormalised(event);
      const start = dragStart;
      dragStart = null;
      if (!end) return;
      const width = Math.abs(end[0] - start[0]);
      const height = Math.abs(end[1] - start[1]);
      // A stray click would otherwise create a degenerate zone that silently matches
      // nothing.
      if (width < MIN_NORMALISED_SIZE || height < MIN_NORMALISED_SIZE) {
        draw();
        return;
      }
      working.push({
        uuid: uuidv4(),
        name: `Zone ${working.length + 1}`,
        kind: 'box',
        pointsJson: [
          [Math.min(start[0], end[0]), Math.min(start[1], end[1])],
          [Math.max(start[0], end[0]), Math.max(start[1], end[1])],
        ],
        referenceWidth,
        referenceHeight,
      });
      renderList();
      draw();
    });

    // ── Footer ──
    const footer = el('div', 'watch-zones-footer');

    const cancelBtn = el('button', 'panel-btn', 'Cancel');
    cancelBtn.type = 'button';
    cancelBtn.addEventListener('click', () => shell.close());
    footer.appendChild(cancelBtn);

    const saveBtn = el('button', 'panel-btn panel-btn--primary', 'Done');
    saveBtn.type = 'button';
    saveBtn.dataset.testid = 'watch-zones-save';
    saveBtn.addEventListener('click', () => {
      settled = true;
      resolve(working);
      shell.close();
    });
    footer.appendChild(saveBtn);
    shell.dialog.appendChild(footer);

    renderList();
    // Open only once the stage has real dimensions in the DOM.
    shell.open();
    draw();
  });
}
