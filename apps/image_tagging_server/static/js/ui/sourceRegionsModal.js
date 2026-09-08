import { createModalShell, toast } from '/app-static/js/ui/utils/index.js';
import { promptRestartAffectedTasks } from '/app-static/js/ui/utils/restartAffectedTasks.js';
import {
  REGION_ROLE_MASK,
  REGION_ROLE_ZONE,
  fetchSourceRegions,
  grabSourceFrame,
  saveSourceRegions,
} from '/app-static/js/shared/api/sourceRegions.js';

/**
 * Region editor — draw zones and masks on a source's frame.
 *
 * A user-drawn box named "litter box area" is more reliable than a litter-box detector,
 * needs no training data, and makes `cat ∩ zone` work today. A mask is the same geometry
 * pointed the other way: it suppresses detections whose centre falls inside it.
 *
 * Three traps this respects:
 *
 * * Coordinates are stored **normalised** (0..1) together with the resolution they were
 *   drawn against, so a camera resolution change doesn't silently move every region.
 * * The canvas container is attached to the DOM **with explicit dimensions before**
 *   anything measures it. A container measured at zero yields a permanent scale of 0.
 * * Region uuids are preserved across a save, because rules point at zones by uuid.
 *   Regenerating them would break every zone rule at once with no error anywhere.
 */

const MIN_NORMALISED_SIZE = 0.01;
const STAGE_WIDTH = 720;
const STAGE_HEIGHT = 405;
// How close (in stage pixels) a click has to land to the first vertex to mean "close it".
// Generous, because the alternative — an extra vertex on top of the first — is invisible.
const CLOSE_SNAP_PX = 12;

const ROLE_STYLES = {
  [REGION_ROLE_ZONE]: { stroke: '#4ade80', fill: 'rgba(74, 222, 128, 0.15)' },
  // Red, and hatched below: a mask *removes* detections, and reading it as a zone is the
  // one mistake that silently produces no events at all.
  [REGION_ROLE_MASK]: { stroke: '#f87171', fill: 'rgba(248, 113, 113, 0.22)' },
};

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
 * Open the region editor for one source.
 *
 * Loads and saves through the API itself rather than taking regions as a parameter:
 * regions outlive any one watcher's editing session, so a caller that passed them in
 * would be handing over a snapshot that another tab may already have changed.
 *
 * @param {{
 *   sourceUuid: string,
 *   sourceName?: string,
 *   maskOptOut?: {disabled: Set<string>, onChange: (next: Set<string>) => void},
 * }} opts
 *   `maskOptOut` is supplied when the modal is opened from a watcher: each mask then
 *   also offers "applies to this watcher". Opened from the Sources pane there is no
 *   watcher to scope it to, so only the geometry is editable.
 * @returns {Promise<Array|null>} the saved regions, or null if cancelled.
 */
export async function openSourceRegionsModal({ sourceUuid, sourceName, maskOptOut = null }) {
  let working = [];
  try {
    working = (await fetchSourceRegions(sourceUuid)).map(r => ({
      ...r,
      pointsJson: (r.pointsJson || []).map(p => [...p]),
    }));
  } catch (e) {
    toast(`Could not load regions: ${e.message}`, 'error');
    return null;
  }

  return new Promise((resolve) => {
    let settled = false;
    const shell = createModalShell({
      onClose: () => { if (!settled) { settled = true; resolve(null); } },
    });

    shell.dialog.classList.add('watch-zones-dialog');
    shell.dialog.appendChild(el('h3', null, `Regions — ${sourceName || sourceUuid}`));

    const hint = el('p', 'panel-hint',
      'Drag to draw a box. For a polygon, click each corner, then close it by clicking the '
      + 'first point again, right-clicking, double-clicking, or pressing Enter — Escape discards it. '
      + 'Zones are named regions a rule can point at; masks suppress any detection whose '
      + 'centre falls inside them, for every rule. '
      + 'Coordinates are stored relative to the frame, so they survive a camera resolution change.');
    shell.dialog.appendChild(hint);

    const body = el('div', 'watch-zones-body');
    shell.dialog.appendChild(body);

    const stageColumn = el('div', 'watch-zones-stage-column');
    body.appendChild(stageColumn);

    // Explicit dimensions BEFORE anything measures this container — a zero-sized parent
    // produces a scale of 0 that never recovers.
    const stageWrap = el('div', 'watch-zones-stage');
    stageWrap.style.width = `${STAGE_WIDTH}px`;
    stageWrap.style.height = `${STAGE_HEIGHT}px`;
    stageWrap.style.position = 'relative';
    stageWrap.dataset.testid = 'watch-zones-stage';
    stageColumn.appendChild(stageWrap);

    const backdrop = document.createElement('img');
    backdrop.className = 'watch-zones-backdrop';
    backdrop.alt = '';
    backdrop.style.position = 'absolute';
    backdrop.style.inset = '0';
    backdrop.style.width = '100%';
    backdrop.style.height = '100%';
    backdrop.style.objectFit = 'fill';
    backdrop.style.background = 'var(--pico-card-background-color, #222)';
    backdrop.dataset.testid = 'watch-zones-backdrop';
    stageWrap.appendChild(backdrop);

    const overlay = document.createElement('canvas');
    overlay.width = STAGE_WIDTH;
    overlay.height = STAGE_HEIGHT;
    overlay.style.position = 'absolute';
    overlay.style.inset = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.cursor = 'crosshair';
    overlay.dataset.testid = 'watch-zones-canvas';
    stageWrap.appendChild(overlay);

    const ctx = overlay.getContext('2d');
    // The camera's true resolution, recorded with each region so a later resolution
    // change is detectable rather than silently shifting everything.
    let referenceWidth = 0;
    let referenceHeight = 0;

    // ── Toolbar ──
    const toolbar = el('div', 'watch-zones-toolbar');
    stageColumn.appendChild(toolbar);

    const grabBtn = el('button', 'panel-btn', 'Grab frame');
    grabBtn.type = 'button';
    grabBtn.dataset.testid = 'watch-zones-grab';
    toolbar.appendChild(grabBtn);

    let drawRole = REGION_ROLE_ZONE;
    const roleSelect = document.createElement('select');
    for (const [key, text] of [[REGION_ROLE_ZONE, 'Draw: Zone'], [REGION_ROLE_MASK, 'Draw: Mask']]) {
      const o = document.createElement('option');
      o.value = key;
      o.textContent = text;
      roleSelect.appendChild(o);
    }
    roleSelect.dataset.testid = 'watch-zones-role';
    roleSelect.addEventListener('change', () => { drawRole = roleSelect.value; });
    toolbar.appendChild(roleSelect);

    let drawKind = 'box';
    const kindSelect = document.createElement('select');
    for (const [key, text] of [['box', 'Shape: Box'], ['polygon', 'Shape: Polygon']]) {
      const o = document.createElement('option');
      o.value = key;
      o.textContent = text;
      kindSelect.appendChild(o);
    }
    kindSelect.dataset.testid = 'watch-zones-kind';
    kindSelect.addEventListener('change', () => {
      drawKind = kindSelect.value;
      pendingPolygon = [];
      hoverPoint = null;
      draw();
    });
    toolbar.appendChild(kindSelect);

    const grabStatus = el('span', 'watch-hint', 'No frame yet — regions are normalised, so drawing on a blank stage still works.');
    grabStatus.dataset.testid = 'watch-zones-grab-status';
    toolbar.appendChild(grabStatus);

    const side = el('div', 'watch-zones-list');
    side.dataset.testid = 'watch-zones-list';
    body.appendChild(side);

    // ── Drawing ──
    /** @type {Array<[number, number]>} in-progress polygon vertices */
    let pendingPolygon = [];
    let dragStart = null;
    let selectedUuid = null;
    /** @type {[number, number]|null} last cursor position, for the rubber-band segment */
    let hoverPoint = null;

    function pointsToPixels(points) {
      return points.map(([x, y]) => [x * overlay.width, y * overlay.height]);
    }

    function drawShape(points, kind, style, { dashed = false, selected = false } = {}) {
      const pts = pointsToPixels(points);
      if (pts.length < 2) return;
      ctx.save();
      ctx.strokeStyle = style.stroke;
      ctx.fillStyle = style.fill;
      ctx.lineWidth = selected ? 3 : 2;
      if (dashed) ctx.setLineDash([6, 4]);
      ctx.beginPath();
      if (kind === 'box') {
        const [[x1, y1], [x2, y2]] = pts;
        ctx.rect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
      } else {
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (const [x, y] of pts.slice(1)) ctx.lineTo(x, y);
        ctx.closePath();
      }
      ctx.fill();
      ctx.stroke();
      ctx.restore();
      return pts;
    }

    function draw() {
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      for (const region of working) {
        const style = ROLE_STYLES[region.role] || ROLE_STYLES[REGION_ROLE_ZONE];
        const pts = drawShape(region.pointsJson || [], region.kind, style, {
          selected: region.uuid === selectedUuid,
        });
        if (!pts?.length) continue;
        ctx.save();
        ctx.fillStyle = style.stroke;
        ctx.font = '13px sans-serif';
        const anchor = pts.reduce((a, p) => (p[1] < a[1] ? p : a), pts[0]);
        const prefix = region.role === REGION_ROLE_MASK ? '⊘ ' : '';
        ctx.fillText(`${prefix}${region.name}`, anchor[0] + 4, anchor[1] + 16);
        ctx.restore();
      }

      // In-progress polygon, dashed so it reads as not-yet-real.
      if (pendingPolygon.length) {
        const style = ROLE_STYLES[drawRole];
        const closing = hoverPoint != null && isNearFirstPoint(hoverPoint);
        ctx.save();
        ctx.strokeStyle = style.stroke;
        ctx.setLineDash([5, 4]);
        ctx.lineWidth = 2;
        ctx.beginPath();
        const pts = pointsToPixels(pendingPolygon);
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (const [x, y] of pts.slice(1)) ctx.lineTo(x, y);
        // Rubber-band the segment under the cursor, snapped to the first vertex when the
        // next click would close the shape — that snap *is* the affordance.
        if (hoverPoint) {
          const [hx, hy] = closing ? pts[0] : pointsToPixels([hoverPoint])[0];
          ctx.lineTo(hx, hy);
        }
        ctx.stroke();
        for (const [x, y] of pts) {
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, Math.PI * 2);
          ctx.fillStyle = style.stroke;
          ctx.fill();
        }
        // A filled halo on the first vertex when clicking there would close the polygon.
        if (closing) {
          ctx.beginPath();
          ctx.setLineDash([]);
          ctx.arc(pts[0][0], pts[0][1], CLOSE_SNAP_PX * 0.75, 0, Math.PI * 2);
          ctx.fillStyle = style.fill;
          ctx.fill();
          ctx.strokeStyle = style.stroke;
          ctx.lineWidth = 2;
          ctx.stroke();
        }
        ctx.restore();
      }
    }

    /**
     * True when `point` (normalised) lands on the polygon's first vertex closely enough to
     * mean "close it". Requires 3 points already, so the second click of a fresh polygon
     * can't collapse it into nothing.
     */
    function isNearFirstPoint(point) {
      if (pendingPolygon.length < 3) return false;
      const [fx, fy] = pendingPolygon[0];
      const dx = (point[0] - fx) * overlay.width;
      const dy = (point[1] - fy) * overlay.height;
      return Math.hypot(dx, dy) <= CLOSE_SNAP_PX;
    }

    function defaultName(role) {
      const count = working.filter(r => r.role === role).length + 1;
      return role === REGION_ROLE_MASK ? `Mask ${count}` : `Zone ${count}`;
    }

    function addRegion(kind, points) {
      const region = {
        uuid: uuidv4(),
        name: defaultName(drawRole),
        role: drawRole,
        kind,
        pointsJson: points,
        referenceWidth,
        referenceHeight,
        order: working.length,
      };
      working.push(region);
      selectedUuid = region.uuid;
      renderList();
      draw();
    }

    function renderList() {
      side.innerHTML = '';
      for (const role of [REGION_ROLE_ZONE, REGION_ROLE_MASK]) {
        const inRole = working.filter(r => r.role === role);
        const heading = el('div', 'watch-zones-list-heading',
          role === REGION_ROLE_ZONE ? 'Zones' : 'Masks');
        side.appendChild(heading);
        if (!inRole.length) {
          side.appendChild(el('div', 'panel-hint',
            role === REGION_ROLE_ZONE
              ? 'None yet. A rule points at a zone by name.'
              : 'None yet. A mask hides detections from every rule.'));
          continue;
        }
        for (const region of inRole) {
          const row = el('div', 'watch-zone-row');
          row.dataset.testid = 'watch-zone-row';
          row.dataset.role = role;
          if (region.uuid === selectedUuid) row.classList.add('watch-zone-row--selected');

          const nameInput = document.createElement('input');
          nameInput.type = 'text';
          nameInput.value = region.name;
          nameInput.dataset.testid = 'watch-zone-name';
          nameInput.addEventListener('input', () => { region.name = nameInput.value; draw(); });
          nameInput.addEventListener('focus', () => { selectedUuid = region.uuid; draw(); });
          row.appendChild(nameInput);

          const kindTag = el('span', 'watch-hint', region.kind === 'polygon' ? 'poly' : 'box');
          row.appendChild(kindTag);

          // "Applies to this watcher" — a mask acts on every detection, so unlike a zone
          // it needs a way to say "not for this one". Ticked by default, which is what
          // makes a mask drawn tomorrow protect the watchers written today.
          if (role === REGION_ROLE_MASK && maskOptOut) {
            const applies = el('label', 'watch-mask-toggle');
            applies.dataset.testid = 'watch-mask-toggle';
            applies.title = 'Apply this mask in the watcher being edited';
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = !maskOptOut.disabled.has(region.uuid);
            cb.dataset.maskUuid = region.uuid;
            cb.addEventListener('change', () => {
              const next = new Set(maskOptOut.disabled);
              if (cb.checked) next.delete(region.uuid);
              else next.add(region.uuid);
              maskOptOut.disabled = next;
              maskOptOut.onChange(next);
            });
            applies.appendChild(cb);
            applies.appendChild(el('span', null, 'apply'));
            row.appendChild(applies);
          }

          const del = el('button', 'panel-btn panel-btn--danger', '×');
          del.type = 'button';
          del.title = 'Remove this region';
          del.addEventListener('click', () => {
            working = working.filter(r => r.uuid !== region.uuid);
            if (selectedUuid === region.uuid) selectedUuid = null;
            renderList();
            draw();
          });
          row.appendChild(del);
          side.appendChild(row);
        }
      }
    }

    function toNormalised(event) {
      const rect = overlay.getBoundingClientRect();
      // Guard against a zero-sized rect: dividing by it yields Infinity and would write
      // a region that can never match anything.
      if (!rect.width || !rect.height) return null;
      return [
        Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
        Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
      ];
    }

    overlay.addEventListener('pointerdown', (event) => {
      // Right-click closes the polygon; it must never also drop a vertex.
      if (event.button !== 0) return;
      if (drawKind === 'polygon') {
        const point = toNormalised(event);
        if (!point) return;
        if (isNearFirstPoint(point)) {
          commitPolygon();
          return;
        }
        pendingPolygon.push(point);
        draw();
        return;
      }
      dragStart = toNormalised(event);
      if (dragStart) overlay.setPointerCapture(event.pointerId);
    });

    overlay.addEventListener('contextmenu', (event) => {
      if (drawKind !== 'polygon' || !pendingPolygon.length) return;
      event.preventDefault();
      commitPolygon();
    });

    // Double-click closes too, but the two clicks that make it up have already added two
    // vertices — drop the stray one so the shape ends where the user stopped clicking.
    overlay.addEventListener('dblclick', (event) => {
      if (drawKind !== 'polygon' || pendingPolygon.length < 4) return;
      event.preventDefault();
      pendingPolygon.pop();
      commitPolygon();
    });

    overlay.addEventListener('pointerleave', () => {
      if (!hoverPoint) return;
      hoverPoint = null;
      draw();
    });

    overlay.addEventListener('pointermove', (event) => {
      if (drawKind === 'polygon') {
        if (!pendingPolygon.length) return;
        hoverPoint = toNormalised(event);
        draw();
        return;
      }
      if (!dragStart) return;
      const current = toNormalised(event);
      if (!current) return;
      draw();
      const style = ROLE_STYLES[drawRole];
      ctx.save();
      ctx.strokeStyle = style.stroke;
      ctx.fillStyle = style.fill;
      ctx.lineWidth = 2;
      const x = dragStart[0] * overlay.width;
      const y = dragStart[1] * overlay.height;
      const w = (current[0] - dragStart[0]) * overlay.width;
      const h = (current[1] - dragStart[1]) * overlay.height;
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
      ctx.restore();
    });

    overlay.addEventListener('pointerup', (event) => {
      if (!dragStart) return;
      const end = toNormalised(event);
      const start = dragStart;
      dragStart = null;
      if (!end) return;
      const width = Math.abs(end[0] - start[0]);
      const height = Math.abs(end[1] - start[1]);
      // A stray click would otherwise create a degenerate region that silently matches
      // nothing.
      if (width < MIN_NORMALISED_SIZE || height < MIN_NORMALISED_SIZE) {
        draw();
        return;
      }
      addRegion('box', [
        [Math.min(start[0], end[0]), Math.min(start[1], end[1])],
        [Math.max(start[0], end[0]), Math.max(start[1], end[1])],
      ]);
    });

    function commitPolygon() {
      if (pendingPolygon.length < 3) {
        toast('A polygon needs at least 3 points', 'error');
        return;
      }
      addRegion('polygon', pendingPolygon);
      pendingPolygon = [];
      hoverPoint = null;
      draw();
    }

    function onKeyDown(event) {
      if (event.key === 'Enter' && pendingPolygon.length) {
        event.preventDefault();
        commitPolygon();
      } else if (event.key === 'Escape' && pendingPolygon.length) {
        // Swallow Escape while a polygon is in progress: closing the whole modal on the
        // keypress meant to cancel one shape loses every other edit too.
        event.preventDefault();
        event.stopPropagation();
        pendingPolygon = [];
        hoverPoint = null;
        draw();
      }
    }
    // Capture phase so the polygon-cancel case beats the modal's own Escape handler.
    document.addEventListener('keydown', onKeyDown, true);

    // ── Frame grab ──
    async function grabFrame() {
      grabBtn.disabled = true;
      grabStatus.textContent = 'Grabbing a frame…';
      try {
        const result = await grabSourceFrame(sourceUuid);
        backdrop.src = `data:image/jpeg;base64,${result.frame}`;
        referenceWidth = result.width || 0;
        referenceHeight = result.height || 0;
        grabStatus.textContent = result.source === 'watcher'
          // Worth saying: a watcher's frame is downscaled to its analysis width, so what
          // the user is drawing on is not the camera's full resolution.
          ? `From the running watcher (${result.width}×${result.height}, analysis size)`
          : `${result.width}×${result.height}`;
      } catch (e) {
        grabStatus.textContent = e.message;
        toast(`Could not grab a frame: ${e.message}`, 'error');
      } finally {
        grabBtn.disabled = false;
      }
    }
    grabBtn.addEventListener('click', grabFrame);

    // ── Footer ──
    const footer = el('div', 'watch-zones-footer');

    const cancelBtn = el('button', 'panel-btn', 'Cancel');
    cancelBtn.type = 'button';
    cancelBtn.addEventListener('click', () => shell.close());
    footer.appendChild(cancelBtn);

    const saveBtn = el('button', 'panel-btn panel-btn--primary', 'Save');
    saveBtn.type = 'button';
    saveBtn.dataset.testid = 'watch-zones-save';
    saveBtn.addEventListener('click', async () => {
      saveBtn.disabled = true;
      let saved = false;
      try {
        const payload = working.map((r, index) => ({
          uuid: r.uuid,
          name: r.name,
          role: r.role,
          kind: r.kind,
          pointsJson: r.pointsJson,
          referenceWidth: r.referenceWidth ?? 0,
          referenceHeight: r.referenceHeight ?? 0,
          order: index,
        }));
        const { regions, orphanedRules } = await saveSourceRegions(sourceUuid, payload);
        if (orphanedRules.length) {
          // A removed zone leaves its rules pointing at nothing. Legitimate, but silent
          // is exactly the failure mode zones-by-uuid invites.
          toast(
            `Saved. ${orphanedRules.length} rule(s) referenced a removed zone: ${orphanedRules.join(', ')}`,
            'error',
          );
        } else {
          toast('Regions saved', 'success');
        }
        settled = true;
        document.removeEventListener('keydown', onKeyDown, true);
        resolve(regions);
        shell.close();
        saved = true;
      } catch (e) {
        toast(`Could not save regions: ${e.message}`, 'error');
      } finally {
        saveBtn.disabled = false;
      }
      // Outside the try: the save has already succeeded and been reported, so a
      // failure in the restart prompt must not come back as "Could not save
      // regions" and send the user round again.
      //
      // Zones and masks are read once per source in _build_pipelines and baked
      // into that source's WatchEngine and MotionDetector, so a running watcher
      // keeps evaluating the old shapes. Prompted after the close so the confirm
      // does not stack on top of this modal. No-op when nothing is running.
      if (saved) await promptRestartAffectedTasks(sourceUuid, { label: "source's regions" });
    });
    footer.appendChild(saveBtn);
    shell.dialog.appendChild(footer);

    const originalClose = shell.close.bind(shell);
    shell.close = () => {
      document.removeEventListener('keydown', onKeyDown, true);
      originalClose();
    };

    renderList();
    // Open only once the stage has real dimensions in the DOM.
    shell.open();
    draw();
    // Try for a real backdrop straight away — drawing against grey works (regions are
    // normalised) but nobody can place a litter-box zone without seeing the litter box.
    grabFrame();
  });
}
