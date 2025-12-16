import { generateUUID, fetchClassByUuid, Logger } from '/app-static/js/ui/utils/index.js';
import { state } from '../state.js';
/**
 * @typedef {Object} CanvasBboxMetadata
 * @property {string} uuid
 * @property {string|null} classUuid
 * @property {string[]|null} tagUuids
 * @property {import('@konva').default.Rect} rect
 * @property {import('@konva').default.Text} text
 * @property {number|null} confidence
 */

/**
 * A Konva.Group subclass representing a bounding box with metadata.
 * @extends Konva.Group
 */
export class CanvasBboxGroup extends Konva.Group {
  /** @type {CanvasBboxMetadata} */
  _metadata;

  /**
  * @param {import('@app_types').RuntimeBbox} bbox
   */
  constructor(bbox) {
    // incoming bbox is not runtime box, it's just a n array
    super({ x: bbox.x, y: bbox.y, draggable: true, name: 'annotation' });

    const color = 'grey';
    const rect = new Konva.Rect({
      width: bbox.width,
      height: bbox.height,
      stroke: color,
      strokeWidth: 2,
      name: 'box',
      draggable: false,
      strokeScaleEnabled: false
    });

    const labelText = bbox.text ?? 'Unknown';
    const text = new Konva.Text({
      text: `${labelText}${bbox.confidence != null ? ` (${(bbox.confidence * 100).toFixed(1)}%)` : ''}`,
      fontSize: 14,
      fill: color,
      x: 0,
      y: -18,
      name: 'class',
    });

    this.add(rect);
    this.add(text);


    // Restore draggable on mouseup or dragend
    this.on('mouseup dragend', () => this.draggable(true));

    // Disable dragging with middle mouse button down
    this.on('mousedown', e => {
      if (e.evt.button === 1) {
        this.draggable(false);
      } else {
        this.draggable(state.currentTool === 'select');
      }
    });

    rect.on('transform', e => {
      // The transformation happened to the rectangle, so convert it to a transformation
      // on the whole group and reset the rectangle
      const newRectPos = {
        x: this.x() + rect.x(),
        y: this.y() + rect.y(),
        width: Math.max(rect.width() * rect.scaleX(), 10),
        height: Math.max(rect.height() * rect.scaleY(), 10)
      };

      this.updatePosition(newRectPos);
    });

    this._metadata = {
      uuid: bbox.uuid ?? generateUUID(),
      classUuid: bbox.classUuid,
      tagUuids: [], // TODO
      rect,
      text,
      confidence: bbox.confidence ?? null,
    };

    this.name('annotation');
    // Resolve class name?
    this.updateMetadata();
  }

  /**
   * Update the bounding box position and/or size.
   *
   * Only properties explicitly provided in `position` are applied; any
   * omitted properties are left unchanged.
   *
   * @param {Object} position
   * @param {number} [position.x]      New X position of the group (canvas coords)
   * @param {number} [position.y]      New Y position of the group (canvas coords)
   * @param {number} [position.width]  New width of the bounding box
   * @param {number} [position.height] New height of the bounding box
   */
  updatePosition(position) {
    const rect = this.metadata.rect;
    const text = this.metadata.text;

    // Update group position if provided
    if (position.x !== undefined || position.y !== undefined) {
      this.position({
        x: position.x !== undefined ? position.x : this.x(),
        y: position.y !== undefined ? position.y : this.y(),
      });
    }

    // Update rect size if provided
    if (position.width !== undefined) {
      rect.width(position.width);
    }
    if (position.height !== undefined) {
      rect.height(position.height);
    }

    // Reset any scaling that may exist
    rect.position({x:0, y:0});
    rect.scaleX(1);
    rect.scaleY(1);

    text.setAttrs({
      x: 0,
      y: -18
    });
    // state.canvas.layer?.batchDraw();
  }

  /**
   * Update an existing bounding box with new metadata, such as class or color.
   * @param {Partial<import('@konva_groups').CanvasBboxMetadata>} metadata
    */
  updateMetadata(metadata = {}) {
    const rectRef = this._metadata.rect;
    const textRef = this._metadata.text;
    if (!rectRef || !textRef) return;

    // Merge the partial metadata with the current metadata
    /** @typedef {import('@konva_groups').CanvasBboxMetadata} */
    const mergedMetadata = { ...this._metadata };
    for (const key in metadata) {
      if (Object.prototype.hasOwnProperty.call(metadata, key)) {
        const value = metadata[key];
        if (value !== undefined) {
          mergedMetadata[key] = value;
        }
      }
    }

    // Kick off async class resolution
    (mergedMetadata.classUuid
      ? fetchClassByUuid(mergedMetadata.classUuid)
      : Promise.resolve(null)
    ).then(resolvedClass => {
      let bboxText;
      const classText = resolvedClass?.metadata?.name ?? 'Unknown';
      const color = resolvedClass?.metadata?.color ?? 'grey';
      bboxText = `${classText}${mergedMetadata.confidence != null ? ` (${(mergedMetadata.confidence * 100).toFixed(1)}%)` : ''}`;

      this._metadata = mergedMetadata;

      textRef.text(bboxText);
      textRef.fill(color);
      rectRef.stroke(color);
      state.canvas.layer?.batchDraw();
    })
      .catch(err => {
        console.error('Failed to fetch class:', err);
        // optionally fallback
      });
  }


  /**
   * Get metadata associated with this group.
   * @returns {CanvasBboxMetadata}
   */
  get metadata() {
    return this._metadata;
  }
}
