import { generateUUID, fetchClassByUuid } from '/app-static/js/ui/utils/index.js';
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
    rect.on('mouseup dragend', () => this.draggable(true));

    // Disable dragging with middle mouse button down
    rect.on('mousedown', e => {
      if (e.evt.button === 1) {
        rect.draggable(false);
      } else {
        rect.draggable(state.currentTool === 'select');
      }
    });

    // Fix the layout on resize
    rect.on('transform', () => {
      const scaleX = rect.scaleX();
      const scaleY = rect.scaleY();

      let newWidth = rect.width() * scaleX;
      let newHeight = rect.height() * scaleY;

      const MIN_SIZE = 10;
      newWidth = Math.max(newWidth, MIN_SIZE);
      newHeight = Math.max(newHeight, MIN_SIZE);

      const rectLeft = rect.x();
      const rectTop = rect.y();

      let newGroupX = this.x() + rectLeft;
      let newGroupY = this.y() + rectTop;

      rect.width(newWidth);
      rect.height(newHeight);

      this.position({
        x: newGroupX,
        y: newGroupY,
      });

      rect.scaleX(1);
      rect.scaleY(1);

      this.applyBoundingBoxLayout();
      state.canvas.layer?.batchDraw();
    });

    this._metadata = {
      uuid: bbox.uuid ?? generateUUID(),
      classUuid: bbox.classUuid,
      tagUuids: [], // TODO
      rect,
      text,
      confidence: bbox.confidence ?? null,
    };

    rect.name('annotation');
    // Resolve class name?
    this.updateMetadata();
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
      const classText = resolvedClass?.metadata?.name ?? mergedMetadata?.text ?? 'Unknown';
      const color = resolvedClass?.metadata?.color ?? 'grey';
      bboxText = `${classText}${mergedMetadata.confidence != null ? ` (${(mergedMetadata.confidence * 100).toFixed(1)}%)` : ''}`;

      this._metadata = mergedMetadata;

      textRef.text(bboxText);
      textRef.fill(color);
      rectRef.stroke(color);

      this.applyBoundingBoxLayout();
      state.canvas.layer?.batchDraw();
    })
      .catch(err => {
        console.error('Failed to fetch class:', err);
        // optionally fallback
      });
  }

  /**
   * Keep rect at (0,0) and class positioned just above.
   */
  applyBoundingBoxLayout() {
    this._metadata.rect.x(0);
    this._metadata.rect.y(0);
    this._metadata.text.x(0);
    this._metadata.text.y(-18);
  }


  /**
   * Get metadata associated with this group.
   * @returns {CanvasBboxMetadata}
   */
  get metadata() {
    return this._metadata;
  }
}
