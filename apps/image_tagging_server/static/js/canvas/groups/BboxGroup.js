import { generateUUID, fetchClassByUuid } from '/app-static/js/ui/utils/index.js';
import { state } from '../state.js';
/**
 * @typedef {Object} BboxGroupMetadata
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
export class BboxGroup extends Konva.Group {
  /** @type {BboxGroupMetadata} */
  _metadata;

  /**
  * @param {import('@app_types').RuntimeBbox} bbox
   */
  constructor(bbox) {
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

    this._metadata = {
      uuid: bbox.uuid ?? generateUUID(),
      classUuid: bbox.classUuid,
      class: null, // fill in later
      rect,
      text,
      confidence: bbox.confidence ?? null,
    };
  }

  /**
   * Update an existing bounding box with new metadata, such as class or color.
   * @param {Partial<import('@konva_groups').BboxGroupMetadata>} metadata
    */
  updateMetadata(metadata = {}) {
    const rectRef = this._metadata.rect;
    const textRef = this._metadata.text;
    if (!rectRef || !textRef) return;

    // Merge the partial metadata with the current metadata
    /** @typedef {import('@konva_groups').BboxGroupMetadata} */
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
   * @returns {BboxGroupMetadata}
   */
  get metadata() {
    return this._metadata;
  }
}
