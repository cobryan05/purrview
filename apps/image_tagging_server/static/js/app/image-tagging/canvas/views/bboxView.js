// @ts-check
export class BboxView extends Konva.Group {
    /** @type {import('@domain_image').UIBbox} */ uiBBox;
    /** @type {import('@konva').default.Rect} */ _rect;
    /** @type {import('@konva').default.Text} */ _text;

  /**
   * @param {import('@domain_image').UIBbox} uiBBox
   * @param {import('@image_tagging_types').ViewportTransform} viewport
   */
  constructor(uiBBox, viewport) {
    // Initialize Konva.Group
    super({
      x: uiBBox.x * viewport.imageWidth * viewport.scale,
      y: uiBBox.y * viewport.imageHeight * viewport.scale,
      width: uiBBox.width * viewport.imageWidth * viewport.scale,
      height: uiBBox.height * viewport.imageHeight * viewport.scale,
      draggable: true,
      name: 'annotation',
      id: uiBBox.uuid
    });

    // Assigning to typed members
    this.uiBBox = uiBBox;

    this._rect = new Konva.Rect({
      width: this.width(),
      height: this.height(),
      stroke: uiBBox.label?.color || 'grey',
      strokeWidth: 2,
      name: 'box',
      strokeScaleEnabled: false
    });

    this._text = new Konva.Text({
      text: uiBBox.label?.text || 'Loading...',
      fill: uiBBox.label?.color || 'grey',
      fontSize: 14,
      y: -18,
      name: 'label'
    });

    this.add(this._rect);
    this.add(this._text);

    this._setupEvents();
  }

  /** @private */
  _setupEvents() {
    this.on('dragend transformend', () => {
      const scaleX = this.scaleX();
      const scaleY = this.scaleY();
      const newWidth = this.width() * scaleX;
      const newHeight = this.height() * scaleY;

      this.setAttrs({
        width: newWidth,
        height: newHeight,
        scaleX: 1,
        scaleY: 1
      });
      this._rect.size({ width: newWidth, height: newHeight });
    });
  }

  /**
   * @param {string} text
   * @param {string} color
   */
  updateAppearance(text, color) {
    this._text.text(text);
    this._text.fill(color);
    this._rect.stroke(color);

    // Safety check if the layer exists to redraw
    this.getLayer()?.batchDraw();
  }

  /**
 * @param {number} width
 * @param {number} height
 */
  updateSize(width, height) {
    // Adjust text position or scale here too?
    this.width(width);
    this.height(height);
    this._rect.size({ width, height });
  }
}