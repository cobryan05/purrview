// @ts-check
export class BboxView extends Konva.Group {
    /** @type {import('@domain_image').UIBbox} */ uiBBox;
    /** @type {import('@konva').default.Rect} */ _rect;
    /** @type {import('@konva').default.Text} */ _text;

  /**
   * @param {import('@domain_image').UIBbox} uiBBox
   * @param {number} scaleWidth - To convert normalized -> pixels
   * @param {number} scaleHeight - To convert normalized -> pixels
   */
  constructor(uiBBox, scaleWidth, scaleHeight) {
    // Initialize Konva.Group
    super({
      x: uiBBox.x * scaleWidth,
      y: uiBBox.y * scaleHeight,
      width: uiBBox.width * scaleWidth,
      height: uiBBox.height * scaleHeight,
      draggable: true,
      name: 'annotation',
      id: uiBBox.uuid
    });

    // Assigning to typed members
    this.uiBBox = uiBBox;

    this._rect = new Konva.Rect({
      width: this.width(),
      height: this.height(),
      stroke: 'grey',
      strokeWidth: 2,
      name: 'box',
      strokeScaleEnabled: false
    });

    this._text = new Konva.Text({
      text: uiBBox.label?.text || 'Loading...',
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
      this.uiBBox.isDirty = true;

      // Logic for snapping/updating normalized data would go here
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
}