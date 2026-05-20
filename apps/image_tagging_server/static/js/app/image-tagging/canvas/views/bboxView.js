// @ts-check
const LABEL_OFFSET_Y = -18;

export class BboxView extends Konva.Group {
    /** @type {import('@domain_image').UIBbox} */ uiBBox;
    /** @type {import('@konva').default.Rect} */ _rect;
    /** @type {import('@konva').default.Text} */ _text;
    /** @type {import('@image_tagging_types').ViewportTransform} */ _viewport;

  /**
   * @param {import('@domain_image').UIBbox} uiBBox
   * @param {import('@image_tagging_types').ViewportTransform} viewport
   */
  constructor(uiBBox, viewport) {
    super({
      x: uiBBox.x * viewport.imageWidth * viewport.scale + viewport.offsetX,
      y: uiBBox.y * viewport.imageHeight * viewport.scale + viewport.offsetY,
      width: uiBBox.width * viewport.imageWidth * viewport.scale,
      height: uiBBox.height * viewport.imageHeight * viewport.scale,
      draggable: true,
      name: 'bbox',
      id: uiBBox.uuid
    });

    this.uiBBox = uiBBox;
    this._viewport = viewport;
    this.dirty = false;

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
      y: LABEL_OFFSET_Y,
      name: 'label'
    });

    this.add(this._rect);
    this.add(this._text);

    this._setupEvents();
  }

  /** @private */
  _setupEvents() {
    // During transform the group's scaleX/Y changes, make sure text doesn't drift or scale
    this.on('transform', () => {
      const scaleX = this.scaleX();
      const scaleY = this.scaleY();
      this._text.scaleX(1 / scaleX);
      this._text.scaleY(1 / scaleY);
      this._text.y(LABEL_OFFSET_Y / scaleY);
    });

    this.on('dragend transformend', () => {
      const scaleX = this.scaleX();
      const scaleY = this.scaleY();
      const newWidth = this.width() * scaleX;
      const newHeight = this.height() * scaleY;

      this.setAttrs({ width: newWidth, height: newHeight, scaleX: 1, scaleY: 1 });
      this._rect.setAttrs({ width: newWidth, height: newHeight, scaleX: 1, scaleY: 1 });
      this._text.setAttrs({ scaleX: 1, scaleY: 1, y: LABEL_OFFSET_Y });

      if (this._viewport) {
        const sw = this._viewport.imageWidth * this._viewport.scale;
        const sh = this._viewport.imageHeight * this._viewport.scale;
        this.uiBBox.x = (this.x() - this._viewport.offsetX) / sw;
        this.uiBBox.y = (this.y() - this._viewport.offsetY) / sh;
        this.uiBBox.width = newWidth / sw;
        this.uiBBox.height = newHeight / sh;
      }
    });
  }

  /** @param {import('@image_tagging_types').CanvasRect} rect */
  updatePosition({ x, y, width, height }) {
    this.setAttrs({ x, y, width, height });
    this._rect.setAttrs({ width, height });
    this.getLayer()?.batchDraw();
  }

  /**
   * @param {string} text
   * @param {string} color
   */
  updateAppearance(text, color) {
    this._text.text(text);
    this._text.fill(color);
    this._rect.stroke(color);
    this.getLayer()?.batchDraw();
  }

  /** @param {object} [config] */
  getClientRect(config) {
    return this._rect.getClientRect(config);
  }

  /** @param {import('@image_tagging_types').ViewportTransform} viewport */
  updateViewport(viewport) {
    this._viewport = viewport;
    const fullScaledWidth = viewport.imageWidth * viewport.scale;
    const fullScaledHeight = viewport.imageHeight * viewport.scale;
    const w = this.uiBBox.width * fullScaledWidth;
    const h = this.uiBBox.height * fullScaledHeight;
    this.setAttrs({
      x: (this.uiBBox.x * fullScaledWidth) + viewport.offsetX,
      y: (this.uiBBox.y * fullScaledHeight) + viewport.offsetY,
      width: w,
      height: h
    });
    this._rect.size({ width: w, height: h });
  }
}
