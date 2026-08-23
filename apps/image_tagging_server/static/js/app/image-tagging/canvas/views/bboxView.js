// @ts-check
import { formatConfidencePct } from '/app-static/js/shared/domain/image/confidence.js';

const LABEL_OFFSET_Y = -18;

export class BboxView extends Konva.Group {
    /** @type {import('@domain_image').UIBbox} */ uiBBox;
    /** @type {import('@konva').default.Rect} */ _rect;
    /** @type {import('@konva').default.Text} */ _text;
    /** @type {import('@konva').default.Text} */ _iconText;
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
      text: this._computeLabelText(uiBBox),
      fill: uiBBox.label?.color || 'grey',
      fontSize: 14,
      y: LABEL_OFFSET_Y,
      name: 'label'
    });

    this._iconText = new Konva.Text({
      text: this._computeIconText(uiBBox.tags),
      fontSize: 14,
      fill: 'white',
      stroke: 'rgba(0,0,0,0.75)',
      strokeWidth: 0.5,
      fillAfterStrokeEnabled: true,
      fontStyle: 'bold',
      x: 4,
      y: 4,
      name: 'icon',
      strokeScaleEnabled: false,
      listening: false,
    });

    // Per-box selection marker. The Transformer draws a single border around
    // the *whole* selection, which says nothing about which boxes are in it
    // once there is more than one — this is the only per-box feedback.
    this._selectionRect = new Konva.Rect({
      width: this.width(),
      height: this.height(),
      stroke: 'white',
      strokeWidth: 2,
      dash: [6, 4],
      name: 'selection-marker',
      strokeScaleEnabled: false,
      listening: false,
      visible: false,
    });

    this.add(this._rect);
    this.add(this._selectionRect);
    this.add(this._text);
    this.add(this._iconText);

    this._setupEvents();
  }

  /**
   * Resize the box body and everything that must track it.
   * @private
   * @param {number} width
   * @param {number} height
   */
  _setBoxSize(width, height) {
    this._rect.setAttrs({ width, height });
    this._selectionRect.setAttrs({ width, height });
  }

  /**
   * Show or hide this box's selection marker.
   * Called only by canvas/selection.js, which owns the selection set.
   * @param {boolean} on
   */
  setSelected(on) {
    this._selectionRect.visible(!!on);
    this.getLayer()?.batchDraw();
  }

  /**
   * Compute the label text, appending the persisted inference confidence
   * (e.g. "cat (82%)") when the bbox carries one.
   *
   * ⚠️ Only reloaded boxes carry `confidence`. A box that has just come back
   * from live inference has the percentage baked into `label.text` by
   * canvas/image.js instead, and no `confidence` — so exactly one of the two
   * paths applies and the suffix is never doubled.
   * @private
   * @param {import('@domain_image').UIBbox} uiBBox
   * @returns {string}
   */
  _computeLabelText(uiBBox) {
    const text = uiBBox.label?.text || 'Loading...';
    const pct = formatConfidencePct(uiBBox.confidence);
    return pct ? `${text} (${pct})` : text;
  }

  /**
   * Compute the icon string to display from the tags array.
   * Collects unique non-null icons and joins them.
   * @private
   * @param {import('@domain_image').UITag[] | undefined} tags
   * @returns {string}
   */
  _computeIconText(tags) {
    const icons = (tags ?? []).map(t => t.icon).filter(Boolean);
    return [...new Set(icons)].join('');
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
      this._iconText.scaleX(1 / scaleX);
      this._iconText.scaleY(1 / scaleY);
    });

    this.on('dragend transformend', () => {
      const scaleX = this.scaleX();
      const scaleY = this.scaleY();
      const newWidth = this.width() * scaleX;
      const newHeight = this.height() * scaleY;

      this.setAttrs({ width: newWidth, height: newHeight, scaleX: 1, scaleY: 1 });
      this._setBoxSize(newWidth, newHeight);
      this._rect.setAttrs({ scaleX: 1, scaleY: 1 });
      this._selectionRect.setAttrs({ scaleX: 1, scaleY: 1 });
      this._text.setAttrs({ scaleX: 1, scaleY: 1, y: LABEL_OFFSET_Y });
      this._iconText.setAttrs({ scaleX: 1, scaleY: 1 });

      if (this._viewport) {
        const sw = this._viewport.imageWidth * this._viewport.scale;
        const sh = this._viewport.imageHeight * this._viewport.scale;
        this.uiBBox.x = (this.x() - this._viewport.offsetX) / sw;
        this.uiBBox.y = (this.y() - this._viewport.offsetY) / sh;
        this.uiBBox.width = newWidth / sw;
        this.uiBBox.height = newHeight / sh;
        this.uiBBox.dirty = true;
      }

      // Re-sync the Transformer's handles to the post-normalization bounds.
      // The Transformer may have cached the pre-normalization geometry; calling
      // forceUpdate() here (synchronously, after attrs are committed) ensures
      // the anchor box reflects the true final size rather than snapping back.
      this.getLayer()?.findOne('Transformer')?.forceUpdate();
    });
  }

  /** @param {import('@image_tagging_types').CanvasRect} rect */
  updatePosition({ x, y, width, height }) {
    this.setAttrs({ x, y, width, height });
    this._setBoxSize(width, height);
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

  /**
   * Update the icon indicator from the current tags array.
   * @param {import('@domain_image').UITag[]} tags
   */
  refreshIcons(tags) {
    this._iconText.text(this._computeIconText(tags));
    this.getLayer()?.batchDraw();
  }

  /** @param {object} [config] */
  getClientRect(config) {
    return this._rect.getClientRect(config);
  }

  /**
   * Temporarily draws attention to this bbox with a colored glow that fades out.
   * Used to flag inference-conflict outcomes (duplicate skipped / replaced / kept-as-is).
   * @param {string} color
   * @param {number} [duration] seconds
   */
  flash(color, duration = 1.2) {
    const glow = new Konva.Rect({
      width: this.width(),
      height: this.height(),
      stroke: color,
      strokeWidth: 8,
      opacity: 0.9,
      listening: false,
      name: 'flash-glow',
      strokeScaleEnabled: false,
    });
    this.add(glow);
    glow.moveToTop();
    new Konva.Tween({
      node: glow,
      duration,
      opacity: 0,
      easing: Konva.Easings.EaseOut,
      onFinish: () => {
        glow.destroy();
        this.getLayer()?.batchDraw();
      },
    }).play();
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
    this._setBoxSize(w, h);
  }
}
