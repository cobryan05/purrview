// @ts-check
import { Field } from './Field.js';
import { TextField } from './TextField.js';
import { ColorSwatchField } from './ColorSwatchField.js';

/**
 * Predefined icon palette. First entry ('') means "no icon".
 * @type {string[]}
 */
const ICON_PALETTE = ['—', '?', '!', '★', '⚠', '✓', '✗'];
const ICON_NONE = null;

/**
 * A composite field for editing/displaying a tag (color + icon + name).
 *
 * getValue() returns { text, color, icon } where icon is null when none is selected.
 */
export class TagField extends Field {
  /**
   * @param {{ text?: string, color?: string, icon?: string|null, uuid?: string|null, placeholder?: string } & Record<string, any>} [params]
   */
  constructor({ text = '', color = '#cccccc', icon = null, uuid = null, ...rest } = {}) {
    super(rest);
    this._textField = new TextField({ value: text, placeholder: this._placeholder, onChange: (v) => { this._text = v; } });
    this._colorField = new ColorSwatchField({ color, onChange: (v) => { this._color = v; } });
    this._text = text;
    this._color = color;
    this._icon = icon;
    this._uuid = uuid;
  }

  // ─── Rendering ───────────────────────────────────────────────────────────

  async renderEdit() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.alignItems = 'flex-start';
    container.style.gap = '0.3em';

    // Row 1: color swatch | icon picker | name text field
    const row1 = document.createElement('div');
    row1.style.display = 'flex';
    row1.style.alignItems = 'center';
    row1.style.gap = '0.3em';
    row1.style.flexWrap = 'wrap';

    row1.appendChild(await this._colorField.renderEdit());
    row1.appendChild(this._renderIconPicker());
    row1.appendChild(await this._textField.renderEdit());

    container.appendChild(row1);

    // Row 2: UUID (read-only, below)
    if (this._uuid) {
      const uuidInfo = document.createElement('div');
      uuidInfo.textContent = `UUID: ${this._uuid}`;
      uuidInfo.style.fontSize = '0.75em';
      uuidInfo.style.color = '#888';
      uuidInfo.style.marginLeft = '0.25em';
      uuidInfo.style.userSelect = 'text';
      uuidInfo.style.fontFamily = 'monospace';
      uuidInfo.style.overflowWrap = 'anywhere';
      container.appendChild(uuidInfo);
    }

    return container;
  }

  async renderView() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.4em';

    container.appendChild(await this._colorField.renderView());

    if (this._icon) {
      const iconSpan = document.createElement('span');
      iconSpan.textContent = this._icon;
      iconSpan.style.fontWeight = 'bold';
      container.appendChild(iconSpan);
    }

    container.appendChild(await this._textField.renderView());
    return container;
  }

  /**
   * Build the icon picker widget — a compact trigger button that opens a
   * floating popup with preset palette buttons and a custom-character input.
   * @private
   * @returns {HTMLElement}
   */
  _renderIconPicker() {
    // ── wrapper (provides the `position:relative` anchor for the popup) ──
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'position:relative; display:inline-block;';
    wrapper.title = 'Tag icon (shown on bbox in canvas)';

    // ── trigger button ────────────────────────────────────────────────────
    const trigger = document.createElement('button');
    trigger.type = 'button';
    trigger.style.cssText = `
      height: 2em;
      padding: 0 0.4em;
      cursor: pointer;
      border: 1px solid var(--pico-form-element-border-color, #888);
      border-radius: 3px;
      font-size: 1em;
      line-height: 1;
      background: var(--pico-form-element-background-color, #f5f5f5);
      color: var(--pico-form-element-color, #333);
      white-space: nowrap;
    `;
    const triggerLabel = () => this._icon ?? 'Icon';
    trigger.textContent = triggerLabel();

    // ── popup ─────────────────────────────────────────────────────────────
    const popup = document.createElement('div');
    popup.style.cssText = `
      display: none;
      position: absolute;
      top: 100%;
      left: 0;
      z-index: 9999;
      background: var(--pico-dropdown-background-color, #fff);
      color: var(--pico-dropdown-color, #333);
      border: 1px solid var(--pico-dropdown-border-color, #888);
      border-radius: 4px;
      padding: 4px 6px;
      white-space: nowrap;
      box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    `;

    let popupOpen = false;
    /** @type {((e: MouseEvent) => void) | null} */
    let outsideListener = null;

    const closePopup = () => {
      popup.style.display = 'none';
      popupOpen = false;
      if (outsideListener) {
        document.removeEventListener('mousedown', outsideListener);
        outsideListener = null;
      }
    };

    const openPopup = () => {
      popup.style.display = 'flex';
      popup.style.gap = '4px';
      popup.style.alignItems = 'center';
      popupOpen = true;
      outsideListener = (e) => {
        if (!wrapper.isConnected || !wrapper.contains(/** @type {Node} */ (e.target))) closePopup();
      };
      document.addEventListener('mousedown', outsideListener);
    };

    trigger.addEventListener('mousedown', (e) => {
      e.preventDefault();
      popupOpen ? closePopup() : openPopup();
    });

    // ── palette buttons inside popup ──────────────────────────────────────
    ICON_PALETTE.forEach((symbol) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = symbol;
      btn.style.cssText = `
        padding: 1px 5px;
        cursor: pointer;
        border: 1px solid var(--pico-form-element-border-color, #888);
        border-radius: 3px;
        font-size: 1em;
        line-height: 1.4;
        min-width: 1.8em;
        background: var(--pico-form-element-background-color, #f5f5f5);
        color: var(--pico-form-element-color, #333);
      `;
      btn.addEventListener('mousedown', (e) => {
        // prevent the document mousedown from closing the popup before click fires
        e.stopPropagation();
      });
      btn.addEventListener('click', () => {
        this._icon = symbol === '—' ? ICON_NONE : symbol;
        trigger.textContent = triggerLabel();
        customInput.value = '';
        closePopup();
      });
      popup.appendChild(btn);
    });

    // ── separator ─────────────────────────────────────────────────────────
    const sep = document.createElement('span');
    sep.textContent = '|';
    sep.style.cssText = 'color:#aaa; margin:0 2px;';
    popup.appendChild(sep);

    // ── custom-character input ────────────────────────────────────────────
    const customInput = document.createElement('input');
    customInput.type = 'text';
    customInput.placeholder = 'custom';
    customInput.maxLength = 2;
    customInput.style.cssText = `
      width: 3.2em;
      padding: 1px 3px;
      font-size: 0.9em;
      border: 1px solid #888;
      border-radius: 3px;
      text-align: center;
    `;
    // Pre-fill if the current icon isn't in the preset palette
    if (this._icon && !ICON_PALETTE.slice(1).includes(this._icon)) {
      customInput.value = this._icon;
    }
    customInput.addEventListener('mousedown', (e) => e.stopPropagation());
    customInput.addEventListener('input', () => {
      const val = customInput.value.trim();
      this._icon = val || ICON_NONE;
      trigger.textContent = triggerLabel();
    });
    customInput.addEventListener('blur', () => closePopup());
    popup.appendChild(customInput);

    wrapper.appendChild(trigger);
    wrapper.appendChild(popup);
    return wrapper;
  }

  // ─── Value ───────────────────────────────────────────────────────────────

  getValue() {
    return {
      text: this._textField.getValue(),
      color: this._colorField.getValue(),
      icon: this._icon,
    };
  }
}
