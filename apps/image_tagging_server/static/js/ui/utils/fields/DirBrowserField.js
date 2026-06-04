import { Field } from './Field.js';
import { openFileBrowserModal } from '/app-static/js/ui/utils/openFileBrowserModal.js';

/**
 * A Field that lets the user pick a directory via a modal file-browser instead of
 * typing a path by hand.  Drop-in replacement for TextField when the schema type is
 * "directory".
 */
export class DirBrowserField extends Field {
  constructor({ ...rest }) {
    super(rest);
    this._input = null;
  }

  setValue(value) {
    this._value = value;
    if (this._input) this._input.value = value ?? '';
  }

  async renderEdit() {
    const wrapper = document.createElement('div');
    wrapper.className = 'fb-dir-field';

    // Read-only text input showing the current value
    const input = document.createElement('input');
    input.type = 'text';
    input.readOnly = true;
    input.value = this._value ?? '';
    input.placeholder = this._placeholder || 'Select a directory…';
    input.style.cursor = 'pointer';
    this._input = input;

    // Clicking the input is equivalent to clicking Browse
    input.addEventListener('click', () => browseBtn.click());

    const browseBtn = document.createElement('button');
    browseBtn.type = 'button';
    browseBtn.textContent = 'Browse…';
    browseBtn.addEventListener('click', async () => {
      const picked = await openFileBrowserModal({
        initialPath: this._value || '/',
        allowCreateDir: true,
        showDbStatus: false,
        showThumbnails: false,
      });
      if (picked !== null) {
        this.setValue(picked);
        this._onChange?.(picked);
      }
    });

    wrapper.append(input, browseBtn);
    return wrapper;
  }

  async renderView() {
    const span = document.createElement('span');
    span.textContent = this._value || '(none)';
    return span;
  }

  getValue() {
    return this._value ?? '';
  }
}
