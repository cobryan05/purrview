import { Logger } from '../logging.js';
import { Field } from './Field.js';
import { LabelDropDownField } from './LabelDropDownField.js';
import { ArrayField } from './ArrayField.js';
import { TextField } from './TextField.js';
import { fetchClassByUuid, fetchClasses, fetchTags, renderBoxed } from '/app-static/js/ui/utils/index.js';
import { fieldFactories } from './Factories.js';
import { SchemaField } from '././SchemaField.js';
// Takes options as an array of strings or objects
// Each object can have 'key', 'text', and 'color' properties

export class BboxInfoField extends Field {
  constructor(params = {}) {
    super(params);
    /** @type {SchemaField} */
    this._schemaField = /** @type {any} */ (null);
  }

  static async create({ bbox, classes, value = [], ...rest } = {}) {
    const instance = new BboxInfoField({ value, ...rest });
    instance._metadata = bbox.metadata;

    const schema = {
      "class_name": {
        "label": "Class",
        "type": "class",
        "required": true,
        "description": "Class of selected bbox",
      },
      "tags": {
        "label": "Tags",
        "type": "tags",
        "description": "Tag applied to the bbox"
      }
    };

    const schema_values = { class_name: instance._metadata.classUuid };
    instance._schemaField = await SchemaField.create({ schema: schema ?? {}, values: schema_values });
    return instance;
  }

  async renderEdit() {
    const span = document.createElement('span');
    span.appendChild(document.createElement('br'));
    span.appendChild(await this._schemaField.renderEdit());
    return renderBoxed(span);
  }

  async renderView() {
    const span = document.createElement('span');
    span.appendChild(document.createElement('br'));
    span.appendChild(await this._schemaField.renderView());
    return renderBoxed(span, {fullWidth: true});
  }

  getValue() {
    return this._schemaField.getValue();
  }
}
