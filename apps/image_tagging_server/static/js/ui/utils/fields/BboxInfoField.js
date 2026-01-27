import { SchemaField } from '././SchemaField.js';
import { Field } from './Field.js';
import { renderBoxed } from '/app-static/js/ui/utils/index.js';
// Takes options as an array of strings or objects
// Each object can have 'key', 'text', and 'color' properties

export class BboxInfoField extends Field {
  /**
   * @param {Object} [params={}] - Field initialization params.
   */
  constructor(params = {}) {
    super(params);
    /** @type {SchemaField} */
    this._schemaField;
    /** @type {import('@app_types').BboxGroup} */
    this._bboxGroup;
  }


  /**
   * @typedef {Object} BboxInfoFieldCreateOptions
   * @property {import('@app_types').BboxGroup} bboxGroup
   * @property {import('@web_api').ClassMetadata[]} classes
   * @property {any[]} [value] - Optional initial selected values.
   *
   * @property {Object.<string, any>} [rest] - Additional parameters passed to Field.
   */

  /**
   * Create a BboxInfoField instance.
   *
   * @param {BboxInfoFieldCreateOptions} options
   * @returns {Promise<BboxInfoField>}
   */
  static async create({ bboxGroup, classes, value = [], ...rest }) {
    const instance = new BboxInfoField({ value, ...rest });
    instance._bboxGroup = bboxGroup;

    const schema = {
      "classUuid": {
        "label": "Class",
        "type": "class",
        "required": true,
        "description": "Class of selected bbox",
      },
      "tagUuids": {
        "label": "Tags",
        "type": "tags",

        "description": "Tag applied to the bbox"
      }
    };

    const schema_values = {
      classUuid: instance._bboxGroup.metadata.runtimeBboxInfo.classUuid,
      tagUuids: instance._bboxGroup.metadata.runtimeBboxInfo.tagUuids
    };
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
    return renderBoxed(span, { fullWidth: true });
  }

  getValue() {
    return { bbox_info: this._schemaField.getValue() };
  }
}
