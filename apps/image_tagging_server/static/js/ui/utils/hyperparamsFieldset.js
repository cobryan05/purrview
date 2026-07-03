/**
 * Batch size / image size / max epochs / early stopping patience fields.
 *
 * Shared by the Training Config form (ui/trainingConfigs.js, per-target
 * training) and the Model Config "Train all → bundle" flow
 * (ui/modelConfigs.js, TrainBundleTask), both of which write into a flat
 * hyperparams_json blob that's merged straight into whichever trainer's
 * job_config.
 *
 * Gotcha (do not "simplify" away): the object-detection trainer reads the
 * hyperparam key `patience`, but the classifier trainers (binary/multi-label)
 * read `early_stopping_patience`. One dropdown here writes BOTH keys with the
 * same value so it works regardless of which trainer ends up consuming it;
 * each trainer ignores keys it doesn't read.
 */

import { makeFieldRow } from '/app-static/js/ui/utils/index.js';

function _numInput({ placeholder, initial, min = 1, title }) {
  const input = document.createElement('input');
  input.type = 'number';
  input.min = String(min);
  input.step = '1';
  input.placeholder = placeholder;
  input.value = initial != null ? String(initial) : '';
  if (title) input.title = title;
  return input;
}

/**
 * @param {Object} [opts]
 * @param {Object} [opts.initial] - snake_case hyperparams_json to prefill from
 *   (batch_size, imgsz, max_epochs, early_stopping_patience|patience).
 * @param {() => void} [opts.onChange] - fired on any field's `input` event.
 * @returns {{ rows: HTMLElement[], getHyperparamsJson: () => object }}
 *   `rows` are ready-made field rows (label + input) to append in order;
 *   `getHyperparamsJson()` returns only the keys the user actually filled in.
 */
export function createHyperparamsFieldset({ initial = {}, onChange } = {}) {
  const batchSizeInput = _numInput({
    placeholder: 'default 16–32', initial: initial.batch_size,
    title: 'Lower this first if you hit a GPU out-of-memory error.',
  });
  const imgszInput = _numInput({
    placeholder: 'default 640', initial: initial.imgsz,
    title: 'Input image size (object detection only). Lower values use less VRAM.',
  });
  const maxEpochsInput = _numInput({ placeholder: 'default 50', initial: initial.max_epochs });
  const patienceInput = _numInput({
    placeholder: 'default 10–20',
    initial: initial.early_stopping_patience ?? initial.patience,
    title: 'Stop early after this many epochs without improvement.',
  });

  if (onChange) {
    for (const input of [batchSizeInput, imgszInput, maxEpochsInput, patienceInput]) {
      input.addEventListener('input', onChange);
    }
  }

  const rows = [
    makeFieldRow('Batch size', batchSizeInput),
    makeFieldRow('Image size', imgszInput),
    makeFieldRow('Max epochs', maxEpochsInput),
    makeFieldRow('Early stopping patience', patienceInput),
  ];

  function getHyperparamsJson() {
    const hyperparamsJson = {};
    const batchSize = parseInt(batchSizeInput.value, 10);
    if (!isNaN(batchSize)) hyperparamsJson.batch_size = batchSize;
    const imgsz = parseInt(imgszInput.value, 10);
    if (!isNaN(imgsz)) hyperparamsJson.imgsz = imgsz;
    const maxEpochs = parseInt(maxEpochsInput.value, 10);
    if (!isNaN(maxEpochs)) hyperparamsJson.max_epochs = maxEpochs;
    const patience = parseInt(patienceInput.value, 10);
    if (!isNaN(patience)) {
      hyperparamsJson.early_stopping_patience = patience;
      hyperparamsJson.patience = patience;
    }
    return hyperparamsJson;
  }

  return { rows, getHyperparamsJson };
}
