/**
 * Batch size / image size / max epochs / early stopping patience fields.
 *
 * Used by the Training Config form (ui/trainingConfigs.js, per-target
 * training), which writes into a flat hyperparams_json blob that's merged
 * straight into whichever trainer's job_config. It was shared with the Model
 * Config "Train all → bundle" flow until that was removed.
 *
 * Gotcha (do not "simplify" away): the object-detection trainer reads the
 * hyperparam key `patience`, but the classifier trainers (binary/multi-label)
 * read `early_stopping_patience`. One dropdown here writes BOTH keys with the
 * same value so it works regardless of which trainer ends up consuming it;
 * each trainer ignores keys it doesn't read.
 *
 * Augmentation preset (off/light/standard/heavy) is a training-config option
 * shared across every trainer: the classifier trainers (binary/multi-label/
 * crop) apply it to their albumentations pipeline (see
 * apps/training_server/augmentation.py), and the Ultralytics YOLO trainer
 * maps it onto the real hsv/degrees/mosaic/mixup kwargs `model.train()`
 * accepts. LibreYOLO9 (`object_detection` trainer_type) does not read this
 * key — its `train()` has no confirmed augmentation kwargs to map onto.
 *
 * The "Remember exact image list" tickbox is the one field here that is NOT a
 * hyperparameter — it rides hyperparams_json only because that is the training
 * config's sole free-form blob. TrainModelTask pops it back out and threads it
 * explicitly (see its `record_image_manifest` handling); the trainers read it in
 * exactly one place, `trainers/base.py:image_manifest_fields`.
 *
 * VRAM-aware suggestions: when the caller knows the target architecture
 * (object-detection only — see vramHyperparams.js), an extra "GPU VRAM"
 * select appears above Batch size / Image size. Choosing a tier PREFILLS
 * (never locks) those two inputs from the per-architecture table; fields
 * stay fully editable afterward, same as manual entry. If no architecture is
 * known (no node selected yet) or the architecture has no table entry
 * (classifiers), the select is omitted entirely rather than guessing.
 */

import { fetchTrainingGpus } from '/app-static/js/shared/api/trainingGpus.js';
import { makeFieldRow } from '/app-static/js/ui/utils/index.js';
import { VRAM_TIERS_GB, getVramTableForArchitecture } from '/app-static/js/ui/utils/vramHyperparams.js';

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
 *   (batch_size, imgsz, max_epochs, early_stopping_patience|patience,
 *   record_image_manifest).
 * @param {() => void} [opts.onChange] - fired on any field's `input` event.
 * @param {?string} [opts.architecture] - object-detection architecture string
 *   (e.g. 'yolov8s', 'yolo_v9_c') used to look up the VRAM suggestion table.
 *   Pass null/omit when the architecture isn't known yet (e.g. no node
 *   selected) — the VRAM select is hidden rather than guessing.
 * @returns {{ rows: HTMLElement[], getHyperparamsJson: () => object,
 *             setArchitecture: (architecture: ?string) => void }}
 *   `rows` are ready-made field rows (label + input) to append in order;
 *   `getHyperparamsJson()` returns only the keys the user actually filled in;
 *   `setArchitecture()` lets a caller update the VRAM table after the target
 *   architecture becomes known/changes post-render (e.g. trainingConfigs.js's
 *   node selector).
 */
export function createHyperparamsFieldset({ initial = {}, onChange, architecture = null } = {}) {
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

  // Opt-OUT, not opt-in: every model trained so far records which images it saw,
  // and the key is absent from every existing hyperparams_json, so anything other
  // than "absent means on" would silently strip the capability from saved configs.
  const recordManifestInput = document.createElement('input');
  recordManifestInput.type = 'checkbox';
  recordManifestInput.checked = initial.record_image_manifest !== false;
  recordManifestInput.dataset.testid = 'hyperparams-record-image-manifest';
  recordManifestInput.title =
    'Write the exact train/validation image lists into the trained model’s sidecar JSON, '
    + 'so you can see later which images produced it. Untick to keep the list out of the artifact.';

  const augmentationSelect = document.createElement('select');
  augmentationSelect.style.cssText = 'width: auto; flex-shrink: 0;';
  augmentationSelect.title =
    'How aggressively to distort training images (crop/flip/rotate/colour jitter). '
    + 'Higher settings help when classes look very similar (e.g. two cats) and the model overfits early. '
    + 'Ignored by the LibreYOLO9 object-detection trainer.';
  for (const [value, label] of [
    ['off', 'Off'],
    ['light', 'Light'],
    ['standard', 'Standard (default)'],
    ['heavy', 'Heavy'],
  ]) {
    const o = document.createElement('option');
    o.value = value;
    o.textContent = label;
    augmentationSelect.appendChild(o);
  }
  augmentationSelect.value = initial.augmentation ?? 'standard';

  if (onChange) {
    for (const input of [batchSizeInput, imgszInput, maxEpochsInput, patienceInput]) {
      input.addEventListener('input', onChange);
    }
    recordManifestInput.addEventListener('change', onChange);
    augmentationSelect.addEventListener('change', onChange);
  }

  // ── VRAM tier select (prefill-only; never disables the fields above) ──────
  const vramSelect = document.createElement('select');
  // Pico sets `select { width: 100% }` globally; inside the flex
  // .panel-field-row this would squeeze the row (see NOTES.md gotcha).
  vramSelect.style.cssText = 'width: auto; flex-shrink: 0;';
  const vramBlankOpt = document.createElement('option');
  vramBlankOpt.value = '';
  vramBlankOpt.textContent = '— none —';
  vramSelect.appendChild(vramBlankOpt);
  for (const gb of VRAM_TIERS_GB) {
    const o = document.createElement('option');
    o.value = String(gb);
    o.textContent = `${gb} GB`;
    vramSelect.appendChild(o);
  }
  vramSelect.addEventListener('change', () => {
    const gb = parseInt(vramSelect.value, 10);
    const table = getVramTableForArchitecture(_architecture);
    const suggestion = table?.[gb];
    if (!suggestion) return;
    batchSizeInput.value = String(suggestion.batch_size);
    imgszInput.value = String(suggestion.imgsz);
    onChange?.();
  });

  // ── GPU select (per-run override of the deployment's TRAINING_GPU) ────────
  // Hidden unless the training host exposes more than one card: with one card there
  // is nothing to choose, and the deployment default already points at it. Like
  // `record_image_manifest` this is not a hyperparameter — it changes nothing about
  // the model — but it rides hyperparams_json because that is the training config's
  // only free-form blob (the trainers read it as `gpu_device`).
  const gpuSelect = document.createElement('select');
  gpuSelect.style.cssText = 'width: auto; flex-shrink: 0;';
  gpuSelect.dataset.testid = 'hyperparams-gpu-device';
  const gpuDefaultOpt = document.createElement('option');
  gpuDefaultOpt.value = '';
  gpuDefaultOpt.textContent = '— deployment default —';
  gpuSelect.appendChild(gpuDefaultOpt);
  if (onChange) gpuSelect.addEventListener('change', onChange);

  const gpuRow = makeFieldRow('GPU', gpuSelect);
  gpuRow.title =
    'Which GPU this run trains on. Leave at the default unless you want to keep a card '
    + 'free — a long run and a concurrent inference load on the same card is what makes '
    + 'inference fall back to the CPU.';
  gpuRow.style.display = 'none'; // shown below only if the host has a choice to offer

  // Deliberately not awaited: the fieldset is synchronous (its rows are appended by
  // the caller immediately), so the options fill in when the training host answers.
  // A host that is down or has no cards leaves the row hidden, which is the same
  // outcome as having nothing to choose.
  fetchTrainingGpus().then(({ gpus, defaultCudaOrdinal }) => {
    if (gpus.length < 2) return;
    const fallback = gpus.find(g => g.cuda_ordinal === defaultCudaOrdinal);
    if (fallback) gpuDefaultOpt.textContent = `— deployment default (${fallback.label}) —`;
    for (const gpu of gpus) {
      const o = document.createElement('option');
      // The CUDA ordinal, not the physical id: CUDA_VISIBLE_DEVICES renumbers from
      // zero, so the label a human reads and the value a trainer needs differ.
      o.value = String(gpu.cuda_ordinal);
      o.textContent = gpu.label;
      gpuSelect.appendChild(o);
    }
    const preselect = initial.gpu_device;
    if (preselect !== undefined && preselect !== null && String(preselect) !== '') {
      gpuSelect.value = String(preselect);
    }
    gpuRow.style.display = '';
  });

  const vramRow = makeFieldRow('GPU VRAM', vramSelect);
  vramRow.title = 'Pick your GPU VRAM to prefill batch size / image size with a suggested starting point. Fields stay editable.';
  vramRow.style.display = 'none'; // shown by _refreshVramVisibility() once an architecture resolves

  let _architecture = architecture;

  function _refreshVramVisibility() {
    const hasTable = !!getVramTableForArchitecture(_architecture);
    vramRow.style.display = hasTable ? '' : 'none';
    if (!hasTable) vramSelect.value = '';
  }
  _refreshVramVisibility();

  /** Update the architecture used for VRAM lookups (e.g. once a node/arch is selected). */
  function setArchitecture(newArchitecture) {
    _architecture = newArchitecture ?? null;
    _refreshVramVisibility();
  }

  const rows = [
    gpuRow,
    vramRow,
    makeFieldRow('Batch size', batchSizeInput),
    makeFieldRow('Image size', imgszInput),
    makeFieldRow('Max epochs', maxEpochsInput),
    makeFieldRow('Early stopping patience', patienceInput),
    makeFieldRow('Augmentation', augmentationSelect),
    makeFieldRow('Remember exact image list', recordManifestInput),
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
    // Only written when opting OUT — an absent key means "record", which is what
    // every already-saved config (and the trainers' own default) means today.
    if (!recordManifestInput.checked) hyperparamsJson.record_image_manifest = false;
    hyperparamsJson.augmentation = augmentationSelect.value;
    // Only written when a card is actually chosen — an absent key means "use the
    // deployment default", which is what every already-saved config means.
    if (gpuSelect.value !== '') hyperparamsJson.gpu_device = gpuSelect.value;
    return hyperparamsJson;
  }

  return { rows, getHyperparamsJson, setArchitecture };
}
