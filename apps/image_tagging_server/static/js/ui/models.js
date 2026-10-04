import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { filterVisibleLabels, labelPickerText } from '/app-static/js/shared/activeModel.js';
import {
  associateLabel,
  clearModelsCache,
  deleteModel,
  detectModel,
  fetchCuratedModels,
  fetchModelDevice,
  fetchModelsLabelMappings,
  fetchModelsList,
  importModelUpload,
  isTrainingServerReachable,
  notifyModelsChanged,
  setModelDevice,
  stageSourceModel,
} from '/app-static/js/shared/api/models.js';
import { DropDownField, EditableField, LabelDropDownField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, Logger, showPaneError, toast } from '/app-static/js/ui/utils/index.js';
import { openTrainingDataModal } from '/app-static/js/ui/trainingDataModal.js';
import { createTaskRunnerSection } from '/app-static/js/ui/utils/taskRunnerSection.js';

export function renderModelLabelAssignments({
  target = "model-labels-box",
  editable = false,
  preselectedModel = null
} = {}) {
  const container = document.getElementById(target);
  container.innerHTML = '';

  const modelSelectDiv = document.createElement('div');
  modelSelectDiv.style.display = 'inline-block';
  container.appendChild(modelSelectDiv);
  // --- Header row: model selection dropdown ---
  fetchModelsList().then(models => {
    const modelDropdownRow = createGenericRow({
      field: new DropDownField({
        value: preselectedModel,
        editMode: true,
        options: models,
        placeholder: 'Choose a model...',
        onChange: value => {
          if (!value) return;
          if (container.querySelector('.emoji-button[title="Save"]') &&
              !window.confirm('You have unsaved changes. Discard them?')) return;
          setTimeout(() => {
            renderModelLabelAssignments({ target, editable, preselectedModel: value });
          }, 0);
        }
      }),
      rightButtons: [
        {
          text: 'Refresh',
          emoji: '🔄',
          onClick: () => {
            if (container.querySelector('.emoji-button[title="Save"]') &&
                !window.confirm('You have unsaved changes. Discard them?')) return;
            clearModelsCache();
            renderModelLabelAssignments({ target, editable, preselectedModel: preselectedModel });
          }
        },
        ...(preselectedModel ? [{
          text: 'Browse training data',
          emoji: '🖼️',
          onClick: () => openTrainingDataModal(preselectedModel),
        }] : []),
        ...(preselectedModel ? [{
          text: 'History',
          emoji: '\u{1F553}',
          // Emoji buttons carry no text for a selector to match — see createGenericRow.
          testid: 'model-history-open',
          onClick: async () => {
            const { showModelVersionsModal } = await import('/app-static/js/ui/modelVersionsModal.js');
            await showModelVersionsModal(preselectedModel);
          },
        }] : []),
        ...(preselectedModel ? [{
          text: 'Delete model',
          emoji: '🗑️',
          onClick: async () => {
            if (!window.confirm(
              `Delete model "${preselectedModel}"? Its previous versions are deleted too. `
              + 'This cannot be undone.')) return;
            try {
              await deleteModel(preselectedModel);
              notifyModelsChanged();
              toast(`Deleted model "${preselectedModel}"`);
              renderModelLabelAssignments({ target, editable, preselectedModel: null });
            } catch (err) {
              showPaneError(modelSelectDiv, err.message);
            }
          }
        }] : []),
      ]
    });
    modelSelectDiv.appendChild(modelDropdownRow);
  }).catch(err => {
    showPaneError(modelSelectDiv, err.message);
  });

  // --- Import model (cold-start): upload a .onnx or download a curated base model ---
  const importSectionDiv = document.createElement('div');
  importSectionDiv.style.marginTop = '0.5em';
  container.appendChild(importSectionDiv);

  const importToggleBtn = document.createElement('button');
  importToggleBtn.type = 'button';
  importToggleBtn.textContent = '+ Import model';
  importToggleBtn.className = 'secondary panel-btn';
  importSectionDiv.appendChild(importToggleBtn);

  const importFormDiv = document.createElement('div');
  importFormDiv.style.display = 'none';
  importFormDiv.style.marginTop = '0.5em';
  importSectionDiv.appendChild(importFormDiv);

  importToggleBtn.onclick = () => {
    const showing = importFormDiv.style.display !== 'none';
    if (showing) {
      importFormDiv.style.display = 'none';
      importFormDiv.innerHTML = '';
    } else {
      importFormDiv.style.display = '';
      renderImportModelForm(importFormDiv, { target, editable });
    }
  };

  if (!preselectedModel) return;

  const deviceDiv = document.createElement('div');
  deviceDiv.style.display = 'flex';
  deviceDiv.style.alignItems = 'center';
  deviceDiv.style.gap = '0.5em';
  container.appendChild(deviceDiv);
  fetchModelDevice(preselectedModel).then(async device => {
    const label = document.createElement('span');
    label.textContent = 'Device:';
    label.style.fontWeight = 'bold';
    deviceDiv.appendChild(label);

    const dropdown = new DropDownField({
      value: device,
      editMode: true,
      options: ['auto', 'gpu', 'cpu'],
      onChange: async value => {
        if (!value || value === device) return;
        try {
          await setModelDevice(preselectedModel, value);
        } catch (err) {
          toast(err.message, 5000, 'error');
        }
        renderModelLabelAssignments({ target, editable, preselectedModel });
      },
    });
    deviceDiv.appendChild(await dropdown.renderEdit());
  }).catch(err => {
    showPaneError(deviceDiv, err.message);
  });

  const labelAssignmentDiv = document.createElement('div');
  container.appendChild(labelAssignmentDiv);

  // --- Load model-label assignments ---
  Promise.all([
    fetchModelsLabelMappings({modelNames: preselectedModel}),
    fetchLabels()
  ]).then(([labelMap, allLabels]) => {
    // Convert labels to the format required by DropDownField
    // Per row, not once: each model class keeps its own assigned label in the options
    // even while dynamic labels are hidden, so the dropdown shows the real assignment
    // instead of falling back to its first entry.
    const labelItemsFor = (assignedUuid) =>
      filterVisibleLabels(allLabels.values(), assignedUuid).map(label => ({
        key: label.uuid,
        text: labelPickerText(label),
        color: label.color || '#cccccc',
      }));

    labelMap.forEach((assignedUuid, modelClass) => {
      const assignedLabel = allLabels.get(assignedUuid);
      const classKey = assignedLabel ? { key: assignedLabel.uuid, text: assignedLabel.name } : assignedUuid
      const deleteButton = {
        text: 'Delete',
        emoji: '🗑️',
        onClick: async () => {
          try {
            await associateLabel(preselectedModel, modelClass, null);
          } catch (err) {
            toast(err.message, 5000, 'error');
          }
          renderModelLabelAssignments({ target, editable, preselectedModel: preselectedModel });
        },
      };
      const assignmentRow = createGenericRow({
        field: new EditableField({
          field: new LabelDropDownField({
            options: labelItemsFor(assignedUuid),
            labelText: modelClass,
            value: classKey
          }),
          editable: true,
          editMode: false,
          buttonsLast: true,
          onSave: async labelUuid => {
            // Save the selected class
            if (labelUuid) {
              await associateLabel(preselectedModel, modelClass, labelUuid);
            }
            renderModelLabelAssignments({ target, editable, preselectedModel: preselectedModel });
          },
          onCancel: () => {
            renderModelLabelAssignments({ target, editable, preselectedModel: preselectedModel });
          },
        }),
        rightButtons: assignedUuid ? [deleteButton] : []
      });

      labelAssignmentDiv.appendChild(assignmentRow);
    });
  }).catch(err => {
    showPaneError(labelAssignmentDiv, err.message);
  });
}

/**
 * Render the "Import model" inline form: a mode toggle between Upload (a local
 * .onnx file, no network) and Curated (a background-downloaded base model),
 * per Phase 5 of the filter-chain plan.
 * @param {HTMLElement} formDiv - container to render into (cleared first).
 * @param {{target: string, editable: boolean}} rerenderArgs - passed back into
 *   renderModelLabelAssignments after a successful import.
 */
function renderImportModelForm(formDiv, { target, editable }) {
  formDiv.innerHTML = '';

  let mode = 'upload';

  const modeRow = document.createElement('div');
  modeRow.style.cssText = 'display:flex; gap:0.5em; margin-bottom:0.5em;';
  formDiv.appendChild(modeRow);

  const modeBody = document.createElement('div');
  formDiv.appendChild(modeBody);

  const uploadBtn = document.createElement('button');
  uploadBtn.type = 'button';
  uploadBtn.textContent = 'Upload';
  const curatedBtn = document.createElement('button');
  curatedBtn.type = 'button';
  curatedBtn.textContent = 'Curated';
  modeRow.append(uploadBtn, curatedBtn);

  function setMode(newMode) {
    mode = newMode;
    uploadBtn.className = mode === 'upload' ? 'panel-btn' : 'secondary panel-btn';
    curatedBtn.className = mode === 'curated' ? 'panel-btn' : 'secondary panel-btn';
    modeBody.innerHTML = '';
    if (mode === 'upload') {
      modeBody.appendChild(buildUploadForm({ target, editable }));
    } else {
      modeBody.appendChild(buildCuratedForm({ target, editable }));
    }
  }

  uploadBtn.onclick = () => setMode('upload');
  curatedBtn.onclick = () => setMode('curated');
  setMode('upload');
}

function _afterImportSuccess(target, editable, modelName) {
  // notify, not just clear: the canvas inference panel and Review's model picker
  // are built elsewhere and stay built, so clearing the cache alone would leave a
  // model you just imported invisible to them until a reload.
  notifyModelsChanged();
  renderModelLabelAssignments({ target, editable, preselectedModel: modelName });
}

function buildUploadForm({ target, editable }) {
  const wrap = document.createElement('div');

  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = '.onnx,.pt,.pth,.torchscript';
  fileInput.style.display = 'block';
  fileInput.style.marginBottom = '0.5em';

  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.placeholder = 'Model name (no extension)';
  nameInput.style.display = 'block';
  nameInput.style.marginBottom = '0.5em';

  const overwriteLabel = document.createElement('label');
  const overwriteCheckbox = document.createElement('input');
  overwriteCheckbox.type = 'checkbox';
  overwriteLabel.appendChild(overwriteCheckbox);
  overwriteLabel.append(' Overwrite if exists');
  overwriteLabel.style.display = 'block';
  overwriteLabel.style.marginBottom = '0.5em';

  wrap.append(fileInput, nameInput, overwriteLabel);

  // A single "Import" action: detects the file's format and either uploads
  // it directly (already .onnx) or — if it's a convertible format such as
  // pytorch — confirms with the user and runs a background ConvertModelTask
  // on the training server, reusing createTaskRunnerSection's SSE plumbing.
  const { el: runnerSection } = createTaskRunnerSection({
    buttonLabel: 'Import',
    runningLabel: 'Converting…',
    buildStartArgs: async () => {
      const file = fileInput.files?.[0];
      const modelName = nameInput.value.trim();
      if (!file) {
        toast('Choose a model file first', 3000, 'warning');
        return null;
      }
      if (!modelName) {
        toast('Enter a model name', 3000, 'warning');
        return null;
      }

      try {
        const headerBase64 = await _fileToBase64(file.slice(0, 512));
        const { isOnnx, conversionSupported, format } = await detectModel({
          filename: file.name,
          headerBase64,
        });

        if (isOnnx) {
          // Already ONNX — do the existing synchronous upload right here;
          // no background task needed.
          const onnxBase64 = await _fileToBase64(file);
          const result = await importModelUpload({
            modelName,
            onnxBase64,
            overwrite: overwriteCheckbox.checked,
          });
          toast(`Imported model '${result.modelName}' (${result.classes.length} classes)`, 4000, 'success');
          _afterImportSuccess(target, editable, result.modelName);
          return null;
        }

        if (!conversionSupported) {
          toast(`Unsupported model format (${format}); cannot import or convert`, 5000, 'error');
          return null;
        }

        if (!window.confirm(`Detected a ${format} model. Convert it to ONNX on the training server?`)) {
          return null;
        }

        const sourceBase64 = await _fileToBase64(file);
        const stagingId = await stageSourceModel({ sourceBase64 });

        return {
          typename: 'ConvertModelTask',
          name: `Convert model: ${modelName}`,
          params: {
            staging_id: stagingId,
            model_name: modelName,
            overwrite: overwriteCheckbox.checked,
          },
        };
      } catch (err) {
        toast(err.message || 'Failed to import model', 5000, 'error');
        return null;
      }
    },
    onSettled: finalTaskInfo => {
      const status = finalTaskInfo?.instance?.status;
      const modelName = nameInput.value.trim();
      if (status === 'completed') {
        toast(`Imported model '${modelName}'`, 4000, 'success');
        _afterImportSuccess(target, editable, modelName);
      } else if (status === 'error') {
        toast('Model conversion failed — see task log', 5000, 'error');
      }
    },
  });
  wrap.appendChild(runnerSection);

  return wrap;
}

function _fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      // dataURL is "data:<mime>;base64,<data>" — strip the prefix.
      const result = /** @type {string} */ (reader.result);
      const comma = result.indexOf(',');
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.onerror = () => reject(reader.error || new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
}

function buildCuratedForm({ target, editable }) {
  const wrap = document.createElement('div');

  const dropdownDiv = document.createElement('div');
  dropdownDiv.style.marginBottom = '0.5em';
  // The pane already holds the "Choose a model..." picker, so a bare
  // `#model-labels-box select` matches two elements — this names the curated one.
  dropdownDiv.dataset.testid = 'curated-model-picker';
  wrap.appendChild(dropdownDiv);

  const conversionBadge = document.createElement('span');
  conversionBadge.textContent = 'via training server';
  conversionBadge.title = 'This model has no directly-published ONNX export; it will be converted from a .pt '
    + 'checkpoint by the training server before import.';
  conversionBadge.style.cssText = 'display:none; font-size:0.8em; padding:0.1em 0.5em; border-radius:1em; '
    + 'background:var(--pico-mark-background-color, #ffe58f); margin-bottom:0.5em;';
  wrap.appendChild(conversionBadge);

  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.placeholder = 'Model name (no extension)';
  nameInput.style.display = 'block';
  nameInput.style.marginBottom = '0.5em';
  wrap.appendChild(nameInput);

  const overwriteLabel = document.createElement('label');
  const overwriteCheckbox = document.createElement('input');
  overwriteCheckbox.type = 'checkbox';
  overwriteLabel.appendChild(overwriteCheckbox);
  overwriteLabel.append(' Overwrite if exists');
  overwriteLabel.style.display = 'block';
  overwriteLabel.style.marginBottom = '0.5em';
  wrap.appendChild(overwriteLabel);

  let selectedKey = null;
  let modelsByKey = new Map();
  let trainingServerReachable = true;

  function _selectedRequiresConversion() {
    return Boolean(modelsByKey.get(selectedKey)?.requiresConversion);
  }

  const { el: runnerSection } = createTaskRunnerSection({
    buttonLabel: 'Download & Import',
    runningLabel: 'Downloading…',
    buildStartArgs: () => {
      const modelName = nameInput.value.trim();
      if (!selectedKey) {
        toast('Choose a curated model first', 3000, 'warning');
        return null;
      }
      if (!modelName) {
        toast('Enter a model name', 3000, 'warning');
        return null;
      }
      if (_selectedRequiresConversion() && !trainingServerReachable) {
        toast('Training server is unreachable — cannot convert this model right now', 4000, 'warning');
        return null;
      }
      const requiresConversion = _selectedRequiresConversion();
      return {
        typename: requiresConversion ? 'ImportCuratedModelTask' : 'ImportModelTask',
        name: `Import model: ${modelName}`,
        params: {
          curated_key: selectedKey,
          model_name: modelName,
          overwrite: overwriteCheckbox.checked,
        },
      };
    },
    onSettled: finalTaskInfo => {
      const status = finalTaskInfo?.instance?.status;
      if (status === 'completed') {
        toast(`Imported model '${nameInput.value.trim()}'`, 4000, 'success');
        _afterImportSuccess(target, editable, nameInput.value.trim());
      } else if (status === 'error') {
        toast('Model import failed — see task log', 5000, 'error');
      }
    },
  });
  wrap.appendChild(runnerSection);

  // createTaskRunnerSection doesn't expose its start button directly; it's
  // always the first <button> appended to the section (before Cancel).
  const startBtn = runnerSection.querySelector('button');

  function _refreshConversionUI() {
    const needsConversion = _selectedRequiresConversion();
    conversionBadge.style.display = needsConversion ? 'inline-block' : 'none';

    const disableDownload = needsConversion && !trainingServerReachable;
    startBtn.disabled = disableDownload;
    startBtn.title = disableDownload
      ? 'Training server is unreachable — cannot convert this model right now.'
      : '';
  }

  Promise.all([fetchCuratedModels(), isTrainingServerReachable()])
    .then(async ([models, reachable]) => {
      modelsByKey = new Map(models.map(m => [m.key, m]));
      trainingServerReachable = reachable;

      const dropdown = new DropDownField({
        value: null,
        editMode: true,
        options: models.map(m => ({
          key: m.key,
          text: m.requiresConversion ? `${m.displayName} (via training server)` : m.displayName,
        })),
        placeholder: 'Choose a curated model...',
        onChange: value => {
          selectedKey = value;
          _refreshConversionUI();
        },
      });
      dropdownDiv.appendChild(await dropdown.renderEdit());
      _refreshConversionUI();
    })
    .catch(err => {
      showPaneError(dropdownDiv, err.message);
    });

  return wrap;
}
