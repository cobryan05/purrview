/**
 * Set up a training collector in three answers instead of thirty.
 *
 * Running `CollectTrainingImagesTask` by hand means creating a Source, then a
 * training collector, then a task config that points at both — three editors,
 * in an order nothing tells you, and the middle one opens on about thirty
 * knobs. Almost every first collector wants the same thing: point at a folder
 * of pictures or video, keep what's new, write it somewhere.
 *
 * So this asks three questions and writes all three records with defaults for
 * the rest. It deliberately does NOT start the run — the config lands in
 * Configuration → Tasks, where the advanced knobs are one click away and the
 * user decides when to set it going.
 *
 * The glob step is the one place the simplification is load-bearing. The
 * provider takes a list of patterns, which is precise and nearly unguessable;
 * here it is three plain choices, with the patterns still reachable under
 * "Only files matching…" for anyone who wants them.
 */
import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';
import { openFileBrowserModal } from '/app-static/js/ui/utils/openFileBrowserModal.js';
import { runPreviewSourceTest } from '/app-static/js/ui/utils/runPreviewSourceTask.js';
import { createSource, fetchSourceList } from '/app-static/js/shared/api/sources.js';
import { createTrainingCollector } from '/app-static/js/shared/api/trainingCollectors.js';
import { createTaskConfig, taskConfigFetcher } from '/app-static/js/shared/api/tasks.js';
import { toast, Logger } from '/app-static/js/ui/utils/logging.js';

/** The filesystem/video provider this wizard configures. */
const PROVIDER = 'Cv2VideoImageProvider';

/** Container id the source-preview helper renders into. */
const PREVIEW_TARGET = 'collect-wizard-preview';

const STEPS = ['Where from', 'Which files', 'Where to'];

/**
 * The glob presets, in the order they are offered.
 *
 * `recursive` and `globs` are what the provider actually reads; everything
 * else here is wording. `**\/*` with recursion is the provider's own default,
 * so the first preset changes nothing about how it behaves — it just says so.
 */
const FILE_PRESETS = [
  {
    id: 'all',
    label: 'Everything in this folder and below',
    detail: 'Sub-folders included. The usual answer.',
    globs: ['**/*'],
    recursive: true,
  },
  {
    id: 'shallow',
    label: 'Only files directly in this folder',
    detail: 'Sub-folders are ignored.',
    globs: ['*'],
    recursive: false,
  },
  {
    id: 'custom',
    label: 'Only files matching…',
    detail: 'Comma-separated patterns, e.g. **/*.jpg, cam1/**/*.png',
    globs: null,
    recursive: true,
  },
];

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  for (const child of children) if (child != null) node.append(child);
  return node;
}

function hint(text) {
  const small = el('small', { textContent: text });
  small.style.cssText = 'display:block; opacity:0.75; margin:-0.3em 0 0.8em 0;';
  return small;
}

/** Turn a folder path into a name and an output directory that read sensibly. */
function suggestNames(path) {
  const leaf = (path || '').replace(/\/+$/, '').split('/').filter(Boolean).pop() || 'images';
  return { name: `Collect from ${leaf}`, outputDir: `collected/${leaf}` };
}

/**
 * Open the collector wizard.
 *
 * @param {Object} [opts]
 * @param {string} [opts.initialPath] - folder to open the browser at
 * @param {() => (void|Promise<void>)} [opts.onSaved]
 */
export async function showCollectWizardModal({ initialPath = '/', onSaved = null } = {}) {
  const state = {
    step: 0,
    /** 'folder' builds a new source; 'existing' reuses one. */
    sourceKind: 'folder',
    folder: null,
    existingSourceUuid: '',
    preset: 'all',
    customGlobs: '',
    frameStride: 1,
    name: '',
    outputDir: '',
    nameTouched: false,
    busy: false,
  };

  let existingSources = [];
  try {
    const map = await fetchSourceList();
    existingSources = [...(map?.values?.() ?? [])];
  } catch (err) {
    Logger.warn('collectWizardModal: could not list sources', err);
  }

  const { dialog, open, close } = createModalShell();
  dialog.dataset.testid = 'collect-wizard-modal';
  Object.assign(dialog.style, {
    width: '620px',
    maxWidth: '94vw',
    maxHeight: '85vh',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75em',
  });

  const heading = el('strong', { textContent: 'Collect training images' });
  const stepLine = el('small', {});
  stepLine.dataset.testid = 'collect-wizard-step';
  stepLine.style.cssText = 'opacity:0.75;';

  const body = el('div', {});
  body.dataset.testid = 'collect-wizard-body';
  body.style.cssText = 'flex:1; overflow:auto; min-height:8em;';

  const backBtn = el('button', { type: 'button', className: 'secondary', textContent: 'Back' });
  backBtn.dataset.testid = 'collect-wizard-back';
  const nextBtn = el('button', { type: 'button', textContent: 'Next' });
  nextBtn.dataset.testid = 'collect-wizard-next';
  const cancelBtn = el('button', { type: 'button', className: 'secondary', textContent: 'Cancel' });
  cancelBtn.addEventListener('click', () => close());

  const footer = el('div', {}, cancelBtn, backBtn, nextBtn);
  footer.style.cssText = 'display:flex; gap:0.5em; justify-content:flex-end; flex-wrap:wrap;';

  dialog.append(heading, stepLine, body, footer);

  // ── the provider params, derived from the answers ──────────────────────

  function currentPreset() {
    return FILE_PRESETS.find(p => p.id === state.preset) ?? FILE_PRESETS[0];
  }

  function providerParams() {
    const preset = currentPreset();
    const globs = preset.globs
      ?? state.customGlobs.split(',').map(g => g.trim()).filter(Boolean);
    return {
      rootdir: state.folder ?? '',
      // A collector run is a pass over what is there, not a live feed. Looping
      // would make the provider a LIVE_STREAM and the task would never finish.
      loop: false,
      recursive: preset.recursive,
      // Only meaningful for videos; harmless for stills, which are one frame.
      frame_stride: Math.max(1, Number(state.frameStride) || 1),
      globs: globs.length ? globs : ['**/*'],
    };
  }

  function canAdvance() {
    if (state.step === 0) {
      return state.sourceKind === 'folder' ? state.folder !== null : Boolean(state.existingSourceUuid);
    }
    if (state.step === 1) {
      return state.preset !== 'custom' || providerParams().globs.length > 0;
    }
    return Boolean(state.name.trim()) && Boolean(state.outputDir.trim());
  }

  // ── rendering ──────────────────────────────────────────────────────────

  function render() {
    stepLine.textContent = `Step ${state.step + 1} of ${STEPS.length} — ${STEPS[state.step]}`;
    body.innerHTML = '';
    backBtn.disabled = state.step === 0 || state.busy;
    nextBtn.textContent = state.step === STEPS.length - 1 ? 'Save configuration' : 'Next';
    if (state.step === 0) renderSource();
    else if (state.step === 1) renderFiles();
    else renderDestination();
    nextBtn.disabled = state.busy || !canAdvance();
  }

  function renderSource() {
    const folderRadio = radio('collect-source-kind', 'folder', state.sourceKind === 'folder',
      'A folder of images or videos');
    const existingRadio = radio('collect-source-kind', 'existing', state.sourceKind === 'existing',
      'A source that already exists');
    for (const [input, kind] of [[folderRadio.input, 'folder'], [existingRadio.input, 'existing']]) {
      input.addEventListener('change', () => { state.sourceKind = kind; render(); });
    }

    body.append(folderRadio.row);
    if (state.sourceKind === 'folder') {
      const chosen = el('code', { textContent: state.folder === null ? '(nothing chosen)' : (state.folder || '(top level)') });
      chosen.dataset.testid = 'collect-wizard-folder';
      const browse = el('button', { type: 'button', className: 'secondary', textContent: 'Choose folder…' });
      browse.dataset.testid = 'collect-wizard-browse';
      browse.addEventListener('click', async () => {
        const picked = await openFileBrowserModal({ initialPath, allowCreateDir: false });
        if (picked === null) return;
        state.folder = picked.replace(/^\/+/, '');
        if (!state.nameTouched) {
          const suggested = suggestNames(state.folder);
          state.name = suggested.name;
          state.outputDir = suggested.outputDir;
        }
        render();
      });
      const row = el('div', {}, browse, chosen);
      row.style.cssText = 'display:flex; gap:0.6em; align-items:center; margin:0 0 0.5em 1.6em; flex-wrap:wrap;';
      body.append(row);
      body.append(hint('Anywhere under the image root — including a folder you bind-mounted into the container.'));
    }

    body.append(existingRadio.row);
    if (state.sourceKind === 'existing') {
      const select = el('select', {});
      select.dataset.testid = 'collect-wizard-existing-source';
      select.style.cssText = 'margin:0 0 0.5em 1.6em; width:calc(100% - 1.6em);';
      select.append(el('option', { value: '', textContent: existingSources.length ? 'Choose a source…' : 'No sources yet' }));
      for (const source of existingSources) {
        select.append(el('option', { value: source.uuid, textContent: `${source.name} (${source.typename})` }));
      }
      select.value = state.existingSourceUuid;
      select.addEventListener('change', () => {
        state.existingSourceUuid = select.value;
        const source = existingSources.find(s => s.uuid === select.value);
        if (source && !state.nameTouched) {
          state.name = `Collect from ${source.name}`;
          state.outputDir = `collected/${source.name.replace(/[^A-Za-z0-9._-]+/g, '_')}`;
        }
        render();
      });
      body.append(select);
      body.append(hint('A camera, stream or URL you already configured. Its own settings are not changed.'));
    }
  }

  function renderFiles() {
    if (state.sourceKind === 'existing') {
      // Nothing to choose: an existing source already decides what it yields.
      body.append(el('p', { textContent: 'This source already decides which frames it produces.' }));
      body.append(hint('Change that in Configuration → Sources. Continue to name the collector.'));
      return;
    }

    for (const preset of FILE_PRESETS) {
      const choice = radio('collect-files-preset', preset.id, state.preset === preset.id, preset.label);
      choice.input.dataset.testid = `collect-wizard-preset-${preset.id}`;
      choice.input.addEventListener('change', () => { state.preset = preset.id; render(); });
      body.append(choice.row, hint(preset.detail));

      if (preset.id === 'custom' && state.preset === 'custom') {
        const input = el('input', { type: 'text', value: state.customGlobs, placeholder: '**/*.jpg, cam1/**/*.png' });
        input.dataset.testid = 'collect-wizard-globs';
        input.style.cssText = 'margin:-0.5em 0 0.8em 1.6em; width:calc(100% - 1.6em);';
        input.addEventListener('input', () => {
          state.customGlobs = input.value;
          nextBtn.disabled = state.busy || !canAdvance();
        });
        body.append(input);
      }
    }

    const stride = el('input', { type: 'number', min: '1', step: '1', value: String(state.frameStride) });
    stride.dataset.testid = 'collect-wizard-stride';
    stride.style.width = '7em';
    stride.addEventListener('input', () => { state.frameStride = stride.value; });
    const strideRow = el('div', {}, el('label', { textContent: 'Take every Nth video frame' }), stride);
    strideRow.style.cssText = 'display:flex; gap:0.6em; align-items:center; margin-top:0.6em;';
    body.append(strideRow);
    body.append(hint('Only affects videos. Consecutive frames of the same scene are near-identical, '
      + 'and deduplication drops most of them anyway — raising this just saves the work.'));

    const testBtn = el('button', { type: 'button', className: 'secondary', textContent: '▶ Test this' });
    testBtn.dataset.testid = 'collect-wizard-test';
    testBtn.addEventListener('click', async () => {
      testBtn.disabled = true;
      try {
        await runPreviewSourceTest({ provider: PROVIDER, providerParams: providerParams(), target: PREVIEW_TARGET });
      } finally {
        testBtn.disabled = false;
      }
    });
    const preview = el('div', { id: PREVIEW_TARGET });
    preview.style.cssText = 'margin-top:0.5em;';
    body.append(testBtn, hint('Pulls one frame, so you can see the patterns match before saving anything.'), preview);
  }

  function renderDestination() {
    const nameInput = el('input', { type: 'text', value: state.name, placeholder: 'Collect from garden cam' });
    nameInput.dataset.testid = 'collect-wizard-name';
    nameInput.addEventListener('input', () => {
      state.name = nameInput.value;
      // Once it is edited by hand, a later folder change must not overwrite it.
      state.nameTouched = true;
      nextBtn.disabled = state.busy || !canAdvance();
    });
    body.append(el('label', { textContent: 'Name' }), nameInput);
    body.append(hint('Used for the source, the collector and the saved job, so they line up in every list.'));

    const outInput = el('input', { type: 'text', value: state.outputDir, placeholder: 'collected/garden' });
    outInput.dataset.testid = 'collect-wizard-output';
    outInput.addEventListener('input', () => {
      state.outputDir = outInput.value.trim();
      nextBtn.disabled = state.busy || !canAdvance();
    });
    body.append(el('label', { textContent: 'Save collected images to' }), outInput);
    body.append(hint('A folder under the image root. It is created on the first frame kept.'));

    const summary = el('p', {});
    summary.dataset.testid = 'collect-wizard-summary';
    summary.textContent = state.sourceKind === 'folder'
      ? `Reads ${state.folder || '(top level)'} · ${providerParams().globs.join(', ')}`
      : `Reads ${existingSources.find(s => s.uuid === state.existingSourceUuid)?.name ?? 'the chosen source'}`;
    body.append(summary);
    body.append(hint('Saved, not started. No model runs and nothing is annotated until you configure one — '
      + 'as it stands, every new frame is kept unannotated. Both are editable in Configuration.'));
  }

  function radio(groupName, value, checked, labelText) {
    const input = el('input', { type: 'radio', name: groupName, value, checked });
    const row = el('label', {}, input, document.createTextNode(` ${labelText}`));
    row.style.cssText = 'display:block; margin-bottom:0.3em;';
    return { input, row };
  }

  // ── saving ─────────────────────────────────────────────────────────────

  async function save() {
    const name = state.name.trim();
    state.busy = true;
    nextBtn.disabled = true;
    nextBtn.textContent = 'Saving…';
    try {
      let sourceUuid = state.existingSourceUuid;
      if (state.sourceKind === 'folder') {
        const source = await createSource({ name, provider: PROVIDER, providerParams: providerParams() });
        sourceUuid = source.uuid;
      }

      // Everything the collector reads except output_dir keeps its default:
      // an empty model_label_config means "keep every frame, unannotated",
      // which is the only honest starting point before a model is chosen.
      const collector = await createTrainingCollector({
        name,
        description: `Created by the collect wizard from ${state.folder ?? 'an existing source'}`,
        params: { output_dir: state.outputDir.trim(), model_label_config: [] },
      });

      const configUuid = await createTaskConfig({
        typename: 'CollectTrainingImagesTask',
        name,
        params: { collector_uuid: collector.uuid, source_uuids: [sourceUuid] },
      });

      // Configuration → Tasks reads through this cache, so a wizard-made
      // config would not appear there until a reload without the clear.
      taskConfigFetcher.clear();
      toast(`Saved "${name}" — start it from Configuration → Tasks`, 6000);
      close();
      await onSaved?.(configUuid);
    } catch (err) {
      Logger.warn('collectWizardModal: save failed', err);
      toast(err.message || 'Could not save the collector', 7000, 'error');
      state.busy = false;
      render();
    }
  }

  backBtn.addEventListener('click', () => {
    if (state.step > 0) state.step -= 1;
    render();
  });
  nextBtn.addEventListener('click', () => {
    if (state.step === STEPS.length - 1) return void save();
    state.step += 1;
    render();
  });

  render();
  open();
}
