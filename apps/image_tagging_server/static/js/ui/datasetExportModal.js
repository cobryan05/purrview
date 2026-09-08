import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';
import { makeFieldRow, toast } from '/app-static/js/ui/utils/index.js';
import { fetchDatasetExportFormats, datasetExportUrl } from '/app-static/js/shared/api/datasets.js';

/**
 * Ask for an export format, then hand the browser the download URL.
 *
 * The archive never passes through JS: `location.assign` on an
 * `attachment`-dispositioned response downloads it without unloading the SPA,
 * and the bytes stream to disk. Reading it into a Blob first would put an
 * entire dataset — images included — in the tab's memory.
 *
 * @param {{uuid: string, name?: string}} dataset
 */
export async function showDatasetExportModal(dataset) {
  const formats = await fetchDatasetExportFormats();
  if (!formats?.length) {
    toast('No export formats available', 5000, 'error');
    return;
  }

  const { dialog, open, close } = createModalShell();
  dialog.dataset.testid = 'dataset-export-modal';

  const title = document.createElement('strong');
  title.textContent = `Export "${dataset.name ?? 'dataset'}"`;

  const formatSelect = document.createElement('select');
  formatSelect.dataset.testid = 'dataset-export-format';
  for (const fmt of formats) {
    const opt = document.createElement('option');
    opt.value = fmt.id;
    opt.textContent = fmt.name;
    formatSelect.appendChild(opt);
  }

  const layout = document.createElement('small');
  layout.dataset.testid = 'dataset-export-layout';
  layout.style.cssText = 'display: block; opacity: 0.75; margin: -0.4em 0 0.6em 0;';
  const syncLayout = () => {
    layout.textContent = formats.find(f => f.id === formatSelect.value)?.description ?? '';
  };
  formatSelect.addEventListener('change', syncLayout);
  syncLayout();

  const splitInput = document.createElement('input');
  splitInput.type = 'number';
  splitInput.min = '0';
  splitInput.max = '0.9';
  splitInput.step = '0.05';
  splitInput.value = '0.2';
  splitInput.dataset.testid = 'dataset-export-val-fraction';

  const splitHint = document.createElement('small');
  splitHint.style.cssText = 'display: block; opacity: 0.75; margin: -0.4em 0 0.6em 0;';
  splitHint.textContent = 'Fraction held out as the validation split (0 = everything in train). '
    + 'Images are split by uuid order, so the same dataset always splits the same way.';

  const verifiedInput = document.createElement('input');
  verifiedInput.type = 'checkbox';
  verifiedInput.checked = true;
  verifiedInput.dataset.testid = 'dataset-export-verified-only';

  const verifiedHint = document.createElement('small');
  verifiedHint.style.cssText = 'display: block; opacity: 0.75; margin: -0.4em 0 0.6em 0;';
  verifiedHint.textContent = 'Export only images where every dataset label has been reviewed — what '
    + 'training uses. Unticking includes unreviewed images, whose missing boxes read as true negatives.';

  const btnRow = document.createElement('div');
  btnRow.style.cssText = 'display: flex; gap: 0.5em; justify-content: flex-end; flex-wrap: wrap;';

  const cancelBtn = document.createElement('button');
  cancelBtn.type = 'button';
  cancelBtn.className = 'secondary';
  cancelBtn.textContent = 'Cancel';
  cancelBtn.addEventListener('click', () => close());

  const downloadBtn = document.createElement('button');
  downloadBtn.type = 'button';
  downloadBtn.textContent = 'Download';
  downloadBtn.dataset.testid = 'dataset-export-download';
  downloadBtn.addEventListener('click', () => {
    const valFraction = Math.min(0.9, Math.max(0, Number(splitInput.value) || 0));
    const url = datasetExportUrl(dataset.uuid, {
      format: formatSelect.value,
      valFraction,
      requireVerification: verifiedInput.checked,
    });
    close();
    // The server assembles the tar as it streams, so a large dataset takes a
    // while to *finish* — but the download starts immediately.
    toast('Preparing export — the download will start shortly', 5000);
    window.location.assign(url);
  });

  btnRow.append(cancelBtn, downloadBtn);
  dialog.append(
    title,
    makeFieldRow('Format', formatSelect),
    layout,
    makeFieldRow('Validation split', splitInput),
    splitHint,
    makeFieldRow('Reviewed images only', verifiedInput),
    verifiedHint,
    btnRow,
  );

  open();
  formatSelect.focus();
}
