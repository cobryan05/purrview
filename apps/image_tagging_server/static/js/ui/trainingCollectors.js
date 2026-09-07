/**
 * Configuration → Collectors.
 *
 * A collector is the collect-training-images parameter set as a reusable entity:
 * which models to run, the per-label confidence bands, the filter chain, dedup,
 * where the frames land. A job names one and supplies the sources and pacing; a
 * watcher will name one too, which is the reason this stopped being thirty knobs
 * on a task config.
 *
 * The form is rendered from `collector_params_schema()` fetched from the server,
 * never restated here — that single definition is the whole point of the split.
 */
import { takePendingSelection } from '/app-static/js/app/utils/tabs.js';
import {
  createTrainingCollector,
  deleteTrainingCollector,
  fetchCollectorParamsSchema,
  fetchTrainingCollectorReviewActions,
  fetchTrainingCollectors,
  updateTrainingCollector,
} from '/app-static/js/shared/api/trainingCollectors.js';
import { buildPanelEditorBar, makeFieldRow, toast } from '/app-static/js/ui/utils/index.js';
import { FIELD_EDIT_EVENT } from '/app-static/js/ui/utils/fields/Field.js';
import { SchemaField } from '/app-static/js/ui/utils/fields/SchemaField.js';
import { createListEditorScreen } from '/app-static/js/ui/utils/listEditorScreen.js';
import { promptRestartAffectedTasks } from '/app-static/js/ui/utils/restartAffectedTasks.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';

const TAB_ID = 'tab-collector-management';

let _rendering = false;
let _navGuardController = null;

/**
 * "Browse unverified" and friends for one collector — works whether or not
 * anything is currently running (see `fetchTrainingCollectorReviewActions`),
 * which is the reason this lives on the collector's own config page rather than
 * only on a running task's status card: a stopped watcher's already-collected
 * frames are exactly as reviewable as a running one's.
 *
 * "Browse unverified" alone is a superset of the narrower buttons beside it —
 * a band match, a confusable flag, and a detector-miss seed all start life
 * unverified — but all of them are still offered here rather than just the one:
 * they're the same buttons a running watcher's Jobs card shows, and hiding some
 * of them here would mean the two surfaces disagree about what this collector
 * produced.
 */
async function _loadReviewActions(box, collectorUuid) {
  let actions;
  try {
    actions = await fetchTrainingCollectorReviewActions(collectorUuid);
  } catch (err) {
    console.error('Could not load review actions for collector', collectorUuid, err);
    return;
  }
  if (!actions.length) return;
  const label = document.createElement('div');
  label.className = 'panel-section-label';
  label.textContent = 'Review this collector’s output';
  box.appendChild(label);
  const row = document.createElement('div');
  row.className = 'collector-review-box__row';
  for (const action of actions) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'panel-btn';
    btn.textContent = action.label;
    btn.dataset.testid = 'collector-review-action';
    btn.addEventListener('click', () => {
      const payload = {
        tagUuids: action.payload?.tag_uuids ?? [],
        imageTagUuids: action.payload?.image_tag_uuids ?? [],
      };
      if (action.payload?.bbox_verification !== undefined) {
        payload.bboxVerification = action.payload.bbox_verification;
      }
      events.publish(EventTypes.OPEN_REVIEW_WITH_FILTERS, payload);
    });
    row.appendChild(btn);
  }
  box.appendChild(row);
}

/**
 * Name/description/enabled, then the schema-rendered params.
 *
 * Edits are picked up by delegation rather than a per-field callback for the same
 * reason ui/tasks.js does it: there is no single hook every field type fires, and
 * button-driven edits inside ArrayField dispatch FIELD_EDIT_EVENT instead of a
 * native one.
 */
async function buildCollectorForm(container, collector, schema, { onAnyChange }) {
  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.value = collector?.name ?? '';
  nameInput.placeholder = 'Required';
  container.appendChild(makeFieldRow('Name', nameInput));

  const descInput = document.createElement('input');
  descInput.type = 'text';
  descInput.value = collector?.description ?? '';
  descInput.placeholder = 'Optional';
  container.appendChild(makeFieldRow('Description', descInput));

  const enabledCheckbox = document.createElement('input');
  enabledCheckbox.type = 'checkbox';
  enabledCheckbox.checked = collector?.enabled !== false;
  container.appendChild(makeFieldRow('Enabled', enabledCheckbox));

  // Only for a collector that's already been saved — a not-yet-created one has no
  // uuid to have collected anything under. Independent of whether anything using
  // it is currently running: the tag a run produces (or would produce) outlives
  // the run, which is the whole point of asking here rather than on a job card.
  if (collector?.uuid) {
    const reviewBox = document.createElement('div');
    reviewBox.className = 'collector-review-box';
    reviewBox.dataset.testid = 'collector-review-actions';
    container.appendChild(reviewBox);
    void _loadReviewActions(reviewBox, collector.uuid);
  }

  const schemaField = await SchemaField.create({
    schema,
    values: collector?.params ?? {},
    editMode: true,
  });
  container.appendChild(await schemaField.renderEdit());

  container.addEventListener('input', onAnyChange);
  container.addEventListener('change', onAnyChange);
  container.addEventListener(FIELD_EDIT_EVENT, onAnyChange);

  return {
    getName: () => nameInput.value.trim(),
    getDescription: () => descInput.value.trim(),
    getEnabled: () => enabledCheckbox.checked,
    getParams: () => schemaField.getValue(),
  };
}

export async function renderTrainingCollectorManager({ target = 'collector-management-box' } = {}) {
  if (_rendering) return;
  _rendering = true;

  _navGuardController?.abort();
  _navGuardController = new AbortController();

  const container = document.getElementById(target);
  if (!container) { _rendering = false; return; }
  container.innerHTML = '';

  try {
    const schema = await fetchCollectorParamsSchema();

    const screen = createListEditorScreen({
      container,
      prefix: 'col',
      title: 'Collectors',
      itemUuidAttr: 'collectorUuid',
      withEditorFooter: true,
      searchable: true,
      searchPlaceholder: 'Filter collectors…',
      newLabel: '+ New Collector',
      onNew: () => _showForm(null),
      ownTabId: TAB_ID,
      navGuardSignal: _navGuardController.signal,
    });
    const { listBody, editorBar, editorBody, editorFooter } = screen;

    let _selectedUuid = null;
    let _switching = false;

    async function _showForm(collector, { force = false } = {}) {
      if (_switching && !force) return;
      _switching = true;
      try {
        if (!force && !await screen.confirmIfDirty()) return;
        screen.setBarHandle(null);
        screen.setDoSave(null);
        _selectedUuid = collector?.uuid ?? null;

        editorBody.innerHTML = '';
        editorFooter.innerHTML = '';

        const barHandle = buildPanelEditorBar({
          container: editorBar,
          item: collector,
          newTitle: 'New Collector',
          deleteTitle: 'Delete collector',
          onSave: () => _doSave(true),
          onDelete: collector ? async () => {
            await deleteTrainingCollector(collector.uuid);
            toast('Collector deleted', 3000);
            await refreshList();
            await _showForm(null, { force: true });
          } : undefined,
        });
        screen.setBarHandle(barHandle);
        const { setDirty, setSaving, updateTitle } = barHandle;

        const formHandle = await buildCollectorForm(
          editorBody,
          collector,
          schema,
          { onAnyChange: () => setDirty(true) },
        );
        setDirty(false);

        async function _doSave(navigateAfterCreate = false) {
          const name = formHandle.getName();
          if (!name) { toast('Name is required', 3000, 'error'); return false; }
          setSaving(true);
          try {
            const payload = {
              name,
              description: formHandle.getDescription(),
              enabled: formHandle.getEnabled(),
              params: formHandle.getParams(),
            };
            if (collector) {
              const updated = await updateTrainingCollector(collector.uuid, payload);
              setDirty(false);
              setSaving(false);
              updateTitle(updated.name);
              toast('Collector saved', 3000);
              await refreshList();
              // A watcher builds its collectors once, in _build_collectors, and holds
              // the CollectorSpec it read then for the life of the run — so every
              // field on this form (dedup thresholds, models, queue caps, eviction
              // strategy) is invisible to anything already running. No-op when
              // nothing is.
              await promptRestartAffectedTasks(collector.uuid, { label: 'collector' });
            } else {
              const created = await createTrainingCollector(payload);
              setDirty(false);
              toast(`Collector "${created.name}" created`, 3000);
              await refreshList();
              if (navigateAfterCreate) {
                await _showForm(created, { force: true });
              } else {
                setSaving(false);
              }
            }
            return true;
          } catch (err) {
            toast(err.message, 5000, 'error');
            setSaving(false);
            return false;
          }
        }

        screen.setDoSave(() => _doSave(false));
        screen.updateListSelection(collector?.uuid ?? null);
      } finally {
        _switching = false;
      }
    }

    async function refreshList() {
      listBody.innerHTML = '';
      const collectorsMap = await fetchTrainingCollectors();
      screen.setCount(collectorsMap.size);

      if (collectorsMap.size === 0) {
        listBody.appendChild(screen.createListEmpty('No collectors yet.'));
      }

      for (const collector of collectorsMap.values()) {
        listBody.appendChild(screen.createListItem({
          uuid: collector.uuid,
          name: collector.name,
          meta: collector.params?.output_dir || 'no output directory',
          onClick: () => _showForm(collector),
        }));
      }

      screen.applyFilter();
      screen.updateListSelection(_selectedUuid);
      return collectorsMap;
    }

    const collectorsMap = await refreshList();
    // Arriving from a job's ⚙ means "show me *this* collector".
    const wantedUuid = takePendingSelection(TAB_ID);
    await _showForm(wantedUuid ? collectorsMap.get(wantedUuid) ?? null : null, { force: true });
  } catch (err) {
    const errMsg = document.createElement('p');
    errMsg.style.color = 'red';
    errMsg.textContent = `Error loading collectors: ${err.message}`;
    container.appendChild(errMsg);
  } finally {
    _rendering = false;
  }
}
