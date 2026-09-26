/**
 * The floating "?" that gets a cancelled tour back.
 *
 * Skipping a tour records the step it was abandoned on, so this offers to resume
 * there rather than restart — the reason a user skips mid-walk is usually that
 * something else needed doing, not that they wanted to start over.
 *
 * Bottom left, not top left, and fixed rather than in the tab strip: the strip
 * slides away on scroll (`body.chrome-hidden`) and its buttons are `flex: 1`, so
 * anything added there either disappears with it or squeezes the tabs. The
 * bottom-right corner is already the Watch tab's "+ New watcher" FAB.
 *
 * Hidden while a tour is running — the tour's own card owns the screen then — and
 * hidden with `visibility` rather than `display` so nothing reflows around it.
 */
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { openAnchoredPopup } from '/app-static/js/ui/utils/anchoredPopup.js';
import { Logger } from '/app-static/js/ui/utils/logging.js';
import { isTourRunning } from './engine.js';
import { runTour, toursForPage } from './index.js';

/** @type {?HTMLElement} */
let button = null;
/** @type {?{close: () => void}} */
let popup = null;
/** @type {import('./index.js').TourOffer[]} */
let offers = [];

function _closePopup() {
  popup?.close();
  popup = null;
}

/** @param {import('./index.js').TourOffer} offer */
function _offerLine(offer) {
  const { tour, startAt, completed } = offer;
  if (startAt > 0) return `Resume at step ${startAt + 1} of ${tour.steps.length}`;
  if (completed) return `Seen — walk it again (${tour.steps.length} steps)`;
  return `${tour.steps.length} steps`;
}

function _buildMenu() {
  const menu = document.createElement('div');
  menu.className = 'tour-help-menu';
  menu.dataset.testid = 'tour-help-menu';

  const header = document.createElement('div');
  header.className = 'tour-help-menu__header';
  header.textContent = 'Guided tours';
  menu.appendChild(header);

  for (const offer of offers) {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'tour-help-menu__item';
    item.dataset.testid = `tour-help-run-${offer.tour.id}`;

    const name = document.createElement('div');
    name.className = 'tour-help-menu__name';
    name.textContent = offer.tour.title;
    const sub = document.createElement('div');
    sub.className = 'tour-help-menu__sub';
    sub.textContent = _offerLine(offer);
    item.append(name, sub);

    item.addEventListener('click', () => {
      _closePopup();
      runTour(offer.tour.id, { startAt: offer.startAt });
    });
    menu.appendChild(item);

    // Only worth offering when it would do something different from the row above.
    if (offer.startAt > 0) {
      const restart = document.createElement('button');
      restart.type = 'button';
      restart.className = 'tour-help-menu__restart';
      restart.dataset.testid = `tour-help-restart-${offer.tour.id}`;
      restart.textContent = 'Start from the beginning';
      restart.addEventListener('click', () => {
        _closePopup();
        runTour(offer.tour.id, { startAt: 0 });
      });
      menu.appendChild(restart);
    }
  }
  return menu;
}

/** @param {string} tabName */
async function _refresh(tabName) {
  if (!button) return;
  try {
    offers = await toursForPage(tabName);
  } catch (err) {
    Logger.warn('tour offers unreadable', err);
    offers = [];
  }
  _syncVisibility();
}

function _syncVisibility() {
  if (!button) return;
  const available = offers.length > 0 && !isTourRunning();
  button.classList.toggle('tour-help-fab--available', available);
  button.setAttribute('aria-hidden', available ? 'false' : 'true');
  button.tabIndex = available ? 0 : -1;
  if (!available) _closePopup();
}

/**
 * Mount the help button. Safe to call once, at boot; a second call is a no-op.
 * @returns {?HTMLElement}
 */
export function mountTourHelpButton() {
  if (button) return button;

  button = document.createElement('button');
  button.type = 'button';
  button.className = 'tour-help-fab';
  button.dataset.testid = 'tour-help-fab';
  button.textContent = '?';
  button.title = 'Guided tours for this page';
  button.setAttribute('aria-label', 'Guided tours for this page');
  document.body.appendChild(button);

  button.addEventListener('click', () => {
    if (popup) { _closePopup(); return; }
    if (!offers.length) return;
    popup = openAnchoredPopup(button, _buildMenu());
  });

  events.subscribe(EventTypes.MAIN_TAB_CHANGED, (payload) => { void _refresh(payload?.tabName ?? ''); });
  // A tour that just ended may have changed what a resume would open on, so the
  // offers are re-read rather than only re-shown.
  events.subscribe(EventTypes.TOUR_STATE_CHANGED, () => {
    _syncVisibility();
    const active = document.querySelector('.main-tab-button.active')?.dataset.tab;
    if (!isTourRunning() && active) void _refresh(active);
  });

  const active = document.querySelector('.main-tab-button.active')?.dataset.tab ?? '';
  void _refresh(active);
  return button;
}
