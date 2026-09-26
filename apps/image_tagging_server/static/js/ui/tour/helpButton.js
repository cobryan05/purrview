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
import { currentTourPage, isAutoStartPending, runTour, toursForPage } from './index.js';
import { markFabHintSeen, readTourState } from './state.js';

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

/** Whether any tour on this page has never been started here. */
function _anyUnseen() {
  return offers.some(offer => offer.unseen);
}

/** Guards against two _refresh results racing to show the callout twice. */
let hintShown = false;

/**
 * The one-time "this page has a tour" callout.
 *
 * The dot on the ? is easy to miss, and a user who never notices it never learns
 * that the deeper per-screen tours exist at all. So the first time a page turns up
 * with an unseen tour, say so once — and only once per workspace. It is a callout
 * rather than an auto-started tour on purpose: the point is to make the button
 * discoverable, not to seize a page the user opened to do something else.
 */
async function _maybeShowHint() {
  if (hintShown || popup || !button) return;
  if (isTourRunning() || isAutoStartPending() || !_anyUnseen()) return;
  let state;
  try {
    state = await readTourState();
  } catch (err) {
    Logger.warn('tour state unreadable, skipping the ? hint', err);
    return;
  }
  // Re-checked after the await: a tour may have started, or the user may have
  // opened the menu themselves, while the state was in flight.
  if (state.fabHintSeen || state.neverShow || hintShown || popup) return;
  if (isTourRunning() || isAutoStartPending() || !_anyUnseen()) return;

  hintShown = true;
  void markFabHintSeen();

  const card = document.createElement('div');
  card.className = 'tour-help-hint';
  card.dataset.testid = 'tour-help-hint';
  const title = document.createElement('div');
  title.className = 'tour-help-hint__title';
  title.textContent = 'There is a tour for this page';
  const body = document.createElement('p');
  body.className = 'tour-help-hint__body';
  body.textContent = 'Whenever this ? is highlighted, the page you are on has a walkthrough you '
    + 'have not seen — including separate, more detailed ones for individual screens. '
    + 'Open it whenever you like; it will not start on its own.';
  const dismiss = document.createElement('button');
  dismiss.type = 'button';
  dismiss.className = 'panel-btn';
  dismiss.dataset.testid = 'tour-help-hint-dismiss';
  dismiss.textContent = 'Got it';
  dismiss.addEventListener('click', () => _closePopup());
  card.append(title, body, dismiss);

  popup = openAnchoredPopup(button, card);
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
      runTour(offer.tour.id, { startAt: offer.startAt, stack: offer.stack });
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
        runTour(offer.tour.id, { startAt: 0, stack: offer.stack });
      });
      menu.appendChild(restart);
    }
  }
  return menu;
}

/** @param {string} tabName @param {string} [subTabName] */
async function _refresh(tabName, subTabName = '') {
  if (!button) return;
  try {
    offers = await toursForPage(tabName, subTabName);
  } catch (err) {
    Logger.warn('tour offers unreadable', err);
    offers = [];
  }
  _syncVisibility();
  void _maybeShowHint();
}

function _syncVisibility() {
  if (!button) return;
  const available = offers.length > 0 && !isTourRunning();
  button.classList.toggle('tour-help-fab--available', available);
  // The dot is the whole of the "there is a tour here you have not seen" signal —
  // the alternative, starting it unasked, is an ambush on a page the user opened
  // to do something else.
  const unseen = available && _anyUnseen();
  button.classList.toggle('tour-help-fab--unseen', unseen);
  button.title = unseen ? 'Guided tour available for this page' : 'Guided tours for this page';
  button.setAttribute('aria-label', button.title);
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

  function _refreshHere() {
    const { main, sub } = currentTourPage();
    if (main) void _refresh(main, sub);
  }

  // Every tier, not only the main one: moving between Configuration's screens
  // changes which tour this button should offer.
  events.subscribe(EventTypes.TAB_CHANGED, _refreshHere);
  // A tour that just ended may have changed what a resume would open on, so the
  // offers are re-read rather than only re-shown.
  events.subscribe(EventTypes.TOUR_STATE_CHANGED, () => {
    _syncVisibility();
    if (!isTourRunning()) _refreshHere();
  });

  _refreshHere();
  return button;
}
