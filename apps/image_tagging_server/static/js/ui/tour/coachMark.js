/**
 * The bubble a tour talks through: a spotlight cut out of a dimmed page, and a card
 * beside it with the step's text and the Back / Next / Skip controls.
 *
 * The spotlight is one element sized to the target with an enormous box-shadow —
 * the dim is the shadow, the hole is the element. It takes no pointer events, so
 * the control it points at stays clickable: a step that says "press + New" must
 * let the user press it.
 *
 * The card is positioned below the target when there is room, above when there is
 * not, and clamped to the viewport; on a touch device it is a bottom sheet instead,
 * for the same reason the filter dropdowns are — nothing on a phone should depend
 * on a bubble finding space beside a 30px button.
 */

import { registerEscHandler, unregisterEscHandler } from '../utils/modalZIndex.js';

const SPOTLIGHT_PAD = 6;
const CARD_GAP = 12;
const VIEWPORT_MARGIN = 12;

/** Touch devices get the sheet; see the dropdown rule in app.css for the same query. */
const sheetQuery = () => window.matchMedia('(hover: none)');

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * @typedef {Object} CoachMarkStep
 * @property {?Element} target - what to spotlight; null centres the card with no dim hole
 * @property {string} title
 * @property {string} body
 * @property {number} index - 0-based
 * @property {number} count
 * @property {() => void} onNext
 * @property {?() => void} onBack - null on the first step
 * @property {() => void} onSkip
 * @property {(never: boolean) => void} onNeverShow
 * @property {boolean} neverShow - current value of the opt-out box
 * @property {boolean} [waiting] - the step advances itself when the user acts; Next reads "Skip this"
 */

/**
 * @returns {{show: (step: CoachMarkStep) => void, hide: () => void, reposition: () => void, isOpen: () => boolean}}
 */
export function createCoachMark() {
  const spotlight = el('div', 'tour-spotlight');
  spotlight.dataset.testid = 'tour-spotlight';
  const card = el('div', 'tour-card');
  card.dataset.testid = 'tour-card';
  card.setAttribute('role', 'dialog');
  card.setAttribute('aria-live', 'polite');

  const progress = el('div', 'tour-card__progress');
  progress.dataset.testid = 'tour-progress';
  const title = el('h4', 'tour-card__title');
  title.dataset.testid = 'tour-title';
  const body = el('p', 'tour-card__body');
  body.dataset.testid = 'tour-body';

  const optOut = el('label', 'tour-card__optout');
  const optOutBox = document.createElement('input');
  optOutBox.type = 'checkbox';
  optOutBox.dataset.testid = 'tour-never-show';
  optOut.append(optOutBox, el('span', null, "Don't offer tours automatically"));

  const actions = el('div', 'tour-card__actions');
  const skipBtn = el('button', 'tour-card__skip', 'Skip tour');
  skipBtn.type = 'button';
  skipBtn.dataset.testid = 'tour-skip';
  const backBtn = el('button', 'panel-btn secondary', 'Back');
  backBtn.type = 'button';
  backBtn.dataset.testid = 'tour-back';
  const nextBtn = el('button', 'panel-btn', 'Next');
  nextBtn.type = 'button';
  nextBtn.dataset.testid = 'tour-next';
  const right = el('div', 'tour-card__nav');
  right.append(backBtn, nextBtn);
  actions.append(skipBtn, right);

  card.append(progress, title, body, optOut, actions);

  /** @type {?CoachMarkStep} */
  let current = null;
  /** @type {?ResizeObserver} */
  let observer = null;
  let raf = 0;

  function place() {
    if (!current) return;
    const target = current.target;
    const sheet = sheetQuery().matches;
    card.classList.toggle('tour-card--sheet', sheet);

    if (!target || !target.isConnected) {
      spotlight.hidden = true;
      card.classList.add('tour-card--centred');
      card.style.top = '';
      card.style.left = '';
      return;
    }
    card.classList.remove('tour-card--centred');

    const rect = target.getBoundingClientRect();
    spotlight.hidden = false;
    spotlight.style.top = `${rect.top - SPOTLIGHT_PAD}px`;
    spotlight.style.left = `${rect.left - SPOTLIGHT_PAD}px`;
    spotlight.style.width = `${rect.width + SPOTLIGHT_PAD * 2}px`;
    spotlight.style.height = `${rect.height + SPOTLIGHT_PAD * 2}px`;

    if (sheet) {
      card.style.top = '';
      card.style.left = '';
      return;
    }

    const cw = card.offsetWidth;
    const ch = card.offsetHeight;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const below = rect.bottom + CARD_GAP;
    const above = rect.top - CARD_GAP - ch;
    let top;
    if (below + ch <= vh - VIEWPORT_MARGIN) top = below;
    else if (above >= VIEWPORT_MARGIN) top = above;
    else top = Math.max(VIEWPORT_MARGIN, Math.min(vh - ch - VIEWPORT_MARGIN, rect.top));
    let left = rect.left;
    left = Math.max(VIEWPORT_MARGIN, Math.min(vw - cw - VIEWPORT_MARGIN, left));
    card.style.top = `${Math.round(top)}px`;
    card.style.left = `${Math.round(left)}px`;
  }

  // Layout settles over a few frames after a tab switch (panes mount, fonts load,
  // sticky bars measure). Chase it briefly rather than trusting the first rect, but
  // stop as soon as the target's rect holds steady for a few frames in a row instead
  // of always burning 30 forced-layout ticks — most tab switches settle in 1-3 frames.
  function settle() {
    cancelAnimationFrame(raf);
    let frames = 0;
    let stableFor = 0;
    let lastKey = '';
    const tick = () => {
      place();
      const r = current?.target?.isConnected ? current.target.getBoundingClientRect() : null;
      const key = r ? `${r.top},${r.left},${r.width},${r.height}` : '';
      stableFor = key === lastKey ? stableFor + 1 : 0;
      lastKey = key;
      if (++frames < 30 && stableFor < 3) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  }

  // Escape goes through the shared stack in modalZIndex.js so a step pointing into an
  // open modal closes only the topmost overlay, not both at once.
  const onEsc = () => current?.onSkip();

  const onKey = (e) => {
    if (!current) return;
    if (e.key === 'ArrowRight' || e.key === 'Enter') {
      // Enter inside a text field belongs to that field — a step that asks the
      // user to type a label name and press Enter must not also advance the tour.
      if (e.key === 'Enter' && e.target !== nextBtn && e.target !== document.body) return;
      e.preventDefault();
      current.onNext();
    } else if (e.key === 'ArrowLeft' && current.onBack) { e.preventDefault(); current.onBack(); }
  };

  skipBtn.addEventListener('click', () => current?.onSkip());
  backBtn.addEventListener('click', () => current?.onBack?.());
  nextBtn.addEventListener('click', () => current?.onNext());
  optOutBox.addEventListener('change', () => current?.onNeverShow(optOutBox.checked));

  function show(step) {
    current = step;
    if (!spotlight.isConnected) document.body.append(spotlight, card);
    progress.textContent = `${step.index + 1} of ${step.count}`;
    title.textContent = step.title;
    body.textContent = step.body;
    optOutBox.checked = !!step.neverShow;
    backBtn.hidden = !step.onBack;
    const last = step.index >= step.count - 1;
    nextBtn.textContent = last ? 'Done' : step.waiting ? 'Skip this step' : 'Next';
    nextBtn.classList.toggle('secondary', !!step.waiting && !last);
    card.hidden = false;

    observer?.disconnect();
    observer = null;
    if (step.target && typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(place);
      observer.observe(step.target);
    }
    window.addEventListener('resize', place);
    document.addEventListener('scroll', place, true);
    document.addEventListener('keydown', onKey, true);
    // show() runs again for every step without an intervening hide(); re-registering
    // without unregistering first would stack a duplicate handler per step.
    unregisterEscHandler(onEsc);
    registerEscHandler(onEsc);
    settle();
    // Focus the card, not Next: a "do this" step wants the user's next keystroke
    // in the control being pointed at, and focusing a button would steal it.
    card.tabIndex = -1;
    if (!step.waiting) nextBtn.focus({ preventScroll: true });
  }

  function hide() {
    current = null;
    cancelAnimationFrame(raf);
    observer?.disconnect();
    observer = null;
    window.removeEventListener('resize', place);
    document.removeEventListener('scroll', place, true);
    document.removeEventListener('keydown', onKey, true);
    unregisterEscHandler(onEsc);
    spotlight.remove();
    card.remove();
    card.hidden = true;
  }

  return { show, hide, reposition: settle, isOpen: () => current != null };
}
