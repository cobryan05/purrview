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
 * @property {string} [tourTitle] - names which tour is counting, once there is more than one
 * @property {?string} [returnTo] - title of the tour this one branched from, if any
 * @property {{label: string, onPick: () => void}[]} [branches] - other tours this step offers
 * @property {?string} [nextLabel] - overrides "Next"
 * @property {() => void} onNext
 * @property {?() => void} onBack - null on the first step
 * @property {() => void} onPause - stop, but resume on coming back to this page
 * @property {() => void} onDismiss - stop for good
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

  // Branches get their own row rather than a fourth button beside Back/Next: on a
  // touch device the card is a bottom sheet and four controls already fill its width.
  const branchRow = el('div', 'tour-card__branches');
  branchRow.dataset.testid = 'tour-branches';

  const actions = el('div', 'tour-card__actions');
  const skipBtn = el('button', 'tour-card__skip', 'Pause');
  skipBtn.type = 'button';
  skipBtn.dataset.testid = 'tour-skip';
  skipBtn.title = 'Stop here — the tour picks up where you left off when you come back';

  // Dismissing is the corner ×, not a second button in the row: pausing is the
  // common case and the one that should be easy to reach by mistake.
  const dismissBtn = el('button', 'tour-card__dismiss', '×');
  dismissBtn.type = 'button';
  dismissBtn.dataset.testid = 'tour-dismiss';
  dismissBtn.title = 'Dismiss this tour';
  dismissBtn.setAttribute('aria-label', 'Dismiss this tour');
  const backBtn = el('button', 'panel-btn secondary', 'Back');
  backBtn.type = 'button';
  backBtn.dataset.testid = 'tour-back';
  const nextBtn = el('button', 'panel-btn', 'Next');
  nextBtn.type = 'button';
  nextBtn.dataset.testid = 'tour-next';
  const right = el('div', 'tour-card__nav');
  right.append(backBtn, nextBtn);
  actions.append(skipBtn, right);

  card.append(dismissBtn, progress, title, body, branchRow, optOut, actions);

  /** @type {?CoachMarkStep} */
  let current = null;
  /** @type {?Element} */
  let marked = null;
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
  // Escape pauses rather than dismisses: it is the reflex for "get this off my
  // screen", which is not the same as "never show me this again".
  const onEsc = () => current?.onPause();
  let escRegistered = false;

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

  skipBtn.addEventListener('click', () => current?.onPause());
  dismissBtn.addEventListener('click', () => current?.onDismiss());
  backBtn.addEventListener('click', () => current?.onBack?.());
  nextBtn.addEventListener('click', () => current?.onNext());
  optOutBox.addEventListener('change', () => current?.onNeverShow(optOutBox.checked));

  /**
   * Flag the spotlit element so CSS can reveal a control that is only visible on
   * hover — the label tree's action cluster is opacity:0 until its row is hovered,
   * and a spotlight around an invisible button points at nothing.
   * @param {?Element} node
   */
  function mark(node) {
    if (marked === node) return;
    marked?.removeAttribute('data-tour-spotlight');
    marked = node ?? null;
    marked?.setAttribute('data-tour-spotlight', '');
  }

  function show(step) {
    current = step;
    mark(step.target);
    if (!spotlight.isConnected) document.body.append(spotlight, card);
    const counted = `${step.index + 1} of ${step.count}`;
    // The tour's own name matters once several exist: "3 of 7" alone cannot tell a
    // user walking the Configure tour that the overview is still waiting behind it.
    progress.textContent = step.tourTitle ? `${step.tourTitle} · ${counted}` : counted;
    progress.title = step.returnTo ? `Returns to ${step.returnTo} when this finishes` : '';
    progress.classList.toggle('tour-card__progress--nested', !!step.returnTo);

    title.textContent = step.title;
    body.textContent = step.body;
    optOutBox.checked = !!step.neverShow;

    branchRow.replaceChildren();
    for (const branch of step.branches ?? []) {
      const btn = el('button', 'panel-btn secondary tour-card__branch', branch.label);
      btn.type = 'button';
      btn.dataset.testid = `tour-branch-${step.branches.indexOf(branch)}`;
      btn.addEventListener('click', () => branch.onPick());
      branchRow.appendChild(btn);
    }
    branchRow.hidden = branchRow.childElementCount === 0;

    backBtn.hidden = !step.onBack;
    const last = step.index >= step.count - 1;
    nextBtn.textContent = last
      ? (step.returnTo ? `Back to ${step.returnTo}` : 'Done')
      : step.nextLabel ?? (step.waiting ? 'Skip this step' : 'Next');
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
    // Registered once for the whole tour, not per step. show() runs again for every
    // step without an intervening hide(), and re-registering would both stack
    // duplicates and push the tour back above any modal a previous step opened —
    // Escape would then skip the tour instead of closing the modal on top of it.
    if (!escRegistered) {
      registerEscHandler(onEsc);
      escRegistered = true;
    }
    settle();
    // Focus the card, not Next: a "do this" step wants the user's next keystroke
    // in the control being pointed at, and focusing a button would steal it.
    card.tabIndex = -1;
    if (!step.waiting) nextBtn.focus({ preventScroll: true });
  }

  function hide() {
    current = null;
    mark(null);
    cancelAnimationFrame(raf);
    observer?.disconnect();
    observer = null;
    window.removeEventListener('resize', place);
    document.removeEventListener('scroll', place, true);
    document.removeEventListener('keydown', onKey, true);
    unregisterEscHandler(onEsc);
    escRegistered = false;
    spotlight.remove();
    card.remove();
    card.hidden = true;
  }

  return { show, hide, reposition: settle, isOpen: () => current != null };
}
