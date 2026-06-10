/**
 * Context-menu type registry + shared open path.
 *
 * Exports:
 *   registerContextProvider(type, fn)
 *   attachContext(el, resolver) → { detach }
 *   getResolver(el) → resolver | undefined
 *   openContextMenuForObjects(objects, anchor) → Promise<boolean>
 */

import { openMenu, closeActiveMenu } from './menu.js';

// ── Provider registry ─────────────────────────────────────────────────────────

/** @type {Map<string, (adapters: object[], ctx: object) => Promise<object[]>|object[]>} */
const _providers = new Map();

/**
 * Register a context-menu provider for the given type.
 * `fn(adapters, ctx) → Section[]` (may be async).
 * Warns on duplicate registration.
 *
 * @param {string} type
 * @param {(adapters: object[], ctx: object) => Promise<object[]>|object[]} fn
 */
export function registerContextProvider(type, fn) {
  if (_providers.has(type)) {
    console.warn(`[contextMenu] Duplicate provider for type "${type}"`);
  }
  _providers.set(type, fn);
}

// ── Element → resolver map ────────────────────────────────────────────────────

/** @type {WeakMap<Element, (event: Event) => object[]>} */
const _resolvers = new WeakMap();

/**
 * Attach a context resolver to a DOM element.
 * `resolver(event) → ContextObject[]` where ContextObject = { adapters: { [type]: Adapter } }.
 *
 * @param {Element} el
 * @param {(event: Event) => object[]} resolver
 * @returns {{ detach: () => void }}
 */
export function attachContext(el, resolver) {
  _resolvers.set(el, resolver);
  return {
    detach() { _resolvers.delete(el); },
  };
}

/**
 * Look up the resolver attached to `el` (or undefined).
 * @param {Element} el
 */
export function getResolver(el) {
  return _resolvers.get(el);
}

// ── Shared open path ──────────────────────────────────────────────────────────

/**
 * Build and open a context menu for the given objects.
 *
 * @param {Array<{ adapters: Record<string, object> }>} objects
 * @param {{ x: number, y: number }} anchor
 * @returns {Promise<boolean>} true if a menu was opened, false if no actions apply
 */
export async function openContextMenuForObjects(objects, anchor) {
  if (!objects?.length) return false;

  // Compute the intersection of types across all objects
  const typeSets = objects.map(o => new Set(Object.keys(o.adapters)));
  const commonTypes = [...typeSets[0]].filter(t => typeSets.every(s => s.has(t)));

  // Keep only types that have a registered provider, in registration order
  const orderedTypes = [..._providers.keys()].filter(t => commonTypes.includes(t));
  if (!orderedTypes.length) return false;

  const ctx = {
    count: objects.length,
    anchor,
    close: () => closeActiveMenu(),
  };

  const allSections = [];
  for (const type of orderedTypes) {
    const provider = _providers.get(type);
    const adapters = objects.map(o => o.adapters[type]);
    const sections = await provider(adapters, ctx);
    if (sections) {
      for (const section of sections) {
        if (section?.items?.length) allSections.push(section);
      }
    }
  }

  if (!allSections.length) return false;

  openMenu({ x: anchor.x, y: anchor.y, sections: allSections });
  return true;
}
