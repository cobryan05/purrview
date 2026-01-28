/** @typedef {(payload: any) => void} EventHandler */

class EventBus {
  constructor() {
    /** @type {Map<string, Set<EventHandler>>} */
    this._listeners = new Map();
  }

  subscribe(event, handler) {
    let handlers = this._listeners.get(event);
    if (!handlers) {
      handlers = new Set();
      this._listeners.set(event, handlers);
    }

    handlers.add(handler);

    return () => {
      handlers.delete(handler);
      if (handlers.size === 0) {
        this._listeners.delete(event);
      }
    };
  }

  publish(event, payload) {
    const handlers = this._listeners.get(event);
    if (!handlers) return;

    [...handlers].forEach((handler) => {
      try {
        handler(payload);
      } catch (err) {
        console.error(`Error in event handler for "${event}"`, err);
      }
    });
  }

  clear() {
    this._listeners.clear();
  }
}

export const events = new EventBus();
