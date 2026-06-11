/**
 * API wrapper for the /api/notices endpoint.
 *
 * Returns an aggregated list of advisory notices from all backend services.
 * Each notice: { key: string, level: "info"|"warning"|"error", message: string }
 */

export async function fetchNotices() {
  try {
    const resp = await fetch('/api/notices');
    if (!resp.ok) return { notices: [] };
    return await resp.json();
  } catch {
    return { notices: [] };
  }
}
