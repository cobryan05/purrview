"""A self-contained HTML landing page for the standalone services.

``inference_server`` and ``training_server`` are the two services that are
meant to be run on their own host (a GPU box, a training box) while the rest
of the stack lives elsewhere. Both are pure REST APIs, so browsing to one of
them gave you an empty page (inference) or a bare 404 (training) — no way to
tell "the process is up and healthy" from "I typed the wrong port".

This renders a small operator page for exactly that moment. It is deliberately
*generic*: it knows nothing about models or training jobs, it just renders the
``summary`` + ``details`` sections that every service's ``GET /status`` already
returns (see :mod:`apps.helpers.serviceStatus`), optionally merged with a
second, heavier endpoint for things that are too expensive to compute on the
status poll (disk usage, directory scans). A service that adds a section to
either endpoint gets it on its landing page with no change here.

The page embeds its own CSS/JS rather than pulling from ``/static``. A service
deployed alone should not need the rest of the stack — or any CDN — to render
its own health page.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

# Actions are rendered as buttons. ``confirm`` (when set) is shown in a
# window.confirm() first — used for anything that throws data away.
Action = dict[str, Any]


def action(
    label: str,
    path: str,
    method: str = "POST",
    confirm: Optional[str] = None,
    danger: bool = False,
    description: str = "",
) -> Action:
    """One button on the landing page.

    Args:
        label: Button text.
        path: URL to call, relative to this service's root.
        method: HTTP method for the call.
        confirm: When set, the text of a confirmation prompt shown first.
        danger: Style the button as destructive.
        description: One-line explanation shown next to the button.
    """
    return {
        "label": label,
        "path": path,
        "method": method,
        "confirm": confirm,
        "danger": danger,
        "description": description,
    }


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: dark;
  --bg: #11141a; --panel: #181c24; --panel-2: #1f2430; --line: #2b3242;
  --fg: #dfe4ee; --muted: #8b95a8; --accent: #5b9dff;
  --ok: #46c07a; --warn: #e0b341; --err: #e0655f;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 900px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
header { display: flex; align-items: baseline; gap: .75rem; flex-wrap: wrap; }
h1 { font-size: 1.5rem; margin: 0; }
.sub { color: var(--muted); }
.dot { width: .6rem; height: .6rem; border-radius: 50%; display: inline-block;
  background: var(--muted); margin-right: .4rem; vertical-align: middle; }
.dot.ok { background: var(--ok); } .dot.warn { background: var(--warn); }
.dot.err { background: var(--err); }
.summary { margin: .75rem 0 1.5rem; padding: .75rem 1rem; background: var(--panel);
  border: 1px solid var(--line); border-radius: 8px; }
.summary .line { font-size: 1.05rem; }
section.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  margin-bottom: 1rem; overflow: hidden; }
section.card > h2 { font-size: .8rem; text-transform: uppercase; letter-spacing: .06em;
  color: var(--muted); margin: 0; padding: .6rem 1rem; background: var(--panel-2);
  border-bottom: 1px solid var(--line); }
.rows { padding: .25rem 0; }
.row { display: flex; gap: 1rem; padding: .35rem 1rem; align-items: baseline; }
.row + .row { border-top: 1px solid rgba(255,255,255,.04); }
.row .label { flex: 0 0 14rem; color: var(--muted); word-break: break-word; }
.row .value { flex: 1 1 auto; font-variant-numeric: tabular-nums; word-break: break-word; }
.empty { padding: .6rem 1rem; color: var(--muted); font-style: italic; }
.actions { display: flex; flex-direction: column; gap: .5rem; padding: .75rem 1rem; }
.action { display: flex; gap: .75rem; align-items: center; flex-wrap: wrap; }
button { font: inherit; color: var(--fg); background: var(--panel-2);
  border: 1px solid var(--line); border-radius: 6px; padding: .35rem .8rem; cursor: pointer; }
button:hover { border-color: var(--accent); }
button.danger { border-color: #5a2f2f; color: #f0b6b3; }
button.danger:hover { border-color: var(--err); }
button:disabled { opacity: .5; cursor: default; }
.action .description { color: var(--muted); font-size: .9rem; }
.links a { color: var(--accent); margin-right: 1rem; }
footer { margin-top: 2rem; color: var(--muted); font-size: .85rem;
  display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; }
#result { margin: .75rem 0 0; padding: .6rem 1rem; border-radius: 6px;
  background: var(--panel-2); border: 1px solid var(--line); white-space: pre-wrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85rem; display: none; }
#result.err { border-color: var(--err); color: #f0b6b3; }
@media (max-width: 600px) { .row { flex-direction: column; gap: 0; } .row .label { flex: none; } }
</style>
</head>
<body>
<main>
  <header>
    <h1>__TITLE__</h1>
    <span class="sub">__SUBTITLE__</span>
  </header>

  <div class="summary">
    <div class="line"><span id="dot" class="dot"></span><span id="summary">loading…</span></div>
    <div class="links" id="links">__LINKS__</div>
  </div>

  <div id="panels"></div>

  <div id="actions-card"></div>
  <pre id="result"></pre>

  <footer>
    <span id="updated">—</span>
    <label><input type="checkbox" id="auto" checked> auto-refresh</label>
    <button id="refresh" type="button">Refresh</button>
  </footer>
</main>
<script>
const STATUS_URL = __STATUS_URL__;
const INFO_URL = __INFO_URL__;
const ACTIONS = __ACTIONS__;
const REFRESH_MS = __REFRESH_MS__;

const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

function renderSection(sec) {
  const card = el("section", "card");
  card.appendChild(el("h2", null, sec.title || ""));
  const rows = sec.rows || [];
  if (!rows.length) {
    card.appendChild(el("div", "empty", sec.empty || "Nothing to show"));
    return card;
  }
  const body = el("div", "rows");
  for (const r of rows) {
    const line = el("div", "row");
    line.appendChild(el("span", "label", r.label ?? ""));
    line.appendChild(el("span", "value", r.value ?? ""));
    body.appendChild(line);
  }
  card.appendChild(body);
  return card;
}

function renderActions() {
  const host = document.getElementById("actions-card");
  host.innerHTML = "";
  if (!ACTIONS.length) return;
  const card = el("section", "card");
  card.appendChild(el("h2", null, "Actions"));
  const box = el("div", "actions");
  for (const a of ACTIONS) {
    const wrap = el("div", "action");
    const btn = el("button", a.danger ? "danger" : null, a.label);
    btn.type = "button";
    btn.addEventListener("click", () => runAction(a, btn));
    wrap.appendChild(btn);
    if (a.description) wrap.appendChild(el("span", "description", a.description));
    box.appendChild(wrap);
  }
  card.appendChild(box);
  host.appendChild(card);
}

function showResult(text, isError) {
  const out = document.getElementById("result");
  out.textContent = text;
  out.className = isError ? "err" : "";
  out.style.display = "block";
}

async function runAction(a, btn) {
  if (a.confirm && !window.confirm(a.confirm)) return;
  btn.disabled = true;
  try {
    const resp = await fetch(a.path, { method: a.method || "POST" });
    const text = await resp.text();
    let body = text;
    try { body = JSON.stringify(JSON.parse(text), null, 2); } catch (e) { /* not json */ }
    showResult(a.label + " → " + resp.status + "\\n" + body, !resp.ok);
  } catch (e) {
    showResult(a.label + " failed: " + e, true);
  } finally {
    btn.disabled = false;
    load();
  }
}

async function fetchJson(url) {
  const resp = await fetch(url, { headers: { Accept: "application/json" } });
  if (!resp.ok) throw new Error(url + " returned " + resp.status);
  return await resp.json();
}

let inFlight = false;
async function load() {
  if (inFlight) return;
  inFlight = true;
  const dot = document.getElementById("dot");
  try {
    const status = await fetchJson(STATUS_URL);
    const sections = (status.details || []).slice();
    document.getElementById("summary").textContent = status.summary || status.status || "ok";
    dot.className = "dot " + (status.status === "ok" ? "ok" : "err");
    if (INFO_URL) {
      try {
        const info = await fetchJson(INFO_URL);
        for (const s of info.details || []) sections.push(s);
      } catch (e) {
        sections.push({ title: "Details unavailable", rows: [{ label: "error", value: String(e) }] });
      }
    }
    const host = document.getElementById("panels");
    host.innerHTML = "";
    for (const s of sections) host.appendChild(renderSection(s));
    document.getElementById("updated").textContent = "updated " + new Date().toLocaleTimeString();
  } catch (e) {
    dot.className = "dot err";
    document.getElementById("summary").textContent = "unreachable: " + e;
  } finally {
    inFlight = false;
  }
}

document.getElementById("refresh").addEventListener("click", load);
renderActions();
load();
setInterval(() => { if (document.getElementById("auto").checked) load(); }, REFRESH_MS);
</script>
</body>
</html>
"""


def render_landing_page(
    title: str,
    subtitle: str = "",
    status_url: str = "/status",
    info_url: Optional[str] = None,
    actions: Optional[Iterable[Action]] = None,
    links: Optional[Iterable[tuple[str, str]]] = None,
    refresh_ms: int = 5000,
) -> str:
    """Render the operator landing page for a standalone service.

    Args:
        title: Service name shown as the heading.
        subtitle: One-line description under the heading.
        status_url: Cheap, frequently-polled health endpoint. Must return the
            ``summary``/``details`` shape from :mod:`apps.helpers.serviceStatus`.
        info_url: Optional second endpoint whose ``details`` sections are
            appended. Use it for anything too expensive for ``/status``.
        actions: Buttons built with :func:`action`.
        links: ``(label, href)`` pairs shown under the summary.
    """
    link_html = "".join(
        f'<a href="{href}">{label}</a>'
        for label, href in (links or [("API docs", "/docs"), ("Status JSON", status_url)])
    )
    return (
        _PAGE.replace("__TITLE__", title)
        .replace("__SUBTITLE__", subtitle)
        .replace("__LINKS__", link_html)
        .replace("__STATUS_URL__", json.dumps(status_url))
        .replace("__INFO_URL__", json.dumps(info_url))
        .replace("__ACTIONS__", json.dumps(list(actions or [])))
        .replace("__REFRESH_MS__", str(int(refresh_ms)))
    )
