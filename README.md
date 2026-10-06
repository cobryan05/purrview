# PurrView

Self-hosted **computer vision for cameras you own**. Watch live camera streams,
detect and track objects on them, turn interactions between those objects into
*events* with video clips, and feed what the detector got wrong back into
annotation and training — all from a single Docker deployment that scales from a
CPU-only box up to a GPU server.

It is three things that deliberately share one database:

- an **event-driven NVR** — rules over tracked objects ("a cat is in the litter
  box", "the car left the garage"), each firing clips, snapshots, timelapses and
  MQTT messages;
- an **annotation tool** — browser-based bounding boxes, hierarchical labels,
  bulk review and search;
- a **training pipeline** — collect images, build datasets, train YOLO models,
  export to ONNX and serve them back to the watchers.

The point of putting them together is the loop: **the NVR records what it
misses, and you train that away.**

```
 cameras ──▶ watchers ──▶ events ──▶ correction ──▶ harvested frames
                 ▲                                        │
                 │                                        ▼
          ONNX models ◀── training ◀── datasets ◀──── annotation
```

![Live view — one card per camera, boxes drawn as the watcher sees them](docs/screenshots/screen2.png)

*Watch → Live: every camera in a running watcher, with what it is tracking right now.*

---

## The loop, concretely

1. A **watcher** runs a model over a camera stream and tracks what it finds.
2. A **rule** turns tracked objects into an **event** ("cat overlaps the litter
   box, for at least 2 s"), which cuts a clip out of the rolling buffer.
3. The event's clip replays with per-frame boxes on it, next to a list of
   **tracking problems** — a dropped detection, an identity flip between two
   subjects, two tracks merging while they overlap. Each one jumps the video to
   the moment it happened.
4. You say **what the watcher got wrong**: per label, leave it alone, relabel
   it, or mark it a false positive. That saves a spread of frames from the clip
   into the image database, left unverified for review, and flags the event so
   you can find it again.
5. You **annotate** those frames, promote a search into a **dataset**, and
   **train**.
6. The new model goes back into the watcher. The miss stops happening.

![An event clip with detection boxes and its tracking problems listed](docs/screenshots/screen1.png)

*Step 3 — the clip, its details, and six tracking problems with jump-to buttons.*

![The label-correction dialog for an event](docs/screenshots/screen5.png)

*Step 4 — correcting an event, and the frames that correction will harvest.*

Nothing in that chain is mandatory — plenty of installs only ever use the NVR,
or only ever use the annotation tool.

---

## The app, by tab

### Watch

- **Live** — one card per camera per watcher: current frame, what is tracked
  right now, which rules are open, and the ingest tier the stream resolved to.
  Cameras can be switched in and out of a running watcher without restarting it.
- **Events** — the recorded history. Events group into **sessions** (one visit,
  however many times the rule re-fired), each card carrying its clip or
  timelapse. Clips play with per-frame detection boxes overlaid, read from a
  `<clip>.mp4.boxes.json` sidecar. Filter by camera, rule, label present,
  review state or time; open one fullscreen; correct its labels; work through
  its tracking problems.

### Annotate

- **Editor** — a Konva canvas for drawing boxes, a file browser, and a sidebar
  with Labels, Inference (run a model on the open image) and Tags.
- **Review** — the bulk surface. A searchable, filterable grid of images *and*
  individual boxes: filter by label, tag, confidence range, source, filename
  pattern or whole-image label state; verify or reject in bulk; find visually
  similar (or deliberately *dissimilar*) images by perceptual hash; run a model,
  a classifier or a filter chain over the current result set; promote the search
  into a dataset.

![The Review grid, filtered, with per-box labels and bulk actions](docs/screenshots/screen4.png)

*Annotate → Review: filter chips across the top, bulk actions beneath, images and
boxes as cards.*

Anything you can filter down to, you can act on in one go — mark a label present
or absent across the whole result set, add or remove tags, settle verification,
or delete. The dialog previews the range it will touch before it does anything:

![The bulk-apply dialog over a filtered result set](docs/screenshots/screen6.png)

*"Apply to all" — the first and last images of the range, the nine operations, and
a label picker that badges whatever is in the active training run.*

### Configuration

| Group | Holds |
|---|---|
| **Vocabulary** | Labels (hierarchical, object- or image-scoped), Tags, and the per-model map from class IDs onto labels |
| **Data** | Datasets (what training consumes), Views (saved searches, promotable), Sources (folders, URLs, camera streams) |
| **Models** | Model Shapes (which model handles which label, hierarchically) and Model Stacks (reusable ordered detector lists, nestable) |
| **Pipelines** | Training definitions, Filter Chains, Collectors, saved Task definitions |
| **Watching** | Watch Rules — what a camera watches for, and what happens when it sees it |

![The Configuration overview, with live counts per section](docs/screenshots/screen8.png)

*Configuration → Overview: every section with a live count, plus the two shortcuts
that skip the setup grind.*

### Jobs

Everything long-running is a task: training, collection, import, classification,
filter chains, model conversion, frame harvesting, and the watchers themselves.
**Running** shows what is in flight and lets you pause, cancel or inspect it;
**Start a job** is the launcher, grouped by what the job is. Tasks persist their
state, so a watcher survives a restart.

![The job launcher, with training recipes showing new data since their last run](docs/screenshots/screen10.png)

*Each training recipe reports how much new annotated data has arrived since it
last ran — which is how you tell what is worth retraining.*

### Settings

Service health, recording and timelapse retention (age *and* size caps, with a
"clean up now"), model-version and training-folder retention, the thumbnail
cache, dedup hash sizes, the MQTT broker, and an optional access token that
gates the whole app.

---

## How watching works

**Sources → ingest.** A source is a URL. PurrView resolves it to an *ingest
tier* and tells you which one it got, because they do not cost the same:

| Tier | Meaning |
|---|---|
| `native` | MediaMTX pulls it directly (RTSP, RTMP, SRT, HLS …). Cheapest. |
| `ffmpeg` | Needs an ffmpeg front-end; copies the codec where it can, transcodes (real per-camera CPU) only when it must. |
| `ring` | Ring cameras, which are event-driven rather than continuous. |
| `none` | Snapshot-only — no rewind timeline. |

Buffered tiers get a **rolling buffer in RAM** (tmpfs `/buffer`, MediaMTX
recording into it), which is what makes pre-roll possible: an event that starts
now can still include the fifteen seconds before it.

**Regions.** You draw regions once per *camera*, not per watcher, because the
geometry describes the view rather than anyone's opinion of it:

- a **zone** is inert until a rule names it — a box called "litter box area" is
  more reliable than a litter-box detector and needs no training data;
- a **mask** acts on everything — any detection whose centre falls inside it is
  suppressed.

![Drawing a zone on a camera frame](docs/screenshots/screen7.png)

*The region editor, grabbing a frame from the running watcher. Coordinates are
stored relative to the frame, so they survive a camera resolution change.*

**Rules.** A rule is a sentence: *subject* — *relation* — *target*, plus timing.

- Subjects and targets are labels (with optional descendants), zones, any, or
  nothing.
- Relations: `present` (tracked at all), `contains`, `overlaps`, `iou`,
  `centroid_in`, `distance`, and `motion` (pixels changed — the one relation
  that fires on a frame with no detections).
- Triggers: `while` (the event spans the predicate), `on_enter` and `on_exit`
  (zero-length impulses — "the car left", where the footage that matters is on
  either side of the instant).
- Timing: `min_duration_s` is the dwell, `rearm_delay_s` the debounce,
  `max_duration_s` the cap on a span.
- Granularity: one event per tracked object, or one per rule for as long as
  anyone matches.

**Actions** hang off rule transitions (`event_started`, `event_ended`,
`track_started`, `object_detected`) or off every analysed frame:

- record a **clip** from the buffer, with pre-roll and post-roll;
- record an **event timelapse** — hours of an event compressed into seconds;
- save a **snapshot** or a standalone **timelapse**;
- **publish to MQTT**, plain or as an image;
- **collect training images** straight into the database.

Output paths and MQTT topics are templated — `{source}`, `{rule}`, `{subject}`,
`{target}`, `{phase}`, `{date}`, `{hour}` and friends — and the editor shows a
live rendering of the template so a typo is visible before it reaches the
broker.

**Trackers** are plugins (ByteTrack and an IoU/SORT-lite tracker ship). The
detection confidence floor is deliberately low: a tracker needs the weak boxes
to associate across frames, and the rule's own match threshold is where you
tighten things up.

One watcher holds all of this — its cameras, its tracker, its rules and its
actions — on one page:

![The watcher editor: sources, tracking, rules and actions](docs/screenshots/screen3.png)

*Configuration → Watch Rules. Each rule row is the sentence; each action row
below says what happens and when.*

---

## Home Assistant

Two channels, because neither is enough alone:

- **MQTT discovery.** An app-level bridge publishes device-based discovery for
  the app, each watcher, each rule, each camera and each tracked subject, so
  states and counts arrive as real HA entities with real history — "detected
  now", "total events", "last seen in the laundry room". Set the broker under
  Settings → MQTT.
- **A custom integration** for the things MQTT is the wrong shape for: an events
  browser with its own camera / subject / event-type / date filters, media
  proxied through HA rather than as raw URLs, and live camera entities fed by
  `publish_mqtt_image` actions.

Flagging comes back the other way: HA can mark an event's labels as wrong, which
shows up in the Events pane as something to fix.

![PurrView entities and an events browser inside Home Assistant](docs/screenshots/screen9.png)

*Last-seen sensors, the filtered events browser, and a tracked subject as its own
HA device.*

---

## Architecture at a glance

One Python codebase (`apps/`) exposed as several FastAPI services, run together
by `supervisord`:

| Service | Port | Role |
|---------|------|------|
| `image_tagging_server` | 7999 | Web UI; a pure proxy + SPA host, no business logic |
| `inference_server` | 8001 | ONNX inference (YOLOv5/v7/v8), GPU pinning with auto-unload |
| `tasks_server` | 8002 | Long-running jobs, including the watchers |
| `db_server` | 8003 | SQLite-backed data store, thumbnails, search |
| `stream_server` | 8004 | Owns `mediamtx`: camera fan-out, stream leases, buffer, clip extraction |
| `training_server` | 8005 | Model training (**optional, GPU**) |

These collapse into **two image tiers**:

- **`app`** — frontend + inference. Built slim/CPU by default (no CUDA/torch);
  inference is pure-ONNX at runtime. Optionally built on a CUDA base for
  GPU-accelerated inference.
- **`training`** — the heavy `torch`/`ultralytics` node. Optional, always GPU,
  and stands alone (it talks to the frontend only over HTTP, so it can live on a
  separate GPU box).

You choose a **topology** per host via compose profiles; `./setup.sh` picks the
right image base and profiles for you.

---

## Prerequisites

- **Docker** with the **Compose v2** plugin (`docker compose …`).
- For any GPU deployment (inference-on-GPU or training): an NVIDIA GPU, recent
  drivers, and the **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**.
- CPU-only deployments need **none** of the NVIDIA pieces.
- An MQTT broker, only if you want the Home Assistant integration.

---

## Quick start

```bash
git clone https://github.com/cobryan05/purrview.git
cd purrview

./setup.sh                     # answer a few questions → writes .env
docker compose up -d --build   # build + start what .env selected
```

Then open the UI at **http://localhost:7999**.

`setup.sh` detects whether you have a GPU and asks what this host should run
(everything / frontend+inference only / training only / a dev workspace), then
writes a `.env`. Even on a GPU host it always offers CPU inference — e.g. to
keep a slim frontend or reserve VRAM for training. `docker compose` reads `.env`
automatically, so from then on plain compose commands work. To reconfigure
later, re-run `./setup.sh`.

**First five minutes:** Configuration → Overview has two buttons that skip the
setup grind — *Import dataset* (YOLO, COCO or Pascal VOC, from a server folder
or an upload) and *Collect images* (a three-step wizard over a folder, a video
or a camera).

---

## Deployment topologies

`setup.sh` maps your answers onto these; you can also set them by hand in `.env`
(`COMPOSE_PROFILES=…`):

| Profile | Runs | Typical host |
|---------|------|--------------|
| `app` | frontend + inference in one container | CPU box, or GPU box (GPU inference) |
| `training` | standalone training node | remote/strong GPU box |
| `app,training` | frontend container + local training container ("split") | one GPU box, two containers |
| `mono` | everything in one container | one GPU box, single container |
| `dev` / `dev-mono` | source-mounted dev workspaces | your dev GPU box |

Common cases:

- **CPU-only NAS / mini-PC** — "Frontend + inference only", no GPU → slim image,
  no CUDA pulled or run. Watching and recording work fine here; training does
  not live on this box.
- **One GPU server, everything** — "Everything on one box" → `mono` (or split).
- **Local frontend + remote GPU trainer** — run `app` locally with
  `TRAINING_SERVER_HOST` pointed at the GPU box, and `training` on that box.

---

## Configuration (`.env`)

`setup.sh` generates `.env`; `.env.example` documents every option. The key
knobs:

| Variable | Values | Purpose |
|----------|--------|---------|
| `COMPOSE_PROFILES` | see table above | which containers run |
| `APP_BASE` | `slim-runtime` (default) / `cuda-runtime` | CPU vs GPU-inference base for the `app` image |
| `ORT_REQUIREMENTS` | `requirements-cpu.txt` / `requirements-gpu.txt` | onnxruntime flavor (pair with `APP_BASE`) |
| `COMPOSE_FILE` | add `:docker-compose.gpu.yaml` | give the `app` container a GPU (for onnxruntime-gpu) |
| `STREAM_BUFFER_SIZE` | e.g. `512m` | cap on the in-RAM rolling buffer — ~30 MB per camera per 60 s at 4 Mbps |
| `TRAINING_SERVER_HOST` | hostname | where the frontend reaches training (defaults to the `training` service) |
| `THUMB_CACHE_MAX_MB` | int | thumbnail cache cap (**`0` means unbounded**, not "no cache") |
| `APPUSER_UID` / `APPUSER_GID` | ints | match host user for clean bind-mount ownership |

Retention for recordings, timelapse frames, model versions and training folders
is set in the UI under **Settings**, not in `.env` — it needs to be changeable
without a restart.

### Recording storage

NVR clips, event snapshots and event timelapses are written under
`/data/recordings/`, inside the `./data:/data` bind mount. To keep recordings on
another disk, move **that whole mount** (`- /mnt/nas/purrview:/data` in
`docker-compose.override.yml`). Everything beneath it belongs to the app, so
this is always safe.

> ⚠️ **Never mount a directory containing anything else at an individual
> recordings root.** Space is reclaimed two ways. Deleting a recording's own
> file follows its database row, so it only ever removes files the app created —
> safe anywhere. But the **orphan sweep** walks the recordings roots and deletes
> every file no database row points at, once older than
> `RECORDINGS_ORPHAN_GRACE_S` (default 900 s). It has no extension filter and
> cannot distinguish your files from its own; its only safety argument is that
> those directories belong to the app alone. Point one at a shared media folder
> and it will be emptied.

`RECORDINGS_CLIP_DIR` / `RECORDINGS_SNAPSHOT_DIR` / `RECORDINGS_TIMELAPSE_DIR`
are deliberately not wired through `.env` for this reason — the safe knob is
exposed, the dangerous one isn't.

---

## Common commands

```bash
docker compose up -d --build     # start / update the stack from .env
docker compose ps                # what's running
docker compose logs -f app       # follow a service's logs
docker compose down              # stop everything
./rebuild.sh                     # down + up --build (uses .env)
```

Exposed ports: **7999** (UI), **8005** (training API, when running), **8553**
(RTSP relay → container 8554), **9001** (supervisord dashboard).

Host directories bind-mounted into the stack: `./data` (database and
recordings), `./models` (trained/loaded models), `./image_datasets` (images),
`./conf`, `./logs`.

---

## Development

Run `./setup.sh` and choose **"Development workspace"**. It offers two layouts:

- **Single container** (`COMPOSE_PROFILES=dev-mono`, service `mono-dev`) —
  everything plus dev tooling in one container.
- **Split** (`COMPOSE_PROFILES=dev`, services `app-dev` + `training-dev`).

The repo is bind-mounted at `/project`, so code changes are live. Bring it up
and run the suites inside the workspace container (`mono-dev` for
single-container, `app-dev` for split):

```bash
./setup.sh                       # -> Development workspace
docker compose up -d --build

scripts/test.sh                  # unit/integration (tests/)
scripts/test.sh playwright       # browser tests
scripts/test.sh both             # both; non-zero if either fails
```

`scripts/test.sh` runs pytest inside the container under `nice`/`ionice`, which
matters when the same box is serving the live app.

Other things worth knowing:

```bash
# Format (from the repo root inside the container, never from /app)
docker exec -w /project purrview_mono_dev \
  sh -c 'black apps/ tests/ playwright_tests/ tests_training/ && isort apps/ tests/ playwright_tests/ tests_training/'

bash generate_openapi_modules.sh              # regenerate + install the API clients
./generate_alembic_revision.sh "describe it"  # after a DB schema change
./scripts/install-git-hooks.sh                # pre-commit formatting checks
```

The dev images bundle the toolchain (formatters, pytest, Playwright +
Chromium). Personal, machine-specific dev mounts (extra datasets, host
`~/.claude`, etc.) go in a **gitignored** `docker-compose.override.yml`, which
compose merges automatically. The shipped override targets the `app-dev`
service; point it at `mono-dev` if you use the single-container dev layout.

`CLAUDE.md` holds the working agreements for this repo (test policy, comment
style, container requirement) and is worth reading before a first PR.

---

## Build-host note

The slim `app` image is CUDA-free to **run and pull**, but *building* the
training / `mono` images compiles the CUDA/torch wheelhouse. For CPU-only
targets, build the images on a capable host (or in CI) and pull them, rather
than building the GPU images on the lean host itself.

---

## License

[GNU AGPL-3.0](LICENSE).
