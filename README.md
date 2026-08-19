# WhiskerWatcher

Self-hosted image **annotation, inference, and training** for object detection and
image classification. Annotate images in the browser, run ONNX models to
auto-detect/label them (including live RTSP camera streams), and optionally train
your own models — all from a single Docker deployment that scales from a CPU-only
box up to a GPU server.

The primary job is **running inference on images**; annotation and training are
features around it. The stack is designed so you only pay for what you deploy: a
CPU-only host never pulls CUDA, and the heavy training stack is an optional,
separable component.

---

## Architecture at a glance

Everything is one Python codebase (`apps/`) exposed as several FastAPI services,
run together by `supervisord`:

| Service | Port | Role |
|---------|------|------|
| `image_tagging_server` | 7999 | Web UI / annotation frontend |
| `inference_server` | 8001 | ONNX model inference |
| `tasks_server` | 8002 | Long-running jobs kicked off by the UI |
| `db_server` | 8003 | SQLite-backed data store |
| `stream_server` | 8004 | Owns `mediamtx`: camera fan-out, stream leases, clip extraction |
| `training_server` | 8005 | Model training (**optional, GPU**) |

These collapse into **two image tiers**:

- **`app`** — the frontend + inference group. Built slim/CPU by default (no
  CUDA/torch); inference is pure-ONNX at runtime. Can optionally be built on a
  CUDA base for GPU-accelerated inference.
- **`training`** — the heavy `torch`/`ultralytics` training node. Optional, always
  GPU, and stands alone (it talks to the frontend only over HTTP, so it can run on
  a separate/remote GPU box).

You choose a **topology** per host (everything in one container, a split
frontend+training pair, a CPU frontend + remote GPU trainer, etc.) via compose
profiles. `./setup.sh` picks the right image base and profiles for you.

---

## Prerequisites

- **Docker** with the **Compose v2** plugin (`docker compose …`).
- For any GPU deployment (inference-on-GPU or training): an NVIDIA GPU, recent
  drivers, and the **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**
  so Docker can expose the GPU to containers.
- CPU-only deployments need **none** of the NVIDIA pieces.

---

## Quick start

```bash
git clone https://github.com/cobryan05/whisker-watcher.git
cd whisker-watcher

./setup.sh                     # answer a few questions → writes .env
docker compose up -d --build   # build + start what .env selected
```

Then open the UI at **http://localhost:7999**.

`setup.sh` detects whether you have a GPU and asks what this host should run
(everything / frontend+inference only / training only / a dev workspace), then
writes a `.env`. Even on a GPU host it always offers CPU inference — e.g. to keep
a slim frontend or reserve VRAM for training. `docker compose` reads `.env`
automatically, so from then on plain compose commands work. To reconfigure later,
re-run `./setup.sh`.

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
  no CUDA pulled or run.
- **One GPU server, everything** — "Everything on one box" → `mono` (or `split`).
- **Local frontend + remote GPU trainer** — run `app` locally with
  `TRAINING_SERVER_HOST` pointed at the GPU box, and `training` on that box.

---

## Configuration (`.env`)

`setup.sh` generates `.env`; `.env.example` documents every option. The key knobs:

| Variable | Values | Purpose |
|----------|--------|---------|
| `COMPOSE_PROFILES` | see table above | which containers run |
| `APP_BASE` | `slim-runtime` (default) / `cuda-runtime` | CPU vs GPU-inference base for the `app` image |
| `ORT_REQUIREMENTS` | `requirements-cpu.txt` / `requirements-gpu.txt` | onnxruntime flavor (pair with `APP_BASE`) |
| `COMPOSE_FILE` | add `:docker-compose.gpu.yaml` | give the `app` container a GPU (for onnxruntime-gpu) |
| `TRAINING_SERVER_HOST` | hostname | where the frontend reaches training (defaults to the `training` service) |
| `APPUSER_UID` / `APPUSER_GID` | ints | match host user for clean bind-mount ownership |

### Recording storage

NVR clips, event snapshots and event timelapses are written under
`/data/recordings/`, inside the `./data:/data` bind mount. To keep recordings on
another disk, move **that whole mount** (`- /mnt/nas/whiskerwatcher:/data` in
`docker-compose.override.yml`). Everything beneath it belongs to the app, so this
is always safe.

> ⚠️ **Never mount a directory containing anything else at an individual
> recordings root.** Space is reclaimed two ways. Deleting a recording's own file
> follows its database row, so it only ever removes files the app created — safe
> anywhere. But the **orphan sweep** walks the recordings roots and deletes every
> file no database row points at, once older than `RECORDINGS_ORPHAN_GRACE_S`
> (default 900 s). It has no extension filter and cannot distinguish your files
> from its own; its only safety argument is that those directories belong to the
> app alone. Point one at a shared media folder and it will be emptied.

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

Host directories bind-mounted into the stack: `./data` (database), `./models`
(trained/loaded models), `./image_datasets` (images), `./conf`, `./logs`.

---

## Development

Run `./setup.sh` and choose **"Development workspace"**. It offers two layouts:

- **Single container** (`COMPOSE_PROFILES=dev-mono`, service `mono-dev`) — everything
  plus dev tooling in one container. This is the old `combined-dev`.
- **Split** (`COMPOSE_PROFILES=dev`, services `app-dev` + `training-dev`).

The repo is bind-mounted at `/project`, so code changes are live. Bring it up and
run the suites inside the workspace container (`mono-dev` for single-container,
`app-dev` for split):

```bash
./setup.sh                       # -> Development workspace
docker compose up -d --build

# Unit tests
docker compose exec mono-dev python -m pytest

# Browser (Playwright) tests
docker compose exec mono-dev bash -c 'cd /project/playwright_tests && python -m pytest'
```

The dev images bundle the toolchain (formatters, pytest, Playwright + Chromium).

Personal, machine-specific dev mounts (extra datasets, host `~/.claude`, etc.) go
in a **gitignored** `docker-compose.override.yml`, which compose merges
automatically. The shipped override targets the `app-dev` service; point it at
`mono-dev` if you use the single-container dev layout.

---

## Build-host note

The slim `app` image is CUDA-free to **run and pull**, but *building* the
training / `mono` images compiles the CUDA/torch wheelhouse. For CPU-only
targets, build the images on a capable host (or in CI) and pull them, rather than
building the GPU images on the lean host itself.

---

## License

[GNU AGPL-3.0](LICENSE).
