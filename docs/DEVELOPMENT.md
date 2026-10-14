# Developing PurrView

Nothing here is needed to *run* PurrView — see the
[README](../README.md#getting-started) for that. This is the contributor's side.

## A dev workspace

Run `./setup.sh` and pick the development workspace. There are two layouts:
single container (`COMPOSE_PROFILES=dev-mono`, service `mono-dev`) and split
(`COMPOSE_PROFILES=dev`, services `app-dev` and `training-dev`). The repo is
bind-mounted at `/project`, so code changes are live.

The dev images bundle the toolchain: formatters, pytest, Playwright and Chromium.
Machine-specific mounts go in `docker-compose.override.yml`, as described in the
[README](../README.md#letting-it-see-your-files), but pointed at `mono-dev` or
`app-dev` rather than the production service.

## Tests

```bash
scripts/test.sh                  # unit and integration tests
scripts/test.sh playwright       # browser tests
scripts/test.sh both             # both; non-zero if either fails
```

`scripts/test.sh` runs pytest inside the container under `nice` and `ionice`,
which matters if the same box is also serving the live app.

## Everything else

```bash
# Format. From the repo root inside the container, never from /app
docker exec -w /project purrview_mono_dev \
  sh -c 'black apps/ tests/ playwright_tests/ tests_training/ && isort apps/ tests/ playwright_tests/ tests_training/'

bash generate_openapi_modules.sh              # regenerate and install the API clients
./generate_alembic_revision.sh "describe it"  # after a DB schema change
./scripts/install-git-hooks.sh                # pre-commit formatting checks
```

## Services

One Python codebase (`apps/`) exposed as several FastAPI services, run together
by supervisord:

| Service | Port | Role |
|---------|------|------|
| `image_tagging_server` | 7999 | Web UI. A pure proxy and SPA host, no business logic |
| `inference_server` | 8001 | ONNX inference (YOLOv5/v7/v8), GPU pinning with auto-unload |
| `tasks_server` | 8002 | Long-running jobs, including the watchers |
| `db_server` | 8003 | SQLite store, thumbnails, search |
| `stream_server` | 8004 | Owns mediamtx: camera fan-out, leases, the buffer, clip extraction |
| `training_server` | 8005 | Training. Optional, GPU |

Those collapse into two images. **`app`** is the frontend plus inference, built
slim and CPU-only by default — no CUDA, no torch, inference is pure ONNX at
runtime — and optionally built on a CUDA base for GPU inference. **`training`** is
the heavy torch/ultralytics node: optional, always GPU, and standalone, since it
talks to the frontend only over HTTP and can therefore live on another box.

## Build hosts

The slim `app` image is CUDA-free to pull and run, but *building* the training or
`mono` images compiles the CUDA and torch wheelhouse. For a CPU-only target,
build elsewhere and pull, rather than building the GPU images on the lean box.
