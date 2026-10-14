# syntax=docker/dockerfile:1.7
# ARGs must come before first FROM statement to be used in a FROM statement and re-declared after a FROM
ARG MEDIAMTX_VERSION=1.21.2
ARG HTMX_VERSION=1.9.12
ARG CUDA_VERSION=12.8.1
ARG PICO_CSS_VERSION=2.1.1
ARG KONVA_VERSION=9.3.20
ARG PYTORCH_VERSION=2.7.1
ARG PYTORCH_CUDA_VERSION=12.8

# ---------------------------------------------------------------------------
# Runtime base selection for the `app` image (frontend + inference group).
#
# The app image runs the frontend/db/tasks/relay group and, optionally, the
# inference server. The inference runtime path is pure-ONNX, so the app image
# never needs torch. Two runtime bases are supported, chosen per deployment:
#
#   APP_BASE=slim-runtime  ORT_REQUIREMENTS=requirements-cpu.txt
#       CPU-only, no CUDA. Smallest image. For low-end / CPU-only hosts.
#
#   APP_BASE=cuda-runtime  ORT_REQUIREMENTS=requirements-gpu.txt
#       GPU-accelerated inference via onnxruntime-gpu, WITHOUT the training
#       torch stack. For GPU hosts that want fast inference but not training.
#
# The heavy training stack lives only in the `training-app` / `combined`
# targets, which are built on the PyTorch base. Defaults below give the slim
# CPU image so a non-GPU host never pulls CUDA.
# ---------------------------------------------------------------------------
ARG APP_BASE=slim-runtime
ARG ORT_REQUIREMENTS=requirements-cpu.txt

# Shared Build Arguments across various target stages
ARG APPUSER_UID=1000
ARG APPUSER_GID=1000


# =========================================================
#  Base MediaMTX Image
# =========================================================
FROM bluenviron/mediamtx:${MEDIAMTX_VERSION} AS mediamtx


# =========================================================
#  OpenAPI / Client Builder (torch-free)
#  Generates the internal API client wheels (db/inference/relay/training/tasks)
#  + the mediamtx client wheel + the TypeScript api-types for the UI.
#
#  This runs on a SLIM base with NO torch/CUDA: the trainer modules (the only
#  torch importers) are imported lazily at trainer-registration time, not at
#  FastAPI-app import / .openapi() time, so schema generation needs only the
#  base runtime deps. Keeping this stage slim means building the CPU `app`
#  image pulls zero CUDA/torch.
# =========================================================
FROM python:3.11-slim AS openapi-builder

ARG MEDIAMTX_VERSION

ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git ca-certificates

WORKDIR /app

# Base runtime deps (+ CPU onnxruntime) so the app modules import cleanly during
# schema generation, plus openapi-generator-cli (bundles its own JDK — no system Java).
COPY requirements.txt requirements-cpu.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt -r requirements-cpu.txt openapi-generator-cli[jdk4py]

# Generate mediamtx_client wheel — needed by stream_server at import time
# and by the app image at runtime.
#
# The spec lives under api/ since 1.19.2 (it 404s at the old apidocs/ path), and it
# needs patching before generation — see scripts/patch_mediamtx_spec.py for which
# schema and why. The patch exits non-zero if the spec no longer matches, so this
# fails the build rather than producing a client that breaks on first use.
#
# BOTH the path above and that patch are pinned to MEDIAMTX_VERSION's schema and
# may need to change, or to go away entirely, on the next bump. Re-read the
# script's header before raising that version — a bump is not a one-line change.
COPY scripts/patch_mediamtx_spec.py ./scripts/
RUN curl -fsSL https://raw.githubusercontent.com/bluenviron/mediamtx/v${MEDIAMTX_VERSION}/api/openapi.yaml \
        -o /tmp/mediamtx-openapi.yaml \
    && python3 ./scripts/patch_mediamtx_spec.py /tmp/mediamtx-openapi.yaml \
    && openapi-generator-cli generate -i /tmp/mediamtx-openapi.yaml -g python \
        -o /tmp/mediamtx_client_src --package-name mediamtx_client \
    && pip wheel --no-deps /tmp/mediamtx_client_src -w ./openapi_wheels/ \
    && pip install --no-cache-dir /tmp/mediamtx_client_src \
    && rm -rf /tmp/mediamtx-openapi.yaml /tmp/mediamtx_client_src

# Copy application layers so the generator can introspect server entrypoints
COPY apps ./apps
COPY generate_openapi_modules.sh generate_alembic_revision.sh ./

# Build the internal dependency wheelhouse (openapi_wheels/) + UI api-types (static/js/api-types/)
RUN ./generate_openapi_modules.sh ./openapi_jsons --wheels-out ./openapi_wheels/


# =========================================================
#  Training Wheelhouse (PyTorch base — heavy, build-time only)
#  Compiles/collects the production training wheels (torch, ultralytics,
#  libreyolo, timm, ...) as cp311 wheels. Kept separate from the openapi
#  builder so the CPU `app` build never triggers a torch download.
# =========================================================
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${PYTORCH_CUDA_VERSION}-cudnn9-runtime AS training-wheels

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git

# Cache production training wheels using the base image's Python 3.11 engine.
COPY requirements-training.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --wheel-dir=/app/training_wheels --prefer-binary -r requirements-training.txt


# =========================================================
#  Runtime OS base: SLIM (CPU-only, no CUDA)
#  python:3.11-slim already ships Python 3.11 + pip; we only add the OS-level
#  runtime libs the app processes need (ffmpeg for streams, supervisor to run
#  the process group, curl for healthchecks/asset fetch).
# =========================================================
FROM python:3.11-slim AS slim-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg supervisor curl ca-certificates libglib2.0-0


# =========================================================
#  Runtime OS base: CUDA (GPU inference, no torch)
#  nvidia/cuda cudnn-runtime provides the CUDA + cuDNN libs onnxruntime-gpu's
#  CUDA execution provider needs, but ships no Python — so we install Python
#  3.11 (deadsnakes) here, mirroring the process group's runtime needs.
# =========================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04 AS cuda-runtime

ENV DEBIAN_FRONTEND=noninteractive

# NOTE: pointing python3 at 3.11 breaks tools built against the distro's python3-apt
# (e.g. add-apt-repository) — any such call in this or a downstream stage must happen
# before the update-alternatives switch below.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
  && apt-get install -y --no-install-recommends software-properties-common curl supervisor ffmpeg \
  && add-apt-repository -y ppa:deadsnakes/ppa \
  && apt-get update \
  && apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-distutils \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
  && ln -s /usr/bin/python3 /usr/bin/python \
  && curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3.11 \
  && python3 -m pip install --upgrade pip


# =========================================================
#  App (frontend + inference group)
#  Built on the selected runtime base (slim-runtime or cuda-runtime). Runs the
#  image_tagging / db / tasks / relay / mediamtx group and, optionally, the
#  inference server. Never contains torch. onnxruntime flavor is chosen by the
#  ORT_REQUIREMENTS build arg (CPU vs GPU), paired with APP_BASE.
# =========================================================
FROM ${APP_BASE} AS app

ARG HTMX_VERSION
ARG PICO_CSS_VERSION
ARG KONVA_VERSION
ARG ORT_REQUIREMENTS
ARG APPUSER_UID
ARG APPUSER_GID

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Created before anything is copied in, so every COPY below can set ownership as it
# writes (COPY --chown) instead of a recursive chown afterwards. That `chown -R /app`
# cost ~50s of every rebuild and, worse, rewrote every file it touched into a fresh
# layer — paying for a second copy of the whole application tree in the image purely
# to change its owner.
RUN groupadd -g $APPUSER_GID appuser && \
    useradd -m -u $APPUSER_UID -g appuser -s /bin/bash appuser

# Runtime Python deps + the selected onnxruntime flavor.
COPY requirements.txt requirements-cpu.txt requirements-gpu.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt -r ${ORT_REQUIREMENTS}

# Fetch minimized versions of frontend UI assets
RUN mkdir -p /logs /app/static/js /app/static/css \
  && curl -L -o /app/static/js/htmx.min.js https://unpkg.com/htmx.org@${HTMX_VERSION}/dist/htmx.min.js \
  && curl -L -o /app/static/css/pico.min.css https://cdn.jsdelivr.net/npm/@picocss/pico@${PICO_CSS_VERSION}/css/pico.blue.min.css \
  \
  # Download Konva tarball and extract only min.js + typings for JSDoc
  && curl -L -o /tmp/konva.tgz https://registry.npmjs.org/konva/-/konva-${KONVA_VERSION}.tgz \
  && mkdir -p /tmp/konva \
  && tar -xzf /tmp/konva.tgz -C /tmp/konva --strip-components=1 \
  && cp /tmp/konva/konva.min.js /app/static/js/ \
  && mkdir -p /app/static/js/konva-types \
  && cp -r /tmp/konva/lib/* /app/static/js/konva-types/ \
  && rm -rf /tmp/konva /tmp/konva.tgz \
  # These were written by root above, and are the only files in /app that no COPY
  # --chown covers. Scoped rather than a blanket -R over /app: it is a handful of
  # vendored assets, so it costs nothing and stays cheap as the app grows.
  && chown -R appuser:appuser /app /logs

COPY --from=mediamtx --chown=appuser:appuser /mediamtx /app/mediamtx

# Copy required foundational application files
COPY --chown=appuser:appuser conf /conf
COPY --chown=appuser:appuser templates ./templates
COPY --chown=appuser:appuser static ./static
COPY --chown=appuser:appuser apps ./apps

# Install all internal API client wheels from the openapi builder
COPY --from=openapi-builder /app/openapi_wheels/ /tmp/openapi_wheels/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install /tmp/openapi_wheels/*.whl && rm -rf /tmp/openapi_wheels

# Copy the generated internal API type sheets for the UI layer
COPY --from=openapi-builder --chown=appuser:appuser /app/static/js/api-types/ /app/static/js/api-types/

# Directory ownership only — the contents were already placed with COPY --chown, so
# this stays a constant-time step however large the application gets. The directories
# themselves still need setting because a COPY creates its destination as root.
RUN mkdir -p /training-artifacts /training-cache \
  && chown appuser:appuser /app /conf /logs /training-artifacts /training-cache

USER appuser

EXPOSE 1935 7999 8001 8002 8003 8004 8554 9001 9997

CMD ["/usr/bin/supervisord", "-c", "/conf/supervisord-inference.conf"]


# =========================================================
#  Training App (standalone ML component, PyTorch base)
#  Stands alone: talks to the frontend only over HTTP (db_client). Optional,
#  and may run on a separate/remote GPU host serving one or more frontends.
# =========================================================
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${PYTORCH_CUDA_VERSION}-cudnn9-runtime AS training-app

ARG APPUSER_UID
ARG APPUSER_GID

ENV DEBIAN_FRONTEND=noninteractive

# opencv runtime libs: ultralytics/libreyolo pull the GUI `opencv-python`
# wheel (not the headless build), whose `import cv2` needs libGL / glib / xcb
# shared objects that the pytorch base image does not ship. Without these,
# `import cv2` — and therefore `import ultralytics`/`libreyolo` — fails outright
# in a stock training-app container.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
  && apt-get install -y --no-install-recommends supervisor curl \
     libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1

WORKDIR /app

# Mirror the application user configuration exactly to handle shared volume writes
# safely. Created before the copies so each can set ownership as it writes — see the
# same note in the `app` stage for why a trailing `chown -R /app` is worth avoiding.
RUN groupadd -g $APPUSER_GID appuser && \
    useradd -m -u $APPUSER_UID -g appuser -s /bin/bash appuser

# Install training dependencies from the compiled wheelhouse
# (Guaranteed cp311 wheels, keeping this stage 100% clean of compilers)
COPY --from=training-wheels /app/training_wheels /tmp/training_wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-index --find-links=/tmp/training_wheels /tmp/training_wheels/*.whl \
  && rm -rf /tmp/training_wheels

# Install only the db_client wheel needed for database tracking during training runs
COPY --from=openapi-builder /app/openapi_wheels/db_client*.whl /tmp/wheels/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install /tmp/wheels/*.whl && rm -rf /tmp/wheels

# Copy application runtime structure
COPY --chown=appuser:appuser conf/supervisord-training.conf /conf/supervisord-training.conf
COPY --chown=appuser:appuser templates ./templates
COPY --chown=appuser:appuser apps ./apps

# Runtime mount points only — empty directories, so this stays a constant-time step.
# /app is included because the pytorch base populated it before appuser existed.
RUN mkdir -p /logs /data /training-artifacts /training-cache \
  && chown appuser:appuser /app /logs /conf /data /training-artifacts /training-cache

USER appuser

EXPOSE 8005 9001

CMD ["/usr/bin/supervisord", "-c", "/conf/supervisord-training.conf"]


# =========================================================
#  Combined App (inference + training in one PRODUCTION container)
#  Layers the training wheelhouse onto the `app` image so a single container
#  runs every service. Only meaningful when `app` is built on the CUDA base
#  (APP_BASE=cuda-runtime, ORT_REQUIREMENTS=requirements-gpu.txt) so torch can
#  use the GPU. Runs everything via supervisord-monolith.conf.
# =========================================================
FROM app AS combined

USER root

# Reuse the prebuilt training wheelhouse instead of re-resolving torch et al.
COPY --from=training-wheels /app/training_wheels /tmp/training_wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-index --find-links=/tmp/training_wheels /tmp/training_wheels/*.whl \
  && rm -rf /tmp/training_wheels

USER appuser

EXPOSE 8005

CMD ["/usr/bin/supervisord", "-c", "/conf/supervisord-monolith.conf"]


# =========================================================
#  Shared Dev Tooling (network-heavy CLI installs, built once)
#  Installs the CLI toolchain (claude, nvm/node, uv, serena-agent) exactly
#  once on a neutral base; every dev variant COPYs the result instead of
#  re-running the same network installers.
# =========================================================
FROM ubuntu:22.04 AS devtools

ARG APPUSER_UID
ARG APPUSER_GID

ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git

RUN groupadd -g $APPUSER_GID appuser && \
    useradd -m -u $APPUSER_UID -g appuser -s /bin/bash appuser

USER appuser
WORKDIR /home/appuser

ENV NVM_DIR="/home/appuser/.nvm"
ENV PATH="/home/appuser/.local/bin:$PATH"

# The node/npm/npx symlinks into ~/.local/bin matter: nvm only puts node on
# PATH when nvm.sh is sourced by an interactive shell, but serena's language
# servers (pyright, typescript) are spawned non-interactively via MCP and need
# `node` resolvable from plain PATH — ~/.local/bin already is in every dev stage.
RUN curl -fsSL https://claude.ai/install.sh | bash \
  && mkdir -p "$NVM_DIR" \
  && curl -LsSf https://astral.sh/uv/install.sh | sh \
  && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash \
  && /bin/bash -c " \
       source \$NVM_DIR/nvm.sh \
       && nvm install --lts \
       && nvm use --lts \
       && nvm alias default 'lts/*' \
       && NODE_BIN=\$(dirname \$(nvm which default)) \
       && ln -sf \$NODE_BIN/node \$NODE_BIN/npm \$NODE_BIN/npx /home/appuser/.local/bin/ \
     " \
  && uv tool install -p 3.13 serena-agent


# =========================================================
#  Dev Layer (app workspace: frontend + inference)
#  Dev overlay on the `app` image. Inherits whatever base `app` was built on
#  (slim for CPU dev, cuda for GPU dev).
# =========================================================
FROM app AS dev-base

USER root

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
  && apt-get install -y --no-install-recommends bash curl git

COPY --chown=appuser:appuser requirements-dev.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt \
  && echo 'cd /project/apps 2>/dev/null || cd /app/apps' >> /home/appuser/.bashrc

# System dependencies for Playwright testing tools. `install-deps` shells out to
# apt-get itself, so it gets the same cache mounts as every other apt step here —
# without them it re-downloads the whole browser dependency set on each rebuild.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    playwright install-deps chromium

USER appuser

# Browser engine for Playwright. Installed on the shared base because CI needs it
# as much as a workstation does.
RUN playwright install chromium


# =========================================================
#  Dev Layer (workstation)
#  dev-base plus the interactive CLI tooling. Split out so the CI image can take
#  the test toolchain without the several gigabytes of editor/agent tooling that
#  only a human workspace uses — a GitHub-hosted runner ships 14 GB of disk.
# =========================================================
FROM dev-base AS dev

# Pull in prebuilt CLI tooling (claude, nvm/node, uv, serena) from the shared devtools stage
COPY --from=devtools --chown=appuser:appuser /home/appuser/.local /home/appuser/.local
COPY --from=devtools --chown=appuser:appuser /home/appuser/.nvm /home/appuser/.nvm

ENV NVM_DIR="/home/appuser/.nvm"
ENV PATH="/home/appuser/.local/bin:$PATH"


# =========================================================
#  CI Layer (GitHub Actions, CPU-only)
#  Both suites in one image, with no CUDA anywhere: ten files under tests/ import
#  torch or ultralytics, so a plain `dev` image cannot collect them, and the CUDA
#  wheelhouse that `combined-dev` uses is far too large for a hosted runner.
#
#  requirements-training.txt stays the single source of what the training stack
#  is. The two substitutions are mechanical, so a package added there reaches CI
#  without a second list to remember:
#    * torch/torchvision come from PyTorch's CPU index at the pins that file sets.
#    * onnxruntime-gpu is dropped; the app layer already installed the CPU build,
#      and both provide the same `onnxruntime` module.
# =========================================================
FROM dev-base AS ci

USER root

COPY requirements-training.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -E '^(torch|torchvision)==' requirements-training.txt > /tmp/torch-pins.txt \
  && pip install --index-url https://download.pytorch.org/whl/cpu -r /tmp/torch-pins.txt \
  && grep -vE '^(torch==|torchvision==|onnxruntime-gpu)' requirements-training.txt > /tmp/train-cpu.txt \
  && pip install -r /tmp/train-cpu.txt \
  && rm -f /tmp/torch-pins.txt /tmp/train-cpu.txt

# ultralytics writes a settings file on first import and warns, every run, when it
# cannot reach ~/.config. Naming a writable directory is cheaper than making one.
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

# compose grants these to every runtime profile as bind mounts, so the suite has
# never had to ask for them. A `docker run` has no compose file, and the thumbnail
# cache writes under /data at import-time defaults — baking them in keeps the CI
# invocation to one mount (the checkout) instead of three.
RUN mkdir -p /data /app/image_datasets \
  && chown appuser:appuser /data /app/image_datasets

USER appuser


# =========================================================
#  Combined Dev Layer (inference + training in one container)
#  Dev counterpart of `combined`: layers the training wheelhouse onto the
#  `dev` image so one container runs every service with full dev tooling.
#  Only meaningful with APP_BASE=cuda-runtime (torch needs the GPU).
# =========================================================
FROM dev AS combined-dev

USER root

COPY --from=training-wheels /app/training_wheels /tmp/training_wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-index --find-links=/tmp/training_wheels /tmp/training_wheels/*.whl \
  && rm -rf /tmp/training_wheels

USER appuser

EXPOSE 8005


# =========================================================
#  Training Dev Layer (ML engineering workspace)
# =========================================================
FROM training-app AS training-dev

USER root

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
  && apt-get install -y --no-install-recommends bash curl git build-essential

COPY --chown=appuser:appuser requirements-dev.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt \
  && echo 'cd /project/apps 2>/dev/null || cd /app/apps' >> /home/appuser/.bashrc

# Pull in prebuilt CLI tooling (claude, nvm/node, uv, serena) from the shared devtools stage
COPY --from=devtools --chown=appuser:appuser /home/appuser/.local /home/appuser/.local
COPY --from=devtools --chown=appuser:appuser /home/appuser/.nvm /home/appuser/.nvm

# Ensure user execution is uniform and permissions match perfectly
USER appuser

ENV NVM_DIR="/home/appuser/.nvm"
ENV PATH="/home/appuser/.local/bin:$PATH"
