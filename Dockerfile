# syntax=docker/dockerfile:1.7
# ARGs must come before first FROM statement to be used in a FROM statement and re-declared after a FROM
ARG MEDIAMTX_VERSION=1.12.0
ARG HTMX_VERSION=1.9.12
ARG CUDA_VERSION=12.8.1
ARG PICO_CSS_VERSION=2.1.1
ARG KONVA_VERSION=9.3.20
ARG PYTORCH_VERSION=2.7.1
ARG PYTORCH_CUDA_VERSION=12.8

# Shared Build Arguments across various target stages
ARG APPUSER_UID=1001
ARG APPUSER_GID=1001


# =========================================================
#  Base MediaMTX Image
# =========================================================
FROM bluenviron/mediamtx:${MEDIAMTX_VERSION} AS mediamtx


# =========================================================
#  App Base (Shared Environment Core)
# =========================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04 AS app-base

ARG MEDIAMTX_VERSION
ARG HTMX_VERSION
ARG PICO_CSS_VERSION
ARG KONVA_VERSION

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

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

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
  && rm -rf /tmp/konva /tmp/konva.tgz

COPY --from=mediamtx /mediamtx /app/mediamtx

# Copy required foundational application files
COPY conf /conf
COPY templates ./templates
COPY static ./static
COPY apps ./apps


# =========================================================
#  Unified Builder (Isolated SDK & Dependency Compiling)
# =========================================================
# Derived from PyTorch base to enforce Python 3.11/cp311 compatibility across target components
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${PYTORCH_CUDA_VERSION}-cudnn9-runtime AS builder

ARG MEDIAMTX_VERSION

WORKDIR /app

# 1. Install build tools needed for wheel compilation
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git

# Install runtime deps + openapi-generator-cli (bundles its own JDK — no system Java needed)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt openapi-generator-cli[jdk4py]

# 2. Cache production training wheels using Python 3.11 engine targeting host constraints.
# Kept ahead of `COPY apps ./apps` below so app-code-only changes (the common dev-loop case)
# don't invalidate this step and force a full torch/wheel re-resolve.
COPY requirements-training.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --wheel-dir=/app/training_wheels --prefer-binary -r requirements-training.txt

# Generate mediamtx_client wheel — needed by relay_buffer_server at import time and by app-base at runtime
RUN curl -fsSL https://raw.githubusercontent.com/bluenviron/mediamtx/v${MEDIAMTX_VERSION}/apidocs/openapi.yaml \
        -o /tmp/mediamtx-openapi.yaml \
    && openapi-generator-cli generate -i /tmp/mediamtx-openapi.yaml -g python \
        -o /tmp/mediamtx_client_src --package-name mediamtx_client \
    && pip wheel --no-deps /tmp/mediamtx_client_src -w ./openapi_wheels/ \
    && pip install --no-cache-dir /tmp/mediamtx_client_src \
    && rm -rf /tmp/mediamtx-openapi.yaml /tmp/mediamtx_client_src

# Copy application layers so parsing logic can read server entrypoints
COPY apps ./apps
COPY generate_openapi_modules.sh generate_alembic_revision.sh ./

# 3. Run reorganized OpenAPI generator script to build internal dependency wheelhouse
RUN ./generate_openapi_modules.sh ./openapi_jsons --wheels-out ./openapi_wheels/


# =========================================================
#  Inference App (Production Core Web Server)
# =========================================================
FROM app-base AS inference-app

ARG APPUSER_UID
ARG APPUSER_GID

# Install all internal API client wheels from the builder stage
COPY --from=builder /app/openapi_wheels/ /tmp/openapi_wheels/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install /tmp/openapi_wheels/*.whl && rm -rf /tmp/openapi_wheels

# Copy the generated internal API type sheets for the UI layer
COPY --from=builder /app/static/js/api-types/ /app/static/js/api-types/

# Create non-root user
RUN groupadd -g $APPUSER_GID appuser && \
    useradd -m -u $APPUSER_UID -g appuser -s /bin/bash appuser

# Set runtime folder ownership safely
RUN mkdir -p /training-artifacts /training-cache && chown -R appuser:appuser /app /logs /conf /training-artifacts /training-cache

USER appuser

EXPOSE 1935 7999 8001 8002 8003 8004 8554 9001 9997

CMD ["/usr/bin/supervisord", "-c", "/conf/supervisord-inference.conf"]


# =========================================================
#  Training App (Production Machine Learning Component)
# =========================================================
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${PYTORCH_CUDA_VERSION}-cudnn9-runtime AS training-app

ARG APPUSER_UID
ARG APPUSER_GID

ENV DEBIAN_FRONTEND=noninteractive

# opencv runtime libs: ultralytics/libreyolo pull the GUI `opencv-python`
# wheel (not the headless build), whose `import cv2` needs libGL / glib / xcb
# shared objects that the pytorch base image does not ship. Without these,
# `import cv2` — and therefore `import ultralytics`/`libreyolo` — fails outright
# in a stock training-app container. (The inference CUDA base happens to carry
# these already, which is why the combined image masked this gap.)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
  && apt-get install -y --no-install-recommends supervisor curl \
     libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1

WORKDIR /app

# Install training dependencies from the builder's compiled wheelhouse
# (Guaranteed cp311 wheels, keeping this stage 100% clean of compilers)
COPY --from=builder /app/training_wheels /tmp/training_wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-index --find-links=/tmp/training_wheels /tmp/training_wheels/*.whl \
  && rm -rf /tmp/training_wheels

# Install only the db_client wheel needed for database tracking during training runs
COPY --from=builder /app/openapi_wheels/db_client*.whl /tmp/wheels/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install /tmp/wheels/*.whl && rm -rf /tmp/wheels

# Copy application runtime structure
COPY conf/supervisord-training.conf /conf/supervisord-training.conf
COPY templates ./templates
COPY apps ./apps

# Mirror the application user configuration exactly to handle shared volume writes safely
RUN groupadd -g $APPUSER_GID appuser && \
    useradd -m -u $APPUSER_UID -g appuser -s /bin/bash appuser

RUN mkdir -p /logs /data /training-artifacts /training-cache && chown -R appuser:appuser /app /logs /conf /data /training-artifacts /training-cache

USER appuser

EXPOSE 8005 9001

CMD ["/usr/bin/supervisord", "-c", "/conf/supervisord-training.conf"]


# =========================================================
#  Shared Dev Tooling (network-heavy CLI installs, built once)
#  `dev` and `training-dev` branch off two different base images
#  (CUDA vs PyTorch) so Docker can't share cache between them.
#  This stage installs the identical CLI toolchain (claude, nvm/node,
#  uv, serena-agent) exactly once; both dev variants COPY the result
#  instead of re-running the same curl/network installers twice.
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
#  Dev-Extra Layer (Inference Workspaces)
# =========================================================
FROM inference-app AS dev

USER root

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
  && apt-get install -y --no-install-recommends bash curl git

COPY requirements-dev.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt \
  && echo 'cd /project/apps 2>/dev/null || cd /app/apps' >> /home/appuser/.bashrc

# System dependencies for Playwright testing tools
RUN playwright install-deps chromium

# Pull in prebuilt CLI tooling (claude, nvm/node, uv, serena) from the shared devtools stage
COPY --from=devtools --chown=appuser:appuser /home/appuser/.local /home/appuser/.local
COPY --from=devtools --chown=appuser:appuser /home/appuser/.nvm /home/appuser/.nvm

USER appuser

ENV NVM_DIR="/home/appuser/.nvm"
ENV PATH="/home/appuser/.local/bin:$PATH"

# Browser engine for Playwright (inference-dev-specific, not part of the shared toolchain)
RUN playwright install chromium


# =========================================================
#  Combined Dev Layer (inference + training in one container)
#  Dev counterpart of the `combined` stage: layers the training
#  wheelhouse onto the inference `dev` image so one container runs
#  every service with full dev tooling. Selected by the `dev-combined`
#  compose profile.
# =========================================================
FROM dev AS combined-dev

USER root

# Reuse the training wheelhouse already built in `builder` instead of
# re-resolving/downloading torch et al. from PyPI a second time.
COPY --from=builder /app/training_wheels /tmp/training_wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-index --find-links=/tmp/training_wheels /tmp/training_wheels/*.whl \
  && rm -rf /tmp/training_wheels

USER appuser

EXPOSE 8005


# =========================================================
#  Combined App (inference + training in one PRODUCTION container)
#  Non-dev counterpart of `combined-dev`: layers the training
#  wheelhouse onto the production inference-app image (no dev
#  tooling). Runs every service via supervisord-monolith.conf.
#  This is the image docker-compose.monolith.yaml targets for a
#  real single-container production deployment.
# =========================================================
FROM inference-app AS combined

USER root

# Reuse the training wheelhouse already built in `builder` instead of
# re-resolving/downloading torch et al. from PyPI a second time.
COPY --from=builder /app/training_wheels /tmp/training_wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-index --find-links=/tmp/training_wheels /tmp/training_wheels/*.whl \
  && rm -rf /tmp/training_wheels

USER appuser

EXPOSE 8005

CMD ["/usr/bin/supervisord", "-c", "/conf/supervisord-monolith.conf"]


# =========================================================
#  Training Dev Layer (ML Engineering Workspaces)
# =========================================================
FROM training-app AS training-dev

USER root

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
  && apt-get install -y --no-install-recommends bash curl git build-essential

COPY requirements-dev.txt .
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