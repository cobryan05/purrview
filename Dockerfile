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

RUN apt-get update \
  && apt-get install -y --no-install-recommends software-properties-common curl supervisor ffmpeg python3-pip \
  && ln -s /usr/bin/python3 /usr/bin/python \
  && python3 -m pip install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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

# Download MediaMTX OpenAPI spec and generate Python client
RUN curl -L -o /openapi.yaml \
  https://raw.githubusercontent.com/bluenviron/mediamtx/v${MEDIAMTX_VERSION}/apidocs/openapi.yaml \
  && openapi-generator-cli generate -i /openapi.yaml -g python -o /tmp/openapi-client --package-name mediamtx_client \
  && pip install --no-cache-dir /tmp/openapi-client \
  && rm -rf /tmp/openapi-client

# Copy required foundational application files
COPY conf /conf
COPY templates ./templates
COPY static ./static
COPY apps ./apps


# =========================================================
#  OpenAPI Client Builder (Isolated SDK Compiling)
# =========================================================
FROM app-base AS openapi-builder

# Generates python wheels and client JS/TS typings for downstream deployment
COPY generate_openapi_modules.sh generate_alembic_revision.sh ./
RUN ./generate_openapi_modules.sh ./openapi_jsons --wheels-out ./openapi_wheels/


# =========================================================
#  Inference App (Production Core Web Server)
# =========================================================
FROM app-base AS inference-app

ARG APPUSER_UID
ARG APPUSER_GID

# Install all internal API client wheels from the builder stage
COPY --from=openapi-builder /app/openapi_wheels/ /tmp/openapi_wheels/
RUN pip install --no-cache-dir /tmp/openapi_wheels/*.whl && rm -rf /tmp/openapi_wheels

# Copy the generated internal API type sheets for the UI layer
COPY --from=openapi-builder /app/static/js/api-types/ /app/static/js/api-types/

# Create non-root user
RUN groupadd -g $APPUSER_GID appuser && \
    useradd -m -u $APPUSER_UID -g appuser -s /bin/bash appuser

# Set runtime folder ownership safely
RUN chown -R appuser:appuser /app /logs /conf

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

RUN apt-get update \
  && apt-get install -y --no-install-recommends supervisor curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-training.txt .
RUN pip install --no-cache-dir -r requirements-training.txt

# Install only the db_client wheel needed for database tracking during training runs
COPY --from=openapi-builder /app/openapi_wheels/db_client*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

# Copy application runtime structure
COPY conf/supervisord-training.conf /conf/supervisord-training.conf
COPY templates ./templates
COPY apps ./apps

# Mirror the application user configuration exactly to handle shared volume writes safely
RUN groupadd -g $APPUSER_GID appuser && \
    useradd -m -u $APPUSER_UID -g appuser -s /bin/bash appuser

RUN mkdir -p /logs /data && chown -R appuser:appuser /app /logs /conf /data

USER appuser

EXPOSE 8005 9001

CMD ["/usr/bin/supervisord", "-c", "/conf/supervisord-training.conf"]


# =========================================================
#  Dev-Extra Layer (Inference Workspaces)
# =========================================================
FROM inference-app AS dev

USER root

RUN apt-get update \
  && apt-get install -y --no-install-recommends bash curl git \
  && rm -rf /var/lib/apt/lists/*

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt \
  && echo 'cd /project/apps 2>/dev/null || cd /app/apps' >> /home/appuser/.bashrc

# System dependencies for Playwright testing tools
RUN playwright install-deps chromium

USER appuser

ENV NVM_DIR="/home/appuser/.nvm"
ENV PATH="/home/appuser/.local/bin:$PATH"

# Browser engines, agent suites, and CLI tooling
RUN playwright install chromium \
  && curl -fsSL https://claude.ai/install.sh | bash \
  && mkdir -p "$NVM_DIR" \
  && curl -LsSf https://astral.sh/uv/install.sh | sh \
  && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash \
  && /bin/bash -c " \
       source \$NVM_DIR/nvm.sh \
       && nvm install --lts \
       && nvm use --lts \
       && nvm alias default 'lts/*' \
     " \
  && uv tool install -p 3.13 serena-agent


# =========================================================
#  Training Dev Layer (ML Engineering Workspaces)
# =========================================================
FROM training-app AS training-dev

USER root

RUN apt-get update \
  && apt-get install -y --no-install-recommends bash curl git \
  && rm -rf /var/lib/apt/lists/*

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt \
  && echo 'cd /project/apps 2>/dev/null || cd /app/apps' >> /home/appuser/.bashrc

# Ensure user execution is uniform and permissions match perfectly
USER appuser

ENV NVM_DIR="/home/appuser/.nvm"
ENV PATH="/home/appuser/.local/bin:$PATH"

# Install everything inside appuser profile scope
RUN curl -fsSL https://claude.ai/install.sh | bash \
  && mkdir -p "$NVM_DIR" \
  && curl -LsSf https://astral.sh/uv/install.sh | sh \
  && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash \
  && /bin/bash -c " \
       source \$NVM_DIR/nvm.sh \
       && nvm install --lts \
       && nvm use --lts \
       && nvm alias default 'lts/*' \
     " \
  && uv tool install -p 3.13 serena-agent