# ARGs must come before first FROM statement to be used in a FROM statement and re-declared after a FROM
ARG MEDIAMTX_VERSION=1.12.0
ARG HTMX_VERSION=1.9.12
ARG CUDA_VERSION=12.8.1
ARG PICO_CSS_VERSION=2.1.1
ARG KONVA_VERSION=9.3.20

FROM bluenviron/mediamtx:${MEDIAMTX_VERSION} AS mediamtx
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04 AS base

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

# Fetch minimized version of HTMX
RUN mkdir -p /logs /app/static/js /app/static/css \
  && curl -L -o /app/static/js/htmx.min.js https://unpkg.com/htmx.org@${HTMX_VERSION}/dist/htmx.min.js \
  && curl -L -o /app/static/css/pico.min.css https://cdn.jsdelivr.net/npm/@picocss/pico@${PICO_CSS_VERSION}/css/pico.blue.min.css \
  \
  # Download Konva tarball and extract only min.js + .d.ts
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

# Copy required files
COPY conf /conf
COPY templates ./templates
COPY static ./static
COPY apps ./apps

# Generates python and js clients for internal apps
COPY generate_openapi_modules.sh ./
RUN ./generate_openapi_modules.sh ./openapi_jsons

# Create non-root user
ARG APPUSER_UID=1001
ARG APPUSER_GID=1001
RUN groupadd -g $APPUSER_GID appuser && \
    useradd -m -u $APPUSER_UID -g appuser -s /bin/bash appuser

# Set permissions on relevant folders
RUN chown -R appuser:appuser /app /logs /conf

USER appuser

EXPOSE 1935 7999 8001 8002 8003 8004 8554 9001 9997

CMD ["/usr/bin/supervisord", "-c", "/conf/supervisord.conf"]


# -------------------------
#   Dev-extra layer only
# -------------------------
FROM base AS dev

USER root

# Development dependencies (CV2 over X11, git, Claude Code)
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  # libgl1 libxext6 libxrender1 libsm6 libx11-6 \
    git \
  && rm -rf /var/lib/apt/lists/* \
  && pip install isort \
  && echo 'cd /project/apps 2>/dev/null || cd /app/apps' >> /home/appuser/.bashrc

USER appuser

# Install claude-cli as appuser; binary lands in ~/.local/bin, credentials in ~/.claude (bind-mounted from host)
RUN curl -fsSL https://claude.ai/install.sh | bash