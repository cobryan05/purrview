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
  && curl -L -o /app/static/js/konva.min.js https://cdn.jsdelivr.net/npm/konva@${KONVA_VERSION}/konva.min.js

COPY --from=mediamtx /mediamtx /app/mediamtx

# Download MediaMTX OpenAPI spec and generate Python client
RUN curl -L -o /openapi.yaml \
  https://raw.githubusercontent.com/bluenviron/mediamtx/v${MEDIAMTX_VERSION}/apidocs/openapi.yaml \
  && openapi-generator-cli generate -i /openapi.yaml -g python -o /tmp/openapi-client --package-name mediamtx_client \
  && pip install --no-cache-dir /tmp/openapi-client \
  && rm -rf /tmp/openapi-client

# Dependencies for CV2 to display images over X11
RUN apt-get update \
  && apt-get install -y --no-install-recommends libgl1 libxext6 libxrender1 libsm6 libx11-6 \
  && rm -rf /var/lib/apt/lists/*

# Copy required files
COPY conf /conf
COPY templates ./templates
COPY static ./static
COPY apps ./apps

# Generates python clients for internal apps
RUN python -m apps.generate_openapi_jsons \
  && openapi-generator-cli generate -i relay_buffer_server_openapi.json -g python -o /tmp/relay_buffer_client --package-name relay_buffer_client \
  && sed -i 's/license = "NoLicense"/license = "Apache-2.0"/' /tmp/relay_buffer_client/pyproject.toml \
  && pip install --no-cache-dir /tmp/relay_buffer_client \
  && rm -rf /tmp/relay_buffer_client \
  && openapi-generator-cli generate -i inference_server_openapi.json -g python -o /tmp/inference_client --package-name inference_client \
  && sed -i 's/license = "NoLicense"/license = "Apache-2.0"/' /tmp/inference_client/pyproject.toml \
  && pip install --no-cache-dir /tmp/inference_client \
  && rm -rf /tmp/inference_client \
  && openapi-generator-cli generate -i tasks_server_openapi.json -g python -o /tmp/tasks_client --package-name tasks_client \
  && sed -i 's/license = "NoLicense"/license = "Apache-2.0"/' /tmp/tasks_client/pyproject.toml \
  && pip install --no-cache-dir /tmp/tasks_client \
  && rm -rf /tmp/tasks_client

EXPOSE 8000 8001 8554 8888 1935 9001 9997

CMD ["/usr/bin/supervisord", "-c", "/conf/supervisord.conf"]