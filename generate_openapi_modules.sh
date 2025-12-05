#!/bin/bash
set -euo pipefail

# Optional JSON output directory parameter
JSON_OUT_DIR="${1:-.}"
mkdir -p "$JSON_OUT_DIR"

# Create temporary directory
TMP_DIR=$(mktemp -d)

# Generate the JSONs
python -m apps.generate_openapi_jsons backend "$JSON_OUT_DIR"

# Generate the python API modules from the JSONs into the temp dir
openapi-generator-cli generate -i "$JSON_OUT_DIR/relay_buffer_server_openapi.json" -g python -o "$TMP_DIR/relay_buffer_client" --package-name relay_buffer_client
openapi-generator-cli generate -i "$JSON_OUT_DIR/inference_server_openapi.json" -g python -o "$TMP_DIR/inference_client" --package-name inference_client
openapi-generator-cli generate -i "$JSON_OUT_DIR/tasks_server_openapi.json" -g python -o "$TMP_DIR/tasks_client" --package-name tasks_client
openapi-generator-cli generate -i "$JSON_OUT_DIR/db_server_openapi.json" -g python -o "$TMP_DIR/db_client" --package-name db_client

# Install the generated clients
pip install --no-cache-dir "$TMP_DIR"/relay_buffer_client "$TMP_DIR"/inference_client "$TMP_DIR"/tasks_client "$TMP_DIR"/db_client

# Generate the Javascript API for the web front end as well
python -m apps.generate_openapi_jsons frontend "$JSON_OUT_DIR"
python -m apps.generate_jsdoc_from_openapi_json "$JSON_OUT_DIR/image_tagging_server_openapi.json" "static/js/api-types.js"

rm -rf "$TMP_DIR"
