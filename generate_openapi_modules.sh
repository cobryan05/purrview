#!/bin/bash
set -euo pipefail

# Optional JSON output directory parameter
JSON_OUT_DIR="${1:-.}"
mkdir -p "$JSON_OUT_DIR"

# Create temporary directory
TMP_DIR=$(mktemp -d)

# Generate documentation for the backends. Must install in order of dependencies
python -m apps.generate_openapi_jsons db relay inference "$JSON_OUT_DIR"

# Generate and install the python API modules from the JSONs
openapi-generator-cli generate -i "$JSON_OUT_DIR/relay_buffer_server_openapi.json" -g python -o "$TMP_DIR/relay_buffer_client" --package-name relay_buffer_client
openapi-generator-cli generate -i "$JSON_OUT_DIR/inference_server_openapi.json" -g python -o "$TMP_DIR/inference_client" --package-name inference_client
openapi-generator-cli generate -i "$JSON_OUT_DIR/db_server_openapi.json" -g python -o "$TMP_DIR/db_client" --package-name db_client

pip install --no-cache-dir "$TMP_DIR"/relay_buffer_client "$TMP_DIR"/inference_client "$TMP_DIR"/db_client

# Now generate the tasks server
python -m apps.generate_openapi_jsons tasks "$JSON_OUT_DIR"
openapi-generator-cli generate -i "$JSON_OUT_DIR/tasks_server_openapi.json" -g python -o "$TMP_DIR/tasks_client" --package-name tasks_client
pip install --no-cache-dir "$TMP_DIR"/tasks_client

# Generate the Javascript API for the web front end as well
python -m apps.generate_openapi_jsons image_tagging "$JSON_OUT_DIR"
python -m apps.generate_jsdoc_from_openapi_json "$JSON_OUT_DIR/image_tagging_server_openapi.json" "static/js/api-types.js"

rm -rf "$TMP_DIR"
