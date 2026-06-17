#!/bin/bash
set -euo pipefail

# Optional JSON output directory parameter
JSON_OUT_DIR="${1:-.}"
mkdir -p "$JSON_OUT_DIR"

# Parse optional --wheels-out DIR flag (supports both --wheels-out=DIR and --wheels-out DIR)
WHEELS_OUT_DIR=""
prev=""
for arg in "$@"; do
    if [[ "$arg" == --wheels-out=* ]]; then
        WHEELS_OUT_DIR="${arg#*=}"
    elif [[ "$prev" == "--wheels-out" ]]; then
        WHEELS_OUT_DIR="$arg"
    fi
    prev="$arg"
done

if [[ -n "$WHEELS_OUT_DIR" ]]; then
    mkdir -p "$WHEELS_OUT_DIR"
fi

# Create temporary directory
TMP_DIR=$(mktemp -d)

# In --wheels-out mode: build a wheel AND install (install is still needed so that
# subsequent generate_openapi_jsons imports work within this same build step).
# Without --wheels-out: install only (original behaviour).
_install_or_wheel() {
    local src_dir="$1"
    if [[ -n "$WHEELS_OUT_DIR" ]]; then
        pip wheel --no-deps "$src_dir" -w "$WHEELS_OUT_DIR"
    fi
    pip install --no-cache-dir "$src_dir"
}

# Generate documentation for the backends. Must install in order of dependencies
python -m apps.generate_openapi_jsons db relay inference "$JSON_OUT_DIR"

# Generate and install the python API modules from the JSONs
openapi-generator-cli generate -i "$JSON_OUT_DIR/relay_buffer_server_openapi.json" -g python -o "$TMP_DIR/relay_buffer_client" --package-name relay_buffer_client
openapi-generator-cli generate -i "$JSON_OUT_DIR/inference_server_openapi.json" -g python -o "$TMP_DIR/inference_client" --package-name inference_client
openapi-generator-cli generate -i "$JSON_OUT_DIR/db_server_openapi.json" -g python -o "$TMP_DIR/db_client" --package-name db_client

_install_or_wheel "$TMP_DIR/relay_buffer_client"
_install_or_wheel "$TMP_DIR/inference_client"
_install_or_wheel "$TMP_DIR/db_client"

# Now generate the training server (depends on db client being installed)
python -m apps.generate_openapi_jsons training "$JSON_OUT_DIR"
openapi-generator-cli generate -i "$JSON_OUT_DIR/training_server_openapi.json" -g python -o "$TMP_DIR/training_client" --package-name training_client
_install_or_wheel "$TMP_DIR/training_client"

# Now generate the tasks server (depends on db/inference/relay/training clients being installed)
python -m apps.generate_openapi_jsons tasks "$JSON_OUT_DIR"
openapi-generator-cli generate -i "$JSON_OUT_DIR/tasks_server_openapi.json" -g python -o "$TMP_DIR/tasks_client" --package-name tasks_client
_install_or_wheel "$TMP_DIR/tasks_client"

# Generate the Javascript API for the web front end as well
python -m apps.generate_openapi_jsons image_tagging "$JSON_OUT_DIR"

# Generate the combined API
python -m apps.merge_openapi_jsons "$JSON_OUT_DIR" "combined-api.json"
openapi-generator-cli generate -i "$JSON_OUT_DIR/combined-api.json" -g typescript-fetch -o ./static/js/api-types --skip-validate-spec --additional-properties=modelOnly=true,typescriptThreePlus=true

rm -rf "$TMP_DIR"
