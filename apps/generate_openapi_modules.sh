#!/bin/bash
set -euo pipefail

# Create temporary directory
TMP_DIR=$(mktemp -d)

# Generate the JSONs (from project root)
python -m apps.generate_openapi_jsons

# Generate the modules from the JSONs into the temp dir
openapi-generator-cli generate -i relay_buffer_server_openapi.json -g python -o "$TMP_DIR/relay_buffer_client" --package-name relay_buffer_client
openapi-generator-cli generate -i inference_server_openapi.json -g python -o "$TMP_DIR/inference_client" --package-name inference_client
openapi-generator-cli generate -i tasks_server_openapi.json -g python -o "$TMP_DIR/tasks_client" --package-name tasks_client
openapi-generator-cli generate -i db_server_openapi.json -g python -o "$TMP_DIR/db_client" --package-name db_client

# Fix license field in all pyproject.toml files
sed -i 's/license = "NoLicense"/license = "Apache-2.0"/' "$TMP_DIR"/*/pyproject.toml

# Install the generated clients
pip install --no-cache-dir "$TMP_DIR"/relay_buffer_client "$TMP_DIR"/inference_client "$TMP_DIR"/tasks_client "$TMP_DIR"/db_client

rm -rf "$TMP_DIR"